#!/usr/bin/env python3
"""Preprocess the OpenBreastUS plane-wave IQ dataset into the training cache.

For every ``sample_XXXXX.npz`` this computes the phase-preserving multi-group
DAS condition tensor and resamples the ground-truth sound-speed map onto the
prediction grid, then stores one small npz per sample (float16 condition,
float32 target).

    python scripts/prepare_cache.py --samples /data/.../dataset/samples \
        --cache cache --workers 32 --speeds 1450,1500,1550 --n-subap 4
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import geometry as G
from data.das import build_condition, condition_channels, resample_target

META_KEYS = ("c_roi_mean", "c_breast_mean", "fibro_frac", "roi_frac")


def _one(args):
    path, out_dir, cond_cfg = args
    name = os.path.basename(path)[:-4]
    out = os.path.join(out_dir, name + ".npz")
    if os.path.isfile(out):
        return name, "cached", 0.0
    t0 = time.perf_counter()
    with np.load(path) as d:
        iq = np.asarray(d["iqdata"])
        xe = np.asarray(d["elpos"])[0]
        angles = np.asarray(d["angles"], np.float32)
        xi, zi = G.x_grid(), G.z_grid()
        cond = build_condition(iq, xe, angles, xi, zi,
                               speeds=cond_cfg["speeds"],
                               ref_speed=cond_cfg["ref_speed"],
                               n_subap=cond_cfg["n_subap"],
                               groups=cond_cfg["groups"],
                               fs=float(d["fs"]), fc=float(d["fd"]))
        c_gt = resample_target(d["c_true"], d["cx_true"], d["cz_true"],
                               float(d["xc_m"]), float(d["z_face_m"]),
                               xi, zi)
        meta = {k: np.float32(d[k]) for k in META_KEYS if k in d}
    np.savez_compressed(out, cond=cond.astype(np.float16), c_gt=c_gt, **meta)
    return name, "ok", time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True,
                    help="directory with sample_*.npz")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--speeds", default="1450,1500,1550",
                    help="assumed beamforming speeds for the 'full' group")
    ap.add_argument("--ref-speed", type=float, default=1500.0,
                    help="reference speed for the per-angle/sub-aperture groups")
    ap.add_argument("--n-subap", type=int, default=4)
    ap.add_argument("--groups", default="full,angle,subap")
    ap.add_argument("--limit", type=int, default=0, help="process at most N")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cond_cfg = {
        "speeds": [float(v) for v in args.speeds.split(",") if v],
        "ref_speed": float(args.ref_speed),
        "n_subap": int(args.n_subap),
        "groups": [g for g in args.groups.split(",") if g],
    }
    n_ch = condition_channels(G.N_ANGLE, len(cond_cfg["speeds"]),
                              cond_cfg["n_subap"], cond_cfg["groups"])
    os.makedirs(args.cache, exist_ok=True)
    files = sorted(os.path.join(args.samples, f)
                   for f in os.listdir(args.samples)
                   if f.startswith("sample_") and f.endswith(".npz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no sample_*.npz under {args.samples}")
    if args.overwrite:
        for f in files:
            out = os.path.join(args.cache, os.path.basename(f)[:-4] + ".npz")
            if os.path.isfile(out):
                os.remove(out)

    with open(os.path.join(args.cache, "meta.json"), "w") as f:
        json.dump({
            "samples_dir": os.path.abspath(args.samples),
            "condition": cond_cfg,
            "cond_channels": n_ch,
            "nx": G.NX, "nz": G.NZ, "dx": G.DX, "dz": G.DZ,
            "x0": float(G.X0), "z0": float(G.Z0),
            "c_ref": G.C_REF, "rho_scale": G.RHO_SCALE,
        }, f, indent=1)
    print(f"condition: {n_ch} channels, groups={cond_cfg['groups']}, "
          f"speeds={cond_cfg['speeds']}, ref={cond_cfg['ref_speed']}, "
          f"n_subap={cond_cfg['n_subap']}")

    t0 = time.perf_counter()
    n_ok = n_cached = 0
    with Pool(args.workers) as pool:
        for k, (name, status, dt) in enumerate(
                pool.imap_unordered(_one, [(p, args.cache, cond_cfg)
                                           for p in files])):
            n_ok += status == "ok"
            n_cached += status == "cached"
            if (k + 1) % 20 == 0 or k + 1 == len(files):
                el = time.perf_counter() - t0
                print(f"  {k+1}/{len(files)}  ({el:.0f}s, "
                      f"{el/(k+1):.2f}s/sample)", flush=True)
    print(f"done: {n_ok} computed, {n_cached} already cached, "
          f"{time.perf_counter()-t0:.0f}s total -> {args.cache}")

    # Write the deterministic split once, from the COMPLETE list, so that a
    # partially populated cache can never freeze an incomplete split.
    split_file = os.path.join(args.cache, "splits.json")
    if not os.path.isfile(split_file):
        names = sorted(os.path.basename(p)[:-4] for p in files)
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(names))
        n_val = max(1, int(round(0.1 * len(names))))
        n_test = max(1, int(round(0.1 * len(names))))
        splits = {
            "train": [names[i] for i in idx[:len(names) - n_val - n_test]],
            "val": [names[i] for i in idx[len(names) - n_val - n_test:
                                           len(names) - n_test]],
            "test": [names[i] for i in idx[len(names) - n_test:]],
        }
        with open(split_file, "w") as f:
            json.dump(splits, f, indent=1)
        print(f"splits: train={len(splits['train'])} val={len(splits['val'])} "
              f"test={len(splits['test'])} -> {split_file}")


if __name__ == "__main__":
    main()
