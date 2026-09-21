#!/usr/bin/env python3
"""DAS (not WFC) of the plane-wave IQ with predicted / true sound-speed maps.

Homogeneous 1500 m/s uses the same constant-speed DAS as the training
condition.  Predicted and ground-truth maps use straight-ray slowness
integrals (no refraction): still delay-and-sum, just with a spatially
varying delay law.

    python scripts/beamform_das.py \
        --samples /data/zhuangyang/openbreast_pw_iq/dataset/samples \
        --cmaps out/pred_flow/cmaps --out out/das --limit 8
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import geometry as G
from data.das import das_angles_const, das_angles_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--cmaps", required=True)
    p.add_argument("--out", default="out/das")
    p.add_argument("--names", default=None)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--which", default="const,pred,true")
    p.add_argument("--n-path", type=int, default=48,
                   help="samples along each straight ray for 2-D maps")
    p.add_argument("--single-angle-idx", type=int, default=6,
                   help="0 deg is index 6 of -12:2:12")
    p.add_argument("--n-panels", type=int, default=0,
                   help="PNG panels to write (0 = all processed samples)")
    return p.parse_args()


def log_compress(img, dr_db=45.0, ref=None):
    env = np.abs(img)
    ref = np.percentile(env, 99.9) if ref is None else ref
    v = np.maximum(env / (ref + 1e-20), 10 ** (-dr_db / 20.0))
    return 20.0 * np.log10(v)


def roi_metrics(env, xg, zg, z_lo=3e-3, z_hi=45e-3):
    mx = np.ones(xg.size, bool)
    mz = (zg >= z_lo) & (zg <= z_hi)
    e = env[np.ix_(mx, mz)]
    mu = e.mean()
    ipr = float((e ** 2).mean() / (mu ** 2 + 1e-20))
    en = e / (mu + 1e-20)
    sharp = float((np.diff(en, axis=0) ** 2).mean()
                  + (np.diff(en, axis=1) ** 2).mean())
    px = (xg >= -5e-3) & (xg <= 5e-3)
    pz = (zg >= 25e-3) & (zg <= 35e-3)
    p = env[np.ix_(px, pz)]
    snr = float(p.mean() / (p.std() + 1e-12))
    return {"ipr": ipr, "sharpness": sharp, "speckle_snr": snr}


def save_panel(path, cmaps, images, xg, zg, name, mode="coh"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("const", "pred", "true") if k in images]
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8.2), squeeze=False)
    x_mm = (xg[0] * 1e3, xg[-1] * 1e3)
    z_mm = (zg[-1] * 1e3, zg[0] * 1e3)
    for j, k in enumerate(keys):
        ax = axes[0][j]
        c = cmaps[k]
        ax.imshow(c["c"].T,
                  extent=[c["cx"][0] * 1e3, c["cx"][-1] * 1e3,
                          c["cz"][-1] * 1e3, c["cz"][0] * 1e3],
                  cmap="turbo", vmin=1420, vmax=1560, aspect="equal")
        ax.set_title(f"c(x,z) {k}")
        ax.set_xlim(*x_mm)
        ax.set_ylim(*z_mm)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax = axes[1][j]
        img = images[k][mode]
        ax.imshow(log_compress(img).T,
                  extent=[xg[0] * 1e3, xg[-1] * 1e3,
                          zg[-1] * 1e3, zg[0] * 1e3],
                  cmap="gray", vmin=-45, vmax=0, aspect="equal")
        m = roi_metrics(np.abs(img), xg, zg)
        ax.set_title(f"DAS {k} ({mode})  ipr={m['ipr']:.2f}")
        ax.set_xlim(*x_mm)
        ax.set_ylim(*z_mm)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
    fig.suptitle(f"{name}   [DAS {mode} compounding]", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compound(A, single_idx):
    coh = np.abs(np.sum(A, axis=0))
    inc = np.sum(np.abs(A), axis=0)
    single = np.abs(A[single_idx])
    return {"coh": coh, "inc": inc, "single": single}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.names:
        names = [s for s in args.names.split(",") if s]
    else:
        names = sorted(f[:-4] for f in os.listdir(args.cmaps)
                       if f.startswith("sample_") and f.endswith(".npz"))
        if args.limit:
            names = names[:args.limit]
    if not names:
        raise SystemExit("no samples to process")
    print(f"{len(names)} samples: {names[:4]}{' ...' if len(names) > 4 else ''}")

    which = [w for w in args.which.split(",") if w]
    xi, zi = G.x_grid(), G.z_grid()
    n_panels = len(names) if args.n_panels <= 0 else args.n_panels
    rows = []

    for k, name in enumerate(names):
        raw = np.load(os.path.join(args.samples, name + ".npz"))
        pred = np.load(os.path.join(args.cmaps, name + ".npz"))
        iq = np.asarray(raw["iqdata"])
        xe = np.asarray(raw["elpos"])[0]
        angles = np.asarray(raw["angles"], np.float32)
        fs, fd = float(raw["fs"]), float(raw["fd"])
        cx = np.asarray(pred["cx"])
        cz = np.asarray(pred["cz"])

        hyp_c = {}
        if "const" in which:
            hyp_c["const"] = np.full((cx.size, cz.size), 1500.0, np.float32)
        if "pred" in which:
            hyp_c["pred"] = np.asarray(pred["c_pred"], np.float32)
        if "true" in which:
            hyp_c["true"] = np.asarray(pred["c_gt"], np.float32)

        images, cmaps, metrics = {}, {}, {}
        t0 = time.perf_counter()
        for key, cmap in hyp_c.items():
            if key == "const":
                A = das_angles_const(iq, 1500.0, xe, xi, zi, angles, fs, fd)
            else:
                A = das_angles_map(iq, cmap, cx, cz, xe, xi, zi, angles,
                                   fs, fd, n_path=args.n_path)
            images[key] = compound(A, args.single_angle_idx)
            cmaps[key] = {"c": cmap, "cx": cx, "cz": cz}
            metrics[key] = {f"{m}_{mk}": mv for m, img in
                            (("coh", images[key]["coh"]),
                             ("inc", images[key]["inc"]),
                             ("single", images[key]["single"]))
                            for mk, mv in roi_metrics(img, xi, zi).items()}
        dt = time.perf_counter() - t0
        row = dict(name=name, n_hyp=len(images), time_s=round(dt, 1))
        for key in images:
            for mk, mv in metrics[key].items():
                row[f"{key}_{mk}"] = round(mv, 5)
        if "pred" in images and "true" in images:
            a, b = images["pred"]["coh"], images["true"]["coh"]
            row["corr_coh_pred_true"] = float(
                np.corrcoef(a.ravel(), b.ravel())[0, 1])
        if "const" in images and "pred" in images:
            a, b = images["const"]["coh"], images["pred"]["coh"]
            row["corr_coh_const_pred"] = float(
                np.corrcoef(a.ravel(), b.ravel())[0, 1])
        rows.append(row)
        print(f"  [{k+1}/{len(names)}] {name}: " + "  ".join(
            f"{key}_ipr={metrics[key]['coh_ipr']:.2f}" for key in images)
            + f"  ({dt:.1f}s)")

        out_npz = dict(xg=xi, zg=zi)
        for key, d in images.items():
            for mode, img in d.items():
                out_npz[f"env_{key}_{mode}"] = img.astype(np.float32)
            out_npz[f"cmap_{key}"] = hyp_c[key]
        np.savez_compressed(os.path.join(args.out, name + ".npz"), **out_npz)
        if k < n_panels:
            save_panel(os.path.join(args.out, name + ".png"), cmaps, images,
                       xi, zi, name, mode="coh")
            save_panel(os.path.join(args.out, name + "_single.png"), cmaps,
                       {kk: {**vv, "coh": vv["single"]}
                        for kk, vv in images.items()},
                       xi, zi, name, mode="coh")

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out}/*.npz, *.png, summary.csv")
    for mode in ("coh", "inc", "single"):
        if all(f"pred_{mode}_ipr" in r and f"const_{mode}_ipr" in r
               for r in rows):
            dp = np.mean([r[f"pred_{mode}_ipr"] - r[f"const_{mode}_ipr"]
                          for r in rows])
            print(f"mean ipr({mode}) pred-const = {dp:+.3f}")
    if "true" in which and "pred" in which:
        dt_ = np.mean([r["true_coh_ipr"] - r["pred_coh_ipr"] for r in rows])
        print(f"mean ipr(coh) true-pred = {dt_:+.3f}")


if __name__ == "__main__":
    main()
