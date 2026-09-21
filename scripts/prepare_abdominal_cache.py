#!/usr/bin/env python3
"""Build an independent, manifest-split abdominal SoS cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import geometry as G
from data.abdominal_rf import read_sample, build_condition, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/data/zhuangyang/SIMUABERV1/simu_kwave_aber_v2")
    p.add_argument("--cache", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit-per-split", type=int, default=0)
    p.add_argument("--chunk", type=int, default=2048)
    args = p.parse_args()
    torch.set_num_threads(4)
    root, cache = Path(args.dataset_root), Path(args.cache)
    manifest = root / "output/dataset/manifest.jsonl"
    raw = manifest.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    source_splits, splits, selected = {}, dict(train=[], val=[], test=[]), []
    seen = set()
    for r in records:
        split = {"validation": "val"}.get(r["split"], r["split"])
        if split not in splits or r["status"] != "complete":
            raise ValueError("Unknown split or incomplete sample")
        source = r["anatomy_source_path"]
        if source in source_splits and source_splits[source] != split:
            raise ValueError("Anatomy source overlaps splits: " + source)
        source_splits[source] = split
        name = r["sample_id"]
        if name in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(name)
        if args.limit_per_split and len(splits[split]) >= args.limit_per_split:
            continue
        if not (manifest.parent / (name + ".h5")).is_file():
            raise FileNotFoundError(name)
        splits[split].append(name)
        selected.append(r)
    metadata = dict(schema="abdominal_sos_v1", cx=G.x_grid().tolist(), cz=G.z_grid().tolist(),
        config=dict(cond_channels=20, grid_shape=[128, 160], speeds=[1450,1500,1550],
            n_subap=4, angles_deg=[-6,0,6], timing="finite_aperture_stored_launch_nominal_burst_center",
            lateral_label_shift="minus_half_cell_for_even_kwave_grid", axial_origin="second_medium_row",
            rf_transform="hilbert_absolute_time_demodulation", direct_wave_removed=False,
            normalization="group_rms_depth_ge_3mm", limit_per_split=args.limit_per_split),
        dataset_root=str(root.resolve()), manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_splits=source_splits, note="Source-file split, not verified independent subjects; FOV depth 0.15-47.85 mm.")
    cache.mkdir(parents=True, exist_ok=True)
    for name, content in (("meta.json", metadata), ("splits.json", splits)):
        path = cache / name
        if path.exists():
            if json.loads(path.read_text()) != content:
                raise ValueError("Incompatible existing cache: " + str(path))
        else:
            path.write_text(json.dumps(content, indent=2) + "\n")
    start = time.monotonic()
    for i, r in enumerate(selected, 1):
        dest = cache / (r["sample_id"] + ".npz")
        if dest.exists():
            with np.load(dest) as saved:
                if saved["cond"].shape != (20,128,160) or saved["c_gt"].shape != (128,160):
                    raise ValueError("Invalid existing cache sample: " + str(dest))
            continue
        source = manifest.parent / (r["sample_id"] + ".h5")
        if hashlib.sha256(source.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("Source checksum mismatch: " + str(source))
        sample = read_sample(source, root)
        if sample["meta"]["sample_id"] != r["sample_id"] or sample["meta"]["split"] != r["split"]:
            raise ValueError("Manifest and HDF5 metadata disagree")
        cond = build_condition(sample, args.device, args.chunk)
        values = targets(sample)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, cond=cond, **values, meta=json.dumps(sample["meta"]))
        os.replace(tmp, dest)
        if i <= 4 or i % 10 == 0 or i == len(selected):
            print(f"{i}/{len(selected)} {dest.name} cond={cond.shape} "
                  f"c=[{values['c_gt'].min():.1f},{values['c_gt'].max():.1f}] "
                  f"wall_pixels={values['wall_mask'].sum()} elapsed={time.monotonic()-start:.1f}s", flush=True)
    (cache / "complete.json").write_text(json.dumps(dict(count=len(selected), split_counts={k:len(v) for k,v in splits.items()}), indent=2))


if __name__ == "__main__":
    main()
