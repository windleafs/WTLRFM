#!/usr/bin/env python3
"""Evaluate abdominal checkpoints and export finite SoS ensembles and metrics."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.abdominal_dataset import (AbdominalSoSDataset, CAVEAT, aggregate_metrics,
                                    cache_fingerprint, sample_metrics, training_mean_map)
from data import geometry as G
from models import SoSMultiplicativeFlowNetwork
from scripts.train_abdominal import prediction_batches


def save_panels(out, cases):
    if not cases:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low = min(min(a["c_gt"][a["valid_mask"]].min(), pred[a["valid_mask"]].min())
              for _, a, pred, _ in cases)
    high = max(max(a["c_gt"][a["valid_mask"]].max(), pred[a["valid_mask"]].max())
               for _, a, pred, _ in cases)
    if high <= low:
        high = low + 1
    error_max = max(1.0, max(float(np.abs(pred - a["c_gt"])[a["valid_mask"]].max())
                             for _, a, pred, _ in cases))
    std_max = max(1.0, max(float(std[a["valid_mask"]].max()) for _, a, _, std in cases))
    for name, a, pred, std in cases:
        x, z = a["cx"] * 1000, a["cz"] * 1000
        dx, dz = (x[1] - x[0]) / 2, (z[1] - z[0]) / 2
        extent = [x[0] - dx, x[-1] + dx, z[-1] + dz, z[0] - dz]
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), constrained_layout=True)
        panels = ((a["c_gt"], "Truth [m/s]", "turbo", low, high),
                  (pred, "Prediction [m/s]", "turbo", low, high),
                  (pred - a["c_gt"], "Prediction - truth [m/s]", "coolwarm", -error_max, error_max),
                  (std, "Ensemble std [m/s]", "viridis", 0, std_max))
        for ax, (image, title, cmap, vmin, vmax) in zip(axes, panels):
            colors = plt.get_cmap(cmap).copy()
            colors.set_bad("0.75")
            visible = np.ma.array(image, mask=~a["valid_mask"])
            im = ax.imshow(visible.T, extent=extent, origin="upper", aspect="equal",
                           cmap=colors, vmin=vmin, vmax=vmax)
            if a["valid_mask"].any() and not a["valid_mask"].all():
                ax.contour(x, z, a["valid_mask"].T, levels=[.5], colors="white", linewidths=.5)
            ax.set(title=title, xlabel="Probe-relative x [mm]", ylabel="Depth [mm]")
            fig.colorbar(im, ax=ax, fraction=.046)
        fig.suptitle(name + " | gray: excluded pixels; white outline: valid tissue")
        fig.savefig(out / f"{name}.png", dpi=130)
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--plot-limit", type=int, default=6)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.batch, args.n_samples, args.ode_steps) < 1 or min(args.limit, args.plot_limit) < 0:
        raise ValueError("Batch/sample/step counts must be positive; limits nonnegative")
    torch.set_num_threads(4)
    dataset = AbdominalSoSDataset(args.cache, args.split, args.limit)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    data_meta = blob.get("data_meta", {})
    fingerprint = cache_fingerprint(dataset.root, dataset.meta, dataset.splits)
    if data_meta.get("cache_fingerprint") != fingerprint or data_meta.get("cache_meta") != dataset.meta:
        raise ValueError("Checkpoint/cache metadata or fingerprint mismatch")
    if data_meta.get("normalization") != {"c_ref": G.C_REF, "rho_scale": G.RHO_SCALE}:
        raise ValueError("Checkpoint geometry normalization mismatch")
    names = data_meta.get("train_names")
    if not names or len(set(names)) != len(names) or not set(names).issubset(dataset.splits["train"]):
        raise ValueError("Checkpoint must provide train-only baseline provenance")
    train = AbdominalSoSDataset(args.cache, "train")
    train.names = names
    # No held-out labels are used to estimate this baseline.
    mean_map, mean_count = training_mean_map(train)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("Prediction output exists; choose a new --out")
    device = torch.device(args.device)
    model = SoSMultiplicativeFlowNetwork.from_checkpoint(args.ckpt).to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    out.mkdir(parents=True)
    (out / "cmaps").mkdir()
    if args.plot_limit:
        (out / "plots").mkdir()
    seed = blob.get("config", {}).get("train", {}).get("eval_seed", 12345)
    rows, cases = [], []
    for batch, predictions, stds in prediction_batches(model, loader, device, args.ode_steps, args.n_samples, seed):
        for i, (pred, std) in enumerate(zip(predictions, stds)):
            name = batch["name"][i]
            a = dataset.load_arrays(name)
            provenance = {"sample": a["meta"], "checkpoint": str(Path(args.ckpt).resolve()),
                          "checkpoint_epoch": blob.get("epoch"), "split": args.split,
                          "cache_fingerprint": fingerprint, "n_samples": args.n_samples,
                          "ode_steps": args.ode_steps, "noise_seed": seed, "std_ddof": 0,
                          "units": "m/s", "coordinate_units": "m", "caveat": CAVEAT}
            np.savez_compressed(out / "cmaps" / f"{name}.npz", pred=pred, std=std,
                                c_pred=pred, c_std=std, truth=a["c_gt"], c_gt=a["c_gt"], valid_mask=a["valid_mask"],
                                wall_mask=a["wall_mask"], segmentation=a["segmentation"],
                                cx=a["cx"], cz=a["cz"], meta=json.dumps(provenance))
            for method, estimate in (("model", pred), ("constant1500", np.full_like(pred, 1500)),
                                     ("constant1540", np.full_like(pred, 1540)), ("train_mean", mean_map)):
                rows.append({"name": name, "method": method,
                             **sample_metrics(estimate, a["c_gt"], a["valid_mask"], a["wall_mask"])})
            if len(cases) < args.plot_limit:
                cases.append((name, a, pred, std))
            print(f"predicted {name}", flush=True)
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"split": args.split, "n_cases": len(dataset), "checkpoint": str(Path(args.ckpt).resolve()),
               "checkpoint_epoch": blob.get("epoch"), "n_samples": args.n_samples, "ode_steps": args.ode_steps,
               "noise_seed": seed, "std_ddof": 0, "cache_fingerprint": fingerprint,
               "metrics": {method: aggregate_metrics([r for r in rows if r["method"] == method])
                           for method in ("model", "constant1500", "constant1540", "train_mean")},
               "train_mean_names": names, "train_mean_fallback": "global valid-tissue training mean where count=0",
               "aggregation": "Pixel-pooled tissue/wall MAE/RMSE; sample-mean legacy metrics; pooled whole-map RMSE",
               "caveat": CAVEAT}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(out / "train_mean.npz", mean=mean_map, count=mean_count,
                        train_names=np.asarray(names), cx=np.asarray(dataset.meta["cx"]), cz=np.asarray(dataset.meta["cz"]))
    save_panels(out / "plots", cases)
    print(json.dumps(summary["metrics"], indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
