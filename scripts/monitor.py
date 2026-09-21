#!/usr/bin/env python3
"""Training monitor: loss/val curves + prediction panels from a checkpoint.

    python scripts/monitor.py --log out/flow/train.log --ckpt out/flow/last.pth \
        --cache cache --out out/monitor --n 3

Produces ``<out>/curves.png`` and ``<out>/pred_<name>.png`` so the run can be
inspected while it is still training.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data.geometry as G
from data.obpw_dataset import load_cached_sample, load_split
from models import SoSMultiplicativeFlowNetwork

TRAIN_RE = re.compile(r"ep\s+(\d+) train L=([\d.]+) sigma_u=([\d.]+)")
VAL_RE = re.compile(r"val mae=([\d.]+) roi_mae=([\d.]+) "
                    r"roi_mean_err=([\d.]+) corr=(-?[\d.]+)")


def parse_log(path):
    train, val = [], []
    ep = None
    for line in open(path, errors="ignore"):
        m = TRAIN_RE.search(line)
        if m:
            ep = int(m.group(1))
            train.append((ep, float(m.group(2)), float(m.group(3))))
        m = VAL_RE.search(line)
        if m and ep is not None:
            val.append((ep, *[float(g) for g in m.groups()]))
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="out/monitor")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, val = parse_log(args.log)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if train:
        ep, L, sig = zip(*train)
        axes[0].plot(ep, L); axes[0].set_title("train velocity MSE")
        axes[0].set_xlabel("epoch"); axes[0].set_yscale("log")
        axes[1].plot(ep, sig); axes[1].set_title("sigma_u")
        axes[1].set_xlabel("epoch")
    if val:
        ep, mae, roi, rmean, corr = zip(*val)
        axes[2].plot(ep, roi, "o-", label="roi_mae")
        axes[2].plot(ep, rmean, "s-", label="roi_mean_err")
        axes[2].axhline(15.0, ls="--", c="gray", label="oracle const ~15")
        axes[2].set_title("val [m/s]"); axes[2].set_xlabel("epoch")
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "curves.png"), dpi=110)
    plt.close(fig)
    print(f"train epochs logged: {train[-1][0] if train else 0}  "
          f"val points: {len(val)}")
    if val:
        print(f"latest val: epoch={val[-1][0]} roi_mae={val[-1][2]:.2f} "
              f"roi_mean_err={val[-1][3]:.2f} corr={val[-1][4]:.3f}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    model = SoSMultiplicativeFlowNetwork.from_checkpoint(args.ckpt)
    model.eval().to(device)
    names = load_split(args.cache, args.split)[:args.n]
    for name in names:
        s = load_cached_sample(args.cache, name)
        cond = torch.from_numpy(s["cond"])[None].to(device)
        with torch.no_grad():
            c = model.sample(cond, n_steps=args.ode_steps,
                             n_samples=args.n_samples)
        c_mean = c.mean(0)[0, 0].cpu().numpy()
        c_std = c.std(0)[0, 0].cpu().numpy()
        c_gt = s["c_gt"]
        xi, zi = G.x_grid() * 1e3, G.z_grid() * 1e3
        ext = [xi[0], xi[-1], zi[-1], zi[0]]
        fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
        for a, (img, ttl, cmap, lim) in zip(ax, (
                (c_gt, "ground truth", "turbo", (1420, 1560)),
                (c_mean, "prediction", "turbo", (1420, 1560)),
                (c_mean - c_gt, "pred - truth", "coolwarm", (-20, 20)),
                (c_std, "ensemble std", "magma", None))):
            im = a.imshow(img.T, extent=ext, cmap=cmap, aspect="auto",
                          **({} if lim is None else dict(vmin=lim[0], vmax=lim[1])))
            a.set_title(ttl); a.set_xlabel("x [mm]"); a.set_ylabel("z [mm]")
            fig.colorbar(im, ax=a, fraction=0.046)
        roi = G.roi_mask()
        fig.suptitle(f"{name}  roi_mae={np.abs(c_mean-c_gt)[roi].mean():.2f} m/s",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"pred_{name}.png"), dpi=110)
        plt.close(fig)
        print(f"  {name}: roi_mae={np.abs(c_mean-c_gt)[roi].mean():.2f} "
              f"roi_mean_err={abs(c_mean[roi].mean()-c_gt[roi].mean()):.2f} "
              f"corr={np.corrcoef(c_mean[roi], c_gt[roi])[0,1]:.3f}")
    print("saved", os.path.join(args.out, "curves.png"), "and panels")


if __name__ == "__main__":
    main()
