#!/usr/bin/env python3
"""Predict 2-D sound-speed maps and save them for the WFC beamformer.

    python predict.py --ckpt out/flow/best.pth --split test --out out/pred_flow

Each sample produces ``<out>/cmaps/<name>.npz`` with
    c_pred [NX, NZ] m/s, c_std [NX, NZ] (flow ensemble std),
    c_gt [NX, NZ], cx [NX], cz [NZ]
which is the input of ``wfc_integration/beamform.py``.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data.geometry as G
from data.obpw_dataset import load_cached_sample, load_split
from engine import map_metrics
from models import SoSMultiplicativeFlowNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="out/pred")
    p.add_argument("--n-samples", type=int, default=8,
                   help="flow ensemble size")
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="0 = deterministic ODE from u_0=0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--n-panels", type=int, default=8)
    return p.parse_args()


def save_panel(path, c_gt, c_pred, c_std, cond, title, vmin=1420, vmax=1560):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xi, zi = G.x_grid() * 1e3, G.z_grid() * 1e3
    ext = [xi[0], xi[-1], zi[-1], zi[0]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    im = axes[0].imshow(c_gt.T, extent=ext, cmap="turbo", vmin=vmin, vmax=vmax,
                        aspect="auto")
    axes[0].set_title("ground truth c(x,z)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im = axes[1].imshow(c_pred.T, extent=ext, cmap="turbo", vmin=vmin, vmax=vmax,
                        aspect="auto")
    axes[1].set_title("prediction")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    err = c_pred - c_gt
    lim = max(5.0, np.abs(err).max())
    im = axes[2].imshow(err.T, extent=ext, cmap="coolwarm", vmin=-lim, vmax=lim,
                        aspect="auto")
    axes[2].set_title("prediction - truth [m/s]")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    if c_std is not None and np.any(c_std > 0):
        im = axes[3].imshow(c_std.T, extent=ext, cmap="magma", aspect="auto")
        axes[3].set_title("flow ensemble std [m/s]")
        fig.colorbar(im, ax=axes[3], fraction=0.046)
    else:
        env = np.sqrt(cond[0] ** 2 + cond[1] ** 2)
        axes[3].imshow(env.T, extent=ext, cmap="gray", aspect="auto")
        axes[3].set_title("condition channel 0/1 magnitude")
    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    os.makedirs(os.path.join(args.out, "cmaps"), exist_ok=True)
    model = SoSMultiplicativeFlowNetwork.from_checkpoint(args.ckpt)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.eval().to(device)
    print(f"model: sigma_u={model.sigma_u:.3f}  epoch={blob.get('epoch')}  "
          f"val={blob.get('val')}  device={device}")

    names = load_split(args.cache, args.split)
    if args.limit:
        names = names[:args.limit]
    rows = []
    agg_pred, agg_gt, agg_const = [], [], []
    for k, name in enumerate(names):
        s = load_cached_sample(args.cache, name)
        cond = torch.from_numpy(s["cond"])[None].to(device)
        with torch.no_grad():
            c = model.sample(cond, n_steps=args.ode_steps,
                             n_samples=args.n_samples,
                             noise_scale=args.noise_scale)
        c_mean = c.mean(dim=0)[0, 0].cpu().numpy()
        c_std = c.std(dim=0)[0, 0].cpu().numpy()
        c_gt = s["c_gt"]
        np.savez_compressed(
            os.path.join(args.out, "cmaps", name + ".npz"),
            c_pred=c_mean.astype(np.float32),
            c_std=c_std.astype(np.float32),
            c_gt=c_gt.astype(np.float32),
            cx=G.x_grid(), cz=G.z_grid(),
            meta=np.array([s["meta"].get("c_roi_mean", np.nan)], np.float32),
        )
        roi = G.roi_mask()
        m = map_metrics(c_mean, c_gt)
        mc = map_metrics(np.full_like(c_gt, G.C_REF), c_gt)
        rows.append(dict(
            name=name, **{f"pred_{k2}": v for k2, v in m.items()},
            **{f"const_{k2}": v for k2, v in mc.items()},
            roi_mean_pred=float(c_mean[roi].mean()),
            roi_mean_gt=float(c_gt[roi].mean()),
            c_roi_mean_label=float(s["meta"].get("c_roi_mean", np.nan))))
        agg_pred.append(c_mean)
        agg_gt.append(c_gt)
        agg_const.append(np.full_like(c_gt, G.C_REF))
        if k < args.n_panels:
            save_panel(os.path.join(args.out, "cmaps", name + ".png"),
                       c_gt, c_mean, c_std, s["cond"],
                       f"{name}  roi_mae={m['roi_mae']:.2f} m/s")
        if (k + 1) % 20 == 0 or k + 1 == len(names):
            print(f"  {k+1}/{len(names)}", flush=True)

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    gt_stack = np.stack(agg_gt)
    overall = map_metrics(np.stack(agg_pred), gt_stack)
    const = map_metrics(np.stack(agg_const), gt_stack)
    # oracle global-constant predictor: honest "how much does a single number
    # buy you" floor (uses the test-set mean, so it is optimistic).
    c_best = float(gt_stack.mean())
    const_best = map_metrics(np.full_like(gt_stack, c_best), gt_stack)
    print(f"\n[{len(names)} samples]")
    print("  predicted  : " + "  ".join(f"{k}={v:.3f}"
                                        for k, v in overall.items()))
    print("  const 1500 : " + "  ".join(f"{k}={v:.3f}"
                                         for k, v in const.items()))
    print(f"  const best ({c_best:.1f}): " + "  ".join(
        f"{k}={v:.3f}" for k, v in const_best.items()))
    print(f"wrote {args.out}/cmaps/*.npz and summary.csv")


if __name__ == "__main__":
    main()
