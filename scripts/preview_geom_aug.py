#!/usr/bin/env python3
"""Preview the train-time geometric perturbation on one cached sample."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data.geometry as G
from data.geom_aug import DEFAULTS, perturb
from data.obpw_dataset import load_cached_sample, load_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="out/geom_aug_preview.png")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    name = args.name or load_split(args.cache, "train")[0]
    s = load_cached_sample(args.cache, name)
    cond, c_gt = s["cond"], s["c_gt"]
    u_gt = G.c_to_u(c_gt).astype(np.float32)[None]
    rng = np.random.default_rng(args.seed)
    cond_w, u_w, c_w = perturb(cond, u_gt, c_gt, DEFAULTS, rng=rng)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xi, zi = G.x_grid() * 1e3, G.z_grid() * 1e3
    ext = [xi[0], xi[-1], zi[-1], zi[0]]
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.2))
    ax[0, 0].imshow(c_gt.T, extent=ext, cmap="turbo", vmin=1420, vmax=1560,
                    aspect="auto")
    ax[0, 0].set_title("c_gt")
    ax[0, 1].imshow(c_w.T, extent=ext, cmap="turbo", vmin=1420, vmax=1560,
                    aspect="auto")
    ax[0, 1].set_title("c_gt  warped")
    env0 = np.sqrt(cond[0] ** 2 + cond[1] ** 2)
    env1 = np.sqrt(cond_w[0] ** 2 + cond_w[1] ** 2)
    ax[0, 2].imshow(env0.T, extent=ext, cmap="gray", aspect="auto")
    ax[0, 2].set_title("DAS |ch0|")
    ax[1, 0].imshow(env1.T, extent=ext, cmap="gray", aspect="auto")
    ax[1, 0].set_title("DAS |ch0|  warped")
    ax[1, 1].imshow((c_w - c_gt).T, extent=ext, cmap="coolwarm",
                    vmin=-40, vmax=40, aspect="auto")
    ax[1, 1].set_title("warped − original c")
    ax[1, 2].imshow((env1 - env0).T, extent=ext, cmap="gray", aspect="auto")
    ax[1, 2].set_title("warped − original |ch0|")
    for a in ax.ravel():
        a.set_xlabel("x [mm]")
        a.set_ylabel("z [mm]")
    fig.suptitle(f"geom_aug preview  {name}  seed={args.seed}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
