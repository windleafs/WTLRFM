#!/usr/bin/env python3
"""Visual sanity check of the cached condition and target maps.

    python scripts/check_cache.py --cache cache --n 3 --out cache/check.png

Shows, for a few samples: the ground-truth SoS map, the magnitude of the
full-aperture images at each assumed speed, the per-angle phases, and the
sub-aperture phase differences (the phase features the network sees).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data.geometry as G
from data.das import condition_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--speeds", default="1450,1500,1550")
    ap.add_argument("--n-subap", type=int, default=4)
    ap.add_argument("--angle-idx", default="0,6,12",
                    help="indices of per-angle images to display")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speeds = [float(v) for v in args.speeds.split(",") if v]
    n_sub = int(args.n_subap)
    ang_show = [int(v) for v in args.angle_idx.split(",") if v]
    n_full, n_ang = len(speeds), G.N_ANGLE
    c_full = 2 * n_full
    c_ang = 2 * n_ang
    n_ch = condition_channels(n_ang, n_full, n_sub)

    # list cached files directly (do NOT touch splits.json, which is written
    # once by prepare_cache.py from the complete sample list)
    names = sorted(f[:-4] for f in os.listdir(args.cache)
                   if f.startswith("sample_") and f.endswith(".npz"))[:args.n]
    xi, zi = G.x_grid() * 1e3, G.z_grid() * 1e3
    ext = [xi[0], xi[-1], zi[-1], zi[0]]
    ncol = 1 + n_full + len(ang_show) + 1
    fig, axes = plt.subplots(len(names), ncol,
                             figsize=(3.1 * ncol, 3.4 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        with np.load(os.path.join(args.cache, name + ".npz")) as d:
            cond = np.asarray(d["cond"], np.float32)
            c_gt = np.asarray(d["c_gt"])
        assert cond.shape[0] == n_ch, f"{cond.shape} vs {n_ch}"
        ax = axes[r][0]
        im = ax.imshow(c_gt.T, extent=ext, cmap="turbo", vmin=1420, vmax=1560,
                       aspect="auto")
        ax.set_title(f"{name}\nc_gt [m/s]", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        col = 1
        for si, sp in enumerate(speeds):
            z = cond[2 * si] + 1j * cond[2 * si + 1]
            ax = axes[r][col]
            ax.imshow(np.abs(z).T, extent=ext, cmap="gray", aspect="auto")
            ax.set_title(f"|full| @{sp:.0f}", fontsize=9)
            col += 1
        for ai in ang_show:
            z = cond[c_full + 2 * ai] + 1j * cond[c_full + 2 * ai + 1]
            ax = axes[r][col]
            ax.imshow(np.angle(z).T, extent=ext, cmap="twilight",
                      vmin=-np.pi, vmax=np.pi, aspect="auto")
            ax.set_title(f"phase angle[{ai}]={ai*2-12:+d}deg", fontsize=9)
            col += 1
        zs = (cond[c_full + c_ang:c_full + c_ang + 2 * n_sub:2]
              + 1j * cond[c_full + c_ang + 1:c_full + c_ang + 2 * n_sub:2])
        ax = axes[r][col]
        ax.imshow(np.angle(zs[0] * np.conj(zs[-1])).T, extent=ext,
                  cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="auto")
        ax.set_title("phase subap[0] vs [-1]", fontsize=9)
        for ax in axes[r]:
            ax.set_xlabel("x [mm]", fontsize=8)
            ax.set_ylabel("z [mm]", fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(args.cache, "check.png")
    fig.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()
