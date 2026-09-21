#!/usr/bin/env python3
"""Zero-shot OpenBreast SoS checkpoint on IMPACT AbdominalMap3.

This is an *adapter experiment*, not a valid transfer.  The checkpoint was
trained on 13-angle 5 MHz / 0.3 mm breast PW IQ.  AbdominalMap3 is 8 MHz /
0.2 mm FMC.  We:

  1. synthesise the same 13 plane-wave angles from FMC (frequency-domain
     delay-and-sum over transmit elements, as in bench_das_vs_wfc_ab.py);
  2. build the 40-channel DAS condition on the model's 128 x 160 grid
     (assumed speeds still 1450/1500/1550);
  3. run ``out/flow/best.pth`` without any fine-tuning.

Expected outcome: a smooth map near 1500 m/s that does not recover the
abdominal 2-D structure.

    python scripts/predict_abdominal_zeroshot.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WFC_DIR = os.path.abspath(os.path.join(ROOT, "..", "wfc_dbua_pw"))
sys.path.insert(0, ROOT)
sys.path.insert(0, WFC_DIR)

import data.geometry as G
from data.das import build_condition, resample_target
from engine import map_metrics
from models import SoSMultiplicativeFlowNetwork
from synth import load_data

DATA_DEFAULT = os.path.join(WFC_DIR, "dataset", "AbdominalMap3.mat")
CKPT_DEFAULT = os.path.join(ROOT, "out", "flow", "best.pth")
ANGLES_DEG = np.arange(-12.0, 12.0 + 1e-6, 2.0)   # 13 angles, OpenBreast set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_DEFAULT)
    p.add_argument("--ckpt", default=CKPT_DEFAULT)
    p.add_argument("--out", default=os.path.join(ROOT, "out",
                                                 "pred_abdominal3_zeroshot"))
    p.add_argument("--c-steer", type=float, default=1540.0,
                   help="speed used to form plane waves from FMC")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--device", default=None)
    return p.parse_args()


def fmc_to_pw(iq_fmc, xe, angles_deg, fs, fd, c_steer):
    """D_pw[rx,a,f] = sum_tx D_fmc[rx,tx,f] exp(-j 2π f_phys x_tx sinθ / c)."""
    iq_fmc = np.asarray(iq_fmc, np.complex64)
    xe = np.asarray(xe, np.float64)
    angles = np.deg2rad(np.asarray(angles_deg, np.float64))
    ne, _, nt = iq_fmc.shape
    nfft = nt
    D = np.fft.fft(iq_fmc, n=nfft, axis=-1)
    fphys = fd + np.fft.fftfreq(nfft, 1.0 / fs)
    iq_pw = np.empty((ne, angles.size, nfft), np.complex64)
    for a, th in enumerate(angles):
        ramp = np.exp(-2j * np.pi * np.outer(xe * np.sin(th) / c_steer, fphys)
                      ).astype(np.complex64)
        iq_pw[:, a, :] = np.fft.ifft(np.einsum("itf,tf->if", D, ramp),
                                     axis=-1).astype(np.complex64)
    return iq_pw


def abdominal_fov_mask(nx=G.NX, nz=G.NZ):
    x, z = G.x_grid(nx), G.z_grid(nz)
    mx = (x >= -10e-3) & (x <= 10e-3)
    mz = (z >= 0.0) & (z <= 40e-3)
    return mx[:, None] & mz[None, :]


def save_panel(path, c_gt, c_pred, c_std, cond, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xi, zi = G.x_grid() * 1e3, G.z_grid() * 1e3
    ext = [xi[0], xi[-1], zi[-1], zi[0]]
    vmin, vmax = 1470.0, 1610.0
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.4))
    im = axes[0].imshow(c_gt.T, extent=ext, cmap="turbo", vmin=vmin, vmax=vmax,
                        aspect="equal")
    axes[0].set_title("AbdominalMap3  C (on model grid)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im = axes[1].imshow(c_pred.T, extent=ext, cmap="turbo", vmin=vmin, vmax=vmax,
                        aspect="equal")
    axes[1].set_title("OpenBreast ckpt  (zero-shot)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    err = c_pred - c_gt
    lim = max(20.0, np.percentile(np.abs(err), 99))
    im = axes[2].imshow(err.T, extent=ext, cmap="coolwarm", vmin=-lim, vmax=lim,
                        aspect="equal")
    axes[2].set_title("pred − truth  [m/s]")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    env = np.sqrt(cond[0] ** 2 + cond[1] ** 2)
    axes[3].imshow(env.T, extent=ext, cmap="gray", aspect="equal")
    axes[3].set_title("DAS condition |ch0| (1450 m/s)")
    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_xlim(-19.05, 19.05)
        ax.set_ylim(47.85, 0.15)
        ax.axvline(-10.0, color="w", ls=":", lw=0.6, alpha=0.7)
        ax.axvline(10.0, color="w", ls=":", lw=0.6, alpha=0.7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    print(f"data {args.data}")
    print(f"ckpt {args.ckpt}")
    print(f"device {device}")
    print("THIS IS A ZERO-SHOT ADAPTER EXPERIMENT, not a valid transfer.")

    t0 = time.perf_counter()
    data = load_data(args.data)
    iq_fmc = np.asarray(data["iqdata"])
    xe = np.asarray(data["elpos"])[0]
    fs, fd = float(data["fs"]), float(data["fd"])
    t0_rec = float(np.asarray(data["t0"]).reshape(-1)[0])
    print(f"FMC {iq_fmc.shape}  fd={fd/1e6:.2f} MHz  fs={fs/1e6:.2f} MHz  "
          f"pitch={np.mean(np.diff(xe))*1e3:.2f} mm  t0={t0_rec*1e6:.3f} us  "
          f"({time.perf_counter()-t0:.1f}s)")

    t1 = time.perf_counter()
    iq_pw = fmc_to_pw(iq_fmc, xe, ANGLES_DEG, fs, fd, args.c_steer)
    print(f"PW synth {iq_pw.shape}  {len(ANGLES_DEG)} angles "
          f"{ANGLES_DEG[0]:.0f}:{ANGLES_DEG[1]-ANGLES_DEG[0]:.0f}:{ANGLES_DEG[-1]:.0f} "
          f"c_steer={args.c_steer:.0f}  ({time.perf_counter()-t1:.1f}s)")

    xi, zi = G.x_grid(), G.z_grid()
    t1 = time.perf_counter()
    cond = build_condition(iq_pw, xe, ANGLES_DEG, xi, zi,
                           speeds=(1450.0, 1500.0, 1550.0),
                           ref_speed=1500.0, n_subap=4,
                           groups=("full", "angle", "subap"),
                           fs=fs, fc=fd, t0=t0_rec)
    print(f"condition {cond.shape}  ({time.perf_counter()-t1:.1f}s)")

    c_gt = resample_target(data["c_true"], data["cx_true"], data["cz_true"],
                           0.0, 0.0, xi, zi)

    model = SoSMultiplicativeFlowNetwork.from_checkpoint(args.ckpt)
    model.eval().to(device)
    cond_t = torch.from_numpy(cond)[None].to(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        c = model.sample(cond_t, n_steps=args.ode_steps,
                         n_samples=args.n_samples)
    c_mean = c.mean(dim=0)[0, 0].cpu().numpy()
    c_std = c.std(dim=0)[0, 0].cpu().numpy()
    print(f"flow sample x{args.n_samples}  ({time.perf_counter()-t1:.1f}s)")

    roi = G.roi_mask()
    fov = abdominal_fov_mask()
    m_roi = map_metrics(c_mean, c_gt)
    err_fov = c_mean[fov] - c_gt[fov]
    corr_fov = float(np.corrcoef(c_mean[fov], c_gt[fov])[0, 1])
    print("\n--- zero-shot metrics (do not treat as a result) ---")
    print("  model ROI 3-45 mm / ±19 mm : "
          + "  ".join(f"{k}={v:.3f}" for k, v in m_roi.items()))
    print(f"  abdominal FOV ±10 x 40 mm : "
          f"mae={np.abs(err_fov).mean():.3f}  "
          f"mean_err={err_fov.mean():+.3f}  corr={corr_fov:.3f}")
    print(f"  mean SoS  pred={c_mean[fov].mean():.1f}  "
          f"gt={c_gt[fov].mean():.1f}  "
          f"const1500 err={1500.0 - c_gt[fov].mean():+.1f}")
    print(f"  pred std (ensemble) mean={c_std[fov].mean():.2f}  "
          f"map std pred={c_mean[fov].std():.2f}  gt={c_gt[fov].std():.2f}")

    np.savez_compressed(
        os.path.join(args.out, "abdominal3_zeroshot.npz"),
        c_pred=c_mean.astype(np.float32),
        c_std=c_std.astype(np.float32),
        c_gt=c_gt.astype(np.float32),
        cond=cond.astype(np.float32),
        cx=xi, cz=zi,
        angles_deg=ANGLES_DEG.astype(np.float32),
        c_steer=np.float32(args.c_steer),
        note=np.array("zero-shot OpenBreast ckpt; not a valid transfer"),
    )
    title = (f"zero-shot OpenBreast → AbdominalMap3   "
             f"FOV mae={np.abs(err_fov).mean():.1f} m/s  "
             f"corr={corr_fov:.2f}  "
             f"pred mean {c_mean[fov].mean():.0f} vs gt {c_gt[fov].mean():.0f}")
    png = os.path.join(args.out, "abdominal3_zeroshot.png")
    save_panel(png, c_gt, c_mean, c_std, cond, title)
    print(f"\nwrote {png}")
    print(f"wrote {os.path.join(args.out, 'abdominal3_zeroshot.npz')}")


if __name__ == "__main__":
    main()
