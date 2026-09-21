#!/usr/bin/env python3
"""One OpenBreast-geometry PW sample whose SoS map follows AbdominalMap3.

k-Wave's CUDA binaries on this machine are Windows-only, so the forward
model is the same 2-D k-space PSTD engine used to build the training set
(``openbreast_pw_iq/code/pstd2d.py``, same update as k-Wave).  Acquisition
matches the SoS checkpoint exactly: 128 x 0.3 mm, 13 plane waves
(-12:2:12 deg), 5 MHz / 20 MHz IQ, scattered field, t0 = wavefront at the
probe centre.

    # JAX env (dbua)
    python scripts/sim_abdominal_openbreast_pw.py sim
    # torch env (py310)
    python scripts/sim_abdominal_openbreast_pw.py predict
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OBPW_CODE = "/data/zhuangyang/openbreast_pw_iq/code"
WFC_DIR = os.path.abspath(os.path.join(ROOT, "..", "wfc_dbua_pw"))
MAT_DEFAULT = os.path.join(WFC_DIR, "dataset", "AbdominalMap3.mat")
OUT_DEFAULT = os.path.join(ROOT, "out", "abdominal_pw_for_sos")

WATER_GAP = 2.0e-3
SCATTER_SIGMA_PX = 0.8
SCATTER_FRAC = 0.008
SEED = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sim", "predict", "all"])
    p.add_argument("--mat", default=MAT_DEFAULT)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--ckpt", default=os.path.join(ROOT, "out", "flow", "best.pth"))
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--scatter", type=float, default=SCATTER_FRAC)
    return p.parse_args()


def load_abdominal_c(mat_path):
    sys.path.insert(0, WFC_DIR)
    from synth import load_data
    d = load_data(mat_path)
    return (np.asarray(d["c_true"], np.float64),
            np.asarray(d["cx_true"], np.float64),
            np.asarray(d["cz_true"], np.float64))


def build_medium(c_ab, cx_ab, cz_ab, scatter_frac, seed):
    sys.path.insert(0, OBPW_CODE)
    import obpw as P

    x_m = P.sim_x_coords(0.0)                 # probe-centred
    z_face_m = P.TOP_MARGIN                   # so iz=0 is domain top
    z_m = P.sim_z_coords(z_face_m)            # probe face at z = TOP_MARGIN
    z_rel = z_m - z_face_m                    # 0 at probe face
    X, Z = np.meshgrid(x_m, z_rel, indexing="ij")

    # AbdominalMap3 z=0 (tissue surface) sits WATER_GAP below the probe.
    col = (X - cx_ab[0]) / (cx_ab[1] - cx_ab[0])
    row = (Z - WATER_GAP - cz_ab[0]) / (cz_ab[1] - cz_ab[0])
    cmap = ndimage.map_coordinates(
        c_ab, np.stack([col.ravel(), row.ravel()]),
        order=1, mode="nearest").reshape(X.shape).astype(np.float32)

    water = Z < WATER_GAP
    cmap = np.where(water, np.float32(P.C_WATER), cmap)

    if scatter_frac > 0:
        rng = np.random.default_rng(seed)
        noise = ndimage.gaussian_filter(
            rng.normal(0.0, 1.0, cmap.shape).astype(np.float32),
            SCATTER_SIGMA_PX)
        tissue = ~water
        cmap = np.where(tissue, cmap * (1.0 + scatter_frac * noise), cmap)
        cmap = np.clip(cmap, 1420.0, 1650.0).astype(np.float32)

    rhomap = P.rho_from_c(cmap).astype(np.float32)
    breast = (np.abs(cmap - P.C_WATER) > 0.5) & (z_rel[None, :] >= 3e-3) & (
        z_rel[None, :] <= P.ROI_DEPTH)
    xe = P.element_positions()
    in_ap = (x_m[:, None] >= xe[0]) & (x_m[:, None] <= xe[-1])
    roi = breast & in_ap
    label = dict(
        c_roi_mean=np.float32(cmap[roi].mean()) if roi.sum() else np.float32(np.nan),
        c_breast_mean=np.float32(cmap[breast].mean()) if breast.any() else np.float32(np.nan),
        fibro_frac=np.float32((cmap[roi] > 1500.0).mean()) if roi.sum() else np.float32(np.nan),
        roi_frac=np.float32(roi.mean()),
        xc_px=np.float32(0.0), z_face_px=np.float32(0.0),
        apex_row=np.float32(0.0),
    )
    med = dict(cmap=cmap, rhomap=rhomap, x_m=x_m.astype(np.float32),
               z_m=z_m.astype(np.float32), z_face_m=np.float32(z_face_m),
               xc_m=np.float32(0.0), z_rel=z_rel.astype(np.float32))
    return med, label, P


def save_phantom_png(path, med, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, z = med["x_m"] * 1e3, med["z_rel"] * 1e3
    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    im = ax.imshow(med["cmap"].T, extent=[x[0], x[-1], z[-1], z[0]],
                   cmap="turbo", vmin=1470, vmax=1610, aspect="equal")
    ax.axhline(2.0, color="w", ls=":", lw=0.7, alpha=0.8)
    ax.set_xlim(-19.05, 19.05)
    ax.set_ylim(48, -1)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]  (0 = probe face)")
    ax.set_title(f"AbdominalMap3 on OpenBreast grid\n"
                 f"ROI mean {label['c_roi_mean']:.0f} m/s")
    fig.colorbar(im, ax=ax, fraction=0.046, label="c [m/s]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cmd_sim(args):
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, OBPW_CODE)
    import obpw as P
    from pstd2d import simulate_pw

    print(f"loading {args.mat}")
    c_ab, cx_ab, cz_ab = load_abdominal_c(args.mat)
    med, label, P = build_medium(c_ab, cx_ab, cz_ab, args.scatter, SEED)
    save_phantom_png(os.path.join(args.out, "phantom.png"), med, label)
    print(f"phantom cmap {med['cmap'].shape}  ROI mean {label['c_roi_mean']:.1f}  "
          f"range {med['cmap'].min():.0f}-{med['cmap'].max():.0f}")

    src_gain, src_tau = P.src_gain_tau(P.ANGLES_DEG)
    ix_rx = P.ix_elements()

    def run(cmap, rhomap):
        rf = simulate_pw(cmap, rhomap, P.DX, P.DT, P.NT, src_gain, src_tau,
                         P.IZ_SRC, ix_rx, P.IZ_SRC, P.FC, P.SIGMA_P, P.C_REF,
                         P.N_ABS, P.SPONGE_STRENGTH, src_zprof=P.SRC_ZPROF)
        return np.asarray(rf, np.float32)

    t0 = time.time()
    water_c = np.full_like(med["cmap"], P.C_WATER)
    water_r = np.full_like(med["rhomap"], P.RHO_WATER)
    print("reference (homogeneous 1500) ...", flush=True)
    ref = run(water_c, water_r)
    print(f"  {ref.shape}  {time.time()-t0:.1f}s")
    t1 = time.time()
    print("scattered (abdominal cmap) ...", flush=True)
    rf = run(med["cmap"], med["rhomap"])
    print(f"  {rf.shape}  {time.time()-t1:.1f}s")
    rf = rf - ref
    shift = P.tx_time_shift(P.ANGLES_DEG)
    iq = P.rf_to_iq(rf, shift_s=shift)
    iq = np.transpose(iq, (1, 0, 2)).astype(np.complex64)
    fp = os.path.join(args.out, "sample_abdominal.npz")
    np.savez_compressed(
        fp,
        iqdata=iq, fs=np.float32(P.FS_IQ), fd=np.float32(P.FC),
        dsf=np.int32(1), t=P.iq_time_axis(), t0=np.float32(0.0),
        angles=P.ANGLES_DEG, txtype=np.array("pw"),
        elpos=np.stack([P.element_positions(),
                        np.zeros(P.NE), np.zeros(P.NE)]).astype(np.float32),
        pitch=np.float32(P.PITCH),
        c_true=med["cmap"], cx_true=med["x_m"], cz_true=med["z_m"],
        z_face_m=med["z_face_m"], xc_m=med["xc_m"],
        txtype_pw_shift=np.asarray(shift, np.float32),
        **label)
    print(f"wrote {fp}  iq {iq.shape}  absmean {np.abs(iq).mean():.4g}")
    print(f"total sim {time.time()-t0:.1f}s")
    return fp


def cmd_predict(args):
    sys.path.insert(0, ROOT)
    import torch
    import data.geometry as G
    from data.das import build_condition, resample_target
    from engine import map_metrics
    from models import SoSMultiplicativeFlowNetwork

    fp = os.path.join(args.out, "sample_abdominal.npz")
    raw = np.load(fp)
    xi, zi = G.x_grid(), G.z_grid()
    print("DAS condition ...", flush=True)
    t0 = time.time()
    cond = build_condition(
        np.asarray(raw["iqdata"]), np.asarray(raw["elpos"])[0],
        np.asarray(raw["angles"]), xi, zi,
        speeds=(1450.0, 1500.0, 1550.0), ref_speed=1500.0, n_subap=4,
        fs=float(raw["fs"]), fc=float(raw["fd"]), t0=float(raw["t0"]))
    print(f"  {cond.shape}  {time.time()-t0:.1f}s")
    c_gt = resample_target(raw["c_true"], raw["cx_true"], raw["cz_true"],
                           float(raw["xc_m"]), float(raw["z_face_m"]),
                           xi, zi)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    model = SoSMultiplicativeFlowNetwork.from_checkpoint(args.ckpt)
    model.eval().to(device)
    with torch.no_grad():
        c = model.sample(torch.from_numpy(cond)[None].to(device),
                         n_steps=args.ode_steps, n_samples=args.n_samples)
    c_mean = c.mean(dim=0)[0, 0].cpu().numpy()
    c_std = c.std(dim=0)[0, 0].cpu().numpy()
    m = map_metrics(c_mean, c_gt)
    roi = G.roi_mask()
    print("predicted : " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print(f"mean SoS   pred={c_mean[roi].mean():.1f}  gt={c_gt[roi].mean():.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, z = xi * 1e3, zi * 1e3
    ext = [x[0], x[-1], z[-1], z[0]]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.4))
    im = axes[0].imshow(c_gt.T, extent=ext, cmap="turbo", vmin=1470, vmax=1610,
                        aspect="equal")
    axes[0].set_title("GT (AbdominalMap3 → model grid)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im = axes[1].imshow(c_mean.T, extent=ext, cmap="turbo", vmin=1470, vmax=1610,
                        aspect="equal")
    axes[1].set_title("SoS prediction")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    err = c_mean - c_gt
    lim = max(15.0, np.percentile(np.abs(err), 99))
    im = axes[2].imshow(err.T, extent=ext, cmap="coolwarm", vmin=-lim, vmax=lim,
                        aspect="equal")
    axes[2].set_title("pred − GT [m/s]")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    env = np.sqrt(cond[0] ** 2 + cond[1] ** 2)
    axes[3].imshow(env.T, extent=ext, cmap="gray", aspect="equal")
    axes[3].set_title("DAS condition |ch0|")
    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_xlim(-19.05, 19.05)
        ax.set_ylim(47.85, 0.15)
    fig.suptitle(
        f"OpenBreast-geometry PW of AbdominalMap3   "
        f"roi_mae={m['roi_mae']:.1f} m/s  corr={m['corr']:.2f}  "
        f"pred {c_mean[roi].mean():.0f} vs gt {c_gt[roi].mean():.0f}",
        fontsize=10)
    fig.tight_layout()
    png = os.path.join(args.out, "predict.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    np.savez_compressed(os.path.join(args.out, "pred.npz"),
                        c_pred=c_mean.astype(np.float32),
                        c_std=c_std.astype(np.float32),
                        c_gt=c_gt.astype(np.float32),
                        cond=cond.astype(np.float32),
                        cx=xi, cz=zi)
    print(f"wrote {png}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.stage in ("sim", "all"):
        cmd_sim(args)
    if args.stage in ("predict", "all"):
        cmd_predict(args)


if __name__ == "__main__":
    main()
