#!/usr/bin/env python3
"""Beamform the plane-wave IQ data with the predicted sound-speed map via WFC.

Consumes
    <samples>/sample_XXXXX.npz          raw PW IQ (from openbreast_pw_iq)
    <cmaps>/sample_XXXXX.npz            predicted map from predict.py
and produces, per sample, WFC images under several sound-speed hypotheses:

    const  : homogeneous 1500 m/s (water)
    pred   : the predicted 2-D map (the pipeline output)
    true   : the ground-truth simulator map (upper-bound reference)

For each hypothesis both compounding modes are stored:
    coh    : coherent compound   |sum_a I_a|   (recommended for PW)
    inc    : incoherent compound sum_a |I_a|   (the engine's `image()`)
    a<idx> : single-angle image, e.g. a6 = 0 deg (diagnostic)

WFC engine is imported from the sibling ``wfc_dbua_pw`` directory (the
multi-angle plane-wave variant of the WFC/DBUA beamformer), so there is a
single implementation of the physics engine.  This script runs in the JAX
environment (e.g. ``conda run -n dbua``).

Note on the point-spread function: a synthetic single-point scatterer written
in this dataset's exact format (same delay convention, 128 elements x 13
angles) images to its true position within 0.04 mm with the same -6 dB width
as DAS, so the engine + convention combination is verified.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WFC_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "wfc_dbua_pw"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True,
                   help="dir with raw sample_*.npz (IQ)")
    p.add_argument("--cmaps", required=True,
                   help="dir with predicted <name>.npz from predict.py")
    p.add_argument("--out", default="out/wfc")
    p.add_argument("--wfc-dir", default=DEFAULT_WFC_DIR)
    p.add_argument("--names", default=None, help="comma-separated sample names")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--which", default="const,pred,true")
    p.add_argument("--fov-z", type=float, default=45e-3)
    p.add_argument("--pad-x", type=float, default=6e-3)
    p.add_argument("--bw", type=float, default=0.55)
    p.add_argument("--dz-img", type=float, default=0.25e-3)
    p.add_argument("--nf", type=int, default=0)
    p.add_argument("--ncx", type=int, default=128)
    p.add_argument("--ncz", type=int, default=160)
    p.add_argument("--single-angle-idx", type=int, default=6,
                   help="index of the diagnostic single-angle image (6 = 0 deg)")
    p.add_argument("--n-panels", type=int, default=0,
                   help="how many samples to save PNG panels for (0 = all)")
    return p.parse_args()


def bilinear_clamped(src, xs, zs, xt, zt):
    """Bilinear resample src[xs, zs] -> (xt, zt), clamped at all boundaries."""
    xt = np.asarray(xt, np.float64)
    zt = np.asarray(zt, np.float64)
    rows = np.empty((xt.size, zs.size), np.float64)
    for j in range(zs.size):
        rows[:, j] = np.interp(xt, xs, src[:, j])
    out = np.empty((xt.size, zt.size), np.float64)
    for i in range(xt.size):
        out[i, :] = np.interp(zt, zs, rows[i, :])
    return out.astype(np.float32)


def log_compress(img, dr_db=45.0, ref=None):
    env = np.abs(img)
    ref = np.percentile(env, 99.9) if ref is None else ref
    v = np.maximum(env / (ref + 1e-20), 10 ** (-dr_db / 20.0))
    return 20.0 * np.log10(v)


def roi_metrics(env, xg, zg, fov, z_lo=3e-3):
    mx = (xg >= fov[0]) & (xg <= fov[1])
    mz = (zg >= z_lo) & (zg <= fov[2])
    e = env[np.ix_(mx, mz)]
    mu = e.mean()
    ipr = float((e ** 2).mean() / (mu ** 2 + 1e-20))
    en = e / (mu + 1e-20)
    sharp = float((np.diff(en, axis=0) ** 2).mean()
                  + (np.diff(en, axis=1) ** 2).mean())
    # speckle SNR proxy in a 10 x 10 mm patch around (0, 30 mm)
    px = (xg >= -5e-3) & (xg <= 5e-3)
    pz = (zg >= 25e-3) & (zg <= 35e-3)
    p = env[np.ix_(px, pz)]
    snr = float(p.mean() / (p.std() + 1e-12))
    return {"ipr": ipr, "sharpness": sharp, "speckle_snr": snr}


def _crop_xz(arr, x, z, fov):
    """Crop arr[x, z] to the linear-array FOV (drop angular-spectrum pad)."""
    x = np.asarray(x)
    z = np.asarray(z)
    mx = (x >= fov[0]) & (x <= fov[1])
    mz = (z >= 0.0) & (z <= fov[2])
    return np.asarray(arr)[np.ix_(mx, mz)], x[mx], z[mz]


def save_panel(path, cmaps, images, xg, zg, name, fov, mode="coh"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("const", "pred", "true") if k in images]
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8.2), squeeze=False)
    x_mm = (fov[0] * 1e3, fov[1] * 1e3)
    z_mm = (fov[2] * 1e3, 0.0)
    for j, k in enumerate(keys):
        ax = axes[0][j]
        c = cmaps.get(k)
        if c is not None:
            cc, cx, cz = _crop_xz(c["c"], c["cx"], c["cz"], fov)
            ax.imshow(cc.T, extent=[cx[0] * 1e3, cx[-1] * 1e3,
                                    cz[-1] * 1e3, cz[0] * 1e3],
                      cmap="turbo", vmin=1420, vmax=1560, aspect="equal")
            ax.set_title(f"c(x,z) {k}")
        ax.set_xlim(*x_mm)
        ax.set_ylim(*z_mm)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax = axes[1][j]
        img = images[k][mode]
        vis, xv, zv = _crop_xz(img, xg, zg, fov)
        ax.imshow(log_compress(vis).T,
                  extent=[xv[0] * 1e3, xv[-1] * 1e3,
                          zv[-1] * 1e3, zv[0] * 1e3],
                  cmap="gray", vmin=-45, vmax=0, aspect="equal")
        m = roi_metrics(np.abs(img), xg, zg, fov)
        ax.set_title(f"WFC {k} ({mode})  ipr={m['ipr']:.2f}")
        ax.set_xlim(*x_mm)
        ax.set_ylim(*z_mm)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
    fig.suptitle(f"{name}   [{mode} compounding]", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    sys.path.insert(0, args.wfc_dir)
    import jax
    import jax.numpy as jnp
    from wfc import WFCConfig, WFCModel

    os.makedirs(args.out, exist_ok=True)
    print(f"jax {jax.__version__} {jax.devices()}")
    print(f"wfc engine: {args.wfc_dir}")

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

    fov = (-19.05e-3, 19.05e-3, args.fov_z)
    which = [w for w in args.which.split(",") if w]
    rows = []
    model = None
    geom = None
    image_from_f0 = None
    n_panels = len(names) if args.n_panels <= 0 else args.n_panels
    for k, name in enumerate(names):
        raw = np.load(os.path.join(args.samples, name + ".npz"))
        pred = np.load(os.path.join(args.cmaps, name + ".npz"))
        iq = np.asarray(raw["iqdata"])
        elpos = np.asarray(raw["elpos"])
        angles = np.asarray(raw["angles"], np.float32)
        fs, fd = float(raw["fs"]), float(raw["fd"])
        t0_rec = (float(np.asarray(raw["t0"]).reshape(-1)[0])
                  if "t0" in raw.files else 0.0)

        this_geom = (iq.shape, elpos.shape, fs, fd, angles.tobytes())
        if model is None or this_geom != geom:
            if model is not None:
                print("  geometry changed -> rebuilding WFCModel")
            cfg = WFCConfig(c0=1500.0, c_min=1350.0, bw_frac=args.bw,
                            pad_x=args.pad_x, dz=0.5e-3, dz_img=args.dz_img,
                            nf=args.nf, ncx=args.ncx, ncz=args.ncz)
            t0 = time.perf_counter()
            model = WFCModel(iq, elpos, fs, fd, cfg, fov=fov, t0=t0_rec,
                             tx="pw", angles_tx_deg=angles)
            geom = this_geom
            # f0 is an argument so swapping IQ via bind_iqdata is visible to jit
            image_from_f0 = jax.jit(lambda c, f0: model.image_from_f0(c, f0))
            print(f"WFCModel: ne={model.ne} nx={model.nx} dx={model.dx*1e3:.3f}mm "
                  f"nf={model.nf} nz_img={model.nz_img} "
                  f"({time.perf_counter()-t0:.1f}s)")
        else:
            model.bind_iqdata(iq, t0=t0_rec)

        # resample every hypothesis onto the WFC inversion grid (cx, cz)
        cx_w, cz_w = np.asarray(model.cx), np.asarray(model.cz)
        cx_p, cz_p = np.asarray(pred["cx"]), np.asarray(pred["cz"])
        hyp = {"const": np.full((cx_w.size, cz_w.size), 1500.0, np.float32)}
        if "pred" in which:
            hyp["pred"] = bilinear_clamped(
                np.asarray(pred["c_pred"]), cx_p, cz_p, cx_w, cz_w)
        if "true" in which:
            hyp["true"] = bilinear_clamped(
                np.asarray(pred["c_gt"]), cx_p, cz_p, cx_w, cz_w)

        images, cmaps, metrics = {}, {}, {}
        t0 = time.perf_counter()
        for key in which:
            if key not in hyp:
                continue
            A = np.asarray(image_from_f0(jnp.asarray(hyp[key]), model.f0_img))
            coh = np.abs(np.sum(A, axis=0))
            inc = np.sum(np.abs(A), axis=0)
            single = np.abs(A[args.single_angle_idx])
            images[key] = {"coh": coh, "inc": inc, "single": single}
            cmaps[key] = {"c": hyp[key], "cx": cx_w, "cz": cz_w}
            metrics[key] = {f"{m}_{mk}": mv for m, img in
                            (("coh", coh), ("inc", inc), ("single", single))
                            for mk, mv in roi_metrics(img, np.asarray(model.xg),
                                                      np.asarray(model.zg),
                                                      fov).items()}
        dt = time.perf_counter() - t0
        row = dict(name=name, n_hyp=len(images), time_s=round(dt, 1))
        for key in images:
            for mk, mv in metrics[key].items():
                row[f"{key}_{mk}"] = round(mv, 5)
        if "pred" in images and "true" in images:
            a, b = images["pred"]["coh"], images["true"]["coh"]
            row["corr_coh_pred_true"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        if "const" in images and "pred" in images:
            a, b = images["const"]["coh"], images["pred"]["coh"]
            row["corr_coh_const_pred"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        rows.append(row)
        print(f"  [{k+1}/{len(names)}] {name}: " + "  ".join(
            f"{key}_ipr={metrics[key]['coh_ipr']:.2f}" for key in images)
            + f"  ({dt:.1f}s)")

        out_npz = dict(xg=np.asarray(model.xg), zg=np.asarray(model.zg),
                       fov=np.asarray(fov, np.float32))
        for key, d in images.items():
            for mode, img in d.items():
                out_npz[f"env_{key}_{mode}"] = img.astype(np.float32)
            out_npz[f"cmap_{key}"] = hyp[key].astype(np.float32)
        np.savez_compressed(os.path.join(args.out, name + ".npz"), **out_npz)
        if k < n_panels:
            save_panel(os.path.join(args.out, name + ".png"), cmaps, images,
                       np.asarray(model.xg), np.asarray(model.zg), name, fov,
                       mode="coh")
            save_panel(os.path.join(args.out, name + "_single.png"), cmaps,
                       {kk: {**vv, "coh": vv["single"]} for kk, vv in images.items()},
                       np.asarray(model.xg), np.asarray(model.zg), name, fov,
                       mode="coh")

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out}/*.npz, *.png, summary.csv")
    for mode in ("coh", "inc", "single"):
        if all(f"pred_{mode}_ipr" in r and f"const_{mode}_ipr" in r for r in rows):
            dp = np.mean([r[f"pred_{mode}_ipr"] - r[f"const_{mode}_ipr"]
                          for r in rows])
            print(f"mean ipr({mode}) pred-const = {dp:+.3f}")
    if "true" in which and "pred" in which:
        dt_ = np.mean([r["true_coh_ipr"] - r["pred_coh_ipr"] for r in rows])
        print(f"mean ipr(coh) true-pred = {dt_:+.3f}  "
              f"(head-room left for the predictor)")


if __name__ == "__main__":
    main()
