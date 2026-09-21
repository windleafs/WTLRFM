#!/usr/bin/env python3
"""Verify the WFC engine against this dataset's exact plane-wave conventions.

Writes a synthetic single-point scatterer in the OpenBreastUS format (same
128-element/13-angle/20 MHz geometry, same two-way delay model, baseband IQ
with the direct wave removed) and checks that

  * the WFC image peaks at the true scatterer position, and
  * its -6 dB point-spread width matches a straight-ray DAS image,

which isolates any WFC-vs-DAS disagreement to the data (multiple scattering,
density contrasts) rather than the implementation.

    python wfc_integration/verify_psf.py --wfc-dir ../wfc_dbua_pw
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WFC_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "wfc_dbua_pw"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wfc-dir", default=DEFAULT_WFC_DIR)
    p.add_argument("--x0", type=float, default=5e-3)
    p.add_argument("--z0", type=float, default=30e-3)
    p.add_argument("--c0", type=float, default=1500.0)
    p.add_argument("--out", default="out/verify_psf.png")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.wfc_dir)
    import jax
    import jax.numpy as jnp
    from wfc import WFCConfig, WFCModel

    # ---- dataset geometry (openbreast_pw_iq) ----
    pitch, ne = 0.3e-3, 128
    xe = (np.arange(ne) - (ne - 1) / 2) * pitch
    angles = np.arange(-12, 13, 2, dtype=np.float32)
    fs, fc, nsamp = 20e6, 5e6, 1300
    t = np.arange(nsamp) / fs
    c0 = args.c0
    x0, z0 = args.x0, args.z0
    sigma = 0.45 / fc

    th = np.deg2rad(angles)
    tau = ((x0 * np.sin(th)[None, :] + z0 * np.cos(th)[None, :]) / c0
           + np.sqrt((x0 - xe[:, None]) ** 2 + z0 ** 2) / c0)   # [ne, nang]
    tt = t[None, None, :] - tau[:, :, None]
    iq = (np.exp(-0.5 * (tt / sigma) ** 2)
          * np.exp(-2j * np.pi * fc * tau)[:, :, None]).astype(np.complex64)

    fov = (-19.05e-3, 19.05e-3, 45e-3)
    cfg = WFCConfig(c0=1500.0, c_min=1350.0, bw_frac=0.55, pad_x=6e-3,
                    dz=0.5e-3, dz_img=0.25e-3, ncx=128, ncz=160)
    elpos = np.vstack([xe, np.zeros(ne), np.zeros(ne)])
    m = WFCModel(iq, elpos, fs, fc, cfg, fov=fov, t0=0.0, tx="pw",
                 angles_tx_deg=angles)
    cm = jnp.full((128, 160), c0, jnp.float32)
    A = np.asarray(jax.block_until_ready(jax.jit(m.image_angles)(cm)))
    xg, zg = np.asarray(m.xg), np.asarray(m.zg)

    # straight-ray DAS reference on the same grid
    X, Z = np.meshgrid(xg, zg, indexing="ij")
    Xf, Zf = X.ravel().astype(np.float64), Z.ravel().astype(np.float64)
    acc = np.zeros(Xf.size, np.complex64)
    for a in range(len(angles)):
        tha = np.deg2rad(angles[a])
        ttx = (Xf * np.sin(tha) + Zf * np.cos(tha)) / c0
        tau_ = ttx[None, :] + (np.sqrt((Xf[:, None] - xe[None, :]) ** 2
                                       + Zf[:, None] ** 2) / c0).T
        idx = tau_ * fs
        i0 = np.floor(idx).astype(np.int32)
        w = idx - i0
        i0c = np.clip(i0, 0, nsamp - 2)
        d = iq[:, a, :]
        v = (np.take_along_axis(d, i0c, -1) * (1 - w)
             + np.take_along_axis(d, i0c + 1, -1) * w)
        acc += (v * np.exp(2j * np.pi * fc * tau_)).sum(0)
    das = np.abs(acc).reshape(xg.size, zg.size)

    results = {}
    for name, img in (("WFC", np.abs(np.sum(A, 0))), ("DAS", das)):
        i, j = np.unravel_index(np.argmax(img), img.shape)
        peak = img[i, j]
        xs = xg[img[:, j] >= peak / 2]
        zs = zg[img[i, :] >= peak / 2]
        results[name] = dict(x=float(xg[i]), z=float(zg[j]),
                             wx=float(xs.max() - xs.min()),
                             wz=float(zs.max() - zs.min()))
        print(f"{name}: peak x={xg[i]*1e3:.2f} mm z={zg[j]*1e3:.2f} mm "
              f"(true {x0*1e3:.2f}/{z0*1e3:.2f})  "
              f"-6dB width x={results[name]['wx']*1e3:.2f} mm "
              f"z={results[name]['wz']*1e3:.2f} mm")
    dx = abs(results["WFC"]["x"] - results["DAS"]["x"])
    dz = abs(results["WFC"]["z"] - results["DAS"]["z"])
    print(f"WFC-DAS peak offset: {dx*1e3:.3f} mm (x), {dz*1e3:.3f} mm (z)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [xg[0] * 1e3, xg[-1] * 1e3, zg[-1] * 1e3, zg[0] * 1e3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, img) in zip(axes, (("WFC", np.abs(np.sum(A, 0))),
                                      ("DAS", das))):
        ax.imshow(20 * np.log10(img / img.max() + 1e-6), extent=ext,
                  cmap="gray", vmin=-40, vmax=0, aspect="auto")
        ax.plot(x0 * 1e3, z0 * 1e3, "r+", ms=12, mew=2)
        ax.set_title(f"{name}  (peak at "
                     f"{results[name]['x']*1e3:.2f},"
                     f"{results[name]['z']*1e3:.2f} mm)")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
