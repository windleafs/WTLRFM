#!/usr/bin/env python3
"""Zero-shot FMC AbdominalMap adapter for the trained 20-channel abdominal model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import geometry as G
from data.abdominal_rf import build_condition
from data.das import resample_target
from data.abdominal_dataset import masked_metrics
from models import SoSMultiplicativeFlowNetwork
from scripts.predict_abdominal import save_panels


def delayed_sum(rf, launch, fs):
    """Sum real FMC [rx,tx,time] with nonnegative launch delays, zero padded."""
    nt = rf.shape[-1]
    if launch.shape[0] != rf.shape[1] or np.min(launch) < 0:
        raise ValueError("Launch delays must match transmitters and be nonnegative")
    length = next_fast_len(nt + int(np.ceil(launch.max() * fs)) + 32)
    freq = rfftfreq(length, 1 / fs)
    spectrum = rfft(rf, n=length, axis=-1)
    out = np.empty((rf.shape[0], launch.shape[1], length), np.float32)
    for a in range(launch.shape[1]):
        ramp = np.exp(-2j * np.pi * launch[:, a, None] * freq[None]).astype(np.complex64)
        out[:, a] = irfft(np.einsum("rtf,tf->rf", spectrum, ramp), n=length).astype(np.float32)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT.parent / "wfc_dbua_pw/dataset/AbdominalMap3.mat"))
    p.add_argument("--ckpt", default=str(ROOT / "out/abdominal_flow_v2/best.pth"))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fc", type=float, default=8e6)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--ode-steps", type=int, default=20)
    args = p.parse_args()
    if min(args.fc, args.n_samples, args.ode_steps) <= 0:
        raise ValueError("Frequency and sampling counts must be positive")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    started = time.monotonic()
    angles = np.array([-6., 0., 6.])
    with h5py.File(args.data, "r") as f:
        t = np.array(f["time"]).ravel()
        fs = 1 / np.mean(np.diff(t))
        if not np.allclose(np.diff(t), 1 / fs, rtol=1e-6, atol=1e-12):
            raise ValueError("Nonuniform time axis")
        xe = np.array(f["rxAptPos"])[0]
        if f["scat"].shape != (len(xe), len(xe), len(t)):
            raise ValueError("Expected AbdominalMap3 HDF5 scat [tx,rx,time]")
        if not np.all(np.diff(xe) > 0):
            raise ValueError("Unsorted array coordinates")
        launch = -xe[:, None] * np.sin(np.deg2rad(angles))[None] / 1540.
        launch -= launch.min(axis=0)
        pw = None
        # HDF5 reverses MATLAB [time,rx,tx] to [tx,rx,time].
        for start in range(0, len(xe), 8):
            rf = np.array(f["scat"][:, start:start + 8, :], dtype=np.float32).transpose(1, 0, 2)
            block = delayed_sum(rf, launch, fs)
            if pw is None:
                pw = np.empty((len(xe), 3, block.shape[-1]), np.float32)
            pw[start:start + 8] = block
        truth = np.array(f["C"], dtype=np.float32)
        xt, zt = np.array(f["x"]).ravel(), np.array(f["z"]).ravel()
        if truth.shape != (xt.size, zt.size):
            raise ValueError("Expected C [x,z] in this MAT storage")
    acquisition_time = t[0] + np.arange(pw.shape[-1]) / fs
    # Retain the dataset's pulse-referenced time convention; burst timing is unknown.
    sample = dict(rf=pw.transpose(2, 0, 1), time=acquisition_time, xe=xe,
                  angles=angles, launch=launch, fs=fs, fc=args.fc,
                  meta={"config": {"probe": {"source_cycles": 0}}})
    cond = build_condition(sample, args.device)
    xi, zi = G.x_grid(), G.z_grid()
    gt = resample_target(truth, xt, zt, 0., 0., xi, zi)
    xx, zz = np.meshgrid(xi, zi, indexing="ij")
    support = (xx >= xt.min()) & (xx <= xt.max()) & (zz >= zt.min()) & (zz <= zt.max())
    valid = support & (abs(xx) <= .010) & (zz >= .003) & (zz <= .040)
    model = SoSMultiplicativeFlowNetwork.from_checkpoint(args.ckpt).to(args.device).eval()
    if model.cfg["cond_channels"] != 20:
        raise ValueError("Requires 20-channel abdominal checkpoint")
    torch.manual_seed(12345)
    with torch.no_grad():
        draws = model.sample(torch.tensor(cond.astype(np.float32)[None], device=args.device),
                             n_steps=args.ode_steps, n_samples=args.n_samples)
    pred = draws.mean(0)[0, 0].cpu().numpy()
    std = draws.std(0, unbiased=False)[0, 0].cpu().numpy()
    if not np.isfinite(pred).all() or not np.isfinite(std).all():
        raise ValueError("Nonfinite model outputs")
    estimates = dict(model=pred, constant1500=np.full_like(gt, 1500), constant1540=np.full_like(gt, 1540))
    results = {}
    for method, estimate in estimates.items():
        m = masked_metrics(estimate, gt, valid)
        a, b = estimate[valid].astype(float), gt[valid].astype(float)
        ac, bc = a - a.mean(), b - b.mean()
        m.update(mean_pred=float(a.mean()), mean_truth=float(b.mean()), bias=float((a-b).mean()),
                 corr=float((ac*bc).sum() / (np.linalg.norm(ac)*np.linalg.norm(bc) + 1e-12)))
        results[method] = m
    note = ("Zero-shot acquisition-domain transfer, not in-domain validation. Training: 3MHz PW, "
            "0.3mm pitch; input: assumed 8MHz FMC, 0.2mm pitch, synthesized PW by linear superposition. "
            "Original source waveform and background subtraction are unverified. Stored time origin "
            "is preserved with no added pulse-centre offset, matching existing IMPACT importer timing. "
            "No target values construct model conditions. Excluded pixels are not evaluated. "
            "Ensemble std is sampling variability, not calibrated confidence.")
    meta = dict(source=str(Path(args.data).resolve()), checkpoint=str(Path(args.ckpt).resolve()),
                fc_hz=args.fc, fs_hz=fs, t0_s=float(t[0]), pitch_m=float(np.mean(np.diff(xe))),
                angles_deg=angles.tolist(), c_steer_mps=1540., added_burst_offset_s=0.,
                n_samples=args.n_samples, ode_steps=args.ode_steps, seed=12345,
                evaluation_fov_m=dict(x=[-.010,.010], z=[.003,.040]), note=note)
    out.mkdir(parents=True)
    np.savez_compressed(out / "AbdominalMap3_prediction.npz", c_pred=pred, c_std=std, c_gt=gt,
                        valid_mask=valid, truth_support_mask=support, cond=cond, cx=xi, cz=zi,
                        launch_delays_s=launch, meta=json.dumps(meta))
    summary = dict(metadata=meta, metrics=results, mean_ensemble_std=float(std[valid].mean()),
                   elapsed_seconds=time.monotonic()-started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    panel_data = dict(c_gt=gt, cx=xi, cz=zi, valid_mask=valid)
    save_panels(out, [("AbdominalMap3_zero_shot", panel_data, pred, std)])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
