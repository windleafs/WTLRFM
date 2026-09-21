#!/usr/bin/env python3
"""Frozen L11 flow SoS on the current 60-slice test set, then Eikonal DAS.

The checkpoint was trained on the original 400/50/50 split.  The live dataset
now has 480/60/60; this run scores every current ``test`` record, including
the ten extra high-echo slices that were not in the original test split.

Sound-speed input is RF-only (36-channel phase-preserving DAS condition).
DAS delays are homogeneous 1540 m/s plus first-arrival Eikonal excess times
from the predicted or true map (coarse-grid fast marching, 16 receive
anchors interpolated across the 192-element aperture).  Constant-1540 DAS
uses the homogeneous law only.

    CUDA_VISIBLE_DEVICES=0 /home/zhuangyang/miniconda3/envs/py310/bin/python \
        scripts/predict_flow_eikonal_das.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skfmm
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

ROOT = Path("/home/zhuangyang/fmmodel/dbua_test/wtlrfm_sos_wfc")
CKPT_RUN = ROOT / "out" / "l11_ultrawave_500_11angle_20260916"
sys.path.insert(0, str(CKPT_RUN / "code"))
sys.path.insert(0, str(ROOT))
from prepare_ultrawave import RFCondition, analytic_baseband  # noqa: E402
from models.sos_mult_flow import SoSMultiplicativeFlowNetwork  # noqa: E402

DATA_ROOT = Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle")
MODEL_NX, MODEL_NZ = 128, 160
NATIVE_NX, NATIVE_NZ = 192, 216
DX_M = 2.0e-4
X0_M = -19.125e-3
Z0_M = 0.075e-3
C0_DAS = 1540.0
N_RX_ANCHORS = 16
COARSE_NZ, COARSE_NX = 64, 48
X_PAD_M = 5.0e-3


def model_axes() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(X0_M, X0_M + (NATIVE_NX - 1) * DX_M, MODEL_NX)
    z = np.linspace(Z0_M, Z0_M + (NATIVE_NZ - 1) * DX_M, MODEL_NZ)
    return x.astype(np.float64), z.astype(np.float64)


def native_axes() -> tuple[np.ndarray, np.ndarray]:
    x = X0_M + np.arange(NATIVE_NX) * DX_M
    z = Z0_M + np.arange(NATIVE_NZ) * DX_M
    return x.astype(np.float64), z.astype(np.float64)


def restore_model(path: Path) -> SoSMultiplicativeFlowNetwork:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = SoSMultiplicativeFlowNetwork(
        unet=blob["cfg"],
        u_source_scale=blob["sigma_u"],
        u_clamp=blob.get("u_clamp", 4.0),
        velocity_clamp=blob.get("velocity_clamp", 8.0),
    )
    model.load_state_dict(blob["state_dict"], strict=True)
    model.cuda().eval()
    return model, blob


def native_from_model(c_xz: torch.Tensor) -> np.ndarray:
    return (
        F.interpolate(c_xz[None, None].transpose(-1, -2), size=(NATIVE_NZ, NATIVE_NX), mode="bilinear", align_corners=True)[0, 0]
        .T.detach()
        .cpu()
        .numpy()
    )


def model_from_native_zx(c_zx: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(c_zx, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=(MODEL_NZ, MODEL_NX), mode="bilinear", align_corners=True)[0, 0].T.numpy()


def map_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    centered_p = prediction.ravel() - prediction.mean()
    centered_t = target.ravel() - target.mean()
    denom = np.linalg.norm(centered_p) * np.linalg.norm(centered_t)
    corr = float(centered_p @ centered_t / denom) if denom > 0 else 0.0
    return {
        "mae_m_s": float(np.mean(np.abs(error))),
        "rmse_m_s": float(np.sqrt(np.mean(error**2))),
        "bias_m_s": float(np.mean(error)),
        "correlation": corr,
    }


def roi_sos_metrics(prediction: np.ndarray, target: np.ndarray, z: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for name, mask in (
        ("all", np.ones(len(z), dtype=bool)),
        ("5_40", (z >= 0.005) & (z < 0.040)),
        ("20_40", (z >= 0.020) & (z < 0.040)),
    ):
        out[name] = map_metrics(prediction[:, mask], target[:, mask])
    return out


def _travel_time(phi: np.ndarray, speed: np.ndarray, dx: tuple[float, float]) -> np.ndarray:
    travel = skfmm.travel_time(phi, speed, dx=dx, order=2)
    if np.ma.isMaskedArray(travel):
        travel = travel.filled(np.nan)
    return np.asarray(travel, dtype=np.float64)


def eikonal_tx_rx_excess(
    sos_zx: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    angles_rad: np.ndarray,
    element_x_m: np.ndarray,
    c0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-arrival Tx/Rx excess time versus homogeneous ``c0``.

    ``sos_zx`` is ``[nz, nx]``.  Returns ``tx [n_angle, nz, nx]`` and
    ``rx [n_elem, nz, nx]`` in seconds.
    """
    x_coarse = np.linspace(x_m[0] - X_PAD_M, x_m[-1] + X_PAD_M, COARSE_NX)
    z_coarse = np.linspace(0.0, z_m[-1], COARSE_NZ)
    dx = float(np.mean(np.diff(x_coarse)))
    dz = float(np.mean(np.diff(z_coarse)))
    zz_c, xx_c = np.meshgrid(z_coarse, x_coarse, indexing="ij")
    row = (zz_c - z_m[0]) / float(np.mean(np.diff(z_m)))
    col = (xx_c - x_m[0]) / float(np.mean(np.diff(x_m)))
    sos_coarse = map_coordinates(np.asarray(sos_zx, np.float64), np.stack([row, col]), order=1, mode="nearest")
    homogeneous = np.full_like(sos_coarse, c0)
    sample_coords = np.stack(np.meshgrid((z_m - z_coarse[0]) / dz, (x_m - x_coarse[0]) / dx, indexing="ij"))

    def excess(phi: np.ndarray) -> np.ndarray:
        delta = _travel_time(phi, sos_coarse, (dz, dx)) - _travel_time(phi, homogeneous, (dz, dx))
        return map_coordinates(delta, sample_coords, order=1, mode="nearest").astype(np.float32)

    tx = np.stack([excess(zz_c * np.cos(angle) + xx_c * np.sin(angle)) for angle in angles_rad], axis=0)
    radius = min(dx, dz)
    anchors = np.linspace(element_x_m[0], element_x_m[-1], N_RX_ANCHORS)
    rx_anchors = np.stack([excess(np.hypot(zz_c, xx_c - anchor) - radius) for anchor in anchors], axis=0)
    index = np.clip(np.searchsorted(anchors, element_x_m, side="right") - 1, 0, N_RX_ANCHORS - 2)
    weight = ((element_x_m - anchors[index]) / np.maximum(anchors[index + 1] - anchors[index], 1e-12)).astype(np.float32)
    rx = (1.0 - weight)[:, None, None] * rx_anchors[index] + weight[:, None, None] * rx_anchors[index + 1]
    return tx.astype(np.float32), rx.astype(np.float32)


@torch.no_grad()
def das_with_excess(
    rf: torch.Tensor,
    *,
    fs: float,
    fc: float,
    angles_deg: np.ndarray,
    refs: np.ndarray,
    xe: torch.Tensor,
    cx: torch.Tensor,
    cz: torch.Tensor,
    c0: float,
    tx_excess_zx: np.ndarray | None,
    rx_excess_zx: np.ndarray | None,
) -> np.ndarray:
    """Return per-angle complex DAS images ``[n_angle, nx, nz]``.

    Excess maps are ``tx [A, nz, nx]`` and ``rx [M, nz, nx]`` in seconds.
    Pixel order matches ``meshgrid(cx, cz, indexing='ij')``.
    """
    iq = analytic_baseband(rf, fs, fc)
    xx, zz = torch.meshgrid(cx, cz, indexing="ij")
    xx = xx.reshape(-1)
    zz = zz.reshape(-1)
    receive = torch.sqrt((xx[None] - xe[:, None]) ** 2 + zz[None] ** 2) / c0
    if rx_excess_zx is None:
        rx_extra = torch.zeros_like(receive)
    else:
        rx_extra = torch.from_numpy(np.transpose(rx_excess_zx, (0, 2, 1)).reshape(xe.numel(), -1)).to(rf.device)
    images = []
    nt = rf.shape[-1]
    for ia, (angle, ref) in enumerate(zip(angles_deg, refs)):
        theta = np.deg2rad(float(angle))
        tx = ref + (xx * np.sin(theta) + zz * np.cos(theta)) / c0
        if tx_excess_zx is not None:
            tx = tx + torch.from_numpy(tx_excess_zx[ia].T.reshape(-1)).to(rf.device)
        tau = tx[None] + receive + rx_extra
        idx = tau * fs
        i0 = idx.floor().long()
        valid = (i0 >= 0) & (i0 < nt - 1)
        frac = idx - i0
        ii = i0.clamp(0, nt - 2)
        sample = (iq[ia].gather(1, ii) * (1 - frac) + iq[ia].gather(1, ii + 1) * frac) * torch.exp(2j * torch.pi * fc * tau) * valid
        images.append(sample.sum(0).reshape(cx.numel(), cz.numel()))
    return torch.stack(images).cpu().numpy()


def das_metrics(per_angle: np.ndarray, z: np.ndarray) -> dict[str, dict[str, float]]:
    coh = np.abs(per_angle.sum(0))
    inc = np.abs(per_angle).sum(0)
    out = {}
    for name, mask in (
        ("all", np.ones(len(z), dtype=bool)),
        ("5_40", (z >= 0.005) & (z < 0.040)),
        ("20_40", (z >= 0.020) & (z < 0.040)),
    ):
        c = coh[:, mask]
        i = inc[:, mask]
        out[name] = {
            "angle_coherence": float(c.mean() / (i.mean() + 1e-30)),
            "mean_envelope": float(c.mean()),
        }
    return out, coh, inc


def envelope_match(pred_env: np.ndarray, true_env: np.ndarray, z: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for name, mask in (
        ("all", np.ones(len(z), dtype=bool)),
        ("5_40", (z >= 0.005) & (z < 0.040)),
        ("20_40", (z >= 0.020) & (z < 0.040)),
    ):
        p = pred_env[:, mask]
        t = true_env[:, mask]
        p_log = np.log(p + 1e-30)
        t_log = np.log(t + 1e-30)
        corr = 0.0 if np.ptp(p) == 0 or np.ptp(t) == 0 else float(np.corrcoef(p.ravel(), t.ravel())[0, 1])
        log_corr = 0.0 if np.ptp(p_log) == 0 or np.ptp(t_log) == 0 else float(np.corrcoef(p_log.ravel(), t_log.ravel())[0, 1])
        scale = float(np.dot(p.ravel(), t.ravel()) / max(np.dot(p.ravel(), p.ravel()), 1e-30))
        nrmse = float(np.sqrt(np.mean((scale * p - t) ** 2)) / (t.mean() + 1e-30))
        out[name] = {"envelope_corr": corr, "log_envelope_corr": log_corr, "gain_nrmse": nrmse}
    return out


def log_compress(env: np.ndarray) -> np.ndarray:
    ref = max(float(np.percentile(env, 99.5)), 1e-30)
    return 20.0 * np.log10(np.maximum(env / ref, 1e-6))


def save_panel(
    path: Path,
    cmaps: dict[str, np.ndarray],
    envelopes: dict[str, np.ndarray],
    x: np.ndarray,
    z: np.ndarray,
    name: str,
    sos_metrics: dict,
    das_metrics_row: dict,
) -> None:
    extent = [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]
    keys = ("constant1540", "flow_pred", "true")
    titles = ("Constant 1540 m/s", "Flow prediction", "Ground-truth SoS")
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.6), constrained_layout=True)
    for j, (key, title) in enumerate(zip(keys, titles)):
        im = axes[0, j].imshow(cmaps[key].T, cmap="turbo", vmin=1400, vmax=1650, extent=extent, aspect="auto")
        fig.colorbar(im, ax=axes[0, j], label="m/s")
        axes[0, j].set(title=title, xlabel="x [mm]", ylabel="z [mm]")
        axes[1, j].imshow(log_compress(envelopes[key]).T, cmap="gray", vmin=-50, vmax=0, extent=extent, aspect="auto")
        coh = das_metrics_row[key]["5_40"]["angle_coherence"]
        axes[1, j].set(title=f"Eikonal DAS  coh={coh:.3f}", xlabel="x [mm]", ylabel="z [mm]")
    mae = sos_metrics["5_40"]["mae_m_s"]
    rmse = sos_metrics["5_40"]["rmse_m_s"]
    corr = sos_metrics["5_40"]["correlation"]
    fig.suptitle(f"{name}  SoS 5–40 mm MAE {mae:.1f} RMSE {rmse:.1f} corr {corr:.3f}", fontsize=12)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mean_nested(rows: list[dict], *keys: str) -> float:
    values = []
    for row in rows:
        value = row
        for key in keys:
            value = value[key]
        values.append(value)
    return float(np.mean(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CKPT_RUN / "best.pth"))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--out-dir", default=str(ROOT / "out" / "l11_flow_eikonal_das_20260917"))
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--c0", type=float, default=C0_DAS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-viz", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    pred_dir = out / "predictions"
    das_dir = out / "das"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    das_dir.mkdir(exist_ok=True)

    data_root = Path(args.data_root)
    index = json.loads((data_root / "index.json").read_text(encoding="utf-8"))
    test = [item for item in index["samples"] if item["split"] == "test"]
    if args.limit:
        test = test[: int(args.limit)]
    orig_cfg = json.loads((CKPT_RUN / "config.json").read_text(encoding="utf-8"))
    train_ids = set(orig_cfg["train_ids"])
    train_records = [item for item in index["samples"] if item["id"] in train_ids]

    cx_np, cz_np = model_axes()
    x_native, z_native = native_axes()
    cx = torch.tensor(cx_np, device="cuda", dtype=torch.float32)
    cz = torch.tensor(cz_np, device="cuda", dtype=torch.float32)
    xe = (torch.arange(NATIVE_NX, device="cuda", dtype=torch.float32) - (NATIVE_NX - 1) / 2.0) * DX_M
    xe_np = xe.detach().cpu().numpy()

    print(f"loading {len(train_records)} original-train maps", flush=True)
    train_native = []
    for record in train_records:
        sample = torch.load(data_root / record["path"], map_location="cpu", weights_only=False)
        train_native.append(sample["c"].T.numpy())
    train_mean_map = np.mean(np.stack(train_native, axis=0), axis=0)
    train_mean_scalar = float(train_mean_map.mean())
    del train_native

    model, blob = restore_model(Path(args.checkpoint))
    first = torch.load(data_root / test[0]["path"], map_location="cpu", weights_only=False)
    conditioner = RFCondition(first["metadata"], xe, cx, cz, 2401)
    angles = np.asarray(first["metadata"]["angles_deg"], dtype=np.float64)
    angles_rad = np.deg2rad(angles)
    refs = np.asarray(first["metadata"]["source_tref_s"], dtype=np.float64)
    c0 = float(args.c0)

    print("identity check: homogeneous 1540 Eikonal excess", flush=True)
    tx0, rx0 = eikonal_tx_rx_excess(np.full((MODEL_NZ, MODEL_NX), c0, np.float32), cx_np, cz_np, angles_rad, xe_np, c0)
    identity_ns = float(max(np.max(np.abs(tx0)), np.max(np.abs(rx0))) * 1e9)
    if identity_ns > 1.0:
        raise RuntimeError(f"homogeneous Eikonal excess {identity_ns:.3f} ns exceeds 1 ns")

    visual_indices = set(np.linspace(0, len(test) - 1, min(args.num_viz, len(test)), dtype=int).tolist())
    rows = []
    start = time.time()
    for i, record in enumerate(test):
        sample = torch.load(data_root / record["path"], map_location="cpu", weights_only=False)
        if not np.allclose(sample["metadata"]["angles_deg"], angles) or not np.allclose(
            sample["metadata"]["source_tref_s"], refs, rtol=0, atol=1e-12
        ):
            raise ValueError(f"{record['id']} acquisition differs from the conditioner")
        rf = sample["rf"].cuda()
        cond = conditioner(rf)[None].cuda()
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(23456 + i)
            draws = model.sample(cond, n_steps=args.ode_steps, n_samples=args.n_samples)[:, 0, 0]
        pred_xz = draws.mean(0)
        pred_std = draws.std(0, unbiased=False)
        pred_native = native_from_model(pred_xz)
        truth_native = sample["c"].T.numpy()
        truth_xz = model_from_native_zx(sample["c"].numpy())
        pred_zx = pred_xz.detach().cpu().numpy().T
        true_zx = truth_xz.T
        constant_native = np.full_like(truth_native, c0)
        sos = {
            "flow_pred": roi_sos_metrics(pred_native, truth_native, z_native),
            "constant1540": roi_sos_metrics(constant_native, truth_native, z_native),
            "train_mean_scalar": roi_sos_metrics(np.full_like(truth_native, train_mean_scalar), truth_native, z_native),
            "train_mean_map": roi_sos_metrics(train_mean_map, truth_native, z_native),
        }

        tx_pred, rx_pred = eikonal_tx_rx_excess(pred_zx, cx_np, cz_np, angles_rad, xe_np, c0)
        tx_true, rx_true = eikonal_tx_rx_excess(true_zx, cx_np, cz_np, angles_rad, xe_np, c0)
        das = {
            "constant1540": das_with_excess(
                rf, fs=conditioner.fs, fc=conditioner.fc, angles_deg=angles, refs=refs,
                xe=xe, cx=cx, cz=cz, c0=c0, tx_excess_zx=None, rx_excess_zx=None,
            ),
            "flow_pred": das_with_excess(
                rf, fs=conditioner.fs, fc=conditioner.fc, angles_deg=angles, refs=refs,
                xe=xe, cx=cx, cz=cz, c0=c0, tx_excess_zx=tx_pred, rx_excess_zx=rx_pred,
            ),
            "true": das_with_excess(
                rf, fs=conditioner.fs, fc=conditioner.fc, angles_deg=angles, refs=refs,
                xe=xe, cx=cx, cz=cz, c0=c0, tx_excess_zx=tx_true, rx_excess_zx=rx_true,
            ),
        }
        das_rows = {}
        envelopes = {}
        for key, image in das.items():
            metrics, coh, inc = das_metrics(image, cz_np)
            das_rows[key] = metrics
            envelopes[key] = coh
        vs_true = {key: envelope_match(envelopes[key], envelopes["true"], cz_np) for key in das}

        row = {
            "id": record["id"],
            "base_anatomy_id": record.get("base_anatomy_id"),
            "high_echo": record["id"] >= "test_050",
            "sos": sos,
            "das": das_rows,
            "das_vs_true": vs_true,
            "prediction_m_s": [float(pred_native.min()), float(pred_native.mean()), float(pred_native.max())],
            "sampling_std_mean_m_s": float(pred_std.mean()),
        }
        rows.append(row)
        np.savez_compressed(
            pred_dir / f"{record['id']}.npz",
            c_pred=pred_xz.detach().cpu().numpy(),
            c_std=pred_std.detach().cpu().numpy(),
            c_native=pred_native,
            c_true_native=truth_native,
            cx=cx_np,
            cz=cz_np,
            cx_native=x_native,
            cz_native=z_native,
        )
        np.savez_compressed(
            das_dir / f"{record['id']}.npz",
            env_constant=envelopes["constant1540"],
            env_pred=envelopes["flow_pred"],
            env_true=envelopes["true"],
            cx=cx_np,
            cz=cz_np,
        )
        if i in visual_indices:
            cmaps = {
                "constant1540": np.full((MODEL_NX, MODEL_NZ), c0, np.float32),
                "flow_pred": pred_xz.detach().cpu().numpy(),
                "true": truth_xz,
            }
            save_panel(
                fig_dir / f"{record['id']}.png",
                cmaps,
                envelopes,
                cx_np,
                cz_np,
                record["id"],
                sos["flow_pred"],
                das_rows,
            )
            np.savez_compressed(
                das_dir / f"{record['id']}_per_angle.npz",
                constant=das["constant1540"],
                pred=das["flow_pred"],
                true=das["true"],
            )
        print(
            f"{i + 1:03d}/{len(test)} {record['id']}  "
            f"sos_rmse={sos['flow_pred']['5_40']['rmse_m_s']:.2f}  "
            f"coh_const={das_rows['constant1540']['5_40']['angle_coherence']:.3f}  "
            f"coh_pred={das_rows['flow_pred']['5_40']['angle_coherence']:.3f}  "
            f"coh_true={das_rows['true']['5_40']['angle_coherence']:.3f}  "
            f"{time.time() - start:.1f}s",
            flush=True,
        )

    def collect(group: str, variant: str, roi: str, metric: str) -> float:
        return mean_nested(rows, group, variant, roi, metric)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(blob["epoch"]),
        "data_root": str(data_root),
        "n_test": len(rows),
        "original_test_in_this_split": 50,
        "extra_high_echo_test": int(sum(row["high_echo"] for row in rows)),
        "protocol": {
            "n_samples": args.n_samples,
            "ode_steps": args.ode_steps,
            "seed_rule": "23456+test_index",
            "network_used_only_RF_condition": True,
            "das": "homogeneous 1540 plus coarse first-arrival Eikonal Tx/Rx excess; 16 receive anchors",
            "c0_das_m_s": c0,
            "homogeneous_eikonal_excess_max_ns": identity_ns,
            "grid": "DAS and network [x=128,z=160]; SoS metrics on native [x=192,z=216]",
        },
        "train_mean_speed_m_s": train_mean_scalar,
        "sos_mean": {
            variant: {roi: {metric: collect("sos", variant, roi, metric) for metric in rows[0]["sos"][variant][roi]} for roi in ("all", "5_40", "20_40")}
            for variant in rows[0]["sos"]
        },
        "das_mean": {
            variant: {roi: {metric: collect("das", variant, roi, metric) for metric in rows[0]["das"][variant][roi]} for roi in ("all", "5_40", "20_40")}
            for variant in rows[0]["das"]
        },
        "das_vs_true_mean": {
            variant: {roi: {metric: collect("das_vs_true", variant, roi, metric) for metric in rows[0]["das_vs_true"][variant][roi]} for roi in ("all", "5_40", "20_40")}
            for variant in rows[0]["das_vs_true"]
        },
        "visualization_ids": [test[i]["id"] for i in sorted(visual_indices)],
        "per_sample": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"sos_mean": summary["sos_mean"]["flow_pred"], "das_mean": summary["das_mean"], "das_vs_true_mean": summary["das_vs_true_mean"]}, indent=2), flush=True)
    print(f"summary: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
