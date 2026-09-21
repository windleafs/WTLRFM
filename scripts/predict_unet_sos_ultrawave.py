#!/usr/bin/env python3
"""Zero-shot NACRF UNet SoS inversion on UltraWave L11 11-angle breast data.

The checkpoint at
``/home/zhuangyang/fmmodel/NACRF/results/unet_sos_inversion_v1`` inverts
compounded Eikonal excess delay + angle-agreement confidence to a 2-D
sound-speed residual around a known reference speed.  It does not take RF.

This experiment therefore builds the *same input representation as training*
from each sample's ground-truth ``c`` map (coarse-grid first-arrival Eikonal,
eight receive anchors, 11 transmit angles).  That is an oracle-input transfer
test, not RF-only sound-speed prediction.  It answers whether the frozen UNet
can invert breast-phantom travel-time residuals after bilinear resampling to
the training tensor size ``[Nz=160, Nx=128]``.

Predictions are interpolated back to the native ``[z=216, x=192]`` label grid
before metrics.  ``c_reference`` is the fixed 1540 m/s assumed speed used by
the original MATLAB dataset config; it is not the per-sample map mean.

    /home/zhuangyang/miniconda3/envs/py310/bin/python \
        scripts/predict_unet_sos_ultrawave.py
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

NACRF_ROOT = Path("/home/zhuangyang/fmmodel/NACRF")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(NACRF_ROOT) not in sys.path:
    sys.path.insert(0, str(NACRF_ROOT))

from scripts.train_kan_sos_inversion import build_model, resolve_device  # noqa: E402


NATIVE_NZ = 216
NATIVE_NX = 192
MODEL_NZ = 160
MODEL_NX = 128
DX_M = 2.0e-4
X0_M = -19.125e-3
Z0_M = 0.075e-3
SHIFT_SCALE_S = 1.0e-6
N_RX_ANCHORS = 8
COARSE_NZ = 48
COARSE_NX = 40
X_PAD_M = 5.0e-3


def native_axes() -> tuple[np.ndarray, np.ndarray]:
    x = X0_M + np.arange(NATIVE_NX) * DX_M
    z = Z0_M + np.arange(NATIVE_NZ) * DX_M
    return x, z


def model_axes() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(X0_M, X0_M + (NATIVE_NX - 1) * DX_M, MODEL_NX)
    z = np.linspace(Z0_M, Z0_M + (NATIVE_NZ - 1) * DX_M, MODEL_NZ)
    return x, z


def resample_z_x(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
    out = F.interpolate(tensor, size=size, mode="bilinear", align_corners=True)
    return out[0, 0].numpy()


def map_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    centered_prediction = prediction.ravel() - prediction.mean()
    centered_target = target.ravel() - target.mean()
    denominator = np.linalg.norm(centered_prediction) * np.linalg.norm(centered_target)
    if denominator > 0:
        correlation = float(centered_prediction @ centered_target / denominator)
    else:
        correlation = 0.0
    return {
        "mae_m_s": float(np.mean(np.abs(error))),
        "rmse_m_s": float(np.sqrt(np.mean(error**2))),
        "bias_m_s": float(np.mean(error)),
        "correlation": correlation,
        "p95_abs_error_m_s": float(np.percentile(np.abs(error), 95)),
    }


def roi_metrics(prediction: np.ndarray, target: np.ndarray, z_m: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for name, mask in (
        ("all", np.ones(len(z_m), dtype=bool)),
        ("5_40", (z_m >= 0.005) & (z_m < 0.040)),
        ("20_40", (z_m >= 0.020) & (z_m < 0.040)),
    ):
        out[name] = map_metrics(prediction[mask], target[mask])
    return out


def aggregate(records: list[dict], variant: str, roi: str) -> dict[str, dict[str, float]]:
    metrics = ("mae_m_s", "rmse_m_s", "bias_m_s", "correlation", "p95_abs_error_m_s")
    result = {}
    for metric in metrics:
        values = np.asarray(
            [record["variants"][variant][roi][metric] for record in records],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
        }
    return result


def _travel_time(phi: np.ndarray, speed: np.ndarray, dx: tuple[float, float]) -> np.ndarray:
    travel = skfmm.travel_time(phi, speed, dx=dx, order=2)
    if np.ma.isMaskedArray(travel):
        travel = travel.filled(np.nan)
    return np.asarray(travel, dtype=np.float64)


def compounded_eikonal_input(
    sos_zx: np.ndarray,
    *,
    x_m: np.ndarray,
    z_m: np.ndarray,
    angles_rad: np.ndarray,
    element_x_m: np.ndarray,
    c_reference: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the MATLAB v1 compounded plane-wave Eikonal label on a coarse grid."""
    x_coarse = np.linspace(x_m[0] - X_PAD_M, x_m[-1] + X_PAD_M, COARSE_NX)
    z_coarse = np.linspace(0.0, z_m[-1], COARSE_NZ)
    dx = float(np.mean(np.diff(x_coarse)))
    dz = float(np.mean(np.diff(z_coarse)))
    zz_c, xx_c = np.meshgrid(z_coarse, x_coarse, indexing="ij")
    row = (zz_c - z_m[0]) / float(np.mean(np.diff(z_m)))
    col = (xx_c - x_m[0]) / float(np.mean(np.diff(x_m)))
    sos_coarse = map_coordinates(sos_zx.astype(np.float64), np.stack([row, col]), order=1, mode="nearest")
    homogeneous = np.full_like(sos_coarse, c_reference)
    sample_coords = np.stack(
        np.meshgrid(
            (z_m - z_coarse[0]) / dz,
            (x_m - x_coarse[0]) / dx,
            indexing="ij",
        )
    )

    def excess(phi: np.ndarray) -> np.ndarray:
        delta = _travel_time(phi, sos_coarse, (dz, dx)) - _travel_time(phi, homogeneous, (dz, dx))
        return map_coordinates(delta, sample_coords, order=1, mode="nearest")

    tx = np.stack([excess(zz_c * np.cos(angle) + xx_c * np.sin(angle)) for angle in angles_rad])
    source_radius = min(dx, dz)
    anchors = np.linspace(element_x_m[0], element_x_m[-1], N_RX_ANCHORS)
    rx = np.stack([excess(np.hypot(zz_c, xx_c - anchor) - source_radius) for anchor in anchors])
    per_angle = tx + rx.mean(axis=0, keepdims=True)
    time_shift = per_angle.mean(axis=0).astype(np.float32)
    confidence = np.exp(-per_angle.std(axis=0) / SHIFT_SCALE_S).astype(np.float32)
    return time_shift, np.clip(confidence, 0.0, 1.0)


def network_input(time_shift: np.ndarray, confidence: np.ndarray, shift_clip: float) -> torch.Tensor:
    normalized = np.clip(time_shift / SHIFT_SCALE_S, -shift_clip, shift_clip).astype(np.float32)
    return torch.from_numpy(np.stack((normalized, confidence), axis=0))[None]


def visualize_sample(
    time_shift: np.ndarray,
    confidence: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    c_reference: float,
    metrics: dict[str, float],
    out_path: Path,
    dpi: int,
) -> None:
    x0_mm, x1_mm = X0_M * 1e3, (X0_M + (NATIVE_NX - 1) * DX_M) * 1e3
    z0_mm, z1_mm = Z0_M * 1e3, (Z0_M + (NATIVE_NZ - 1) * DX_M) * 1e3
    extent = (x0_mm, x1_mm, z1_mm, z0_mm)
    error = prediction - target
    speed_min = float(min(target.min(), prediction.min(), baseline.min(), 1400.0))
    speed_max = float(max(target.max(), prediction.max(), baseline.max(), 1650.0))
    error_limit = max(20.0, float(np.percentile(np.abs(error), 99)))
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.8), constrained_layout=True)
    panels = [
        (time_shift * 1e9, "Oracle Eikonal time shift", "coolwarm", None, None, "ns"),
        (confidence, "Angle-agreement confidence", "viridis", 0.0, 1.0, None),
        (baseline, f"Constant {c_reference:.0f} m/s", "turbo", speed_min, speed_max, "m/s"),
        (target, "Ground-truth SoS", "turbo", speed_min, speed_max, "m/s"),
        (prediction, "UNet prediction", "turbo", speed_min, speed_max, "m/s"),
        (error, "Prediction error", "RdBu_r", -error_limit, error_limit, "m/s"),
    ]
    for axis, (image, title, cmap, vmin, vmax, unit) in zip(axes.ravel(), panels):
        view = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Lateral x (mm)")
        axis.set_ylabel("Depth z (mm)")
        colorbar = figure.colorbar(view, ax=axis, fraction=0.046, pad=0.03)
        if unit:
            colorbar.set_label(unit)
    figure.suptitle(
        f"MAE {metrics['mae_m_s']:.2f} m/s | RMSE {metrics['rmse_m_s']:.2f} m/s | "
        f"corr {metrics['correlation']:.3f}",
        fontsize=13,
    )
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def load_split_maps(root: Path, records: list[dict]) -> np.ndarray:
    maps = []
    for record in records:
        sample = torch.load(root / record["path"], map_location="cpu", weights_only=False)
        maps.append(np.asarray(sample["c"], dtype=np.float32))
    return np.stack(maps, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="/home/zhuangyang/fmmodel/NACRF/results/unet_sos_inversion_v1/best.pt",
    )
    parser.add_argument(
        "--data-root",
        default="/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "out" / "unet_sos_l11_ultrawave_20260917"),
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--c-reference", type=float, default=1540.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-viz", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    if str(cfg["model"].get("bottleneck_type", "kan")) != "unet":
        raise ValueError(f"expected UNet bottleneck, got {cfg['model']}")
    device = resolve_device(args.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    root = Path(args.data_root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    records = [item for item in index["samples"] if item["split"] == args.split]
    train_records = [item for item in index["samples"] if item["split"] == "train"]
    if not records:
        raise ValueError(f"no samples for split {args.split}")
    if args.limit:
        records = records[: int(args.limit)]

    x_model, z_model = model_axes()
    x_native, z_native = native_axes()
    element_x = (np.arange(NATIVE_NX) - (NATIVE_NX - 1) / 2.0) * DX_M
    shift_clip = float(cfg["dataset"]["shift_clip"])
    c_reference = float(args.c_reference)

    print(f"loading {len(train_records)} train maps for baselines", flush=True)
    train_maps = load_split_maps(root, train_records)
    train_mean_map = train_maps.mean(axis=0)
    train_mean_scalar = float(train_maps.mean())
    del train_maps

    out_dir = Path(args.out_dir)
    figure_dir = out_dir / "figures"
    pred_dir = out_dir / "predictions"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    visual_indices = set(
        np.linspace(0, len(records) - 1, min(args.num_viz, len(records)), dtype=int).tolist()
    )
    rows = []
    start = time.time()
    for index_i, record in enumerate(records):
        sample = torch.load(root / record["path"], map_location="cpu", weights_only=False)
        target = np.asarray(sample["c"], dtype=np.float32)
        angles = np.deg2rad(np.asarray(sample["metadata"]["angles_deg"], dtype=np.float64))
        sos_model = resample_z_x(target, (MODEL_NZ, MODEL_NX))
        time_shift, confidence = compounded_eikonal_input(
            sos_model,
            x_m=x_model,
            z_m=z_model,
            angles_rad=angles,
            element_x_m=element_x,
            c_reference=c_reference,
        )
        inputs = network_input(time_shift, confidence, shift_clip).to(device)
        reference = torch.tensor([c_reference], dtype=torch.float32, device=device)
        prediction_model = model(inputs, reference)["sos"][0, 0].cpu().numpy().astype(np.float32)
        prediction = resample_z_x(prediction_model, (NATIVE_NZ, NATIVE_NX))
        constant = np.full_like(target, c_reference)
        variants = {
            "unet_oracle_eikonal": prediction,
            "constant_reference": constant,
            "train_mean_scalar": np.full_like(target, train_mean_scalar),
            "train_mean_map": train_mean_map,
        }
        row = {
            "id": record["id"],
            "path": record["path"],
            "base_anatomy_id": record.get("base_anatomy_id"),
            "c_reference_m_s": c_reference,
            "input": {
                "time_shift_ns": [float(time_shift.min() * 1e9), float(time_shift.max() * 1e9)],
                "time_shift_mean_ns": float(time_shift.mean() * 1e9),
                "confidence": [float(confidence.min()), float(confidence.max())],
                "clipped_fraction": float(np.mean(np.abs(time_shift) > shift_clip * SHIFT_SCALE_S)),
            },
            "prediction_m_s": [float(prediction.min()), float(prediction.mean()), float(prediction.max())],
            "variants": {name: roi_metrics(value, target, z_native) for name, value in variants.items()},
        }
        rows.append(row)
        np.savez_compressed(
            pred_dir / f"{record['id']}.npz",
            c_pred=prediction,
            c_gt=target,
            time_shift=time_shift,
            confidence=confidence,
            c_pred_model=prediction_model,
            cx=x_native,
            cz=z_native,
        )
        if index_i in visual_indices:
            visualize_sample(
                resample_z_x(time_shift, (NATIVE_NZ, NATIVE_NX)),
                resample_z_x(confidence, (NATIVE_NZ, NATIVE_NX)),
                target,
                prediction,
                constant,
                c_reference,
                row["variants"]["unet_oracle_eikonal"]["all"],
                figure_dir / f"{record['id']}.png",
                args.dpi,
            )
        print(
            f"{index_i + 1:03d}/{len(records)} {record['id']}  "
            f"mae={row['variants']['unet_oracle_eikonal']['all']['mae_m_s']:.2f}  "
            f"rmse={row['variants']['unet_oracle_eikonal']['all']['rmse_m_s']:.2f}  "
            f"corr={row['variants']['unet_oracle_eikonal']['all']['correlation']:.3f}  "
            f"{time.time() - start:.1f}s",
            flush=True,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_root": str(root),
        "split": args.split,
        "n_samples": len(rows),
        "oracle_input": True,
        "warning": (
            "Input time_shift/confidence were built from ground-truth c via coarse "
            "Eikonal first arrivals. This is not RF-only inference."
        ),
        "c_reference_m_s": c_reference,
        "grid": {
            "native_zx": [NATIVE_NZ, NATIVE_NX],
            "model_zx": [MODEL_NZ, MODEL_NX],
            "dx_m": DX_M,
            "x_m": [float(x_native[0]), float(x_native[-1])],
            "z_m": [float(z_native[0]), float(z_native[-1])],
        },
        "train_mean_speed_m_s": train_mean_scalar,
        "visualization_ids": [records[i]["id"] for i in sorted(visual_indices)],
        "aggregate": {
            variant: {roi: aggregate(rows, variant, roi) for roi in ("all", "5_40", "20_40")}
            for variant in rows[0]["variants"]
        },
        "per_sample": rows,
    }
    report_path = out_dir / f"eval_{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"]["unet_oracle_eikonal"], indent=2), flush=True)
    print(f"report: {report_path}", flush=True)
    print(f"figures: {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
