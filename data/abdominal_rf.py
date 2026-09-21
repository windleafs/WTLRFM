"""k-Wave RF adapter. Arrays exposed to the model use [x, z] order."""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.signal import hilbert

from . import geometry as G


def read_sample(path, dataset_root):
    reader = Path(dataset_root) / "python/liver_rf_dataset.py"
    spec = importlib.util.spec_from_file_location("liver_reader", reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    read = module._read_matlab_dataset
    # Travel-time labels are neither network inputs nor SoS targets.
    with h5py.File(path, "r") as f:
        meta = json.loads(bytes(read(f, "/meta/json").astype(np.uint8)).decode())
        out = {k: read(f, p) for k, p in {
            "rf": "/rf/channel_data", "time": "/rf/time_s",
            "xe": "/probe/element_x_m", "angles": "/probe/angles_deg",
            "launch": "/probe/launch_delays_s", "c": "/medium/sound_speed_mps",
            "seg": "/medium/segmentation", "x": "/medium/x_m",
            "z": "/medium/z_m"}.items()}
        if f["/rf/channel_data"].attrs["units"] not in ("Pa", b"Pa"):
            raise ValueError("Expected real pressure RF in Pa")
    out["meta"] = meta
    t = out["time"]
    fs = 1.0 / np.median(np.diff(t))
    expected = (len(t), len(out["xe"]), len(out["angles"]))
    if out["rf"].shape != expected or not np.allclose(np.diff(t), 1 / fs, rtol=1e-6, atol=1e-12):
        raise ValueError("RF dimensions or time sampling do not match metadata")
    if not np.allclose(out["angles"], [-6, 0, 6]):
        raise ValueError("This cache configuration requires angles [-6, 0, 6]")
    if not np.isfinite(out["rf"]).all():
        raise ValueError("Nonfinite RF")
    out["fs"] = fs
    out["fc"] = float(meta["config"]["probe"]["center_frequency_hz"])
    if not np.isclose(fs, meta["config"]["probe"]["receive_sample_rate_hz"]):
        raise ValueError("Time axis disagrees with sampling metadata")
    return out


def analytic_iq(rf, time, fc):
    """Mix on the absolute acquisition clock; restore carrier at query time."""
    analytic = hilbert(np.asarray(rf, np.float32), axis=0)
    return (analytic * np.exp(-2j * np.pi * fc * time[:, None, None])).transpose(
        1, 2, 0).astype(np.complex64)


@torch.no_grad()
def build_condition(sample, device="cpu", chunk=2048):
    """Finite-aperture earliest-arrival DAS using stored launch delays.

    Native-time rounding is not recorded; use nominal acquisition delays.
    Burst timing is acquisition metadata, never inferred from SoS truth.
    """
    xi, zi = G.x_grid(), G.z_grid()
    dev = torch.device(device)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    iq = torch.as_tensor(analytic_iq(sample["rf"], sample["time"], sample["fc"]), device=dev)
    xe, launch = tensor(sample["xe"]), tensor(sample["launch"])
    X, Z = np.meshgrid(xi, zi, indexing="ij")
    x, z = tensor(X.ravel()), tensor(Z.ravel())
    ne, na, nt = iq.shape
    fs, fc, t0 = sample["fs"], sample["fc"], float(sample["time"][0])
    burst_center = sample["meta"]["config"]["probe"]["source_cycles"] / (2 * fc)
    full = torch.zeros((3, x.numel()), dtype=torch.complex64, device=dev)
    angle = torch.zeros((na, x.numel()), dtype=torch.complex64, device=dev)
    subap = torch.zeros((4, x.numel()), dtype=torch.complex64, device=dev)
    for start in range(0, x.numel(), chunk):
        sl = slice(start, start + chunk)
        dist = torch.sqrt((x[sl][None] - xe[:, None]).square() + z[sl][None].square())
        for si, speed in enumerate((1450., 1500., 1550.)):
            rx = dist / speed
            for a in range(na):
                tx = (rx + launch[:, a, None]).amin(dim=0) + burst_center
                tau = rx + tx[None]
                index = (tau - t0) * fs
                i0 = index.floor().long()
                valid = (i0 >= 0) & (i0 < nt - 1)
                frac = index - i0
                safe = i0.clamp(0, nt - 2)
                d = iq[:, a]
                v = (torch.gather(d, 1, safe) * (1 - frac)
                     + torch.gather(d, 1, safe + 1) * frac)
                v = (v * torch.exp(2j * torch.pi * fc * tau) * valid).to(torch.complex64)
                image = v.sum(dim=0)
                full[si, sl] += image
                if si == 1:
                    angle[a, sl] = image
                    for k, group in enumerate(torch.tensor_split(v, 4, dim=0)):
                        subap[k, sl] += group.sum(dim=0)
    channels = []
    # Exclude near-source pressure from normalization without using tissue labels.
    norm_mask = torch.as_tensor(Z.ravel() >= 3e-3, device=dev)
    for group in (full, angle, subap):
        scale = group[:, norm_mask].abs().square().mean(dim=1).sqrt().mean().clamp_min(1e-20)
        mag = torch.asinh(group.abs() / scale) / np.arcsinh(3.)
        phase = group.angle()
        pair = torch.stack((mag * phase.cos(), mag * phase.sin()), dim=1)
        channels.append(pair.flatten(0, 1))
    cond = torch.cat(channels).reshape(20, len(xi), len(zi)).cpu().numpy()
    if not np.isfinite(cond).all():
        raise ValueError("Nonfinite DAS condition")
    return cond.astype(np.float16)


def targets(sample):
    xi, zi = G.x_grid(), G.z_grid()
    x, z = sample["x"].astype(float), sample["z"].astype(float)
    dx, dz = np.diff(x)[0], np.diff(z)[0]
    # Even k-Wave lateral grids start at -N*dx/2, not -(N-1)*dx/2.
    x = x - (dx / 2 if len(x) % 2 == 0 else 0)
    z = z - dz  # Source and receiver occupy the second axial row.
    xx, zz = np.meshgrid(xi, zi, indexing="ij")
    coords = [(zz - z[0]) / dz, (xx - x[0]) / dx]
    c = map_coordinates(sample["c"], coords, order=1, mode="nearest").astype(np.float32)
    seg = map_coordinates(sample["seg"], coords, order=0, mode="nearest").astype(np.uint8)
    support = (xx >= x[0]) & (xx <= x[-1]) & (zz >= z[0]) & (zz <= z[-1])
    valid = support & (seg >= 2) & (seg <= 9) & (zz >= 3e-3)
    wall = valid & (seg <= 6)
    if not np.isfinite(c).all() or (c <= 0).any() or not valid.any():
        raise ValueError("Invalid SoS target or empty tissue mask")
    return dict(c_gt=c, segmentation=seg, valid_mask=valid, wall_mask=wall, cx=xi, cz=zi)
