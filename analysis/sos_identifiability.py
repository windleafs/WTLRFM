"""Low-dimensional local identifiability analysis for SoS fields.

The routines in this module operate on a physically interpretable basis B of
sound-speed perturbations and never require an explicit full image Jacobian.
For an observation map F(c), finite differences give columns J B. Their Gram
matrix approximates B^T J^T J B. A generalized eigendecomposition against
B^T B then returns the most/least observable directions in the span of B.
"""

from __future__ import annotations

import numpy as np


def make_roi_mask(x, z, z_min=3e-3, z_max=43e-3):
    x = np.asarray(x, np.float64)
    z = np.asarray(z, np.float64)
    if x.ndim != 1 or z.ndim != 1:
        raise ValueError("x and z must be one-dimensional coordinate arrays")
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(z) > 0)):
        raise ValueError("x and z must be strictly increasing")
    return np.broadcast_to(((z >= z_min) & (z <= z_max))[None, :],
                           (len(x), len(z))).copy()


def _bounded_mode(value, roi):
    value = np.asarray(value, np.float64) * roi
    peak = float(np.max(np.abs(value)))
    if not np.isfinite(peak) or peak <= 1e-12:
        raise ValueError("degenerate basis mode")
    return value / peak


def build_physical_basis(x, z, *, z_min=3e-3, z_max=43e-3,
                         depth_slabs=5, lateral_modes=4, axial_modes=4,
                         gaussian_x=3, gaussian_z=2,
                         gaussian_sigma_mm=4.0):
    """Build bounded, interpretable SoS perturbation modes on an (x,z) grid.

    The basis contains a global offset, adjacent depth-slab contrasts, 1-D
    lateral/axial cosine modes and mean-removed local Gaussian blobs. Modes
    are bounded to max(abs(mode)) == 1 inside the ROI, so delta_c in a
    finite-difference experiment is also a hard bound on the pointwise speed
    perturbation. The basis need not be orthogonal; solve_generalized_spectrum
    accounts for B^T B explicitly.
    """
    x = np.asarray(x, np.float64)
    z = np.asarray(z, np.float64)
    roi = make_roi_mask(x, z, z_min, z_max)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    modes, names = [], []

    modes.append(_bounded_mode(np.ones_like(xx), roi))
    names.append("global")

    depth_slabs = int(depth_slabs)
    if depth_slabs >= 2:
        edges = np.linspace(z_min, z_max, depth_slabs + 1)
        slabs = [((zz >= edges[i]) & (zz < edges[i + 1])).astype(np.float64)
                 for i in range(depth_slabs)]
        slabs[-1] = ((zz >= edges[-2]) & (zz <= edges[-1])).astype(np.float64)
        for i in range(depth_slabs - 1):
            modes.append(_bounded_mode(slabs[i] - slabs[i + 1], roi))
            names.append(f"depth_contrast_{i}_{i+1}")

    x0, x1 = float(x.min()), float(x.max())
    xn = (xx - x0) / max(x1 - x0, np.finfo(float).eps)
    zn = (zz - z_min) / max(z_max - z_min, np.finfo(float).eps)
    for k in range(1, int(lateral_modes) + 1):
        modes.append(_bounded_mode(np.cos(np.pi * k * xn), roi))
        names.append(f"lateral_cos_{k}")
    for k in range(1, int(axial_modes) + 1):
        modes.append(_bounded_mode(np.cos(np.pi * k * zn), roi))
        names.append(f"axial_cos_{k}")

    gx, gz = int(gaussian_x), int(gaussian_z)
    sigma = float(gaussian_sigma_mm) * 1e-3
    if gx > 0 and gz > 0 and sigma > 0:
        xs = np.linspace(x0 + .15 * (x1 - x0), x1 - .15 * (x1 - x0), gx)
        zs = np.linspace(z_min + .2 * (z_max - z_min),
                         z_max - .2 * (z_max - z_min), gz)
        roi_count = max(int(roi.sum()), 1)
        for iz, zc in enumerate(zs):
            for ix, xc in enumerate(xs):
                g = np.exp(-((xx - xc) ** 2 + (zz - zc) ** 2) / (2 * sigma ** 2)) * roi
                g = (g - float(g.sum()) / roi_count) * roi
                modes.append(_bounded_mode(g, roi))
                names.append(f"gaussian_x{ix}_z{iz}")

    basis = np.stack(modes).astype(np.float32)
    if not np.isfinite(basis).all():
        raise FloatingPointError("non-finite basis")
    return basis, names, roi


def state_gram(basis, roi_mask=None):
    """Return the SoS-space metric B^T B as a mean inner product."""
    b = np.asarray(basis, np.float64)
    if b.ndim != 3:
        raise ValueError("basis must have shape [K,nx,nz]")
    if roi_mask is None:
        mask = np.ones(b.shape[1:], bool)
    else:
        mask = np.asarray(roi_mask, bool)
        if mask.shape != b.shape[1:]:
            raise ValueError("roi_mask shape must match basis spatial shape")
    flat = b[:, mask]
    return (flat @ flat.T) / max(flat.shape[1], 1)


def relative_response(plus, minus, baseline, delta):
    """Central-difference response whitened by baseline RMS and feature count.

    The returned flattened vector has norm equal to the relative RMS change
    per m/s. Complex observations are kept complex; real_gram uses the real
    part of the Hermitian inner product, equivalent to stacking Re/Im.
    """
    plus = np.asarray(plus)
    minus = np.asarray(minus)
    base = np.asarray(baseline)
    if plus.shape != minus.shape or plus.shape != base.shape:
        raise ValueError("plus, minus and baseline must have identical shapes")
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    rms = float(np.sqrt(np.mean(np.abs(base) ** 2)))
    if not np.isfinite(rms) or rms <= 1e-20:
        raise ValueError("baseline observation has zero/invalid RMS")
    d = (plus - minus) / (2.0 * delta)
    dtype = np.complex64 if np.iscomplexobj(d) else np.float32
    return (d.reshape(-1) / (rms * np.sqrt(max(d.size, 1)))).astype(dtype, copy=False)


def real_gram(responses):
    """Gram matrix using Re(<a,b>), valid for real or complex responses."""
    responses = list(responses)
    k = len(responses)
    if k == 0:
        raise ValueError("at least one response is required")
    out = np.empty((k, k), np.float64)
    for i in range(k):
        a = np.asarray(responses[i]).reshape(-1)
        for j in range(i + 1):
            b = np.asarray(responses[j]).reshape(-1)
            if a.size != b.size:
                raise ValueError("response vectors must have the same size")
            value = float(np.vdot(a, b).real)
            out[i, j] = out[j, i] = value
    return out


def solve_generalized_spectrum(observation_gram, state_metric, rcond=1e-8):
    """Solve (JB)^T(JB) a = lambda (B^T B) a without SciPy.

    Returns singular values in ascending order and coefficient vectors whose
    SoS-space norm a^T(B^T B)a is one. Near-dependent basis directions are
    removed according to rcond.
    """
    gy = np.asarray(observation_gram, np.float64)
    gc = np.asarray(state_metric, np.float64)
    if gy.shape != gc.shape or gy.ndim != 2 or gy.shape[0] != gy.shape[1]:
        raise ValueError("Gram matrices must be square and have the same shape")
    gy = .5 * (gy + gy.T)
    gc = .5 * (gc + gc.T)
    s, q = np.linalg.eigh(gc)
    keep = s > max(float(rcond), 0.0) * max(float(s.max()), 1e-30)
    if not np.any(keep):
        raise ValueError("state metric has no numerically independent directions")
    white = q[:, keep] / np.sqrt(s[keep])[None, :]
    h0 = white.T @ gy @ white
    h = .5 * (h0 + h0.T)
    lam, v = np.linalg.eigh(h)
    lam = np.maximum(lam, 0.0)
    coeff = white @ v
    for j in range(coeff.shape[1]):
        i = int(np.argmax(np.abs(coeff[:, j])))
        if coeff[i, j] < 0:
            coeff[:, j] *= -1
    return {
        "singular_values": np.sqrt(lam),
        "eigenvalues": lam,
        "coefficients": coeff.T,
        "state_rank": int(keep.sum()),
    }


def synthesize_modes(coefficients, basis):
    coeff = np.asarray(coefficients, np.float64)
    b = np.asarray(basis, np.float64)
    if coeff.ndim == 1:
        coeff = coeff[None]
    if coeff.shape[1] != b.shape[0]:
        raise ValueError("coefficient width must match basis count")
    return np.einsum("mk,kxz->mxz", coeff, b)


def network_mode_diagnostics(coefficients, state_metric, network_gram,
                             state_network_cross):
    """Aligned gain/leakage of a network Jacobian along selected SoS modes."""
    coeff = np.asarray(coefficients, np.float64)
    gc = np.asarray(state_metric, np.float64)
    gn = np.asarray(network_gram, np.float64)
    cross = np.asarray(state_network_cross, np.float64)
    rows = []
    for a in coeff:
        denom = float(a @ gc @ a)
        if denom <= 0:
            raise ValueError("mode has zero SoS-space norm")
        gain = float(a @ cross @ a) / denom
        pred_norm2 = max(float(a @ gn @ a) / denom, 0.0)
        pred_norm = float(np.sqrt(pred_norm2))
        leakage = float(np.sqrt(max(pred_norm2 - gain * gain, 0.0)))
        rows.append((gain, pred_norm, leakage))
    return np.asarray(rows, np.float64)
