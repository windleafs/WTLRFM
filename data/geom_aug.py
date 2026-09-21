"""Random geometric perturbation of (DAS condition, SoS map) pairs.

The same sampling grid is applied to every condition channel and to the
sound-speed target, so the pair stays aligned.  This is domain randomisation
(the probe is no longer exactly at the top of a rigid grid), not a physically
exact re-acquisition: the point is to stop the network from memorising the
OpenBreast dome layout.

Transforms (all optional, composed on the sampling coordinates):
  * anisotropic stretch (x / z independently)
  * rotation about the image centre
  * translation
  * smooth elastic warp (low-pass random displacement)
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


DEFAULTS = dict(
    p=0.8,
    translate_x_px=10.0,
    translate_z_px=16.0,
    rotate_deg=12.0,
    scale_x=(0.85, 1.20),
    scale_z=(0.80, 1.25),
    elastic_amp_px=5.0,
    elastic_sigma_px=14.0,
    cond_cval=0.0,
    c_cval=1500.0,
)


def _as_range(v):
    if v is None:
        return None
    if np.isscalar(v):
        v = float(v)
        return (-v, v) if v >= 0 else (v, -v)
    a, b = float(v[0]), float(v[1])
    return (a, b) if a <= b else (b, a)


def _scale_factor(rng, spec):
    """``spec`` is a (lo, hi) factor range, or a single number s meaning [1/s, s]."""
    if spec is None:
        return 1.0
    if np.isscalar(spec):
        s = float(spec)
        lo, hi = (1.0 / s, s) if s >= 1.0 else (s, 1.0 / s)
    else:
        lo, hi = float(spec[0]), float(spec[1])
        if lo > hi:
            lo, hi = hi, lo
    return float(rng.uniform(lo, hi))


def sample_coords(nx, nz, rng, cfg):
    """Return (Is, Js) source coordinates for output pixels [nx, nz]."""
    i0 = 0.5 * (nx - 1)
    j0 = 0.5 * (nz - 1)
    I, J = np.meshgrid(np.arange(nx, dtype=np.float64),
                       np.arange(nz, dtype=np.float64), indexing="ij")
    di, dj = I - i0, J - j0

    sx = _scale_factor(rng, cfg.get("scale_x", DEFAULTS["scale_x"]))
    sz = _scale_factor(rng, cfg.get("scale_z", DEFAULTS["scale_z"]))
    ang = np.deg2rad(rng.uniform(*_as_range(cfg.get("rotate_deg",
                                                   DEFAULTS["rotate_deg"]))))
    ca, sa = np.cos(ang), np.sin(ang)
    di_s, dj_s = sx * di, sz * dj
    di_r = ca * di_s - sa * dj_s
    dj_r = sa * di_s + ca * dj_s

    tx = rng.uniform(*_as_range(cfg.get("translate_x_px",
                                        DEFAULTS["translate_x_px"])))
    tz = rng.uniform(*_as_range(cfg.get("translate_z_px",
                                        DEFAULTS["translate_z_px"])))
    Is = di_r + i0 + tx
    Js = dj_r + j0 + tz

    amp = float(cfg.get("elastic_amp_px", DEFAULTS["elastic_amp_px"]) or 0.0)
    sig = float(cfg.get("elastic_sigma_px", DEFAULTS["elastic_sigma_px"]) or 1.0)
    if amp > 0:
        ex = gaussian_filter(rng.normal(size=(nx, nz)), sigma=sig, mode="nearest")
        ez = gaussian_filter(rng.normal(size=(nx, nz)), sigma=sig, mode="nearest")
        ex *= amp / (ex.std() + 1e-8)
        ez *= amp / (ez.std() + 1e-8)
        Is = Is + ex
        Js = Js + ez
    return Is.astype(np.float64), Js.astype(np.float64)


def _warp_field(field, Is, Js, cval):
    """Bilinear sample ``field[..., nx, nz]`` at (Is, Js)."""
    field = np.asarray(field)
    nx, nz = field.shape[-2:]
    coords = np.stack([Is.ravel(), Js.ravel()], axis=0)
    lead = field.shape[:-2]
    flat = field.reshape((-1, nx, nz))
    out = np.empty((flat.shape[0], nx, nz), dtype=np.float32)
    for k in range(flat.shape[0]):
        out[k] = map_coordinates(flat[k].astype(np.float64), coords, order=1,
                                 mode="constant", cval=float(cval)
                                 ).reshape(nx, nz)
    return out.reshape(lead + (nx, nz)).astype(np.float32, copy=False)


def perturb(cond, u_gt, c_gt, cfg=None, rng=None):
    """Warp condition and targets with one shared random grid.

    Args:
        cond: [C, nx, nz]
        u_gt: [1, nx, nz] or [nx, nz]
        c_gt: [nx, nz] or [1, nx, nz]
    Returns:
        cond, u_gt, c_gt with the same shapes as the inputs.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() >= float(cfg.get("p", 1.0)):
        return cond, u_gt, c_gt
    cond = np.asarray(cond, np.float32)
    u_was_2d = np.asarray(u_gt).ndim == 2
    c_was_2d = np.asarray(c_gt).ndim == 2
    u_gt = np.asarray(u_gt, np.float32)
    c_gt = np.asarray(c_gt, np.float32)
    if u_gt.ndim == 2:
        u_gt = u_gt[None]
    if c_gt.ndim == 2:
        c_gt = c_gt[None]
    nx, nz = cond.shape[-2:]
    Is, Js = sample_coords(nx, nz, rng, cfg)
    u_cval = 0.0  # log(1500/1500)/scale
    cond = _warp_field(cond, Is, Js, cfg.get("cond_cval", 0.0))
    u_gt = _warp_field(u_gt, Is, Js, u_cval)
    c_gt = _warp_field(c_gt, Is, Js, cfg.get("c_cval", 1500.0))
    if u_was_2d:
        u_gt = u_gt[0]
    if c_was_2d:
        c_gt = c_gt[0]
    return cond, u_gt, c_gt
