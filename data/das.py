"""DAS beamforming of the OpenBreastUS multi-angle plane-wave IQ data.

The IQ data are baseband (analytic) signals whose time origin is the instant
the transmitted plane wavefront crosses the probe centre (x = 0, z = 0), and
the direct wave has already been removed.  The two-way delay is therefore

    tau(x, z, theta) = (x sin(theta) + z cos(theta)) / c_bf      (transmit)
                     + sqrt((x - x_e)^2 + z^2) / c_bf           (receive)

and each delayed baseband sample carries a factor exp(-i 2 pi f_c tau), so the
carrier phase exp(+i 2 pi f_c tau) must be re-applied before coherent summing
(equivalent to up-mixing to RF first).  This mirrors ``code/validate_dataset.py``
of the dataset repository.

Phase handling (important for sound-speed estimation)
-----------------------------------------------------
The DAS images are *linear, phase-preserving* functionals of the raw IQ: no
envelope detection and no per-component nonlinearity.  The network condition
is built in polar form -- the magnitude is asinh-compressed and the phase is
kept exactly -- so the RF/IQ phase information survives into the network.
"""

import numpy as np

from . import geometry as G


def das_complex(iq, c_bf, xe, xi, zi, angles_deg, fs=G.FS, fc=G.FC,
                chunk=8192):
    """Delay-and-sum beamforming at a constant assumed sound speed.

    Args:
        iq: complex64 [n_elem, n_angle, n_sample] baseband scattered field.
        c_bf: assumed beamforming sound speed [m/s].
        xe: [n_elem] element x positions [m].
        xi: [nx] lateral pixel coordinates [m].
        zi: [nz] axial pixel coordinates [m] (depth below probe face).
        angles_deg: [n_angle] transmit plane-wave angles [deg].
        fs, fc: IQ sample rate / centre frequency [Hz].
        chunk: pixel-chunk size to bound temporary memory.

    Returns:
        complex64 [nx, nz] coherently compounded image (all angles summed).
    """
    return das_groups(iq, xe, angles_deg, xi, zi, fs, fc,
                      speeds=(float(c_bf),))["full"][0]


def das_groups(iq, xe, angles_deg, xi, zi, fs=G.FS, fc=G.FC,
               speeds=(1450.0, 1500.0, 1550.0), ref_speed=1500.0,
               n_subap=4, chunk=8192, t0=0.0):
    """Beamform the IQ data into several phase-preserving image groups.

    Returns a dict of complex64 arrays:
        ``full``      [n_speeds, nx, nz]  full-aperture, one image per speed
        ``per_angle`` [n_angle,  nx, nz]  one image per transmit angle
                                          (full receive aperture, ref_speed)
        ``subap``     [n_subap,  nx, nz]  receive sub-aperture images
                                          (ref_speed)

    All groups are linear functionals of the same raw IQ, so the relative
    complex phases between channels within a group are physically meaningful
    (they encode the arrival-time mismatch caused by a wrong assumed speed).
    """
    iq = np.asarray(iq)
    xe = np.asarray(xe, np.float64)
    xi = np.asarray(xi, np.float32)
    zi = np.asarray(zi, np.float32)
    angles_deg = np.asarray(angles_deg, np.float64)
    n_ang = iq.shape[1]
    nsamp = iq.shape[-1]
    ne = xe.size

    X, Z = np.meshgrid(xi, zi, indexing="ij")
    Xf, Zf = X.ravel().astype(np.float64), Z.ravel().astype(np.float64)
    npix = Xf.size

    # receive sub-apertures (contiguous element blocks)
    n_subap = max(1, int(n_subap))
    edges = np.linspace(0, ne, n_subap + 1).round().astype(int)
    sub_idx = [np.arange(edges[k], edges[k + 1]) for k in range(n_subap)]

    speeds = tuple(float(c) for c in speeds)
    if not any(np.isclose(ref_speed, c) for c in speeds):
        raise ValueError(f"ref_speed {ref_speed} must be one of speeds {speeds}")
    full = np.zeros((len(speeds), npix), np.complex64)
    per_angle = None
    subap = None

    for si, c_bf in enumerate(speeds):
        is_ref = np.isclose(c_bf, ref_speed)
        if is_ref:
            per_angle = np.zeros((n_ang, npix), np.complex64)
            subap = np.zeros((n_subap, npix), np.complex64)
        for a in range(n_ang):
            th = np.deg2rad(angles_deg[a])
            tau_tx = (Xf * np.sin(th) + Zf * np.cos(th)) / c_bf
            d = iq[:, a, :]
            acc_ang = np.zeros(npix, np.complex64)
            acc_sub = np.zeros((n_subap, npix), np.complex64)
            for s in range(0, npix, chunk):
                sl = slice(s, min(s + chunk, npix))
                tau_rx = np.sqrt((Xf[sl, None] - xe[None, :]) ** 2
                                 + Zf[sl, None] ** 2) / c_bf      # [npc, ne]
                tau = tau_tx[sl][None, :] + tau_rx.T              # [ne, npc]
                idx = (tau - t0) * fs
                i0 = np.floor(idx).astype(np.int32)
                w = idx - i0
                valid = (i0 >= 0) & (i0 < nsamp - 1)
                i0c = np.clip(i0, 0, nsamp - 2)
                v = (np.take_along_axis(d, i0c, axis=-1) * (1.0 - w)
                     + np.take_along_axis(d, i0c + 1, axis=-1) * w)
                v = np.where(valid, v, 0.0) * np.exp(2j * np.pi * fc * tau)
                acc_ang[sl] = v.sum(axis=0)
                for k, eidx in enumerate(sub_idx):
                    acc_sub[k, sl] = v[eidx].sum(axis=0)
            full[si] += acc_ang
            if is_ref:
                per_angle[a] = acc_ang
                subap += acc_sub

    out = {"full": full.reshape(len(speeds), xi.size, zi.size)}
    out["per_angle"] = (None if per_angle is None
                        else per_angle.reshape(n_ang, xi.size, zi.size))
    out["subap"] = (None if subap is None
                    else subap.reshape(n_subap, xi.size, zi.size))
    return out


def _sum_delayed(iq_a, tau, fs, fc, nsamp, chunk=8192):
    """Linear-interpolate baseband IQ at delays tau and restore carrier phase.

    iq_a: [ne, nsamp], tau: [ne, npix] -> complex64 [npix]
    """
    npix = tau.shape[1]
    acc = np.zeros(npix, np.complex64)
    for s in range(0, npix, chunk):
        sl = slice(s, min(s + chunk, npix))
        t = tau[:, sl]
        idx = t * fs
        i0 = np.floor(idx).astype(np.int32)
        w = idx - i0
        valid = (i0 >= 0) & (i0 < nsamp - 1)
        i0c = np.clip(i0, 0, nsamp - 2)
        v = (np.take_along_axis(iq_a, i0c, axis=-1) * (1.0 - w)
             + np.take_along_axis(iq_a, i0c + 1, axis=-1) * w)
        v = np.where(valid, v, 0.0) * np.exp(2j * np.pi * fc * t)
        acc[sl] = v.sum(axis=0)
    return acc


def das_angles_const(iq, c_bf, xe, xi, zi, angles_deg, fs=G.FS, fc=G.FC,
                     chunk=8192):
    """Homogeneous plane-wave DAS. Returns complex64 [n_angle, nx, nz]."""
    iq = np.asarray(iq)
    xe = np.asarray(xe, np.float64)
    xi = np.asarray(xi, np.float32)
    zi = np.asarray(zi, np.float32)
    angles_deg = np.asarray(angles_deg, np.float64)
    X, Z = np.meshgrid(xi, zi, indexing="ij")
    Xf, Zf = X.ravel().astype(np.float64), Z.ravel().astype(np.float64)
    nsamp = iq.shape[-1]
    out = np.empty((iq.shape[1], xi.size, zi.size), np.complex64)
    c_bf = float(c_bf)
    tau_rx = np.sqrt((Xf[:, None] - xe[None, :]) ** 2
                     + Zf[:, None] ** 2) / c_bf          # [npix, ne]
    for a, th in enumerate(np.deg2rad(angles_deg)):
        tau_tx = (Xf * np.sin(th) + Zf * np.cos(th)) / c_bf
        tau = tau_tx[None, :] + tau_rx.T                 # [ne, npix]
        out[a] = _sum_delayed(iq[:, a, :], tau, fs, fc, nsamp, chunk).reshape(
            xi.size, zi.size)
    return out


def _sample_c(cmap, cx, cz, xq, zq):
    """Bilinear sample cmap[x, z] at query points (nearest outside)."""
    from scipy.ndimage import map_coordinates
    dx = float(cx[1] - cx[0])
    dz = float(cz[1] - cz[0])
    ix = (np.asarray(xq, np.float64) - float(cx[0])) / dx
    iz = (np.asarray(zq, np.float64) - float(cz[0])) / dz
    shp = ix.shape
    return map_coordinates(np.asarray(cmap, np.float64),
                           [ix.ravel(), iz.ravel()],
                           order=1, mode="nearest").reshape(shp)


def _path_time(cmap, cx, cz, x0, z0, x1, z1, n_path):
    """Straight-ray travel time from (x0,z0) to (x1,z1) through cmap."""
    s = np.linspace(0.0, 1.0, int(n_path), dtype=np.float64)
    w = np.ones(s.size, np.float64)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    x0 = np.asarray(x0, np.float64)
    z0 = np.asarray(z0, np.float64)
    x1 = np.asarray(x1, np.float64)
    z1 = np.asarray(z1, np.float64)
    length = np.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
    xs = x0[..., None] + (x1 - x0)[..., None] * s
    zs = z0[..., None] + (z1 - z0)[..., None] * s
    xs, zs = np.broadcast_arrays(xs, zs)
    slowness = 1.0 / np.clip(_sample_c(cmap, cx, cz, xs, zs), 1200.0, 1800.0)
    return (slowness * w).sum(axis=-1) * np.broadcast_to(length, slowness.shape[:-1])


def das_angles_map(iq, cmap, cx, cz, xe, xi, zi, angles_deg,
                   fs=G.FS, fc=G.FC, n_path=48, chunk=8192, ebatch=8):
    """Straight-ray DAS through a 2-D sound-speed map (no refraction).

    Transmit: slowness integral along the plane-wave direction to each pixel.
    Receive: slowness integral along the geometric element-to-pixel ray.
    Carrier phase is restored exactly as in ``das_complex``.

    Returns complex64 [n_angle, nx, nz].
    """
    iq = np.asarray(iq)
    cmap = np.asarray(cmap, np.float64)
    cx = np.asarray(cx, np.float64)
    cz = np.asarray(cz, np.float64)
    xe = np.asarray(xe, np.float64)
    xi = np.asarray(xi, np.float32)
    zi = np.asarray(zi, np.float32)
    angles_deg = np.asarray(angles_deg, np.float64)
    X, Z = np.meshgrid(xi.astype(np.float64), zi.astype(np.float64),
                       indexing="ij")
    nsamp = iq.shape[-1]
    npix = X.size
    Xf, Zf = X.ravel(), Z.ravel()

    tau_rx = np.empty((xe.size, npix), np.float64)
    for e0 in range(0, xe.size, ebatch):
        e1 = min(e0 + ebatch, xe.size)
        tau_rx[e0:e1] = _path_time(
            cmap, cx, cz, xe[e0:e1, None], 0.0, Xf[None, :], Zf[None, :],
            n_path)

    out = np.empty((iq.shape[1], xi.size, zi.size), np.complex64)
    for a, th in enumerate(np.deg2rad(angles_deg)):
        st, ct = np.sin(th), np.cos(th)
        L = X * st + Z * ct
        x_wf = X - L * st
        z_wf = Z - L * ct
        tau_tx = _path_time(cmap, cx, cz, x_wf, z_wf, X, Z, n_path)
        # plane-wave time is signed (pixels "behind" the reference wavefront)
        tau_tx = np.where(L >= 0.0, tau_tx, -tau_tx)
        tau = tau_tx.ravel()[None, :] + tau_rx
        out[a] = _sum_delayed(iq[:, a, :], tau, fs, fc, nsamp, chunk).reshape(
            xi.size, zi.size)
    return out


def _polar_compress(z, scale, k=3.0):
    """Magnitude asinh-compressed to ~[0,1], phase preserved exactly."""
    mag = np.abs(z)
    m = np.arcsinh(mag / (scale + 1e-30)) / np.arcsinh(k)
    ph = np.angle(z)
    return m * np.cos(ph), m * np.sin(ph)


def _group_scale(arrays):
    """One scalar per group (preserves inter-channel amplitude relations)."""
    rms = [np.sqrt(np.mean(np.abs(a) ** 2)) for a in arrays if a is not None]
    return float(np.mean(rms)) if rms else 1.0


def build_condition(iq, xe, angles_deg, xi, zi, speeds=(1450.0, 1500.0, 1550.0),
                    ref_speed=1500.0, n_subap=4, groups=("full", "angle", "subap"),
                    fs=G.FS, fc=G.FC, t0=0.0):
    """Phase-preserving multi-group DAS condition tensor.

    Returns:
        float32 [2*C, nx, nz] with C = n_speeds + (n_angle if "angle") +
        (n_subap if "subap"), channels ordered (Re, Im) per beam image.

    The three groups probe complementary phase information:
        full  : how the total coherent energy changes with the assumed speed
                (global SoS mismatch);
        angle : the transmit-angle dependence of the residual phase
                (the transmit leg is (x sin theta + z cos theta)/c, so the
                angle-to-angle phase slope measures the SoS error);
        subap : the receive-aperture dependence of the residual phase
                (arrival-time slope across the array = local aberration).
    """
    g = das_groups(iq, xe, angles_deg, xi, zi, fs=fs, fc=fc,
                   speeds=speeds, ref_speed=ref_speed, n_subap=n_subap,
                   t0=t0)
    chans = []
    if "full" in groups:
        sc = _group_scale(g["full"])
        for im in g["full"]:
            chans.extend(_polar_compress(im, sc))
    if "angle" in groups and g["per_angle"] is not None:
        sc = _group_scale(g["per_angle"])
        for im in g["per_angle"]:
            chans.extend(_polar_compress(im, sc))
    if "subap" in groups and g["subap"] is not None:
        sc = _group_scale(g["subap"])
        for im in g["subap"]:
            chans.extend(_polar_compress(im, sc))
    return np.stack(chans, axis=0).astype(np.float32)


def condition_channels(n_angle=G.N_ANGLE, n_speeds=3, n_subap=4,
                       groups=("full", "angle", "subap")):
    """Number of condition channels for a given group selection."""
    c = 0
    if "full" in groups:
        c += 2 * n_speeds
    if "angle" in groups:
        c += 2 * n_angle
    if "subap" in groups:
        c += 2 * n_subap
    return c


def resample_target(c_true, cx_true, cz_true, xc_m, z_face_m, xi, zi):
    """Resample the simulator-grid ground-truth SoS map onto the prediction grid.

    ``c_true`` lives on the 2-D simulation grid with absolute phantom
    coordinates; the prediction grid is probe-relative (x from the probe
    centre, z from the probe face).
    """
    from scipy.ndimage import map_coordinates

    c_true = np.asarray(c_true)
    x_rel = np.asarray(cx_true, np.float64) - float(xc_m)
    z_rel = np.asarray(cz_true, np.float64) - float(z_face_m)
    ix = (np.asarray(xi, np.float64) - x_rel[0]) / (x_rel[1] - x_rel[0])
    iz = (np.asarray(zi, np.float64) - z_rel[0]) / (z_rel[1] - z_rel[0])
    II, JJ = np.meshgrid(ix, iz, indexing="ij")
    out = map_coordinates(c_true, [II.ravel(), JJ.ravel()], order=1,
                          mode="nearest")
    return out.reshape(len(xi), len(zi)).astype(np.float32)


def sample_grids():
    """Prediction grid arrays (xi, zi) [m]."""
    return G.x_grid(), G.z_grid()
