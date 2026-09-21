"""Structured RF/IQ condition construction for acquisition-aware SoS models.

The original condition tensor assigns physical transmit angles to fixed channel
positions.  This module instead returns the per-event complex DAS images and
the physical geometry that produced them.  A small encoder can then aggregate a
variable event set into canonical condition slots without pretending that
``channel 3`` always means the same acquisition angle.
"""

import numpy as np
import torch
import torch.nn.functional as F


def analytic_baseband(rf, fs, fc, t0_s=0.):
    """Zero-padded Hilbert transform followed by absolute-clock demodulation."""
    n = rf.shape[-1]
    padded = F.pad(rf, (n, n))
    m = padded.shape[-1]
    h = rf.new_zeros(m)
    h[0] = 1
    h[1:(m + 1)//2] = 2
    if m % 2 == 0:
        h[m//2] = 1
    a = torch.fft.ifft(torch.fft.fft(padded)*h)[..., n:2*n]
    t = float(t0_s) + torch.arange(n, device=rf.device, dtype=rf.dtype)/float(fs)
    return a*torch.exp(-2j*torch.pi*float(fc)*t)


def subap_edges(n_elem, n_subap=4):
    """Contiguous receiver sub-apertures as ``(start, stop)`` index pairs."""
    n_elem, n_subap = int(n_elem), int(n_subap)
    if n_subap < 1 or n_subap > n_elem:
        raise ValueError('n_subap must be between 1 and the receiver count')
    edges = np.linspace(0, n_elem, n_subap+1).round().astype(int)
    return [(int(s), int(e)) for s, e in zip(edges[:-1], edges[1:])]


def event_geometry(angles_deg, tx_t_ref_s, max_angle_deg=45.):
    """Normalised physical features for every transmit event.

    Columns are ``sin(theta), cos(theta), theta/max_angle, tx_ref_us``.  The
    encoder recovers radians with ``atan2`` from the first two columns, while
    the bounded third column is convenient for choosing a near-broadside
    relative-phase reference.
    """
    angles = np.asarray(angles_deg, np.float64)
    refs = np.asarray(tx_t_ref_s, np.float64)
    if angles.ndim != 1 or refs.shape != angles.shape or not np.isfinite([angles, refs]).all():
        raise ValueError('Transmit angles and reference delays must be finite [n_event]')
    theta = np.deg2rad(angles)
    return np.stack([np.sin(theta), np.cos(theta), angles/float(max_angle_deg),
                     refs*1e6], axis=1).astype(np.float32)


def subap_geometry(xe, n_subap=4, element_mask=None):
    """Normalised physical features for every receive sub-aperture."""
    xe = np.asarray(xe, np.float64)
    if xe.ndim != 1 or xe.size < 2 or not np.isfinite(xe).all():
        raise ValueError('Receiver coordinates must be finite [n_elem]')
    active = np.ones(xe.size, bool) if element_mask is None else np.asarray(element_mask, bool)
    if active.shape != xe.shape:
        raise ValueError('element_mask must match receiver coordinates')
    half = max(abs(xe[0]), abs(xe[-1]), np.finfo(float).eps)
    aperture = max(np.ptp(xe), np.finfo(float).eps)
    out = []
    for s, e in subap_edges(xe.size, n_subap):
        idx = np.flatnonzero(active[s:e])
        if idx.size:
            idx = idx + s
            centre = float(np.mean(xe[idx]))/half
            width = float(np.ptp(xe[idx]))/aperture if idx.size > 1 else 0.
            ordinal = float(np.mean(idx)/(xe.size-1)*2-1)
            fill = float(idx.size/(e-s))
        else:
            centre = width = ordinal = fill = 0.
        out.append([centre, width, ordinal, fill])
    return np.asarray(out, np.float32)


def global_geometry(fs_hz, fc_hz, c_steer, bandwidth_fraction, xe, angles_deg,
                    t0_s=0., event_mask=None, element_mask=None):
    """Normalised acquisition metadata used for FiLM conditioning."""
    xe = np.asarray(xe, np.float64)
    angles = np.asarray(angles_deg, np.float64)
    event_mask = np.ones(len(angles), bool) if event_mask is None else np.asarray(event_mask, bool)
    element_mask = np.ones(len(xe), bool) if element_mask is None else np.asarray(element_mask, bool)
    if event_mask.shape != angles.shape or element_mask.shape != xe.shape:
        raise ValueError('Acquisition masks must match event/receiver geometry')
    pitch = float(np.median(np.diff(xe))) if len(xe) > 1 else 0.
    aperture = float(np.ptp(xe)) if len(xe) > 1 else 0.
    vals = [fc_hz/1e7, fs_hz/4e7, (c_steer-1500.)/100., bandwidth_fraction,
            pitch/2e-4, aperture/3.84e-2, len(angles)/16., t0_s*1e6/10.,
            event_mask.mean(), element_mask.mean()]
    if not np.isfinite(vals).all():
        raise ValueError('Acquisition metadata contains nonfinite values')
    return np.asarray(vals, np.float32)


def _bool_tensor(value, shape, device):
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    out = torch.as_tensor(value, dtype=torch.bool, device=device)
    if out.shape != torch.Size(shape):
        raise ValueError('Acquisition mask has the wrong shape')
    return out


@torch.no_grad()
def das_speed_events(rf, xe, angles_deg, tx_t_ref_s, xi, zi, fs, fc,
                     speeds=(1450., 1500., 1550.), ref_speed=1500.,
                     n_subap=4, event_mask=None, element_mask=None,
                     t0_s=0., chunk=4096):
    """Per-speed/per-event complex DAS images plus explicit acquisition masks.

    Args:
        rf: real RF tensor ``[n_event, n_elem, n_time]``.
        speeds: assumed homogeneous speeds for the full-aperture group.
        ref_speed: speed used for the per-event and receive-subaperture groups.

    Returns:
        Dict containing ``speed_events [n_speed, n_event, nx, nz]`` and
        ``subap [n_subap, nx, nz]`` as complex64 tensors, and boolean
        ``event_mask``, ``subap_mask`` and ``element_mask``.
    """
    rf = torch.as_tensor(rf)
    if rf.ndim != 3 or torch.is_complex(rf) or not torch.isfinite(rf).all() or not bool((rf != 0).any()):
        raise ValueError('RF must be finite, nonzero real [event, receiver, time]')
    device = rf.device
    n_event, n_elem, nt = rf.shape
    angles = np.asarray(angles_deg, np.float64)
    refs = np.asarray(tx_t_ref_s, np.float64)
    xe_t = torch.as_tensor(xe, dtype=torch.float64, device=device)
    xi_t = torch.as_tensor(xi, dtype=torch.float64, device=device)
    zi_t = torch.as_tensor(zi, dtype=torch.float64, device=device)
    if angles.ndim != 1 or len(angles) != n_event or refs.shape != angles.shape or xe_t.ndim != 1 or len(xe_t) != n_elem:
        raise ValueError('RF shape and acquisition geometry disagree')
    speeds = tuple(float(c) for c in speeds)
    if not speeds or not np.isfinite(speeds).all() or min(speeds) <= 0:
        raise ValueError('Assumed speeds must be finite and positive')
    ref_candidates = np.flatnonzero(np.isclose(speeds, float(ref_speed)))
    if ref_candidates.size != 1:
        raise ValueError('ref_speed must occur exactly once in speeds')
    ref_index = int(ref_candidates[0])
    event_mask_t = _bool_tensor(event_mask, (n_event,), device)
    element_mask_t = _bool_tensor(element_mask, (n_elem,), device)
    if not event_mask_t.any() or not element_mask_t.any():
        raise ValueError('At least one event and one receiver must remain active')

    iq = analytic_baseband(rf, float(fs), float(fc), float(t0_s))
    xx, zz = torch.meshgrid(xi_t, zi_t, indexing='ij')
    xx, zz = xx.reshape(-1), zz.reshape(-1)
    npix = xx.numel()
    speed_events = torch.zeros(len(speeds), n_event, npix,
                               dtype=torch.complex64, device=device)
    subap_events = torch.zeros(n_event, int(n_subap), npix,
                                dtype=torch.complex64, device=device)
    edges = subap_edges(n_elem, n_subap)

    for si, speed in enumerate(speeds):
        receive = torch.sqrt((xx[None, :] - xe_t[:, None])**2 + zz[None, :]**2)/speed
        for ia, (angle, ref) in enumerate(zip(angles, refs)):
            if not bool(event_mask_t[ia]):
                continue
            theta = np.deg2rad(float(angle))
            tx = float(ref) + (xx*np.sin(theta) + zz*np.cos(theta))/speed
            for start in range(0, npix, int(chunk)):
                stop = min(start+int(chunk), npix)
                sl = slice(start, stop)
                tau = tx[sl][None, :] + receive[:, sl]
                idx = (tau-float(t0_s))*float(fs)
                i0 = idx.floor().long()
                frac = (idx-i0).to(iq.dtype)
                valid = (i0 >= 0) & (i0 < nt-1)
                ii = i0.clamp(0, nt-2)
                v = (iq[ia].gather(1, ii)*(1-frac)
                     + iq[ia].gather(1, ii+1)*frac)
                v = torch.where(valid, v, torch.zeros((), dtype=v.dtype, device=device))
                v = v*torch.exp(2j*torch.pi*float(fc)*tau)*element_mask_t[:, None]
                speed_events[si, ia, sl] = v.sum(0)
                if si == ref_index:
                    for k, (s, e) in enumerate(edges):
                        subap_events[ia, k, sl] = v[s:e].sum(0)

    nx, nz = len(xi_t), len(zi_t)
    subap_mask = torch.tensor([bool(element_mask_t[s:e].any()) for s, e in edges],
                              dtype=torch.bool, device=device)
    subap_events = subap_events.reshape(n_event, int(n_subap), nx, nz)
    subap = subap_events.sum(dim=0)
    return {
        'speed_events': speed_events.reshape(len(speeds), n_event, nx, nz),
        'subap': subap,
        'subap_events': subap_events,
        'event_mask': event_mask_t,
        'subap_mask': subap_mask,
        'element_mask': element_mask_t,
    }


def structured_condition(rf, xe, angles_deg, tx_t_ref_s, xi, zi, fs, fc, c_steer,
                         bandwidth_fraction, speeds=(1450., 1500., 1550.),
                         ref_speed=1500., n_subap=4, event_mask=None,
                         element_mask=None, t0_s=0., max_angle_deg=45.,
                         chunk=4096):
    """Build the complete structured condition package for one acquisition."""
    das = das_speed_events(rf, xe, angles_deg, tx_t_ref_s, xi, zi, fs, fc,
                           speeds=speeds, ref_speed=ref_speed, n_subap=n_subap,
                           event_mask=event_mask, element_mask=element_mask,
                           t0_s=t0_s, chunk=chunk)
    event_mask_np = das['event_mask'].detach().cpu().numpy()
    element_mask_np = das['element_mask'].detach().cpu().numpy()
    return {
        **das,
        'event_geom': torch.from_numpy(event_geometry(angles_deg, tx_t_ref_s, max_angle_deg)),
        'subap_geom': torch.from_numpy(subap_geometry(xe, n_subap, element_mask_np)),
        'global_geom': torch.from_numpy(global_geometry(
            fs, fc, c_steer, bandwidth_fraction, xe, angles_deg, t0_s,
            event_mask_np, element_mask_np)),
        'event_indices': torch.arange(len(np.asarray(angles_deg)), dtype=torch.int16),
    }
