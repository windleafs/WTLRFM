"""Physically constrained RF-domain augmentation for acquisition robustness.

Only measurement-level perturbations are applied: timing offsets, channel and
global gain, smooth frequency response, additive noise, and missing transmit /
receiver elements.  The perturbations keep every active channel tied to the
same propagated field; they never randomise phase independently per pixel or
channel, which would destroy the very cue used for sound-speed inference.
"""

import numpy as np
import torch


DEFAULT_AUGMENTATION = {
    'time_shift_s': (-50e-9, 50e-9),
    'event_jitter_s': (-10e-9, 10e-9),
    'global_gain_db': (-3., 3.),
    'channel_gain_db': (-1.5, 1.5),
    'spectral_gain_db': (-1., 1.),
    'noise_snr_db': (35., 60.),
    'event_dropout_p': 0.,
    'min_events': 6,
    'receiver_dropout_p': 0.,
    'min_elements': 32,
}


def _range(value, name, nonnegative=False):
    if value is None:
        return None
    lo, hi = (float(v) for v in value)
    if not np.isfinite([lo, hi]).all() or lo > hi or (nonnegative and lo < 0):
        raise ValueError(f'Invalid augmentation range {name}')
    return lo, hi


def validate_augmentation(cfg):
    cfg = {**DEFAULT_AUGMENTATION, **(cfg or {})}
    for key in ('time_shift_s', 'event_jitter_s', 'global_gain_db',
                'channel_gain_db', 'spectral_gain_db'):
        cfg[key] = _range(cfg.get(key), key)
    cfg['noise_snr_db'] = _range(cfg.get('noise_snr_db'), 'noise_snr_db', True)
    for key in ('event_dropout_p', 'receiver_dropout_p'):
        value = float(cfg.get(key, 0.))
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f'{key} must be in [0,1)')
        cfg[key] = value
    for key in ('min_events', 'min_elements'):
        cfg[key] = max(1, int(cfg[key]))
    return cfg


def _uniform(rng, interval, size=None):
    if interval is None:
        return 0. if size is None else np.zeros(size)
    lo, hi = interval
    return rng.uniform(lo, hi, size=size)


def _time_shift(rf, shifts_samples):
    """Return ``out[n] = in[n - shift]``; positive shift delays the record."""
    nt = rf.shape[-1]
    base = torch.arange(nt, device=rf.device, dtype=rf.dtype)
    shifts = torch.as_tensor(shifts_samples, dtype=rf.dtype, device=rf.device)
    src = base - shifts.view(*shifts.shape, *([1]*(rf.ndim-shifts.ndim)))
    i0 = src.floor().long()
    frac = (src-i0).to(rf.dtype)
    valid = (i0 >= 0) & (i0 < nt-1)
    ii = i0.clamp(0, nt-2).expand(rf.shape)
    out = rf.gather(-1, ii)*(1-frac) + rf.gather(-1, ii+1)*frac
    return torch.where(valid, out, torch.zeros((), dtype=rf.dtype, device=rf.device))


def _spectral_gain(rf, coeffs_db):
    spectrum = torch.fft.rfft(rf, dim=-1)
    freq = torch.linspace(0., 1., spectrum.shape[-1], device=rf.device)
    gain_db = (coeffs_db[0] + coeffs_db[1]*torch.cos(torch.pi*freq)
               + coeffs_db[2]*torch.cos(2*torch.pi*freq))
    gained = torch.fft.irfft(spectrum*(10**(gain_db/20.)).to(spectrum.dtype),
                             n=rf.shape[-1], dim=-1)
    return gained


def _drop_mask(rng, count, p, minimum):
    keep = rng.random(count) >= p
    if keep.sum() < minimum:
        keep[:] = False
        keep[rng.choice(count, size=minimum, replace=False)] = True
    return keep


def augment_rf(rf, fs_hz, rng=None, cfg=None):
    """Apply one physically plausible acquisition perturbation.

    Args:
        rf: real RF tensor ``[n_event, n_elem, n_time]``.
        fs_hz: RF sampling rate used for bounded timing shifts.
        rng: ``numpy.random.Generator`` or integer seed.
        cfg: ranges/counts; see ``DEFAULT_AUGMENTATION``.

    Returns:
        ``augmented_rf, provenance`` where provenance contains masks and the
        exact sampled parameters.  Missing events/receivers are both zeroed in
        RF and reported by their masks.
    """
    cfg = validate_augmentation(cfg)
    rng = np.random.default_rng(rng)
    rf = torch.as_tensor(rf)
    if rf.ndim != 3 or torch.is_complex(rf) or not torch.isfinite(rf).all() or not bool((rf != 0).any()):
        raise ValueError('RF must be finite, nonzero real [event, receiver, time]')
    n_event, n_elem, _ = rf.shape
    if cfg['min_events'] > n_event or cfg['min_elements'] > n_elem:
        raise ValueError('Minimum active acquisition size exceeds the array')
    out = rf.clone()
    applied = {'time_shift_s': 0., 'event_jitter_s': np.zeros(n_event).tolist(),
                   'global_gain_db': 0., 'channel_gain_db': np.zeros(n_elem).tolist(),
                   'spectral_coeffs_db': np.zeros(3).tolist(), 'noise_snr_db': None}

    common = float(_uniform(rng, cfg['time_shift_s']))
    jitter = np.asarray(_uniform(rng, cfg['event_jitter_s'], n_event), np.float64)
    shifts = (common+jitter)*float(fs_hz)
    if np.any(shifts):
        out = _time_shift(out, torch.tensor(shifts, dtype=out.dtype, device=out.device)[:, None])
    applied['time_shift_s'] = common
    applied['event_jitter_s'] = jitter.tolist()

    global_gain = float(_uniform(rng, cfg['global_gain_db']))
    channel_gain = np.asarray(_uniform(rng, cfg['channel_gain_db'], n_elem), np.float64)
    gains = torch.tensor(10**((global_gain+channel_gain)/20.), dtype=out.dtype,
                         device=out.device)
    out = out*gains[None, :, None]
    applied['global_gain_db'] = global_gain
    applied['channel_gain_db'] = channel_gain.tolist()

    if cfg['spectral_gain_db'] is not None:
        coeffs = np.asarray(_uniform(rng, cfg['spectral_gain_db'], 3), np.float64)
        out = _spectral_gain(out, coeffs)
        applied['spectral_coeffs_db'] = coeffs.tolist()

    if cfg['noise_snr_db'] is not None:
        snr = float(_uniform(rng, cfg['noise_snr_db']))
        rms = out.square().mean().sqrt().clamp_min(1e-30)
        sigma = rms*(10**(-snr/20.))
        out = out + torch.randn_like(out)*sigma
        applied['noise_snr_db'] = snr

    event_mask = _drop_mask(rng, n_event, cfg['event_dropout_p'], cfg['min_events'])
    element_mask = _drop_mask(rng, n_elem, cfg['receiver_dropout_p'], cfg['min_elements'])
    out[~torch.tensor(event_mask, device=out.device)] = 0
    out[:, ~torch.tensor(element_mask, device=out.device)] = 0
    if not torch.isfinite(out).all() or not bool((out != 0).any()):
        raise FloatingPointError('RF augmentation produced an invalid record')
    applied.update(event_mask=event_mask.tolist(), element_mask=element_mask.tolist(),
                   active_events=int(event_mask.sum()), active_elements=int(element_mask.sum()))
    return out, applied
