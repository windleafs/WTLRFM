import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt


def validate_acquisition(a):
    rf = np.asarray(a['rf'])
    xe, angles, refs = (np.asarray(a[k], dtype=float) for k in ('xe', 'angles_deg', 'tx_t_ref_s'))
    if rf.ndim != 3 or rf.shape[:2] != (angles.size, xe.size) or rf.shape[-1] < 4:
        raise ValueError('RF must be [angle, receiver, time] with matching geometry')
    if np.iscomplexobj(rf) or not np.isfinite(rf).all() or not np.any(rf):
        raise ValueError('Expected finite, nonzero real RF')
    if xe.ndim != 1 or xe.size < 8 or not np.all(np.diff(xe) > 0):
        raise ValueError('Need at least 8 ordered receiver coordinates')
    if angles.ndim != 1 or len(np.unique(angles)) != len(angles) or np.any(abs(angles) >= 90):
        raise ValueError('Angles must be unique and between -90 and 90 degrees')
    if refs.shape != angles.shape or not all(np.isfinite(v).all() for v in (xe, angles, refs)):
        raise ValueError('Transmit references must match finite geometry')
    scalars = [float(a[k]) for k in ('fs_hz', 'fc_hz', 't0_s', 'c_steer', 'bandwidth_fraction')]
    fs, fc, _, c, bw = scalars
    if not np.isfinite(scalars).all() or not 0 < fc < fs/2 or c <= 0 or not 0 < bw < 2:
        raise ValueError('Invalid sampling, carrier, steering speed or bandwidth')
    if fc*(1+bw/2) >= fs/2:
        raise ValueError('Requested physical bandwidth exceeds RF Nyquist')
    return a


def load_acquisition(path):
    with np.load(path, allow_pickle=False) as d:
        a = {k: d[k] for k in d.files if k not in ('metadata', 'c_gt', 'c_true')}
        a['metadata'] = json.loads(str(d['metadata'].item())) if 'metadata' in d else {}
    return validate_acquisition(a)


def load_real_acquisition(prefix):
    prefix = Path(prefix)
    md = loadmat(prefix.with_suffix('.mat'), simplify_cells=True)['metadata']
    p = md['param']
    raw = np.fromfile(prefix.with_suffix('.bin'), dtype='<i2')
    if raw.nbytes != int(md['expected_bytes']):
        raise ValueError('RF byte count disagrees with acquisition metadata')
    rf = raw.reshape((int(md['rx_ch']), int(md['sample_num']), int(md['angle_num'])),
                     order='F').transpose(2, 0, 1).astype(np.float64)
    b, a = butter(5, 1.5e6/(float(md['fs'])/2), btype='highpass')
    rf = filtfilt(b, a, rf, axis=-1, padlen=15)
    with h5py.File(str(prefix) + '_iq.mat') as f:
        gain = np.asarray(f['metadata/tgc_gain']).ravel()
    if gain.shape != (rf.shape[-1],) or not np.isfinite(gain).all() or (gain <= 0).any():
        raise ValueError('Invalid stored TGC')
    xe = (np.arange(rf.shape[1])-(rf.shape[1]-1)/2)*float(p['pitch'])
    order = np.argsort(md['angles_deg'])
    angles = np.asarray(md['angles_deg'], float)[order]
    acquisition = dict(rf=rf[order].astype(np.float32), tgc_gain=gain.astype(np.float32),
                       xe=xe, angles_deg=angles,
                       tx_t_ref_s=max(abs(xe))*abs(np.sin(np.deg2rad(angles)))/float(p['c']),
                       fs_hz=float(md['fs']), fc_hz=float(p['fc']), t0_s=float(p['t0']),
                       c_steer=float(p['c']), bandwidth_fraction=float(p['bandwidth'])/100,
                       metadata={'source': str(prefix.resolve()), 'preprocessing': '1.5MHz highpass; no TGC',
                                 'tx_reference': 'inferred minimum-delay steering convention',
                                 'unverified': ['system_delay', 'source_waveform', 'transmit_apodization'],
                                 'hardware_calibrated': False})
    return validate_acquisition(acquisition)


def apply_calibration(a, calibration):
    allowed = {'t0_offset_s', 'tx_reference_offset_s', 'fc_hz', 'bandwidth_fraction', 'provenance'}
    if set(calibration)-allowed:
        raise ValueError('Unsupported calibration fields: ' + str(set(calibration)-allowed))
    out = dict(a)
    out['t0_s'] = float(a['t0_s']) + float(calibration.get('t0_offset_s', 0))
    out['tx_t_ref_s'] = np.asarray(a['tx_t_ref_s']) + np.asarray(calibration.get('tx_reference_offset_s', 0))
    for key in ('fc_hz', 'bandwidth_fraction'):
        if key in calibration:
            out[key] = float(calibration[key])
    out['metadata'] = {**a.get('metadata', {}), 'calibration_overrides': calibration,
                       'hardware_calibrated': False}
    return validate_acquisition(out)


def audit_acquisition(a, training=None):
    validate_acquisition(a)
    actual = {k: float(a[k]) for k in ('fs_hz', 'fc_hz', 't0_s', 'c_steer', 'bandwidth_fraction')}
    actual.update(angles_deg=np.asarray(a['angles_deg']).tolist(),
                  pitch_m=float(np.median(np.diff(a['xe']))),
                  aperture_m=float(np.ptp(a['xe'])),
                  tx_t_ref_s=np.asarray(a['tx_t_ref_s']).tolist(), rf_shape=list(a['rf'].shape))
    mismatches = {}
    for key, expected in (training or {}).items():
        if key not in actual:
            continue
        measured = actual[key]
        left, right = np.asarray(measured), np.asarray(expected)
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-5, atol=1e-12):
            mismatches[key] = {'training': right.tolist(), 'acquired': left.tolist()}
    return dict(actual=actual, mismatches=mismatches, hardware_calibrated=False,
                metadata=a.get('metadata', {}),
                warnings=['Metadata agreement is not hardware calibration.',
                          'Changing demodulation frequency does not change the physical acquisition band.',
                          'Unrecorded transmit waveform and apodization remain model assumptions.'])


def fit_pulse_echo_timing(depth_m, arrival_s):
    z, t = np.asarray(depth_m, float), np.asarray(arrival_s, float)
    if z.ndim != 1 or t.shape != z.shape or len(z) < 3 or not np.isfinite([z, t]).all() or np.ptp(z) <= 0:
        raise ValueError('Need at least three finite known depths and arrival times')
    slope, intercept = np.polyfit(z, t, 1)
    if slope <= 0:
        raise ValueError('Arrival time must increase with depth')
    return dict(speed_m_s=float(2/slope), time_offset_s=float(intercept),
                rms_residual_s=float(np.sqrt(np.mean((t-slope*z-intercept)**2))))
