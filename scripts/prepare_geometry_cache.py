"""Build structured acquisition caches for the geometry-aware SoS flow.

Unlike the legacy 36-channel cache, each record stores per-speed/per-event
complex DAS images plus explicit angle, receiver and acquisition metadata.  The
network can therefore aggregate the actual event set instead of relying on
fixed channel positions.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.rf_augment import augment_rf, validate_augmentation
from data.rf_geometry import structured_condition


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--splits', default='train,val')
    p.add_argument('--limit-per-split', type=int, default=0)
    p.add_argument('--ids', default='')
    p.add_argument('--speeds', default='1450,1500,1550')
    p.add_argument('--ref-speed', type=float, default=1500.)
    p.add_argument('--n-subap', type=int, default=4)
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--nz', type=int, default=160)
    p.add_argument('--x0', type=float, default=-.019125)
    p.add_argument('--x1', type=float, default=.019075)
    p.add_argument('--z0', type=float, default=.000075)
    p.add_argument('--z1', type=float, default=.043075)
    p.add_argument('--canonical-angle', type=float, default=8.)
    p.add_argument('--pitch-m', type=float, default=None,
                   help='fallback element pitch when source metadata does not provide it')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--augment-variants', type=int, default=0)
    p.add_argument('--augment-all', action='store_true',
                   help='augment every split; default augments train only')
    p.add_argument('--augment-seed', type=int, default=20260921)
    p.add_argument('--time-shift-ns', type=float, nargs=2, default=[-50., 50.])
    p.add_argument('--event-jitter-ns', type=float, nargs=2, default=[-10., 10.])
    p.add_argument('--global-gain-db', type=float, nargs=2, default=[-3., 3.])
    p.add_argument('--channel-gain-db', type=float, nargs=2, default=[-1.5, 1.5])
    p.add_argument('--spectral-gain-db', type=float, nargs=2, default=[-1., 1.])
    p.add_argument('--noise-snr-db', type=float, nargs=2, default=[35., 60.])
    p.add_argument('--event-dropout', type=float, default=.1)
    p.add_argument('--receiver-dropout', type=float, default=.02)
    p.add_argument('--min-events', type=int, default=6)
    p.add_argument('--min-elements', type=int, default=96)
    return p.parse_args()


def bandwidth_fraction(metadata, fc):
    if 'bandwidth_fraction' in metadata:
        return float(metadata['bandwidth_fraction'])
    band = np.asarray(metadata.get('band_hz', []), float)
    if band.shape == (2,):
        return float((band[1]-band[0])/(2*fc))
    return .6


def augmentation_cfg(args):
    return validate_augmentation({
        'time_shift_s': [v*1e-9 for v in args.time_shift_ns],
        'event_jitter_s': [v*1e-9 for v in args.event_jitter_ns],
        'global_gain_db': args.global_gain_db,
        'channel_gain_db': args.channel_gain_db,
        'spectral_gain_db': args.spectral_gain_db,
        'noise_snr_db': args.noise_snr_db,
        'event_dropout_p': args.event_dropout,
        'min_events': args.min_events,
        'receiver_dropout_p': args.receiver_dropout,
        'min_elements': args.min_elements,
    })


def make_record(sample, record, args, device, variant=0):
    rf = sample['rf'].to(device)
    metadata = sample['metadata']
    applied = None
    event_mask = element_mask = None
    if variant:
        seed = (args.augment_seed + 1009*variant
                + int.from_bytes(hashlib.sha256(record['id'].encode()).digest()[:4], 'little'))
        rf, applied = augment_rf(rf, float(metadata['fs_hz']),
                                 rng=np.random.default_rng(seed),
                                 cfg=args.augmentation)
        event_mask = applied['event_mask']
        element_mask = applied['element_mask']
    angles = np.asarray(metadata['angles_deg'], np.float64)
    refs = np.asarray(metadata['source_tref_s'], np.float64)
    fs = float(metadata['fs_hz'])
    fc = float(metadata['source_f0_hz'])
    if rf.shape[:2] != (len(angles), rf.shape[1]) or rf.shape[0] != len(angles):
        raise ValueError('RF event count does not match angle metadata')
    pitch = float(metadata.get('pitch_m', args.pitch_m or 2e-4))
    n_elem = rf.shape[1]
    xe = (np.arange(n_elem)-(n_elem-1)/2)*pitch
    xi = np.linspace(args.x0, args.x1, args.nx, dtype=np.float32)
    zi = np.linspace(args.z0, args.z1, args.nz, dtype=np.float32)
    speeds = [float(v) for v in args.speeds.split(',')]
    c_steer = float(metadata.get('c_steer_m_s', metadata.get('c_steer', args.ref_speed)))
    condition = structured_condition(
        rf, xe, angles, refs, xi, zi, fs, fc, c_steer,
        bandwidth_fraction(metadata, fc), speeds=speeds,
        ref_speed=args.ref_speed, n_subap=args.n_subap,
        event_mask=event_mask, element_mask=element_mask,
        t0_s=float(metadata.get('t0_s', 0.)), chunk=args.chunk)
    target = F.interpolate(sample['c'].cpu()[None, None], size=(args.nz, args.nx),
                           mode='bilinear', align_corners=True)[0, 0].T.contiguous()
    u_gt = torch.log(target/1500.)/.05
    arrays = {
        'speed_events': condition['speed_events'].cpu().numpy().astype(np.complex64),
        'subap': condition['subap'].cpu().numpy().astype(np.complex64),
        'subap_events': condition['subap_events'].cpu().numpy().astype(np.complex64),
        'event_geom': condition['event_geom'].numpy().astype(np.float32),
        'subap_geom': condition['subap_geom'].numpy().astype(np.float32),
        'global_geom': condition['global_geom'].numpy().astype(np.float32),
        'event_mask': condition['event_mask'].cpu().numpy(),
        'subap_mask': condition['subap_mask'].cpu().numpy(),
        'element_mask': condition['element_mask'].cpu().numpy(),
        'event_indices': np.arange(len(angles), dtype=np.int16),
        'c_gt': target.numpy().astype(np.float32),
        'u_gt': u_gt.numpy().astype(np.float32),
    }
    record_id = record['id'] if variant == 0 else f"{record['id']}_aug{variant}"
    meta = {
        'id': record_id, 'source_id': record['id'], 'split': record['split'],
        'case': record.get('case'), 'base_anatomy_id': record.get('base_anatomy_id'),
        'source_path': record.get('path'), 'backend': record.get('backend'),
        'augmentation': applied, 'angles_deg': angles.tolist(), 'tx_t_ref_s': refs.tolist(),
        'fs_hz': fs, 'fc_hz': fc, 'pitch_m': pitch, 'c_steer': c_steer,
        'bandwidth_fraction': bandwidth_fraction(metadata, fc),
        'source_waveform_calibrated': False,
        'transmit_apodization_calibrated': False,
        'hardware_calibrated': False,
    }
    return arrays, meta


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh structured cache directory')
    root = args.root
    index = json.loads((root/'index.json').read_text())
    args.pitch_m = index.get('acquisition', {}).get('pitch_m')
    wanted_ids = set(v for v in args.ids.split(',') if v)
    splits = set(v for v in args.splits.split(',') if v)
    records = [r for r in index['samples']
               if r.get('split') in splits and (not wanted_ids or r['id'] in wanted_ids)]
    if args.limit_per_split:
        records = [r for split in splits for r in
                   [x for x in records if x['split'] == split][:args.limit_per_split]]
    if not records:
        raise FileNotFoundError('No source records selected')
    args.augmentation = augmentation_cfg(args)
    args.out.mkdir(parents=True)
    device = torch.device(args.device)
    start = time.time()
    manifest_records = []
    total_variants = sum(1 + (args.augment_variants if args.augment_all or r['split'] == 'train' else 0)
                         for r in records)
    done = 0
    for record in records:
        sample = torch.load(root/record['path'], map_location='cpu', weights_only=False)
        variants = args.augment_variants if (args.augment_all or record['split'] == 'train') else 0
        for variant in range(variants+1):
            arrays, meta = make_record(sample, record, args, device, variant)
            name = meta['id'] + '.npz'
            np.savez_compressed(args.out/name, **arrays)
            manifest_records.append({**meta, 'cache_file': name, 'status': 'complete'})
            done += 1
            print(json.dumps({'cached': done, 'total': total_variants, 'id': meta['id'],
                              'source': record['id'], 'seconds': round(time.time()-start, 1)}),
                  flush=True)
    manifest = {
        'version': 2,
        'kind': 'structured_acquisition_sos',
        'data_root': str(root.resolve()),
        'records': manifest_records,
        'grid': {'x0_m': args.x0, 'x1_m': args.x1, 'z0_m': args.z0, 'z1_m': args.z1,
                 'nx': args.nx, 'nz': args.nz},
        'condition': {'speeds': [float(v) for v in args.speeds.split(',')],
                      'ref_speed': args.ref_speed, 'n_subap': args.n_subap,
                      'canonical_angle_deg': args.canonical_angle,
                      'event_geom_dim': 4, 'subap_geom_dim': 4, 'global_geom_dim': 10,
                      'event_order': 'physical angles, not fixed channel positions',
                      'subap_layout': 'event-resolved [event, subap, x, z] plus compounded legacy view'},
        'augmentation': {'variants_per_train_record': args.augment_variants,
                         'augment_all': args.augment_all, 'seed': args.augment_seed,
                         'ranges': args.augmentation,
                         'constraint': 'RF-level timing/gain/spectral/noise/dropout; no independent phase randomisation'},
        'provenance': {'source_waveform_calibrated': False,
                       'transmit_apodization_calibrated': False,
                       'hardware_calibrated': False},
    }
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cache': str(args.out), 'records': len(manifest_records),
                      'seconds': round(time.time()-start, 1)}), flush=True)


if __name__ == '__main__':
    main()
