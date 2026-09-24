"""Pack decoupled counterfactual raws into a cache root for prepare_geometry_cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path,
                   default=Path('/data/zhuangyang/tmp/decoupled_raw'))
    p.add_argument('--out-root', type=Path,
                   default=Path('/data/zhuangyang/tmp/decoupled_root'))
    args = p.parse_args()
    (args.out_root/'shards').mkdir(parents=True, exist_ok=True)
    samples = []
    for path in sorted(args.raw_dir.glob('*.npz')):
        if 'meta_json' not in np.load(path, allow_pickle=False).files:
            continue  # e.g. the L12-5 homogeneous reference, not a record
        meta = json.loads(str(np.load(path, allow_pickle=False)['meta_json'].item()))
        d = np.load(path, allow_pickle=False)
        shard = {'rf': torch.from_numpy(d['rf']), 'c': torch.from_numpy(d['c']),
                 'm': torch.from_numpy(d['m']),
                 'metadata': {
                     'id': meta['id'],
                     'split': meta.get('split', 'train'),
                     'case': meta.get('case', meta.get('family', 'probe')),
                     'backend': 'ultrawave',
                     'base_anatomy_id': meta.get(
                         'base_anatomy_id', meta['id']),
                     'scatter_seed': meta.get('scatter_seed', 0),
                     'variant': meta.get('variant', meta.get('family', '')),
                     'angles_deg': meta['angles_deg'],
                     'source_tref_s': meta['source_tref_s'],
                     'fs_hz': meta['fs_hz'], 'band_hz': meta['band_hz'],
                     'source_f0_hz': meta['fc_hz'], 'native_dt_s': 2.5e-9,
                     'native_nt': 24001, 'space_order': 8, 'dx_m': 5e-5,
                     'preset': 'dual_scale',
                     'simulation': ('decoupled counterfactual/probe-swap ('
                                    + (meta.get('variant')
                                       or meta.get('family', '')) + ')')}}
        torch.save(shard, args.out_root/'shards'/f"{meta['id']}.pt")
        samples.append({'id': meta['id'],
                        'split': meta.get('split', 'train'),
                        'backend': 'ultrawave',
                        'variant': (meta.get('variant')
                                    or meta.get('family', '')),
                        'source_id': meta.get('source_id', meta['id']),
                        'path': f"shards/{meta['id']}.pt",
                        'status': 'complete'})
        print(f'packed {meta["id"]}', flush=True)
    acquisition = {'angles_deg': [-8., -6.4, -4.8, -3.2, -1.6, 0., 1.6, 3.2,
                                  4.8, 6.4, 8.], 'elements': 192,
                   'pitch_m': 2e-4, 'fs_hz': 40e6, 'band_hz': [4e6, 7.5e6],
                   'rf_samples': 2401, 'source_f0_hz': 7.5e6,
                   'native_dt_s': 2.5e-9, 'native_nt': 24001}
    counts = {}
    for s in samples:
        counts[s['variant']] = counts.get(s['variant'], 0) + 1
    (args.out_root/'index.json').write_text(json.dumps(
        {'version': 1, 'backend': 'ultrawave',
         'simulation': 'decoupled reflectivity/SoS counterfactuals',
         'variant_counts': counts, 'acquisition': acquisition,
         'samples': samples}, indent=2)+'\n')
    print(f'total {len(samples)} shards -> {args.out_root}; counts={counts}')


if __name__ == '__main__':
    main()
