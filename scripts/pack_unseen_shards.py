"""Pack unseen-angle raw simulations into mini cache roots.

Reads the raw npz files written by sim_unseen_angles.py plus the original
11-angle shards (for the identical SoS truth), and writes two dataset roots:

  unseen5 — RF restricted to the 5-angle unseen set
  unseen9 — all 9 unseen angles

Each root has the index.json + shards/*.pt layout that
scripts/prepare_geometry_cache.py consumes, so the same DAS/cache pipeline
produces evaluation records bit-compatible with the training cache.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SRC_ROOT = Path('/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle')
UNION_ANGLES = [-7.5, -6.0, -4.5, -3.0, -1.5, 0.5, 2.5, 4.5, 6.5]
UNSEEN_5 = [-7.5, -4.5, -1.5, 2.5, 6.5]


def pack_root(sim_dir, out_root, subset, ids):
    idx = [i for i, a in enumerate(UNION_ANGLES) if a in subset]
    if len(idx) != len(subset):
        raise ValueError('subset angles missing from the union set')
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root/'shards').mkdir(exist_ok=True)
    samples = []
    for sid in ids:
        raw = np.load(sim_dir/f'raw_{sid}.npz')['rf']            # [9,192,2401]
        meta = json.loads((sim_dir/f'raw_{sid}.meta.json').read_text())
        orig = torch.load(SRC_ROOT/'shards'/f'{sid}.pt', map_location='cpu',
                          weights_only=False)
        src = meta['source_record']
        shard_meta = {
            'id': sid, 'split': 'val', 'case': src['case'], 'h5': src['h5'],
            'z_index': src['z_index'], 'scatter_seed': src['scatter_seed'],
            'backend': src['backend'],
            'base_anatomy_id': src['base_anatomy_id'],
            'anatomy_repeated': src['anatomy_repeated'],
            'angles_deg': [UNION_ANGLES[i] for i in idx],
            'source_tref_s': [meta['source_tref_s'][i] for i in idx],
            'fs_hz': 40e6, 'band_hz': [4e6, 7.5e6], 'source_f0_hz': 7.5e6,
            'native_dt_s': 2.5e-9, 'native_nt': 24001, 'space_order': 8,
            'dx_m': 5e-5, 'preset': 'dual_scale',
            'simulation': '2D linear acoustic full-wave (unseen-angle re-sim)',
        }
        torch.save({'rf': torch.from_numpy(np.ascontiguousarray(raw[idx])),
                    'c': orig['c'], 'm': orig['m'], 'metadata': shard_meta},
                   out_root/'shards'/f'{sid}.pt')
        samples.append({'id': sid, 'split': 'val', 'backend': src['backend'],
                        'case': src['case'], 'path': f'shards/{sid}.pt',
                        'status': 'complete'})
    acquisition = {'angles_deg': list(subset), 'elements': 192,
                   'pitch_m': 2e-4, 'fs_hz': 40e6, 'band_hz': [4e6, 7.5e6],
                   'rf_samples': 2401, 'source_f0_hz': 7.5e6,
                   'native_dt_s': 2.5e-9, 'native_nt': 24001}
    (out_root/'index.json').write_text(json.dumps(
        {'version': 1, 'backend': 'ultrawave',
         'simulation': 'unseen-angle re-simulation of val anatomy',
         'acquisition': acquisition, 'samples': samples}, indent=2)+'\n')
    print(f'packed {len(samples)} shards -> {out_root} ({len(subset)} angles)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-dir', type=Path,
                   default=Path('/data/zhuangyang/tmp/unseen_sim'))
    p.add_argument('--out-base', type=Path,
                   default=Path('/data/zhuangyang/tmp'))
    p.add_argument('--ids', default=','.join(f'val_{i:03d}' for i in range(0, 60, 4)))
    args = p.parse_args()
    ids = [v for v in args.ids.split(',') if v]
    pack_root(args.sim_dir, args.out_base/'unseen5_root', UNSEEN_5, ids)
    pack_root(args.sim_dir, args.out_base/'unseen9_root', UNION_ANGLES, ids)


if __name__ == '__main__':
    main()
