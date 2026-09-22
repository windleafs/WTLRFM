"""Trivial-baseline sanity check for the geometry-flow SoS model.

Before touching the network again, ask whether the trained model actually
beats two deliberately dumb predictors on the same val split and grid:

  1. constant 1540 m/s for every sample
  2. the pixel-wise training-set mean SoS map for every sample

Metrics reuse ``grid_metrics`` from the training script so the numbers are
directly comparable with history.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from data.geometry_dataset import StructuredSoSDataset
from train_geometry_flow import grid_metrics


def collect_gt(dataset):
    """Raw target maps without train-time augmentation (no mirror/dropout)."""
    maps = [np.asarray(dataset._load(r)['c_gt'], np.float64) for r in dataset.records]
    return np.stack(maps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default='/data/zhuangyang/geometry_flow_v2_cache')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--compare', type=Path,
                   default=ROOT/'out/geometry_flow_v2_20260921/history.jsonl',
                   help='model history.jsonl whose best val row is shown alongside')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; existing runs are never overwritten')
    args.out.mkdir(parents=True)

    train_set = StructuredSoSDataset(args.cache, 'train')
    val_set = StructuredSoSDataset(args.cache, 'val')
    grid = train_set.manifest['grid']
    print(f'train={len(train_set)} val={len(val_set)} grid={grid["nx"]}x{grid["nz"]}')

    c_train = collect_gt(train_set)
    c_val = collect_gt(val_set)
    mean_map = c_train.mean(axis=0)
    print(f'train c_gt: {c_train.min():.1f}..{c_train.max():.1f} m/s, '
          f'mean-map range {mean_map.min():.1f}..{mean_map.max():.1f} m/s')

    x = np.linspace(grid['x0_m'], grid['x1_m'], int(grid['nx']))
    z = np.linspace(grid['z0_m'], grid['z1_m'], int(grid['nz']))
    roi = ((x[:, None] >= x.min()) & (x[:, None] <= x.max())
           & (z[None, :] >= 3e-3) & (z[None, :] <= 45e-3))
    # Reference only: per-sample constant equal to that sample's GT ROI mean
    # (cheats with the answer, bounds what "global speed offset alone" achieves).
    per_sample = c_val[..., roi].mean(axis=-1)[:, None, None]
    oracle = np.broadcast_to(per_sample, c_val.shape).copy()

    baselines = {
        'const_1540': np.full_like(c_val, 1540.),
        'train_mean_map': np.repeat(mean_map[None], len(c_val), axis=0),
        'oracle_per_sample_mean': oracle,
    }
    results = {}
    for name, pred in baselines.items():
        m = grid_metrics(pred, c_val, grid)
        results[name] = m
        print(f'{name:>14}: ' + '  '.join(f'{k}={v:.4f}' for k, v in m.items()))

    rows = [{'baseline': k, 'val': v} for k, v in results.items()]
    if args.compare and args.compare.exists():
        best = None
        for line in args.compare.read_text().splitlines():
            row = json.loads(line)
            if 'val' in row and (best is None
                                 or row['val']['roi_mae'] < best['val']['roi_mae']):
                best = row
        if best:
            results['model_best'] = best['val']
            rows.append({'baseline': f"model_best(epoch {best['epoch']})",
                         'val': best['val']})

    report = {
        'cache': str(args.cache),
        'n_train': len(train_set), 'n_val': len(val_set),
        'baselines': rows,
        'note': ('Same val split, grid and grid_metrics as training; '
                 'baselines ignore the input entirely.'),
    }
    (args.out/'eval.json').write_text(json.dumps(report, indent=2)+'\n')

    print('\n| predictor | mae | roi_mae | roi_mean_err | corr |')
    print('|---|---|---|---|---|')
    for r in rows:
        v = r['val']
        print(f"| {r['baseline']} | {v['mae']:.2f} | {v['roi_mae']:.2f} "
              f"| {v['roi_mean_err']:.3f} | {v['corr']:.4f} |")


if __name__ == '__main__':
    main()
