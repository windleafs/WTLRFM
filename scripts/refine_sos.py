import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from data.acquisition import load_acquisition
from scripts.generalize_sos import plot_maps, save_json
from wfc_integration.generalization import build_model, refine_lowdim, subset_fields


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--ablation', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--controls', type=int, nargs=2, default=[2, 6])
    p.add_argument('--lr', type=float, default=.05)
    p.add_argument('--prior-weight', type=float, default=.002)
    p.add_argument('--smooth-weight', type=float, default=.002)
    p.add_argument('--max-log-change', type=float, default=.05)
    p.add_argument('--flow-start', default='smooth2_alpha1')
    args = p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Use a fresh refinement directory')
    summary = json.loads((args.ablation/'summary.json').read_text())
    protocol = summary['protocol']
    a = load_acquisition(args.ablation/'acquisition.npz')
    with np.load(args.ablation/'candidates.npz', allow_pickle=False) as d:
        x, z = d['cx'], d['cz']
        starts = {'constant1540': d['constant1540'], 'scalar_fit': d['scalar_fit'], 'flow': d[args.flow_start]}
    model, project, roi, splits, current_protocol = build_model(
        a, x, z, protocol['backend'], protocol['dz_mm'], protocol['fov_m'][2]*1000,
        protocol['transmit_window_assumed'])
    if splits != protocol['event_indices']:
        raise ValueError('Event partitions changed since the ablation')
    args.out.mkdir(parents=True)
    fields = {key: subset_fields(model, indices) for key, indices in splits.items()}
    options = dict(steps=args.steps, controls_shape=tuple(args.controls), learning_rate=args.lr,
                   prior_weight=args.prior_weight, smooth_weight=args.smooth_weight,
                   max_log_change=args.max_log_change)
    results, maps = {}, {}
    for name, base in starts.items():
        def progress(row):
            with (args.out/f'{name}_history.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps({'initialization': name, **row}), flush=True)
        result = refine_lowdim(model.image_from_f0, fields, roi, project(base), callback=progress, **options)
        maps[name] = result.pop('c_pred')
        controls = result.pop('controls')
        np.savez_compressed(args.out/f'{name}.npz', c_pred=maps[name], cx=model.cx, cz=model.cz,
                            controls=controls, input_event_indices=np.unique(protocol['prior_event_indices']+splits['train']+splits['val']))
        results[name] = result
        save_json(args.out/f'{name}_metrics.json', result)
        jax.clear_caches()
    save_json(args.out/'summary.json', dict(stage='lowdim_refinement', results=results, options=options,
                protocol={**protocol, **current_protocol}, flow_start=args.flow_start,
                ablation=str(args.ablation.resolve()), accuracy_metrics_available=False))
    plot_maps(args.out/'maps.png', maps, np.asarray(model.cx), np.asarray(model.cz))
    print(json.dumps({'completed': list(results), 'out': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
