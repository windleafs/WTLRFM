"""Perturbation robustness evaluation for a trained geometry-aware SoS flow.

Tests whether the checkpoint really consumes the acquisition geometry rather
than channel positions:

  * random-K   — keep a random subset of the K transmit events (mask the rest;
                  the encoder recomputes compounded views from active events)
  * shift      — rotate every event angle by +d degrees in event_geom only
  * permute    — rebind each event's RF to another event's geometry row,
                  breaking the (y_i, g_i) pairing while keeping the wave data

Metrics use the training script's grid_metrics on the same val split.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from data.geometry_dataset import StructuredSoSDataset, geometry_collate
from engine import set_seed
from models.geometry_flow import GeometryAwareSoSFlow
from train_geometry_flow import grid_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default='/data/zhuangyang/geometry_flow_v2_cache')
    p.add_argument('--ckpt', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--ode-steps', type=int, default=10)
    p.add_argument('--n-samples', type=int, default=2)
    p.add_argument('--sample-seed', type=int, default=20260921)
    p.add_argument('--ids', default='',
                   help='comma list of val record ids; default = whole val split')
    p.add_argument('--seeds', type=int, default=3,
                   help='random repetitions for drop/permute modes')
    p.add_argument('--ks', default='11,9,7,5,3')
    p.add_argument('--shifts', default='1,3,5')
    p.add_argument('--modes', default='full,drop,shift,permute',
                   help='comma list of perturbation families to run')
    p.add_argument('--model', default='flow',
                   choices=('flow', 'deterministic'),
                   help='checkpoint family to evaluate')
    return p.parse_args()


def load_model(ckpt, device, model_kind):
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    if model_kind == 'flow':
        model = GeometryAwareSoSFlow(
            unet=blob['unet'], encoder=blob['encoder_cfg'],
            u_clamp=blob.get('u_clamp', 4.), u_source_scale=-1.,
            velocity_clamp=blob.get('velocity_clamp', 8.),
            reflow_t_schedule=blob.get('reflow_t_schedule', 'stratified'))
    else:
        from models.geometry_deterministic import GeometryDeterministicSoS
        model = GeometryDeterministicSoS(unet=blob['unet'],
                                         encoder=blob['encoder_cfg'])
    model.load_state_dict(blob['state_dict'])
    if 'sigma_u' in blob and hasattr(model, 'flow'):
        model.flow._u_scale.fill_(float(blob['sigma_u']))
        model.flow._u_initialised = True
    return model.to(device).eval(), blob


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def clone_condition(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in batch['condition'].items()}


def perturb_drop(batch, k, seed, base_index):
    cond = clone_condition(batch)
    mask = cond['event_mask']
    n_events = mask.shape[1]
    for b in range(mask.shape[0]):
        rng = np.random.default_rng(1000003 + 1009*seed + 37*int(base_index+b))
        keep = rng.choice(n_events, size=int(k), replace=False)
        row = torch.zeros(n_events, dtype=torch.bool, device=mask.device)
        row[torch.from_numpy(keep).to(mask.device)] = True
        mask[b] = row
    return cond


def perturb_shift(batch, delta_deg):
    cond = clone_condition(batch)
    geom = cond['event_geom']
    theta = torch.atan2(geom[..., 0], geom[..., 1]) + float(np.deg2rad(delta_deg))
    geom[..., 0] = torch.sin(theta)
    geom[..., 1] = torch.cos(theta)
    geom[..., 2] = (geom[..., 2] + float(delta_deg)/45.)
    return cond


def perturb_permute(batch, seed, base_index):
    cond = clone_condition(batch)
    geom = cond['event_geom']
    for b in range(geom.shape[0]):
        rng = np.random.default_rng(2000003 + 1009*seed + 37*int(base_index+b))
        perm = torch.from_numpy(rng.permutation(geom.shape[1])).to(geom.device)
        geom[b] = geom[b][perm]
    return cond


@torch.no_grad()
def run_pass(model, dataset, device, args, cfg):
    """One evaluation pass over val with a per-batch perturbation hook."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        collate_fn=geometry_collate, pin_memory=True)
    preds, gts = [], []
    for batch in loader:
        b = move(batch, device)
        cond = cfg(b) if cfg else b['condition']
        if args.model == 'flow':
            c = model.sample(cond, n_steps=args.ode_steps,
                             n_samples=args.n_samples)
            preds.append(c.mean(0)[:, 0].cpu().numpy())
        else:
            c = model(cond)
            preds.append(c[:, 0].cpu().numpy())
        gts.append(b['c_gt'][:, 0].cpu().numpy())
    return grid_metrics(np.concatenate(preds), np.concatenate(gts),
                        dataset.manifest['grid'])


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; existing runs are never overwritten')
    args.out.mkdir(parents=True)
    device = torch.device(args.device)
    set_seed(args.sample_seed)
    model, blob = load_model(args.ckpt, device, args.model)
    dataset = StructuredSoSDataset(args.cache, 'val')
    if args.ids:
        wanted = [v for v in args.ids.split(',') if v]
        dataset.records = [r for r in dataset.records if r['id'] in wanted]
        if not dataset.records:
            raise FileNotFoundError('No matching val ids in the cache')
    ks = [int(v) for v in args.ks.split(',')]
    shifts = [float(v) for v in args.shifts.split(',')]

    # Batch index bookkeeping for per-sample RNG stability across modes.
    state = {'index': 0}

    def with_index(fn):
        def hook(batch):
            out = fn(batch, state['index'])
            state['index'] += batch['condition']['event_geom'].shape[0]
            return out
        return hook

    results = {}

    def record(name, metrics):
        results[name] = metrics
        print(f'{name:>18}: ' + '  '.join(f'{k}={v:.4f}' for k, v in metrics.items()),
              flush=True)

    for k in ks:
        if k == ks[0] and k >= 11:
            if 'full' in args.modes.split(','):
                with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                    torch.manual_seed(args.sample_seed + 17)
                    state['index'] = 0
                    record(f'full_{k}', run_pass(model, dataset, device, args, None))
            continue
        if 'drop' not in args.modes.split(','):
            continue
        per_seed = []
        for seed in range(args.seeds):
            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                torch.manual_seed(args.sample_seed + 17)
                state['index'] = 0
                m = run_pass(model, dataset, device, args,
                             with_index(lambda b, i: perturb_drop(b, k, seed, i)))
            per_seed.append(m)
        agg = {key: float(np.mean([m[key] for m in per_seed])) for key in per_seed[0]}
        agg['std_roi_mae'] = float(np.std([m['roi_mae'] for m in per_seed]))
        agg['n_seeds'] = len(per_seed)
        record(f'random_{k}', agg)

    if 'shift' in args.modes.split(','):
        for delta in shifts:
            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                torch.manual_seed(args.sample_seed + 17)
                state['index'] = 0
                record(f'shift+{delta:g}deg', run_pass(
                    model, dataset, device, args,
                    lambda b: perturb_shift(b, delta)))

    if 'permute' in args.modes.split(','):
        for seed in range(args.seeds):
            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                torch.manual_seed(args.sample_seed + 17)
                state['index'] = 0
                record(f'permute_seed{seed}', run_pass(
                    model, dataset, device, args,
                    with_index(lambda b, i: perturb_permute(b, seed, i))))
        perm = [v for k_, v in results.items() if k_.startswith('permute')]
        results['permute'] = {key: float(np.mean([m[key] for m in perm])) for key in perm[0]}
        results['permute']['std_roi_mae'] = float(np.std([m['roi_mae'] for m in perm]))

    report = {
        'ckpt': str(args.ckpt), 'model': args.model, 'cache': str(args.cache),
        'val_samples': len(dataset), 'ode_steps': args.ode_steps,
        'n_samples': args.n_samples, 'seeds': args.seeds,
        'results': results,
        'note': ('event_geom = [sin(theta), cos(theta), theta/45deg, tx_ref_us]; '
                 'shift rewrites the three angle columns only; permute rebinds '
                 'RF rows to permuted geometry rows, breaking (y_i, g_i) pairing.'),
    }
    (args.out/'eval.json').write_text(json.dumps(report, indent=2)+'\n')

    print('\n| perturbation | roi_mae (m/s) | mae | corr |')
    print('|---|---|---|---|')
    for name, m in results.items():
        std = f" ±{m['std_roi_mae']:.2f}" if 'std_roi_mae' in m else ''
        print(f"| {name} | {m['roi_mae']:.2f}{std} | {m['mae']:.2f} | {m['corr']:.4f} |")


if __name__ == '__main__':
    main()
