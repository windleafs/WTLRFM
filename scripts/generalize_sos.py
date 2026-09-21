import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from data.acquisition import (apply_calibration, audit_acquisition, load_acquisition,
                              load_real_acquisition)
from data.sos_generalization import candidate_maps, score_images, validate_map
from wfc_integration.generalization import build_model


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def plot_maps(path, maps, x, z):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(maps), figsize=(4*len(maps), 5), squeeze=False)
    for ax, (name, c) in zip(axes[0], maps.items()):
        im = ax.imshow(c.T, extent=[x[0]*1e3, x[-1]*1e3, z[-1]*1e3, z[0]*1e3],
                       vmin=1400, vmax=1650, cmap='viridis', aspect='auto')
        ax.set(title=name, xlabel='x [mm]', ylabel='z [mm]')
        fig.colorbar(im, ax=ax, label='m/s')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--real-prefix', type=Path)
    source.add_argument('--acquisition', type=Path)
    p.add_argument('--prediction', required=True, type=Path)
    p.add_argument('--training-manifest', type=Path)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--backend', type=Path, default=ROOT.parent/'wfc_dbua_pw')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--dz-mm', type=float, default=.4)
    p.add_argument('--depth-mm', type=float, default=40.)
    p.add_argument('--tx-window', choices=('flat', 'hann'), default='flat')
    p.add_argument('--sigmas-mm', type=float, nargs='+', default=[1., 2., 4.])
    p.add_argument('--alphas', type=float, nargs='+', default=[.25, .5, 1.])
    p.add_argument('--scalar-speeds', type=float, nargs='+', default=[1420., 1460., 1500., 1540., 1580., 1620.])
    args = p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Use a new output directory; previous experiments are never overwritten')
    if not np.isfinite(args.scalar_speeds).all() or min(args.scalar_speeds) < 1350 or max(args.scalar_speeds) > 1800:
        raise ValueError('Scalar candidates must lie in [1350,1800] m/s')
    a = load_real_acquisition(args.real_prefix) if args.real_prefix else load_acquisition(args.acquisition)
    if args.calibration:
        a = apply_calibration(a, json.loads(args.calibration.read_text()))
    with np.load(args.prediction, allow_pickle=False) as d:
        c, x, z = (np.asarray(d[k]) for k in ('c_pred', 'cx', 'cz'))
        prior_events = d['input_event_indices'].tolist() if 'input_event_indices' in d else list(range(len(a['angles_deg'])))
    validate_map(c, x, z)
    training = None
    if args.training_manifest:
        m = json.loads(args.training_manifest.read_text())
        training = dict(angles_deg=m['angles_deg'], fc_hz=m.get('carrier_hz', m.get('fc_hz')),
                        fs_hz=m['fs_hz'])
    audit = audit_acquisition(a, training)
    model, project, roi, splits, protocol = build_model(a, x, z, args.backend, args.dz_mm, args.depth_mm, args.tx_window)
    if not set(prior_events).issubset(range(len(a['angles_deg']))):
        raise ValueError('Prediction event provenance is outside the acquisition')
    protocol.update(event_indices=splits, prior_event_indices=prior_events,
                    independent_validation=not bool(set(prior_events) & set(splits['val'])),
                    independent_test=not bool(set(prior_events) & set(splits['test'])),
                    caveat='Legacy predictions may use every event: held out from optimization does not then mean independent of the prior.')
    args.out.mkdir(parents=True)
    save_json(args.out/'audit.json', audit)
    np.savez_compressed(args.out/'acquisition.npz', **{k: v for k, v in a.items() if k != 'metadata'},
                        metadata=json.dumps(a['metadata']))
    image_fn = jax.jit(model.image_from_f0)
    selection_splits = {k: splits[k] for k in ('train', 'val')}
    scalar_rows = []
    for speed in args.scalar_speeds:
        image = np.asarray(image_fn(jnp.full((64, 80), speed, dtype=jnp.float32), model.f0_img))
        metrics = score_images(image, roi, {'train': splits['train']})
        scalar_rows.append({'speed': speed, **metrics['train']})
    base = max(scalar_rows, key=lambda row: row['coherence'])['speed']
    maps = candidate_maps(c, x, z, base, args.sigmas_mm, args.alphas)
    metrics = {}
    for name, candidate in maps.items():
        image = np.asarray(image_fn(jnp.asarray(project(candidate)), model.f0_img))
        metrics[name] = score_images(image, roi, selection_splits)
        print(json.dumps({'candidate': name, **metrics[name]}), flush=True)
    selected = max(metrics, key=lambda name: metrics[name]['val']['coherence'])
    comparisons = dict.fromkeys(['constant1540', 'scalar_fit', 'flow_mean', 'flow', selected])
    for name in comparisons:
        image = np.asarray(image_fn(jnp.asarray(project(maps[name])), model.f0_img))
        metrics[name].update(score_images(image, roi, {'test': splits['test']}))
    np.savez_compressed(args.out/'candidates.npz', cx=x, cz=z, **maps)
    np.savez_compressed(args.out/'selected.npz', c_pred=maps[selected], cx=x, cz=z,
                        input_event_indices=np.unique(prior_events + splits['train'] + splits['val']))
    summary = dict(stage='ablation', selected_by_validation=selected, base_speed_m_s=base,
                   scalar_scan_train_only=scalar_rows, metrics=metrics, protocol=protocol,
                   prediction_source=str(args.prediction.resolve()), accuracy_metrics_available=False)
    save_json(args.out/'summary.json', summary)
    plot_maps(args.out/'maps.png', {name: maps[name] for name in dict.fromkeys(['constant1540', 'flow', 'scalar_fit', selected])}, x, z)
    print(json.dumps({'selected': selected, 'out': str(args.out), 'independent_test': protocol['independent_test']}), flush=True)


if __name__ == '__main__':
    main()
