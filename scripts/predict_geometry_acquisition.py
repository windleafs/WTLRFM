"""Predict a SoS map from a real/generic RF acquisition with the geometry-aware flow."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.acquisition import audit_acquisition, load_acquisition, load_real_acquisition
from data.rf_geometry import structured_condition
from models.geometry_flow import GeometryAwareSoSFlow


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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--real-prefix', type=Path)
    source.add_argument('--acquisition', type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--training-manifest', type=Path)
    p.add_argument('--speeds', default='1450,1500,1550')
    p.add_argument('--ref-speed', type=float, default=1500.)
    p.add_argument('--n-subap', type=int, default=4)
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--nz', type=int, default=160)
    p.add_argument('--x0', type=float, default=-.019125)
    p.add_argument('--x1', type=float, default=.019075)
    p.add_argument('--z0', type=float, default=.000075)
    p.add_argument('--z1', type=float, default=.043075)
    p.add_argument('--ode-steps', type=int, default=10)
    p.add_argument('--n-samples', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=20260921)
    return p.parse_args()


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh prediction directory')
    a = load_real_acquisition(args.real_prefix) if args.real_prefix else load_acquisition(args.acquisition)
    training = None
    if args.training_manifest:
        m = json.loads(args.training_manifest.read_text())
        training = dict(angles_deg=m.get('angles_deg'), fc_hz=m.get('carrier_hz', m.get('fc_hz')),
                        fs_hz=m.get('fs_hz'))
    audit = audit_acquisition(a, training)
    device = torch.device(args.device)
    rf = torch.from_numpy(np.asarray(a['rf'] * a.get('tgc_gain', 1.), np.float32)).to(device)
    xi = np.linspace(args.x0, args.x1, args.nx, dtype=np.float32)
    zi = np.linspace(args.z0, args.z1, args.nz, dtype=np.float32)
    speeds = [float(v) for v in args.speeds.split(',')]
    condition = structured_condition(
        rf, a['xe'], a['angles_deg'], a['tx_t_ref_s'], xi, zi,
        float(a['fs_hz']), float(a['fc_hz']), float(a['c_steer']),
        float(a['bandwidth_fraction']), speeds=speeds, ref_speed=args.ref_speed,
        n_subap=args.n_subap, t0_s=float(a['t0_s']))
    batch = {k: (v[None].to(device) if torch.is_tensor(v) else v)
             for k, v in condition.items() if k != 'event_indices'}
    model = GeometryAwareSoSFlow.from_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    torch.manual_seed(args.seed)
    with torch.no_grad():
        samples, encoded, aux = model.sample(batch, n_steps=args.ode_steps,
                                             n_samples=args.n_samples,
                                             return_cond=True)
    mean = samples.mean(0)[0, 0].cpu().numpy().astype(np.float32)
    std = samples.std(0)[0, 0].cpu().numpy().astype(np.float32)
    event_indices = np.flatnonzero(condition['event_mask'].cpu().numpy()).astype(np.int16)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'prediction.npz', c_pred=mean, c_std=std, cx=xi, cz=zi,
                        input_event_indices=event_indices,
                        canonical_slot_angles_rad=model.encoder.slot_angles.detach().cpu().numpy(),
                        encoded_condition=encoded[0].cpu().numpy())
    weights = aux['event_weights'][0].cpu().numpy()
    summary = dict(stage='geometry_aware_prediction', checkpoint=str(args.checkpoint.resolve()),
                   acquisition=audit, output_grid_m={'x0': args.x0, 'x1': args.x1,
                                                     'z0': args.z0, 'z1': args.z1,
                                                     'nx': args.nx, 'nz': args.nz},
                   condition={'speeds': speeds, 'ref_speed': args.ref_speed,
                              'n_subap': args.n_subap,
                              'input_event_indices': event_indices.tolist(),
                              'event_attention_minmax': [float(weights.min()), float(weights.max())]},
                   sampling={'ode_steps': args.ode_steps, 'n_samples': args.n_samples,
                             'seed': args.seed,
                             'ensemble_std_is_uncertainty': False},
                   accuracy_metrics_available=False)
    save_json(args.out/'summary.json', summary)
    np.save(args.out/'event_weights.npy', weights)
    plot_maps(args.out/'prediction.png', {'geometry_flow': mean}, xi, zi)
    print(json.dumps({'prediction': str(args.out/'prediction.npz'),
                      'active_events': event_indices.tolist(),
                      'speed_range_m_s': [float(mean.min()), float(mean.max())]}), flush=True)


if __name__ == '__main__':
    main()
