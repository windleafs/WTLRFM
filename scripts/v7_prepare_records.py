"""Stage 1 (torch) of the v7 physics-objective validation.

Selects records covering original/new probe x breast/homogeneous media,
computes the v5 model prediction for each, and writes one npz per record
with RF, acquisition parameters, truth map and prediction on a shared grid.
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

from data import geometry as G
from data.rf_geometry import structured_condition
from models.geometry_flow import GeometryAwareSoSFlow

DATA = Path('/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle')

# deterministic coordinate axes of the simulator's truth maps (50um grid,
# window i0=176 laterally / face=60 axially, 4x block mean -> 200um cells)
X_T = ((np.arange(192)*4 + 176) - 560)*50e-6
Z_T = (np.arange(216)*4)*50e-6


def pick_records():
    out = []
    for i, sid in enumerate(('train_000', 'train_016', 'train_032', 'train_048')):
        out.append((f'orig_breast_{i}', 'orig_breast', DATA/'raw'/f'{sid}.npz'))
    for i, tag in enumerate(('c1450_r_normal_curve', 'c1500_r_normal',
                             'c1540_r_normal', 'c1580_r_normal_curve')):
        out.append((f'orig_homo_{i}', 'orig_homo',
                    Path(f'/data/zhuangyang/tmp/calib_sim/rf_{tag}.npz')))
    probe = Path('/data/zhuangyang/tmp/probe_raw')
    for i, sid in enumerate(('train_000', 'train_016', 'train_032', 'train_048')):
        out.append((f'l125_breast_{i}', 'l125_breast', probe/f'{sid}_l125breast.npz'))
    flats = sorted(probe.glob('flat*_l125flat.npz'))[:4]
    for i, p in enumerate(flats):
        out.append((f'l125_flat_{i}', 'l125_flat', p))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--ckpt', default=str(ROOT/'out/geometry_flow_v5_depth_20260923/best.pth'))
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    blob = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = dict(blob['encoder_cfg'])
    cfg.update(depth_tgc=True, q_rel_eps=.05)
    model = GeometryAwareSoSFlow(unet=blob['unet'], encoder=cfg,
                                 u_source_scale=-1.).to(device).eval()
    model.load_state_dict(blob['state_dict'])

    index = {r['id']: r for r in
             (json.loads((DATA/'index.json').read_text()))['samples']}
    manifest = []
    for rid, group, path in pick_records():
        d = np.load(path, allow_pickle=False)
        rf = d['rf']
        if 'meta_json' in d.files:
            meta = json.loads(str(d['meta_json'].item()))
            angles = np.asarray(meta['angles_deg'])
            trefs = np.asarray(meta['source_tref_s'])
            fs, fc = float(meta['fs_hz']), float(meta['fc_hz'])
        else:                                    # dataset raw npz
            meta = json.loads(str(d['metadata_json'].item()))
            angles = np.asarray(meta['angles_deg'])
            trefs = np.asarray(meta['source_tref_s'])
            fs, fc = float(meta['fs_hz']), float(meta['source_f0_hz'])
        n_el = rf.shape[1]
        xe = (np.arange(n_el) - (n_el-1)/2)*2e-4
        cond = structured_condition(
            rf, xe, angles, trefs, G.x_grid(), G.z_grid(), fs, fc, 1500.,
            0.2333, t0_s=0., chunk=4096)
        condition = {k: (v.to(device)[None] if torch.is_tensor(v) else v)
                     for k, v in cond.items()}
        torch.manual_seed(20260923)
        with torch.no_grad():
            draws = model.sample(condition, n_steps=10, n_samples=16)
        pred = draws.mean(0)[0, 0].cpu().numpy()          # [nx, nz] model grid
        # resample prediction onto the truth grid (x, z separable)
        xi, zi = G.x_grid().astype(np.float64), G.z_grid().astype(np.float64)
        step = np.stack([np.interp(Z_T, zi, pred[i]) for i in range(len(xi))])
        pred_t = np.stack([np.interp(X_T, xi, step[:, j])
                           for j in range(len(Z_T))], axis=1)  # [192, 216]
        if 'c' in d.files:
            truth = d['c'].T.astype(np.float64)            # raw stores [z,x]
            if truth.shape != (len(X_T), len(Z_T)):
                truth = d['c'].astype(np.float64)
        else:                                             # homogeneous records
            truth = np.full((len(X_T), len(Z_T)), float(meta['c_gt']))
        np.savez_compressed(args.out/f'{rid}.npz', rf=rf, truth=truth,
                            pred=pred_t, xe=xe, angles=angles, trefs=trefs,
                            fs=fs, fc=fc, x=X_T, z=Z_T)
        manifest.append({'id': rid, 'group': group, 'source': str(path),
                         'c_gt_mean': float(truth.mean())})
        print(f'{rid}: rf {rf.shape} gt_mean {truth.mean():.1f} '
              f'pred_mean {pred_t.mean():.1f}', flush=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
