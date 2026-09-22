"""Homogeneous-SoS counterfactual: reflectivity-only media through GeometryFlow.

Builds full-wave simulations in which the sound speed is exactly 1500 m/s
everywhere while ALL echo contrast comes from density (impedance) structure:
specular bright points, an arc reflector, hyper/hypo-echogenic patches
(speckle-density variation), anechoic cysts and a strong reflecting boundary.
A SoS inverter should return ~1500 m/s everywhere; structure-correlated
deviations are direct evidence of an echogenicity shortcut.

Stage ``sim`` (torch-free, NVIDIA HPC SDK on PATH) runs the UltraWave solver
with the dataset's exact recipe; stage ``infer`` beamforms the structured
condition, samples GeometryFlow, and reports per-region SoS deviations.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# --- counterfactual layout (mm, model-grid coordinates) --------------------
BRIGHT_POINTS = [(-9., 13.), (0., 16.), (9., 13.)]
POINT_SIGMA, POINT_AMP = .25, 60.
ARC_CENTER, ARC_RADIUS = (0., -12.), 33.
ARC_SIGMA, ARC_AMP = .3, 25.
HYPER_RECT = ((-16., -6.), (26., 34.))     # x range, z range
HYPER_GAIN = 2.0                            # speckle std multiplier
HYPO_RECT = ((6., 16.), (26., 34.))
HYPO_GAIN = 0.4
CYSTS = [(-8., 38.5), (8., 38.5)]
CYST_RADIUS = 3.
BOUNDARY_Z = 24.5
BOUNDARY_STEP = 40.                          # rho jump below the boundary
BASE_SPECKLE = 8.                            # rho noise std (kg/m^3)
C_HOMOGENEOUS = 1500.

REGIONS = {'background': None, 'bright_points': None, 'arc': None,
           'hyperechoic': None, 'hypoechoic': None, 'anechoic': None,
           'strong_boundary': None}


def build_maps(shape, x, z, seed=20260922):
    """Density-only reflectivity on an exactly homogeneous SoS medium."""
    zz, xx = np.meshgrid(z, x, indexing='ij')            # [nz, nx] metres
    xmm, zmm = xx*1e3, zz*1e3
    rng = np.random.default_rng(seed)
    speckle = rng.normal(0., BASE_SPECKLE, shape)
    in_tissue = zmm >= 0.
    speckle = np.where(in_tissue, speckle, 0.)
    rho = 1000. + speckle

    for cx, cz in BRIGHT_POINTS:
        rho = rho + POINT_AMP*np.exp(-(((xmm-cx)**2+(zmm-cz)**2)
                                       / (2.*POINT_SIGMA**2)))
    d_arc = np.sqrt((xmm-ARC_CENTER[0])**2 + (zmm-ARC_CENTER[1])**2) - ARC_RADIUS
    rho = rho + ARC_AMP*np.exp(-(d_arc**2)/(2.*ARC_SIGMA**2))
    rho = rho + BOUNDARY_STEP*(zmm > BOUNDARY_Z)

    hyper = ((xmm >= HYPER_RECT[0][0]) & (xmm <= HYPER_RECT[0][1])
             & (zmm >= HYPER_RECT[1][0]) & (zmm <= HYPER_RECT[1][1]))
    hypo = ((xmm >= HYPO_RECT[0][0]) & (xmm <= HYPO_RECT[0][1])
            & (zmm >= HYPO_RECT[1][0]) & (zmm <= HYPO_RECT[1][1]))
    extra = rng.normal(0., BASE_SPECKLE, shape)
    rho = rho + np.where(hyper, (HYPER_GAIN-1.)*speckle
                         + (HYPER_GAIN-1.)*extra, 0.)
    rho = rho + np.where(hypo, (HYPO_GAIN-1.)*speckle, 0.)
    for cx, cz in CYSTS:                                  # anechoic: kill speckle
        inside = ((xmm-cx)**2 + (zmm-cz)**2) <= CYST_RADIUS**2
        mean_rho = 1000. + BOUNDARY_STEP*(cz > BOUNDARY_Z)
        rho = np.where(inside, mean_rho, rho)

    maps = {
        'sound_speed': np.full(shape, C_HOMOGENEOUS, np.float32),
        'density': rho.astype(np.float32),
        'alpha_coeff': np.full(shape, .002, np.float32),
        'BonA': np.zeros(shape, np.float32),
    }
    assert np.allclose(maps['sound_speed'], C_HOMOGENEOUS, atol=0)
    return maps


def stage_sim(args):
    sys.path.insert(0, '/home/zhuangyang/fmmodel/neural_asp')
    sys.path.insert(0, '/home/zhuangyang/fmmodel/UltraWave/benchmarks')
    sys.path.insert(0, '/data/zhuangyang/NumerialBreastPhantoms')
    if 'torch' in sys.modules:
        raise RuntimeError('torch must not be imported in the sim process')
    import scripts.generate_l11_ultrawave_raw as gen
    gen.configure_gpu()
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    case = gen.geometry_case()
    maps = build_maps((len(case['z']), len(case['x'])), case['x'], case['z'])
    case['maps'] = maps
    gen.bench.validate_case(case)
    fit = gen.absorption_model(case)
    solver = gen.solver_for(case, maps, fit)
    refs = np.load(args.root/'reference_native.npz')['rf_native']
    angles = np.linspace(-8., 8., 11)
    rf_list, trefs = [], []
    for ai, angle in enumerate(angles):
        tref = gen.set_angle(solver, case, float(angle))
        total, timing = solver.run()
        rf, _ = gen.sim.analytic_channels(total - refs[:, :, ai], gen.DT,
                                          band=[4e6, 7.5e6])
        if rf.shape != (2401, 192):
            raise RuntimeError(f'unexpected analytic shape {rf.shape}')
        rf_list.append(rf.T.astype(np.float32))
        trefs.append(float(tref))
        print(f'[sim] angle={angle:+.1f} solve={timing["solve_readback_s"]:.2f}s '
              f'elapsed={time.monotonic()-start:.1f}s', flush=True)
    rf = np.stack(rf_list)
    if not np.isfinite(rf).all() or rf.std() <= 0:
        raise RuntimeError('invalid counterfactual RF')
    meta = {'angles_deg': angles.tolist(), 'source_tref_s': trefs,
            'fs_hz': 40e6, 'fc_hz': 7.5e6, 'band_hz': [4e6, 7.5e6],
            'c_homogeneous': C_HOMOGENEOUS, 'layout': {
                'bright_points': BRIGHT_POINTS, 'arc_center': ARC_CENTER,
                'arc_radius': ARC_RADIUS, 'hyper_rect': HYPER_RECT,
                'hypo_rect': HYPO_RECT, 'cysts': CYSTS,
                'cyst_radius': CYST_RADIUS, 'boundary_z': BOUNDARY_Z}}
    np.savez_compressed(args.out/'counterfactual_rf.npz', rf=rf,
                        meta_json=np.asarray(json.dumps(meta)))
    print(f'[sim] wrote {args.out}/"counterfactual_rf.npz" '
          f'rf_rms={float(rf.std()):.1f}', flush=True)


def region_masks(xi, zi):
    """Boolean masks [nx, nz] for every layout region, in metres."""
    xmm, zmm = np.meshgrid(xi*1e3, zi*1e3, indexing='ij')
    points = np.zeros_like(xmm, bool)
    for cx, cz in BRIGHT_POINTS:
        points |= ((xmm-cx)**2 + (zmm-cz)**2) <= 1.0**2
    d_arc = (np.sqrt((xmm-ARC_CENTER[0])**2 + (zmm-ARC_CENTER[1])**2)
             - ARC_RADIUS)
    arc = np.abs(d_arc) <= 1.0
    def rect(r, shrink=1.0):
        return ((xmm >= r[0][0]+shrink) & (xmm <= r[0][1]-shrink)
                & (zmm >= r[1][0]+shrink) & (zmm <= r[1][1]-shrink))
    hyper, hypo = rect(HYPER_RECT), rect(HYPO_RECT)
    cysts = np.zeros_like(xmm, bool)
    for cx, cz in CYSTS:
        cysts |= ((xmm-cx)**2 + (zmm-cz)**2) <= (CYST_RADIUS-1.)**2
    boundary = np.abs(zmm-BOUNDARY_Z) <= 1.0
    masks = {'bright_points': points, 'arc': arc, 'hyperechoic': hyper,
             'hypoechoic': hypo, 'anechoic': cysts, 'strong_boundary': boundary}
    all_struct = np.zeros_like(xmm, bool)
    for m in masks.values():
        all_struct |= m
    # dilate the union by 2.5 mm via distance on the coarse grid
    from scipy.ndimage import binary_dilation
    struct_big = binary_dilation(all_struct, iterations=25)
    masks['background'] = (~struct_big) & (zmm >= 3.) & (zmm <= 43.)
    return masks


def stage_infer(args):
    import torch
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT/'scripts'))
    from data import geometry as G
    from data.rf_geometry import structured_condition
    from models.geometry_flow import GeometryAwareSoSFlow
    from train_geometry_flow import grid_metrics

    blob_npz = np.load(args.out/'counterfactual_rf.npz', allow_pickle=False)
    meta = json.loads(str(blob_npz['meta_json'].item()))
    rf = blob_npz['rf']                                     # [11, 192, 2401]
    xe = (np.arange(rf.shape[1]) - (rf.shape[1]-1)/2)*2e-4
    cond = structured_condition(
        rf, xe, np.asarray(meta['angles_deg']),
        np.asarray(meta['source_tref_s']), G.x_grid(), G.z_grid(),
        meta['fs_hz'], meta['fc_hz'], 1500., 0.2333, t0_s=0., chunk=4096)
    device = torch.device(args.device)
    condition = {k: (v.to(device)[None] if torch.is_tensor(v) else v)
                 for k, v in cond.items()}
    blob = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = GeometryAwareSoSFlow(unet=blob['unet'], encoder=blob['encoder_cfg'],
                                 u_source_scale=-1.).to(device).eval()
    model.load_state_dict(blob['state_dict'])
    torch.manual_seed(20260922)
    with torch.no_grad():
        draws = model.sample(condition, n_steps=10, n_samples=16)
    pred = draws.mean(0)[0, 0].cpu().numpy()
    std = draws.std(0, unbiased=False)[0, 0].cpu().numpy()

    masks = region_masks(G.x_grid().astype(np.float64),
                         G.z_grid().astype(np.float64))
    stats = {}
    for name, m in masks.items():
        if not m.any():
            raise RuntimeError(f'empty region {name}')
        stats[name] = dict(mean=float(pred[m].mean()),
                           std=float(pred[m].std()),
                           dev=float(pred[m].mean()-C_HOMOGENEOUS),
                           post_std=float(std[m].mean()),
                           n=int(m.sum()))
    truth = np.full_like(pred, C_HOMOGENEOUS)
    metrics = grid_metrics(pred, truth, dict(x0_m=G.x_grid()[0],
                                             x1_m=G.x_grid()[-1],
                                             z0_m=G.z_grid()[0],
                                             z1_m=G.z_grid()[-1],
                                             nx=len(G.x_grid()),
                                             nz=len(G.z_grid())))
    # corr between prediction deviation and a reflectivity-structure proxy
    struct_proxy = np.zeros_like(pred)
    for name in ('bright_points', 'arc', 'hyperechoic', 'strong_boundary'):
        struct_proxy[masks[name]] = 1.
    struct_proxy[masks['hypoechoic']] = .5
    a = (pred-C_HOMOGENEOUS)[masks['background'] | masks['bright_points']
                             | masks['arc'] | masks['hyperechoic']
                             | masks['hypoechoic'] | masks['strong_boundary']
                             | masks['anechoic']]
    b = struct_proxy[masks['background'] | masks['bright_points']
                     | masks['arc'] | masks['hyperechoic']
                     | masks['hypoechoic'] | masks['strong_boundary']
                     | masks['anechoic']]
    shortcut_corr = float(np.corrcoef(a, b)[0, 1])
    report = {'metrics_vs_1500': metrics, 'regions': stats,
              'shortcut_corr': shortcut_corr, 'meta': meta}
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'regions': {k: round(v['dev'], 1) for k, v in stats.items()},
                      'roi_mae': round(metrics['roi_mae'], 2),
                      'shortcut_corr': round(shortcut_corr, 3)}, indent=2),
          flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xi, zi = G.x_grid(), G.z_grid()
    extent = [xi[0]*1e3, xi[-1]*1e3, zi[-1]*1e3, zi[0]*1e3]
    env = np.abs(cond['speed_events'].numpy()[1].sum(axis=0))
    bmode = 20*np.log10(np.maximum(env/np.percentile(env[env > 0], 98), 1e-3))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    axes[0].imshow(bmode.T, cmap='gray', vmin=-45, vmax=0, extent=extent,
                   aspect='equal')
    axes[0].set_title('B-mode (reflectivity-only medium)')
    im = axes[1].imshow(pred.T, cmap='turbo', vmin=1440, vmax=1560,
                        extent=extent, aspect='equal')
    fig.colorbar(im, ax=axes[1], shrink=.85, label='SoS [m/s]')
    axes[1].set_title('GeometryFlow prediction (truth = 1500)')
    dev = pred-C_HOMOGENEOUS
    im = axes[2].imshow(dev.T, cmap='RdBu_r', vmin=-30, vmax=30,
                        extent=extent, aspect='equal')
    fig.colorbar(im, ax=axes[2], shrink=.85, label='deviation [m/s]')
    axes[2].set_title('deviation from 1500')
    names = list(stats)
    axes[3].barh(names, [stats[n]['dev'] for n in names], xerr=[stats[n]['std']/np.sqrt(stats[n]['n'])*1.96 for n in names])
    axes[3].axvline(0., color='k', lw=.8)
    axes[3].set_xlabel('mean SoS deviation [m/s]')
    axes[3].set_title('per-structure deviation')
    for ax in axes[:3]:
        ax.set(xlabel='x [mm]', ylabel='Depth [mm]')
    fig.tight_layout()
    fig.savefig(args.out/'counterfactual.png', dpi=160)
    np.savez_compressed(args.out/'prediction.npz', c_pred=pred, c_std=std,
                        bmode_db=bmode, cx=xi, cz=zi,
                        **{f'mask_{k}': v for k, v in masks.items()})
    print(f"[done] -> {args.out/'counterfactual.png'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('sim', 'infer'))
    p.add_argument('--root', type=Path,
                   default=Path('/data/zhuangyang/NumerialBreastPhantoms/'
                                'l11_ultrawave_500_11angle'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--ckpt', default=str(ROOT/'out/geometry_flow_v2_20260921/best.pth'))
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    if args.stage == 'sim':
        stage_sim(args)
    else:
        stage_infer(args)


if __name__ == '__main__':
    main()
