"""Absolute-SoS calibration benchmark and encoder identifiability test.

Sweeps homogeneous sound speed c in {1450..1580} crossed with speckle
texture density {sparse, normal, dense, anechoic-like} on otherwise
reflectivity-only media, then measures:

  * per-condition ROI-mean predictions for any number of checkpoints
  * calibration fit  c_pred = a * c_GT + b  (S_cal = a, E_cal = mean|err|)
  * the 4x4 causal matrix (prediction should follow c, ignore texture)
  * encoder feature distances  D(c_i, c_j)  vs  D(r_i, r_j)  — is absolute
    SoS evidence even representable in the condition features?

Stage sim: torch-free, NVIDIA HPC SDK on PATH, shardable across GPUs.
Stage infer/encoder: torch.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

C_MATRIX = [1460., 1500., 1540., 1580.]
C_EXTRA = [1450., 1470., 1520., 1560., 1580.]
TEXTURES = {'sparse': .3, 'normal': 1., 'dense': 2., 'anechoic': .05}
BASE_SPECKLE = 8.      # kg/m^3
SEED = 20260923


def conditions_list(curve_only=False):
    """(tag, c, texture_multiplier) triples; matrix + finer normal curve."""
    out = []
    if not curve_only:
        for c in C_MATRIX:
            for rname, mult in TEXTURES.items():
                out.append((f'c{int(c)}_r_{rname}', c, mult))
    for c in C_EXTRA:
        out.append((f'c{int(c)}_r_normal_curve', c, 1.))
    return out


def build_maps(shape, x, z, c, speckle_mult, seed=SEED):
    zz, xx = np.meshgrid(z, x, indexing='ij')
    in_tissue = zz*1e3 >= 0.
    rng = np.random.default_rng(seed)
    speckle = rng.normal(0., BASE_SPECKLE*speckle_mult, shape)*in_tissue
    maps = {
        'sound_speed': np.full(shape, float(c), np.float32),
        'density': (1000. + speckle).astype(np.float32),
        'alpha_coeff': np.full(shape, .002, np.float32),
        'BonA': np.zeros(shape, np.float32),
    }
    return maps


def stage_sim(args):
    sys.path.insert(0, '/home/zhuangyang/fmmodel/neural_asp')
    sys.path.insert(0, '/home/zhuangyang/fmmodel/UltraWave/benchmarks')
    sys.path.insert(0, '/data/zhuangyang/NumerialBreastPhantoms')
    if 'torch' in sys.modules:
        raise RuntimeError('torch must not be imported in the sim process')
    import scripts.generate_l11_ultrawave_raw as gen
    gen.configure_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    refs = np.load(args.root/'reference_native.npz')['rf_native']
    conds = conditions_list()
    i, n = (int(v) for v in args.shard.split('/'))
    work = conds[i::n]
    case0 = gen.geometry_case()
    start = time.monotonic()
    for tag, c, mult in work:
        out_path = args.out/f'rf_{tag}.npz'
        if out_path.exists():
            continue
        case = gen.geometry_case()
        case['maps'] = build_maps((len(case['z']), len(case['x'])),
                                  case['x'], case['z'], c, mult)
        gen.bench.validate_case(case)
        fit = gen.absorption_model(case)
        solver = gen.solver_for(case, case['maps'], fit)
        angles = np.linspace(-8., 8., 11)
        rf_list, trefs = [], []
        for ai, angle in enumerate(angles):
            tref = gen.set_angle(solver, case, float(angle))
            total, timing = solver.run()
            rf, _ = gen.sim.analytic_channels(total - refs[:, :, ai], gen.DT,
                                              band=[4e6, 7.5e6])
            rf_list.append(rf.T.astype(np.float32))
            trefs.append(float(tref))
        rf = np.stack(rf_list)
        if not np.isfinite(rf).all() or rf.std() <= 0:
            raise RuntimeError(f'{tag}: invalid RF')
        meta = {'tag': tag, 'c_gt': float(c), 'texture_mult': float(mult),
                'angles_deg': angles.tolist(), 'source_tref_s': trefs,
                'fs_hz': 40e6, 'fc_hz': 7.5e6}
        tmp = out_path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, rf=rf, meta_json=np.asarray(json.dumps(meta)))
        tmp.replace(out_path)
        print(json.dumps({'tag': tag, 'elapsed': round(time.monotonic()-start, 1)}),
              flush=True)
    print('[sim] shard done', flush=True)


def load_condition(path, device):
    import torch
    sys.path.insert(0, str(ROOT))
    from data import geometry as G
    from data.rf_geometry import structured_condition
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta_json'].item()))
    rf = d['rf']
    xe = (np.arange(rf.shape[1]) - (rf.shape[1]-1)/2)*2e-4
    cond = structured_condition(
        rf, xe, np.asarray(meta['angles_deg']),
        np.asarray(meta['source_tref_s']), G.x_grid(), G.z_grid(),
        meta['fs_hz'], meta['fc_hz'], 1500., 0.2333, t0_s=0., chunk=4096)
    return {k: (v.to(device)[None] if torch.is_tensor(v) else v)
            for k, v in cond.items()}, meta


def roi_mean(pred):
    from data import geometry as G
    xi, zi = G.x_grid(), G.z_grid()
    m = (((zi[None, :] >= 3e-3) & (zi[None, :] <= 45e-3))
         & (xi[:, None] >= xi.min()) & (xi[:, None] <= xi.max()))
    return float(pred[m].mean())


def stage_infer(args):
    import torch
    sys.path.insert(0, str(ROOT))
    from models.geometry_flow import GeometryAwareSoSFlow
    device = torch.device(args.device)
    ckpts = [(name, path) for name, path in
             (p.split('=') for p in args.ckpts.split(','))]
    files = sorted(args.out.glob('rf_*.npz'))
    if not files:
        raise FileNotFoundError('run --stage sim first')
    models = {}
    for name, path in ckpts:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        model = GeometryAwareSoSFlow(unet=blob['unet'],
                                     encoder=blob['encoder_cfg'],
                                     u_source_scale=-1.).to(device).eval()
        model.load_state_dict(blob['state_dict'])
        models[name] = model
    rows = []
    for path in files:
        condition, meta = load_condition(path, device)
        torch.manual_seed(SEED)
        with torch.no_grad():
            draws = models[next(iter(models))].sample(
                condition, n_steps=10, n_samples=16)
        row = {'tag': meta['tag'], 'c_gt': meta['c_gt'],
               'texture_mult': meta['texture_mult']}
        for name, model in models.items():
            if len(models) > 1:
                torch.manual_seed(SEED)
                with torch.no_grad():
                    draws = model.sample(condition, n_steps=10, n_samples=16)
            row[f'mean_{name}'] = roi_mean(draws.mean(0)[0, 0].cpu().numpy())
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.out/'calibration_rows.json').write_text(json.dumps(rows, indent=2)+'\n')

    report = {}
    for name, _ in ckpts:
        key = f'mean_{name}'
        curve = [r for r in rows if r['texture_mult'] == 1.]
        c_gt = np.array([r['c_gt'] for r in curve])
        c_pred = np.array([r[key] for r in curve])
        a, b = np.polyfit(c_gt, c_pred, 1)
        matrix = {}
        for r in rows:
            if r['tag'].endswith('_curve'):
                continue
            tex = r['tag'].split('_r_')[1]
            matrix.setdefault(tex, {})[str(int(r['c_gt']))] = round(r[key], 1)
        dcdc = {}
        for tex in TEXTURES:
            col = sorted([r for r in rows if r['tag'].endswith(f'_r_{tex}')],
                         key=lambda r: r['c_gt'])
            cs = np.array([r['c_gt'] for r in col])
            ps = np.array([r[key] for r in col])
            dcdc[tex] = round(float((ps[-1]-ps[0])/(cs[-1]-cs[0])), 3)
        dcdr = {}
        for cval in C_MATRIX:
            col = [r for r in rows if r['c_gt'] == cval
                   and not r['tag'].endswith('_curve')]
            ps = {r['tag'].split('_r_')[1]: r[key] for r in col}
            dcdr[str(int(cval))] = round(
                float(max(ps.values())-min(ps.values())), 1)
        report[name] = {'S_cal_slope': round(float(a), 3),
                        'intercept': round(float(b), 1),
                        'E_cal': round(float(np.abs(c_pred-c_gt).mean()), 2),
                        'dcdc_per_texture': dcdc, 'dcdr_per_c': dcdr,
                        'matrix': matrix}
    (args.out/'calibration_report.json').write_text(
        json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def stage_encoder(args):
    import torch
    sys.path.insert(0, str(ROOT))
    from models.geometry_flow import GeometryAwareSoSFlow
    device = torch.device(args.device)
    ckpts = [(name, path) for name, path in
             (p.split('=') for p in args.ckpts.split(','))]
    files = sorted(args.out.glob('rf_*.npz'))
    feats = {name: {} for name, _ in ckpts}
    metas = {}
    for path in files:
        condition, meta = load_condition(path, device)
        metas[meta['tag']] = meta
        for name, cpath in ckpts:
            blob = torch.load(cpath, map_location='cpu', weights_only=False)
            model = GeometryAwareSoSFlow(unet=blob['unet'],
                                         encoder=blob['encoder_cfg'],
                                         u_source_scale=-1.).to(device).eval()
            model.load_state_dict(blob['state_dict'])
            with torch.no_grad():
                e = model.encoder(condition).flatten().float()
            feats[name][meta['tag']] = e
    report = {}
    for name in feats:
        def dist(t1, t2):
            return float((feats[name][t1]-feats[name][t2]).norm())
        pairs_c, pairs_r = [], []
        tags = list(metas)
        curve_tags = [t for t in tags if t.endswith('_r_normal_curve')
                      or t == 'c1500_r_normal']
        curve_tags = sorted(curve_tags, key=lambda t: metas[t]['c_gt'])
        for i in range(len(curve_tags)):
            for j in range(i+1, len(curve_tags)):
                dc = abs(metas[curve_tags[i]]['c_gt']-metas[curve_tags[j]]['c_gt'])
                pairs_c.append(dist(curve_tags[i], curve_tags[j])/dc)
        tex_tags = {tex: f'c1500_r_{tex}' for tex in TEXTURES}
        base = tex_tags['normal']
        for tex, tag in tex_tags.items():
            if tex != 'normal' and tag in feats[name]:
                pairs_r.append(dist(base, tag))
        scale = float(feats[name][base].norm())
        report[name] = {
            'feature_norm': round(scale, 1),
            'D_per_mps_same_texture_mean': round(float(np.mean(pairs_c)), 4),
            'D_texture_change_at_1500': {tex: round(dist(base, tag), 1)
                                         for tex, tag in tex_tags.items()
                                         if tag in feats[name] and tex != 'normal'},
            'note': 'D_per_mps: encoder-feature L2 distance per m/s of true '
                    'SoS change at fixed texture; D_texture: distance for '
                    'texture change at fixed c=1500'}
    (args.out/'encoder_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('sim', 'infer', 'encoder'))
    p.add_argument('--root', type=Path,
                   default=Path('/data/zhuangyang/NumerialBreastPhantoms/'
                                'l11_ultrawave_500_11angle'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--ckpts', default=(
        'v2=out/geometry_flow_v2_20260921/best.pth,'
        'v3=out/geometry_flow_v3_decoupled_20260922/best.pth'))
    args = p.parse_args()
    {'sim': stage_sim, 'infer': stage_infer, 'encoder': stage_encoder}[args.stage](args)


if __name__ == '__main__':
    main()
