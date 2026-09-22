"""Generate decoupled reflectivity/SoS training media (counterfactual mix).

For each source training plane four variants are simulated with the dataset's
exact UltraWave recipe:

  v1 reseed   - same macro medium, new sub-resolution scatterer realization
                (same c, different reflectivity)
  v2 texture  - v1 plus density-only texture edits: an anechoic cyst, a
                hyper- and a hypo-echogenic patch (same c, stronger)
  v3 cswap    - original scatter realization kept, macro sound speed of
                gland/fat redrawn independently (same reflectivity,
                different c)
  v4 homoc    - breast-textured density with exactly homogeneous c=1500

Torch-free; run with the NVIDIA HPC SDK on PATH.  Sharded via --shard i/n so
several GPUs can generate in parallel; existing outputs are skipped.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

NEURAL_ASP = '/home/zhuangyang/fmmodel/neural_asp'
ULTRAWAVE_BENCH = '/home/zhuangyang/fmmodel/UltraWave/benchmarks'
PHANTOM_DIR = '/data/zhuangyang/NumerialBreastPhantoms'
for p in (NEURAL_ASP, ULTRAWAVE_BENCH, PHANTOM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import scripts.generate_l11_ultrawave_raw as gen  # noqa: E402

VARIANTS = ('reseed', 'texture', 'cswap', 'homoc')
# dual_scale table values (build_acoustic_l11.DUAL_SCALE_TISSUES)
GLAND_C, FAT_C = 1560., 1455.
GLAND_NEW = (1510., 1590.)   # redraw range for gland macro speed
FAT_NEW = (1445., 1500.)
TEXTURE_AMP = 12.            # kg/m^3 extra speckle std, ~ gland level


def macro_density(rho, dx=5e-5):
    """Local scatterer-free density (~1 mm box mean)."""
    return uniform_filter(rho, size=int(1e-3/dx) | 1)


def apply_texture_edits(rho, x, z, seed):
    """Density-only edits: anechoic cyst + hyper/hypo patches (c untouched)."""
    zz, xx = np.meshgrid(z, x, indexing='ij')
    xmm, zmm = xx*1e3, zz*1e3
    rng = np.random.default_rng(seed)
    out = rho.copy()
    cyst_done = patch_done = 0
    while cyst_done < 1:
        cx, cz = rng.uniform(-13., 13.), rng.uniform(10., 38.)
        r = rng.uniform(2.5, 4.)
        inside = (xmm-cx)**2 + (zmm-cz)**2 <= r**2
        if inside.mean() < 1e-4:
            continue
        macro = macro_density(rho)
        out[inside] = macro[inside]
        cyst_done = 1
    for gain in (1. + TEXTURE_AMP/8., 0.35):
        while patch_done < 2:
            w, h = rng.uniform(6., 12.), rng.uniform(6., 12.)
            x0 = rng.uniform(-16., 16.-w)
            z0 = rng.uniform(8., 38.-h)
            rect = ((xmm >= x0) & (xmm <= x0+w) & (zmm >= z0) & (zmm <= z0+h))
            if rect.mean() < 1e-4:
                continue
            macro = macro_density(rho)
            if gain > 1.:
                out[rect] += rng.normal(0., TEXTURE_AMP, out.shape)[rect]
            else:
                out[rect] = macro[rect] + gain*(out[rect]-macro[rect])
            patch_done += 1
            break
    return np.clip(out, 800., 1300.).astype(np.float32)


def remap_sound_speed(c, codes, dx=5e-5, seed=0):
    """Redraw gland/fat macro speeds, keeping interfaces smooth."""
    rng = np.random.default_rng(seed)
    g_new = rng.uniform(*GLAND_NEW)
    f_new = rng.uniform(*FAT_NEW)
    sigma = int(0.2e-3/dx) | 1
    s_g = gaussian_filter((codes == 2).astype(np.float32), sigma)
    s_f = gaussian_filter((codes == 3).astype(np.float32), sigma)
    delta = s_g*(g_new-GLAND_C) + s_f*(f_new-FAT_C)
    out = np.clip(c + delta, 1400., 1700.).astype(np.float32)
    return out


def make_variant(plane, case, base_seed, variant, rng_seed):
    args = (plane, case['x'], case['z'])
    if variant == 'reseed':
        maps, *_ = gen.medium_builder.build_medium(*args, seed=base_seed+101,
                                                   preset='dual_scale')
    elif variant == 'texture':
        maps, *_ = gen.medium_builder.build_medium(*args, seed=base_seed+101,
                                                   preset='dual_scale')
        maps['density'] = apply_texture_edits(maps['density'], case['x'],
                                              case['z'], base_seed+303)
    elif variant == 'cswap':
        maps, codes, *_ = gen.medium_builder.build_medium(*args,
                                                           seed=base_seed,
                                                           preset='dual_scale')
        maps['sound_speed'] = remap_sound_speed(maps['sound_speed'], codes,
                                                seed=base_seed+404)
    elif variant == 'homoc':
        maps, *_ = gen.medium_builder.build_medium(*args, seed=base_seed+202,
                                                   preset='dual_scale')
        maps['sound_speed'] = np.full_like(maps['sound_speed'], 1500.)
    else:
        raise ValueError(variant)
    gel_rows = (case['z']*1e3) < 1.0
    if variant == 'homoc':
        maps['sound_speed'][gel_rows, :] = 1500.
    return maps


def simulate_record(root, rec, out_dir, variant):
    out_path = out_dir/f"{rec['id']}_{variant}.npz"
    if out_path.exists():
        return 'skip'
    with h5py.File(rec['h5'], 'r') as f:
        plane = np.asarray(f['phan'][rec['z_index']])
    case = gen.geometry_case()
    maps = make_variant(plane, case, rec['scatter_seed'], variant,
                        rec['scatter_seed'])
    case['maps'] = maps
    gen.bench.validate_case(case)
    fit = gen.absorption_model(case)
    solver = gen.solver_for(case, maps, fit)
    refs = np.load(root/'reference_native.npz')['rf_native']
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
        raise RuntimeError(f"{rec['id']}/{variant}: invalid RF")
    c, m, _ = gen.truth_maps(maps, case['x'], case['face'])
    meta = {'id': f"{rec['id']}_{variant}", 'source_id': rec['id'],
            'variant': variant, 'split': 'train',
            'case': rec['case'], 'z_index': rec['z_index'],
            'scatter_seed': rec['scatter_seed'],
            'angles_deg': angles.tolist(), 'source_tref_s': trefs,
            'fs_hz': 40e6, 'fc_hz': 7.5e6, 'band_hz': [4e6, 7.5e6]}
    tmp = out_path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, rf=rf, c=c.astype(np.float32),
                        m=m.astype(np.float32),
                        meta_json=np.asarray(json.dumps(meta)))
    tmp.replace(out_path)
    return 'done'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path,
                   default=Path('/data/zhuangyang/NumerialBreastPhantoms/'
                                'l11_ultrawave_500_11angle'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--planes', type=int, default=120,
                   help='training planes to use (every 4th of 480)')
    p.add_argument('--shard', default='0/1', help='i/n worker shard')
    args = p.parse_args()
    if 'torch' in sys.modules:
        raise RuntimeError('torch must not be imported in this process')
    gen.configure_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.root/'index.json').read_text())
    trains = [r for r in index['samples'] if r['split'] == 'train']
    planes = trains[::4][:args.planes]
    i, n = (int(v) for v in args.shard.split('/'))
    work = [(rec, v) for rec in planes for v in VARIANTS][i::n]
    start = time.monotonic()
    for k, (rec, variant) in enumerate(work, 1):
        status = simulate_record(args.root, rec, args.out, variant)
        print(json.dumps({'worker': args.shard, 'done': k, 'total': len(work),
                          'id': rec['id'], 'variant': variant,
                          'status': status,
                          'elapsed': round(time.monotonic()-start, 1)}),
              flush=True)


if __name__ == '__main__':
    main()
