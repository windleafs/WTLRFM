"""Generate probe-swap training data: same simulator, L12-5 probe acquisition.

Swaps ONLY the probe in the dataset's simulation stack — 256 elements,
0.1953 mm pitch (49.8 mm aperture), fc = 5.403 MHz carrier, fs = 25 MHz —
matching PhantomL12-5-50mm.mat, while keeping the wave physics, angle grid
and recipe identical.  Two media families:

  breast  - real breast planes with their natural SoS (probe generalisation)
  flat    - uniform-speckle reflectivity with random homogeneous c
            (absolute calibration on the new probe)

Torch-free; NVIDIA HPC SDK on PATH; shardable.  The homogeneous reference
for the L12-5 configuration is simulated once and cached.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

NEURAL_ASP = '/home/zhuangyang/fmmodel/neural_asp'
ULTRAWAVE_BENCH = '/home/zhuangyang/fmmodel/UltraWave/benchmarks'
PHANTOM_DIR = '/data/zhuangyang/NumerialBreastPhantoms'
for p in (NEURAL_ASP, ULTRAWAVE_BENCH, PHANTOM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import scripts.generate_l11_ultrawave_raw as gen  # noqa: E402

# L12-5 acquisition parameters (measured from PhantomL12-5-50mm.mat).
# The solver requires receiver centres on the 50 um simulation grid, so the
# 0.1953 mm pitch is snapped to 0.2 mm x 249 elements = 49.80 mm aperture,
# which reproduces the real probe's aperture exactly (+2.3% pitch error).
L125_NE = 249
L125_PITCH = 200e-6
L125_FC = 5.403e6
L125_FS = 25e6
L125_BAND = [0.533*L125_FC, L125_FC]     # same fractional band as the L11 recipe


def install_probe(sim):
    """Mutate the simulator module globals to the L12-5 probe."""
    sim.NE = L125_NE
    sim.PITCH = L125_PITCH
    sim.XE = (np.arange(sim.NE) - (sim.NE-1)/2)*sim.PITCH
    sim.FS = L125_FS


def l125_case():
    install_probe(gen.sim)
    case = gen.geometry_case()
    case['f0'] = L125_FC
    gen.bench.validate_case(case)
    return case


def flat_maps(shape, x, z, c, mult, seed, n_disk=0):
    zz, xx = np.meshgrid(z, x, indexing='ij')
    rng = np.random.default_rng(seed)
    speckle = rng.normal(0., 8.*mult, shape)*(zz*1e3 >= 0.)
    c_map = np.full(shape, float(c), np.float32)
    for _ in range(int(n_disk)):
        cx = rng.uniform(-15., 15.); cz = rng.uniform(8., 38.)
        r = rng.uniform(2., 5.)
        dc = rng.uniform(20., 80.)*rng.choice((-1., 1.))
        inside = (xx-cx)**2 + (zz-cz)**2 <= (r*1e-3)**2
        c_map[inside] = float(c) + float(dc)
    return {'sound_speed': c_map,
            'density': (1000.+speckle).astype(np.float32),
            'alpha_coeff': np.full(shape, .002, np.float32),
            'BonA': np.zeros(shape, np.float32)}


def simulate_maps(case, maps, fit, solver, refs, angles):
    case = dict(case)
    case['maps'] = maps
    gen.bench.validate_case(case)
    fit = gen.absorption_model(case)
    solver = gen.solver_for(case, maps, fit)
    rf_list, trefs = [], []
    for ai, angle in enumerate(angles):
        tref = gen.set_angle(solver, case, float(angle))
        total, _ = solver.run()
        rf, _ = gen.sim.analytic_channels(total - refs[:, :, ai], gen.DT,
                                          band=L125_BAND)
        rf_list.append(rf.astype(np.float32).T)      # [n_elem, n_samples]
        trefs.append(float(tref))
    return np.stack(rf_list), trefs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path,
                   default=Path('/data/zhuangyang/NumerialBreastPhantoms/'
                                'l11_ultrawave_500_11angle'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--n-breast', type=int, default=100)
    p.add_argument('--n-flat', type=int, default=20)
    p.add_argument('--n-incl', type=int, default=20)
    p.add_argument('--val-breast', type=int, default=20)
    p.add_argument('--val-flat', type=int, default=4)
    p.add_argument('--val-incl', type=int, default=4)
    p.add_argument('--shard', default='0/1')
    args = p.parse_args()
    if 'torch' in sys.modules:
        raise RuntimeError('torch must not be imported in this process')
    gen.configure_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(-8., 8., 11)

    # homogeneous L12-5 reference (once, shared by all records)
    ref_path = args.out/'reference_l125.npz'
    if ref_path.exists():
        refs = np.load(ref_path)['rf_native']
    else:
        case0 = l125_case()
        fit0 = gen.absorption_model(case0)
        solver0 = gen.solver_for(case0, case0['maps'], fit0)
        values = []
        for angle in angles:
            gen.set_angle(solver0, case0, float(angle))
            total, _ = solver0.run()
            values.append(total)
        refs = np.stack(values, axis=-1)
        np.savez(ref_path, rf_native=refs)
        print('[ref] L12-5 homogeneous reference simulated', flush=True)

    index = json.loads((args.root/'index.json').read_text())
    trains = [r for r in index['samples'] if r['split'] == 'train']
    pool = trains[::4]
    breast_planes = pool[:args.n_breast]
    val_planes = pool[args.n_breast:args.n_breast+args.val_breast]
    jobs = []
    for rec in breast_planes:
        jobs.append((f"{rec['id']}_l125breast", 'breast', rec, 'train'))
    for rec in val_planes:
        jobs.append((f"{rec['id']}_l125breastVAL", 'breastVAL', rec, 'val'))
    rng_master = np.random.default_rng(20260924)
    def flat_job(k, family, split, seed_offset):
        r = np.random.default_rng(20260924 + seed_offset + k)
        return (f'flat{k:03d}_l125{family}', family,
                dict(c=float(r.uniform(1450., 1580.)),
                     mult=float(r.uniform(.3, 2.2)),
                     seed=int(r.integers(1 << 30)),
                     n_disk=(0 if family in ('flat', 'flatVAL') else
                             int(r.integers(2, 5)))), split)
    for k in range(args.n_flat):
        jobs.append(flat_job(k, 'flat', 'train', 0))
    for k in range(args.n_incl):
        jobs.append(flat_job(k, 'incl', 'train', 555))
    for k in range(args.val_flat):
        jobs.append(flat_job(k, 'flatVAL', 'val', 909))
    for k in range(args.val_incl):
        jobs.append(flat_job(k, 'inclVAL', 'val', 1111))
    i, n = (int(v) for v in args.shard.split('/'))
    work = jobs[i::n]
    start = time.monotonic()
    for k, (job_id, family, spec, split) in enumerate(work, 1):
        out_path = args.out/f'{job_id}.npz'
        if out_path.exists():
            continue
        case = l125_case()
        if family in ('breast', 'breastVAL'):
            with h5py.File(spec['h5'], 'r') as f:
                plane = np.asarray(f['phan'][spec['z_index']])
            maps, *_ = gen.medium_builder.build_medium(
                plane, case['x'], case['z'], seed=spec['scatter_seed']+707,
                preset='dual_scale')
        else:
            maps = flat_maps((len(case['z']), len(case['x'])),
                             case['x'], case['z'], spec['c'], spec['mult'],
                             spec['seed'], n_disk=spec.get('n_disk', 0))
        rf, trefs = simulate_maps(case, maps, None, None, refs, angles)
        if not np.isfinite(rf).all() or rf.std() <= 0:
            raise RuntimeError(f'{job_id}: invalid RF')
        c, m, _ = gen.truth_maps(maps, case['x'], case['face'])
        meta = {'id': job_id, 'family': family.rstrip('VAL'), 'split': split,
                'angles_deg': angles.tolist(), 'source_tref_s': trefs,
                'fs_hz': L125_FS, 'fc_hz': L125_FC,
                'band_hz': list(L125_BAND), 'n_elements': L125_NE,
                'pitch_m': L125_PITCH,
                'c_gt_mean': float(c.mean())}
        tmp = out_path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, rf=rf, c=c.astype(np.float32),
                            m=m.astype(np.float32),
                            meta_json=np.asarray(json.dumps(meta)))
        tmp.replace(out_path)
        print(json.dumps({'id': job_id, 'done': k, 'total': len(work),
                          'rf_shape': list(rf.shape),
                          'elapsed': round(time.monotonic()-start, 1)}),
              flush=True)
    print('[sim] shard done', flush=True)


if __name__ == '__main__':
    main()
