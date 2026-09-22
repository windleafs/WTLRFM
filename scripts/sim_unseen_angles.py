"""Re-simulate selected val cases at transmit angles never seen in training.

The training cache was built from plane-wave RF simulated on the 11-angle grid
linspace(-8, 8, 11).  This driver rebuilds the exact same medium (same phantom
slice + scatter seed) with the original UltraWave solver and re-runs only a
custom angle set whose locations are off that grid, so the encoder sees truly
unseen angle positions with correct physics.

Must run in a torch-free process with the NVIDIA HPC SDK on PATH (the
generator module enforces this for OpenACC runtime reasons).

Outputs raw npz files (rf / c / trefs / metadata), not torch shards; a second
torch-enabled step packs them into cache roots.
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

# Union of unseen angle sets; none of these lie on the 1.6-degree training grid.
UNION_ANGLES = [-7.5, -6.0, -4.5, -3.0, -1.5, 0.5, 2.5, 4.5, 6.5]
UNSEEN_5 = [-7.5, -4.5, -1.5, 2.5, 6.5]


def solver_case(maps=None):
    case = gen.geometry_case(maps)
    fit = gen.absorption_model(case)
    return case, fit, gen.solver_for(case, case['maps'], fit)


def run_angle(solver, case, angle):
    tref = gen.set_angle(solver, case, float(angle))
    total, timing = solver.run()
    return tref, total, timing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path,
                   default=Path('/data/zhuangyang/NumerialBreastPhantoms/'
                                'l11_ultrawave_500_11angle'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--ids', default=','.join(f'val_{i:03d}' for i in range(0, 60, 4)))
    p.add_argument('--sanity-only', action='store_true')
    args = p.parse_args()
    if 'torch' in sys.modules:
        raise RuntimeError('torch must not be imported in this process')
    gen.configure_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.root/'index.json').read_text())
    by_id = {r['id']: r for r in index['samples']}

    # --- Sanity 1: homogeneous reference at the original -8 degree matches.
    ref_stored = np.load(args.root/'reference_native.npz')['rf_native']
    case, fit, solver = solver_case()
    tref, total, _ = run_angle(solver, case, -8.0)
    err = np.max(np.abs(total-ref_stored[:, :, 0]))/np.max(np.abs(ref_stored))
    print(f'[sanity] reference -8.0deg rel-max-err={err:.3e} tref={tref*1e6:.4f}us',
          flush=True)
    if err > 1e-4:
        raise RuntimeError('Reference re-simulation does not match the stored one')

    # --- Sanity 2: val_000 medium rebuild + solve at -8 degree matches raw.
    rec = by_id['val_000']
    with h5py.File(rec['h5'], 'r') as f:
        plane = np.asarray(f['phan'][rec['z_index']])
    case = gen.geometry_case()
    maps, *_ = gen.medium_builder.build_medium(
        plane, case['x'], case['z'], seed=rec['scatter_seed'], preset='dual_scale')
    case['maps'] = maps
    gen.bench.validate_case(case)
    fit = gen.absorption_model(case)
    solver = gen.solver_for(case, maps, fit)
    tref, total, _ = run_angle(solver, case, -8.0)
    rf, _ = gen.sim.analytic_channels(total - ref_stored[:, :, 0], gen.DT,
                                      band=[4e6, 7.5e6])
    raw_stored = np.load(args.root/'raw'/'val_000.npz')['rf']
    err = np.max(np.abs(rf.T.astype(np.float32)-raw_stored[0])) \
        / max(float(np.abs(raw_stored[0]).max()), 1e-30)
    print(f'[sanity] val_000 -8.0deg rf rel-max-err={err:.3e}', flush=True)
    if err > 1e-3:
        raise RuntimeError('Sample re-simulation does not match the stored raw RF')
    if args.sanity_only:
        return

    # --- Reference fields at the unseen angles (shared across samples).
    refs_path = args.out/'reference_unseen.npz'
    if refs_path.exists():
        refs = np.load(refs_path)['rf_native']
        print('[refs] reusing cached unseen-angle reference', flush=True)
    else:
        case, fit, solver = solver_case()
        values = []
        start = time.monotonic()
        for angle in UNION_ANGLES:
            _, total, timing = run_angle(solver, case, angle)
            values.append(total)
            print(f'[refs] angle={angle:+.1f} solve={timing["solve_readback_s"]:.2f}s '
                  f'elapsed={time.monotonic()-start:.1f}s', flush=True)
        refs = np.stack(values, axis=-1)
        tmp = refs_path.with_suffix('.tmp.npz')
        np.savez(tmp, rf_native=refs, angles_deg=np.asarray(UNION_ANGLES))
        tmp.replace(refs_path)

    # --- Per-sample simulation at the unseen angles.
    manifest = []
    for sid in [v for v in args.ids.split(',') if v]:
        rec = by_id[sid]
        out_path = args.out/f'raw_{sid}.npz'
        if out_path.exists():
            print(f'[skip] {sid}', flush=True)
            manifest.append(json.loads(out_path.with_suffix('.meta.json').read_text()))
            continue
        start = time.monotonic()
        with h5py.File(rec['h5'], 'r') as f:
            plane = np.asarray(f['phan'][rec['z_index']])
        case = gen.geometry_case()
        maps, *_ = gen.medium_builder.build_medium(
            plane, case['x'], case['z'], seed=rec['scatter_seed'],
            preset='dual_scale')
        case['maps'] = maps
        gen.bench.validate_case(case)
        fit = gen.absorption_model(case)
        solver = gen.solver_for(case, maps, fit)
        rf_list, trefs = [], []
        for ai, angle in enumerate(UNION_ANGLES):
            tref, total, timing = run_angle(solver, case, angle)
            rf, _ = gen.sim.analytic_channels(total - refs[:, :, ai], gen.DT,
                                              band=[4e6, 7.5e6])
            if rf.shape != (2401, 192):
                raise RuntimeError(f'{sid}: unexpected analytic shape {rf.shape}')
            rf_list.append(rf.T.astype(np.float32))
            trefs.append(float(tref))
        rf = np.stack(rf_list)
        if not np.isfinite(rf).all():
            raise RuntimeError(f'{sid}: nonfinite RF')
        meta = {'id': sid, 'source_record': {k: rec[k] for k in
                                             ('id', 'split', 'case', 'h5', 'z_index',
                                              'scatter_seed', 'base_anatomy_id',
                                              'anatomy_repeated', 'backend')},
                'angles_deg': UNION_ANGLES, 'source_tref_s': trefs,
                'elapsed_s': time.monotonic()-start}
        tmp = out_path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, rf=rf)
        tmp.replace(out_path)
        out_path.with_suffix('.meta.json').write_text(json.dumps(meta)+'\n')
        manifest.append(meta)
        print(f'[done] {sid} elapsed={meta["elapsed_s"]:.1f}s '
              f'tref={trefs[0]*1e6:.4f}us', flush=True)

    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'[all] wrote {len(manifest)} raw files to {args.out}', flush=True)


if __name__ == '__main__':
    main()
