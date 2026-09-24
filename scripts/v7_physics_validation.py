"""Stage 2 (JAX/dbua env) of the v7 physics-objective validation.

For each prepared record this measures the WFC angle-coherence C(c) of
candidate speed maps (GT, global offsets, smoothed/perturbed GT, model
prediction, constants), then runs 8 full-map Adam steps that update ONLY the
speed map to maximise coherence (with the low-energy guard), logging
coherence, SoS MAE and mean bias per step.

Gate criteria (see v7 plan): the loss must separate meaningful speed errors,
and small steps must overall improve SoS error.  Run with the dbua python.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wfc_integration.generalization import build_model, subset_fields  # noqa: E402

BACKEND = Path('/home/zhuangyang/fmmodel/dbua_test/wfc_dbua_pw')


def coherence_metrics(model, fields, roi, c_speed):
    import jax
    import jax.numpy as jnp
    mask = jnp.asarray(roi, jnp.float32)
    imgs = model.image_from_f0(jnp.asarray(c_speed), fields)
    num = jnp.sum(jnp.abs(jnp.sum(imgs, axis=0))*mask)
    den = jnp.sum(jnp.sum(jnp.abs(imgs), axis=0)*mask)
    return float(num/jnp.maximum(den, 1e-20)), float(den/jnp.sum(mask))


def main():
    import jax
    import jax.numpy as jnp
    import optax

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--dz-mm', type=float, default=.4)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--lr', type=float, default=.5)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.records/'manifest.json').read_text())
    if args.limit:
        manifest = manifest[:args.limit]

    report = {}
    for rec in manifest:
        d = np.load(args.records/f"{rec['id']}.npz")
        x, z = d['x'], d['z']
        acquisition = dict(rf=d['rf'].astype(np.float32), xe=d['xe'],
                           angles_deg=d['angles'], tx_t_ref_s=d['trefs'],
                           fs_hz=float(d['fs']), fc_hz=float(d['fc']),
                           t0_s=0., c_steer=1500., bandwidth_fraction=.2333)
        model, project, roi, splits, _ = build_model(
            acquisition, x, z, BACKEND, dz_mm=args.dz_mm, depth_mm=40.)
        fields = subset_fields(model, splits['train'])
        gt_speed = project(d['truth'])
        gt_speed = jnp.asarray(gt_speed)
        pred_speed = jnp.asarray(project(d['pred']))

        # --- candidate coherence table
        rng = np.random.default_rng(7)
        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(d['truth'], sigma=15, mode='nearest')
        perturbed = gaussian_filter(rng.standard_normal(d['truth'].shape),
                                    sigma=10, mode='nearest')
        perturbed *= 20./max(perturbed.std(), 1e-12)   # fixed 20 m/s RMS
        perturb = d['truth'] + perturbed
        cands = {'gt': d['truth'], 'smooth_gt': smooth, 'perturb_gt': perturb,
                 'pred_v5': d['pred'],
                 'const_1500': np.full_like(d['truth'], 1500.),
                 'const_1540': np.full_like(d['truth'], 1540.)}
        for off in (10., 20., 40.):
            cands[f'gt{off:+.0f}'] = d['truth'] + off
            cands[f'gt{-off:+.0f}'] = d['truth'] - off
        roi_np = np.asarray(roi, bool)
        table = {}
        for name, c in cands.items():
            cs = project(c)
            coh, energy = coherence_metrics(model, fields, roi, cs)
            dd = np.asarray(cs)-np.asarray(gt_speed)
            # metrics inside the ROI only: the WFC grid pads ~54% of its
            # area beyond the record support with clipped boundary copies,
            # which would dilute mae/bias roughly twofold
            table[name] = {'coherence': coh, 'energy': energy,
                           'mae_vs_gt': float(np.abs(dd[roi_np]).mean()),
                           'bias': float(dd[roi_np].mean())}

        # --- full-map gradient probe
        probes = {}
        for start_name, start in (('gt+40', d['truth']+40.),
                                  ('gt-40', d['truth']-40.),
                                  ('perturb_gt', perturb),
                                  ('pred_v5', d['pred']),
                                  ('const_1500', np.full_like(d['truth'], 1500.))):
            c0 = np.asarray(project(start), np.float32)
            e0 = coherence_metrics(model, fields, roi, jnp.asarray(c0))[1]
            cv = jnp.asarray(c0)

            @jax.jit
            def loss_and_grad(c):
                def loss(cc):
                    imgs = model.image_from_f0(cc, fields)
                    mask = jnp.asarray(roi, jnp.float32)
                    num = jnp.sum(jnp.abs(jnp.sum(imgs, axis=0))*mask)
                    den = jnp.sum(jnp.sum(jnp.abs(imgs), axis=0)*mask)
                    coh = num/jnp.maximum(den, 1e-20)
                    energy = den/jnp.sum(mask)
                    guard = jnp.maximum(.1-energy/max(e0, 1e-30), 0.)**2
                    return -coh+guard
                return jax.value_and_grad(loss)(c)

            opt = optax.chain(optax.clip_by_global_norm(50.), optax.adam(args.lr))
            state = opt.init(cv)
            hist = [{'step': 0,
                     'coherence': coherence_metrics(model, fields, roi, cv)[0],
                     'mae': float(np.abs((np.asarray(cv)-np.asarray(gt_speed))[roi_np]).mean()),
                     'bias': float((np.asarray(cv)-np.asarray(gt_speed))[roi_np].mean())}]
            for step in range(1, args.steps+1):
                v, g = loss_and_grad(cv)
                if not np.isfinite(float(v)) or not np.isfinite(np.asarray(g)).all():
                    hist.append({'step': step, 'error': 'nonfinite'})
                    break
                upd, state = opt.update(g, state, cv)
                cv = optax.apply_updates(cv, upd)
                hist.append({'step': step,
                             'coherence': coherence_metrics(model, fields, roi, cv)[0],
                             'mae': float(np.abs((np.asarray(cv)-np.asarray(gt_speed))[roi_np]).mean()),
                             'bias': float((np.asarray(cv)-np.asarray(gt_speed))[roi_np].mean())})
            probes[start_name] = hist
        report[rec['id']] = {'group': rec['group'], 'table': table,
                             'probes': probes}
        best_gt = table['gt']['coherence']
        print(f"{rec['id']} ({rec['group']}): C(gt)={best_gt:.4f} "
              f"C(gt+40)={table['gt+40']['coherence']:.4f} "
              f"C(pred)={table['pred_v5']['coherence']:.4f} "
              f"C(1540)={table['const_1540']['coherence']:.4f}", flush=True)
    (args.out/f'physics_validation_dz{args.dz_mm}.json').write_text(
        json.dumps(report, indent=2)+'\n')
    print('saved', args.out/f'physics_validation_dz{args.dz_mm}.json')


if __name__ == '__main__':
    main()
