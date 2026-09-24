"""Persistent JAX (dbua env) WFC coherence worker for v6C fine-tuning.

Builds the WFC models for a fixed cohort of records once, then serves

    L_phys(c) = 1 - C(c) + low-energy guard

and its gradient with respect to the speed map on the RECORD (x, z) grid
via an exact bilinear-adjoint scatter of the WFC-grid gradient.

Protocol: the torch trainer drops ``req_<tag>.npz`` (fields: key, c [nx, nz]
float32 on the record grid) into the spool; the worker answers
``resp_<tag>.npz`` (loss float32 scalar, grad [nx, nz] float32, ok bool).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from wfc_integration.generalization import build_model, subset_fields  # noqa: E402

BACKEND = Path('/home/zhuangyang/fmmodel/dbua_test/wfc_dbua_pw')


def bilinear_adjoint(grad_speed, cx, cz, x, z):
    """Exact adjoint of RegularGridInterpolator((z, x), ·) at points (cz, cx).

    grad_speed [ncx, ncz]; returns grad on the record grid [nz, nx].
    """
    nz, nx = len(z), len(x)
    out = np.zeros((nz, nx), np.float64)
    g = np.asarray(grad_speed, np.float64)
    # forward projection clips out-of-range query points onto the boundary
    # (fill_value=None with pre-clipped coordinates); match that here so
    # padded WFC-grid points accumulate exactly onto the boundary nodes.
    cx = np.clip(cx, x[0], x[-1])
    cz = np.clip(cz, z[0], z[-1])
    j0 = np.clip(np.searchsorted(x, cx, side='right')-1, 0, nx-2)
    j1 = j0+1
    wx = (cx-x[j0])/(x[j1]-x[j0])
    i0 = np.clip(np.searchsorted(z, cz, side='right')-1, 0, nz-2)
    i1 = i0+1
    wz = (cz-z[i0])/(z[i1]-z[i0])
    # accumulate with the four bilinear weights (sample points are the full
    # (cx, cz) outer product; g is indexed [ix, iz])
    W00 = ((1-wx)[:, None]*(1-wz)[None, :])
    W01 = ((1-wx)[:, None]*wz[None, :])
    W10 = (wx[:, None]*(1-wz)[None, :])
    W11 = (wx[:, None]*wz[None, :])
    for W, ii, jj in ((W00, i0, j0), (W01, i1, j0), (W10, i0, j1), (W11, i1, j1)):
        np.add.at(out, (ii[None, :].repeat(len(cx), 0).ravel(),
                        jj[:, None].repeat(len(cz), 1).ravel()),
                  (g*W).ravel())
    return out


def build_cohort(manifest_path, records_dir, dz_mm):
    import jax
    import jax.numpy as jnp
    manifest = json.loads(Path(manifest_path).read_text())
    cohort = {}
    for rec in manifest:
        d = np.load(Path(records_dir)/f"{rec['id']}.npz")
        acquisition = dict(rf=d['rf'].astype(np.float32), xe=d['xe'],
                           angles_deg=d['angles'], tx_t_ref_s=d['trefs'],
                           fs_hz=float(d['fs']), fc_hz=float(d['fc']),
                           t0_s=0., c_steer=1500., bandwidth_fraction=.2333)
        model, project, roi, splits, _ = build_model(
            acquisition, d['x'], d['z'], BACKEND, dz_mm=dz_mm, depth_mm=40.)
        fields = subset_fields(model, splits['train'])
        mask = jnp.asarray(roi, jnp.float32)
        gt_speed = jnp.asarray(project(d['truth']))
        imgs0 = model.image_from_f0(gt_speed, fields)
        e0 = jnp.array(float(jnp.sum(jnp.sum(jnp.abs(imgs0), axis=0)*mask)
                             / jnp.sum(mask)))

        @jax.jit
        def loss_and_grad(c_speed, mask=mask, fields=fields, e0=e0):
            def loss(c):
                imgs = model.image_from_f0(c, fields)
                num = jnp.sum(jnp.abs(jnp.sum(imgs, axis=0))*mask)
                den = jnp.sum(jnp.sum(jnp.abs(imgs), axis=0)*mask)
                coh = num/jnp.maximum(den, 1e-20)
                energy = den/jnp.sum(mask)
                guard = jnp.maximum(.1-energy/jnp.maximum(e0, 1e-30), 0.)**2
                return -coh+guard
            return jax.value_and_grad(loss)(c_speed)

        loss_and_grad(gt_speed)                    # warm the jit
        cohort[rec['id']] = dict(project=project, x=np.asarray(d['x']),
                                 z=np.asarray(d['z']),
                                 cx=np.asarray(model.cx, np.float64),
                                 cz=np.asarray(model.cz, np.float64),
                                 loss_and_grad=loss_and_grad,
                                 group=rec.get('group', ''))
        print(f"[worker] ready {rec['id']}", flush=True)
    return cohort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--records', default='/data/zhuangyang/tmp/v7_records')
    p.add_argument('--spool', required=True, type=Path)
    p.add_argument('--dz-mm', type=float, default=.4)
    p.add_argument('--poll-s', type=float, default=.05)
    args = p.parse_args()
    args.spool.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort(args.manifest, args.records, args.dz_mm)
    (args.spool/'READY').write_text(json.dumps(
        {'cohort': list(cohort), 'dz_mm': args.dz_mm})+'\n')
    print(f'[worker] serving {len(cohort)} records', flush=True)
    handled = 0
    while True:
        requests = sorted(args.spool.glob('req_*.npz'))
        if not requests:
            time.sleep(args.poll_s)
            continue
        for req in requests:
            tag = req.stem[4:]
            resp = args.spool/f'resp_{tag}.npz'
            try:
                d = np.load(req, allow_pickle=False)
                key = str(d['key'])
                entry = cohort[key]
                c = np.asarray(d['c'], np.float64)            # [nx, nz]
                # project() transposes x-major input internally
                c_speed = np.asarray(entry['project'](c), np.float32)
                value, grad_speed = entry['loss_and_grad'](c_speed)
                grad = bilinear_adjoint(np.asarray(grad_speed), entry['cx'],
                                        entry['cz'], entry['x'],
                                        entry['z']).T          # [nx, nz]
                np.savez(resp, loss=np.float32(float(value)),
                         grad=grad.astype(np.float32), ok=np.bool_(True))
                handled += 1
                if handled % 20 == 0:
                    print(f'[worker] {handled} requests served', flush=True)
            except Exception as exc:                           # noqa: BLE001
                np.savez(resp, loss=np.float32(np.nan),
                         grad=np.zeros_like(np.asarray(d['c'], np.float32)),
                         ok=np.bool_(False), error=str(exc))
            finally:
                req.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
