"""Torch-side client for the v6C JAX WFC worker + finite-difference check."""

import time
import uuid
from pathlib import Path

import numpy as np


class WFCPhysClient:
    def __init__(self, spool, timeout_s=60.):
        self.spool = Path(spool)
        self.timeout = float(timeout_s)

    def request(self, key, c):
        tag = uuid.uuid4().hex[:12]
        req = self.spool/f'req_{tag}.npz'
        resp = self.spool/f'resp_{tag}.npz'
        np.savez(req, key=key, c=np.asarray(c, np.float32))
        deadline = time.time()+self.timeout
        while time.time() < deadline:
            if resp.exists():
                time.sleep(.01)                      # let the write finish
                d = np.load(resp, allow_pickle=False)
                resp.unlink(missing_ok=True)
                if not bool(d['ok']):
                    raise RuntimeError(f'worker error: {str(d["error"])}')
                return float(d['loss']), d['grad']
            time.sleep(.02)
        req.unlink(missing_ok=True)
        raise TimeoutError(f'WFC worker did not answer {tag}')


def finite_difference_check(client, records=('orig_breast_0', 'l125_flat_0'),
                            n_dir=4, delta=1.0):
    rng = np.random.default_rng(20260924)
    import json
    manifest = {r['id']: r for r in json.loads(
        (Path('/data/zhuangyang/tmp/v7_records')/'manifest.json').read_text())}
    report = {}
    for key in records:
        d = np.load(f'/data/zhuangyang/tmp/v7_records/{key}.npz')
        c0 = d['truth'].astype(np.float64)
        loss0, grad = client.request(key, c0)
        rows = []
        for i in range(n_dir):
            direction = rng.standard_normal(c0.shape)
            direction /= np.linalg.norm(direction)/np.sqrt(direction.size)
            lp, _ = client.request(key, c0+delta*direction)
            lm, _ = client.request(key, c0-delta*direction)
            numeric = (lp-lm)/(2*delta)
            analytic = float((grad*direction).sum())
            rows.append(abs(numeric-analytic)/max(abs(numeric), 1e-6))
        report[key] = {'loss0': loss0,
                       'rel_err_median': float(np.median(rows)),
                       'rel_err_max': float(np.max(rows))}
        print(f'[fd] {key}: loss {loss0:.4f}, relative direction-derivative '
              f'error median {np.median(rows):.3e} max {np.max(rows):.3e}',
              flush=True)
    return report


if __name__ == '__main__':
    client = WFCPhysClient('/data/zhuangyang/tmp/v6c_spool')
    finite_difference_check(client)
