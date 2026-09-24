"""Build the SoS-latent patch bank for geometry-aware latent FM (v6B).

Extracts overlapping patches of the normalised log-SoS targets from the v4/v5
training mixture, fits a PCA latent basis (E_c), and stores a subsampled
patch library in latent coordinates for k-NN local-covariance estimation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.geometry_dataset import StructuredSoSDataset  # noqa: E402


def collect_patches(caches, split, patch, stride, max_bank, seed):
    patches = []
    rng = np.random.default_rng(seed)
    for cache in caches:
        ds = StructuredSoSDataset(cache, split)
        for rec in ds.records:
            d = ds._load(rec)
            u = np.asarray(d['u_gt'], np.float32)
            H, W = u.shape
            rows = range(0, H-patch+1, stride)
            cols = range(0, W-patch+1, stride)
            take = rng.random((len(rows), len(cols))) < .12
            for i, r in enumerate(rows):
                for j, c in enumerate(cols):
                    if take[i, j]:
                        patches.append(u[r:r+patch, c:c+patch].reshape(-1))
    patches = np.stack(patches)
    if len(patches) > max_bank:
        idx = rng.choice(len(patches), max_bank, replace=False)
        patches = patches[idx]
    return patches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--caches', default=(
        '/data/zhuangyang/geometry_flow_v2_cache,'
        '/data/zhuangyang/geometry_flow_v2_decv4_cache'))
    p.add_argument('--patch', type=int, default=8)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--dim', type=int, default=16)
    p.add_argument('--max-bank', type=int, default=60000)
    p.add_argument('--seed', type=int, default=20260923)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('fresh output path required')
    caches = [Path(v) for v in args.caches.split(',')]
    patches = collect_patches(caches, 'train', args.patch, args.stride,
                              args.max_bank, args.seed)
    print(f'bank patches: {patches.shape}', flush=True)
    center = patches.mean(axis=0, keepdims=True)
    x = torch.from_numpy(patches - center)
    # PCA via SVD on the (subsampled) covariance
    basis, _, _ = torch.linalg.svd(x.T, full_matrices=False)
    basis = basis[:, :args.dim].contiguous()          # [patch*patch, dim]
    z = (x @ basis).numpy().astype(np.float32)
    ev = float(torch.var(x @ basis, dim=0).sum())
    tv = float(torch.var(x, dim=0).sum())
    args.out.mkdir(parents=True)
    torch.save({'basis': basis, 'center': torch.from_numpy(center[0]),
                'bank_z': torch.from_numpy(z),
                'meta': dict(patch=args.patch, stride=args.stride,
                             dim=args.dim, n_patches=len(patches),
                             sources=[str(c) for c in caches],
                             evr=ev/max(tv, 1e-12))},
               args.out/'patch_bank.pt')
    print(json.dumps({'patch': args.patch, 'stride': args.stride,
                      'dim': args.dim, 'n': len(patches),
                      'explained_variance_ratio': round(ev/max(tv, 1e-12), 4)}))


if __name__ == '__main__':
    main()
