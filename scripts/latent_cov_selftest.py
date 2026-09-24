"""Self-test for the v6B latent-covariance path (post-review fixes).

Reproduces the reviewer's probes:
  1. t=0 start statistics: per-pixel std ratio vs white noise and horizontal
     neighbour correlation (train start must equal the inference start).
  2. var(u_0)/sigma_u^2 internal ratio.
  3. Eigendecomposition robustness on worst-case duplicate-only neighbour
     sets, float32 path.
  4. Endpoint residual noise (t=1 structured term) in m/s, median/p95.
  5. Effective-rank distribution of the local covariances on real patches.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from data.geometry_dataset import StructuredSoSDataset  # noqa: E402
from models.latent_cov import LatentCovarianceNoise  # noqa: E402

BANK = '/data/zhuangyang/geometry_flow_v2_patchbank/patch_bank.pt'
DEV = 'cuda:2' if torch.cuda.is_available() else 'cpu'


def main():
    torch.manual_seed(20260924)
    lc = LatentCovarianceNoise(BANK).to(DEV)
    ds = StructuredSoSDataset('/data/zhuangyang/geometry_flow_v2_cache', 'train')
    ds.records = ds.records[:8]
    maps = torch.stack([torch.from_numpy(np.asarray(ds._load(r)['u_gt']))
                        for r in ds.records]).unsqueeze(1).float().to(DEV)
    sigma = torch.tensor(0.59)

    # --- 1+2: t=0 start statistics
    t0 = torch.zeros(len(maps), device=DEV)
    u0, _ = lc(maps, t0, sigma)
    white = torch.randn_like(u0)*sigma
    std_ratio = float(u0.std()/white.std())
    corr = float(torch.corrcoef(torch.stack([u0[0, 0, :, :-1].flatten(),
                                             u0[0, 0, :, 1:].flatten()]))[0, 1])
    wcorr = float(torch.corrcoef(torch.stack([white[0, 0, :, :-1].flatten(),
                                              white[0, 0, :, 1:].flatten()]))[0, 1])
    var_ratio = float(u0.var()/sigma**2)
    print(f'[1] t=0 per-pixel std ratio vs white: {std_ratio:.4f} (target 1.0)')
    print(f'[1] horizontal neighbour corr: {corr:+.4f} '
          f'(white reference {wcorr:+.4f})')
    print(f'[2] var(u0)/sigma_u^2: {var_ratio:.4f} (target 1.0)')

    # --- 3: duplicate-only neighbour sets (worst case)
    dup = lc.bank_z[:1].repeat(64, 1)
    try:
        sq = lc._local_sqrt(dup)
        eigmin = float(torch.linalg.eigvalsh(sq).min())
        print(f'[3] duplicate-only eigh OK; min eigenvalue of sqrt(Gamma) = '
              f'{eigmin:.2e}')
    except Exception as e:
        print(f'[3] FAILED: {e}')

    # --- 4+5: endpoint noise and effective ranks on real patches
    t1 = torch.ones(len(maps), device=DEV)
    _, v1 = lc(maps, t1, sigma)
    end_noise_u = (v1 - maps).abs()                       # |alpha s s(G eps) - s eta|
    # isolate the structured part: rerun with eta fixed zero
    # structured-only term, recomputed directly
    with torch.no_grad():
        patches = lc._patches(maps)
        z = (patches - lc.center) @ lc.basis
        sqrtg = lc._local_sqrt(z.reshape(-1, z.shape[-1])).reshape(
            len(maps), -1, lc.dim, lc.dim)
        eff_rank = (sqrtg.diagonal(dim1=2, dim2=3).sum(-1)**2
                    / (sqrtg**2).sum(dim=(2, 3)).clamp_min(1e-12))
        eps = torch.randn(z.shape[0], z.shape[1], lc.dim, device=DEV)
        struct = lc._scatter(torch.einsum('bnij,bnj->bni', sqrtg, eps)
                             @ lc.basis.T)
        end_mps = (0.5*float(sigma)*struct).abs()*75.
    print(f'[4] endpoint structured noise: median {end_mps.median():.3f} m/s, '
          f'p95 {end_mps.quantile(.95):.2f} m/s')
    print(f'[5] Gamma effective rank: median {eff_rank.median():.1f}, '
          f'p10 {eff_rank.quantile(.1):.1f}, p90 {eff_rank.quantile(.9):.1f} '
          f'(truncated at r=8)')


if __name__ == '__main__':
    main()
