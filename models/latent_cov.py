"""Geometry-aware latent FM noise shaping (v6B).

Implements the planned path covariance

    Sigma_t(z_i) = [(1-t) I + t alpha Gamma_i^{1/2}]^2

where Gamma_i is the truncated local covariance of the SoS-latent patch
z_i = E_c(patch_i(u_gt)) over its k nearest neighbours in a precomputed
patch bank (scripts/build_patch_bank.py).  Only the TRAINING path and the
velocity target change; ODE sampling stays exactly as in the isotropic model,
so checkpoints remain compatible with every evaluation script.

Concretely, with white patch noise eps and the fixed PCA encoder E_c:

    n_t(i)   = [(1-t) sigma_u I + t alpha sigma_u Gamma_i^{1/2}] eps_i
    u_t      = t u_gt + scatter(n_t)                  (overlap-normalised)
    v*(i)    = u_gt + scatter(sigma_u (alpha Gamma_i^{1/2} - I) eps_i)

v* is the exact time derivative of the path at fixed eps.
"""

import torch
import torch.nn as nn


class LatentCovarianceNoise(nn.Module):
    """Turns u_gt batches into (u_t, v_star) with k-NN local covariance noise."""

    def __init__(self, bank_path, k=32, rank=8, alpha=.5, eps=1e-3,
                 sigma_ref=.59, chunk_neighbors=4096):
        super().__init__()
        blob = torch.load(bank_path, map_location='cpu', weights_only=False)
        meta = blob['meta']
        self.patch, self.stride = int(meta['patch']), int(meta['stride'])
        self.dim = int(meta['dim'])
        self.k, self.rank = int(k), int(rank)
        self.alpha, self.eps = float(alpha), float(eps)
        self.sigma_ref = float(sigma_ref)
        self.chunk = int(chunk_neighbors)
        self.register_buffer('basis', blob['basis'].float())        # [P*P, d]
        self.register_buffer('center', blob['center'].float())      # [P*P]
        self.register_buffer('bank_z', blob['bank_z'].float())      # [N, d]
        # overlap-count weights for stride-4 scatter on the 128x160 grid
        self.register_buffer('scatter_norm', self._overlap_counts(),
                             persistent=False)

    def _overlap_counts(self):
        H, W, p, s = 128, 160, self.patch, self.stride
        nrm = torch.zeros(1, 1, H, W)
        for r in range(0, H-p+1, s):
            for c in range(0, W-p+1, s):
                nrm[0, 0, r:r+p, c:c+p] += 1.
        return nrm.clamp_min(1.)

    def _patches(self, u):
        """[B,1,H,W] -> [B, R, P*P] unfolded patches."""
        p, s = self.patch, self.stride
        out = u.unfold(2, p, s).unfold(3, p, s)          # [B,1,R,C,p,p]
        b, _, R, C, _, _ = out.shape
        return out.reshape(b, R*C, p*p)

    def _scatter(self, patch_field):
        """[B, RC, p*p] -> [B,1,H,W] overlap-normalised (F.fold overlap sum)."""
        b, n, pp = patch_field.shape
        img = torch.nn.functional.fold(
            patch_field.transpose(1, 2), output_size=(128, 160),
            kernel_size=self.patch, stride=self.stride)
        return img/self.scatter_norm

    def _local_sqrt(self, z):
        """z [n, d] -> Gamma^{1/2} [n, d, d] via k-NN covariance, rank-truncated."""
        n = z.shape[0]
        sq = []
        for start in range(0, n, self.chunk):
            zc = z[start:start+self.chunk]                     # [m, d]
            d2 = (torch.cdist(zc, self.bank_z)**2)              # [m, N]
            nb = d2.topk(self.k, largest=False).indices         # [m, k]
            zn = self.bank_z[nb]                                # [m, k, d]
            zn = zn - zn.mean(dim=1, keepdim=True)
            cov = torch.einsum('mki,mkj->mij', zn, zn)/self.k    # [m, d, d]
            evals, evecs = torch.linalg.eigh(cov)
            evals = evals.clamp_min(0.)
            keep = torch.zeros_like(evals)
            keep[:, -self.rank:] = 1.
            lam = (evals.sqrt())*keep + self.eps
            sq.append((evecs*lam.unsqueeze(1)) @ evecs.transpose(1, 2))
        return torch.cat(sq)

    def forward(self, u_gt, t, sigma_u, generator=None):
        """Return (u_t, v_star) for per-sample times t [B]."""
        b = u_gt.shape[0]
        patches = self._patches(u_gt)                          # [B, n, pp]
        z = (patches - self.center) @ self.basis               # [B, n, d]
        n = z.shape[1]
        sqrt_g = self._local_sqrt(z.reshape(-1, z.shape[-1])).reshape(b, n,
                                                                      self.dim,
                                                                      self.dim)
        eps = torch.randn(b, n, self.dim, device=u_gt.device,
                          dtype=u_gt.dtype, generator=generator)
        eye = torch.eye(self.dim, device=u_gt.device, dtype=u_gt.dtype)
        a_t = ((1.-t).view(b, 1, 1, 1)*eye
               + (t*self.alpha).view(b, 1, 1, 1)*sqrt_g)        # [(1-t)I + taG^1/2]
        a_v = (self.alpha*sqrt_g - eye)                        # d/dt bracket
        noise_lat = torch.einsum('bnij,bnj->bni', a_t, eps)    # [B, n, d]
        vcorr_lat = torch.einsum('bnij,bnj->bni', a_v, eps)
        basis = self.basis                                     # [pp, d]
        noise_px = torch.einsum('bnd,pd->bnp', noise_lat, basis) \
            + (patches*0.)                                     # [B, n, pp] + center 0
        vcorr_px = torch.einsum('bnd,pd->bnp', vcorr_lat, basis)
        sigma = sigma_u.reshape(1, 1, 1)   # scalar broadcast
        n_t = self._scatter(noise_px*sigma)
        v_corr = self._scatter(vcorr_px*sigma)
        t_e = t.view(b, 1, 1, 1)
        u_t = t_e*u_gt + n_t
        v_star = u_gt + v_corr
        return u_t, v_star
