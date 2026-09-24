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
        """z [n, d] -> Gamma^{1/2} [n, d, d] via k-NN covariance, rank-truncated.

        The eigendecomposition runs in float64 on a jittered covariance:
        exact-duplicate neighbours make the float32 covariance singular and
        eigh can fail outright (verified on the pre-dedup bank).  rank <= 0
        keeps NO directions (pure eps*I), never all of them.
        """
        n = z.shape[0]
        sq = []
        eye = torch.eye(self.dim, dtype=torch.float64, device=z.device)
        for start in range(0, n, self.chunk):
            zc = z[start:start+self.chunk]                     # [m, d]
            d2 = (torch.cdist(zc, self.bank_z)**2)              # [m, N]
            nb = d2.topk(self.k, largest=False).indices         # [m, k]
            zn = self.bank_z[nb].double()                       # [m, k, d]
            zn = zn - zn.mean(dim=1, keepdim=True)
            cov = torch.einsum('mki,mkj->mij', zn, zn)/self.k    # [m, d, d]
            cov = cov + (self.eps*torch.diagonal(cov, dim1=1, dim2=2)
                         .mean(dim=1, keepdim=True).clamp_min(1e-12))[:, None]*eye
            evals, evecs = torch.linalg.eigh(cov)
            evals = evals.clamp_min(0.)
            keep = torch.zeros_like(evals)
            if self.rank > 0:
                keep[:, -self.rank:] = 1.
            lam = (evals.sqrt())*keep + self.eps
            sq.append(((evecs*lam.unsqueeze(1)) @ evecs.transpose(1, 2))
                      .to(z.dtype))
        return torch.cat(sq)

    def forward(self, u_gt, t, sigma_u, generator=None):
        """Return (u_t, v_star) for per-sample times t [B].

        Additive split of the planned covariance path:

            n_t = (1-t) sigma_u eta + t alpha sigma_u s(Gamma^{1/2} eps)

        with eta a FULL-RANK per-pixel white field, so the t=0 marginal is
        exactly N(0, sigma_u^2 I) and matches the inference-time white
        start (the previous patch-decoded start had per-pixel std 0.26x
        and neighbour correlation 0.58, a train/inference mismatch).  The
        t=1 endpoint keeps alpha sigma_u s(Gamma^{1/2} eps) by design: the
        target is distributional on the local SoS manifold neighbourhood.
        v_star is the exact time derivative at fixed (eta, eps).
        """
        b = u_gt.shape[0]
        patches = self._patches(u_gt)                          # [B, n, pp]
        z = (patches - self.center) @ self.basis               # [B, n, d]
        n = z.shape[1]
        sqrt_g = self._local_sqrt(z.reshape(-1, z.shape[-1])).reshape(b, n,
                                                                      self.dim,
                                                                      self.dim)
        eps = torch.randn(b, n, self.dim, device=u_gt.device,
                          dtype=u_gt.dtype, generator=generator)
        eta = torch.randn(u_gt.shape, device=u_gt.device, dtype=u_gt.dtype,
                          generator=generator)
        struct_lat = torch.einsum('bnij,bnj->bni', sqrt_g, eps)
        struct_px = self._scatter(torch.einsum('bnd,pd->bnp', struct_lat,
                                               self.basis))
        sigma = sigma_u.reshape(1, 1, 1, 1).to(device=u_gt.device,
                                              dtype=u_gt.dtype)
        t_e = t.view(b, 1, 1, 1)
        n_t = (1.-t_e)*sigma*eta + (t_e*self.alpha)*sigma*struct_px
        u_t = t_e*u_gt + n_t
        v_star = u_gt + self.alpha*sigma*struct_px - sigma*eta
        return u_t, v_star
