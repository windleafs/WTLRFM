"""Multiplicative flow matching for 2-D sound-speed maps.

Adapted from WTLRFM's ``ComplexMagPhaseResidualFlowNetwork``
(``/data/zhuangyang/WTLRFM/models/flow_matching_network.py``), the
"multiplicative flow" that factorises a complex field as
``Z = M_S * exp(rho) * exp(i dphi)`` with a frozen structure provider ``M_S``.

Why the frozen provider is gone
-------------------------------
In WTLRFM the frozen ``M_S`` was **necessary**: the flow modelled a *residual*
log-amplitude on top of an externally supplied magnitude estimate, because the
network only predicted velocity fields for `rho` and `dphi` and had no way to
produce the magnitude scale itself.  For a real, strictly positive target like
a sound-speed map the exponential map is a complete parameterisation by
itself:

    c(x, z) = C_REF * exp(RHO_SCALE * u(x, z)),   u ~ flow(cond)

so the frozen baseline network is pure overhead here (an extra training stage,
an extra checkpoint, and a second source of bias).  The multiplicative
character is kept where it matters: the flow lives in the **logarithmic
(multiplicative) domain**, and the output is composed multiplicatively with
the reference speed.

What is kept from WTLRFM
------------------------
  * conditional flow matching with an Euclidean linear path
    ``u_t = (1-t) u_0 + t u_1``, ``u_0 ~ N(0, sigma_u^2)``;
  * ``sigma_u`` auto-estimated from the data by a warmup EMA and stored as a
    registered buffer (so it stays in sync with the model EMA copy -- the
    260702 WTLRFM fix);
  * velocity-matching loss only, ``L = MSE(v_theta, u_1 - u_0)``;
  * Euler ODE sampling with per-step velocity clamp and final state clamp;
  * the same WTLR-UNet backbone (multi-level Haar DWT + gated fusion).

Sampling several trajectories gives an ensemble: the mean is the point
estimate and the std is an uncertainty map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import data.geometry as G
from wtlrfm import WTLRUNet


class SoSMultiplicativeFlowNetwork(nn.Module):
    """Conditional flow matching on the normalised log-SoS map u.

    Args:
        unet: WTLR-UNet config; ``in_channel`` must be cond_channels + 1.
        u_clamp: clamp of the final log-SoS state (sampling only).
        velocity_clamp: per-step velocity clamp (sampling only).
        u_source_scale: fixed prior std; <=0 -> auto-estimate from the data.
        reflow_t_schedule: 'uniform' | 'stratified' | 'lognormal'.
    """

    def __init__(self, unet, u_clamp=4.0, u_source_scale=-1.0,
                 velocity_clamp=8.0, reflow_t_schedule="stratified",
                 u_warmup_batches=200, u_ema_decay=0.95):
        super().__init__()
        self.cfg = dict(unet)
        self.net = WTLRUNet(**self.cfg)

        self.u_clamp = float(u_clamp)
        self.velocity_clamp = float(velocity_clamp)
        self.reflow_t_schedule = str(reflow_t_schedule)
        self.u_source_scale = float(u_source_scale)

        # sigma_u: prior std of the log-SoS state, auto-estimated then frozen.
        # Registered buffer -> synced (copied) into the model EMA copy.
        self.register_buffer("_u_scale", torch.tensor(1.0))
        if self.u_source_scale > 0:
            self._u_scale.fill_(self.u_source_scale)
            self._u_initialised = True
        else:
            self._u_initialised = False
        self._u_ema = None
        self._u_ema_decay = float(u_ema_decay)
        self._u_warmup_batches = int(u_warmup_batches)
        self._u_steps = 0
        self.last_loss_dict = {}

    @property
    def sigma_u(self):
        return float(self._u_scale.item())

    def _sample_time(self, b, device):
        if self.reflow_t_schedule == "uniform":
            return torch.rand(b, device=device)
        if self.reflow_t_schedule == "lognormal":
            return torch.sigmoid(torch.randn(b, device=device) * 0.5)
        # stratified: one sample per equal-probability bin (lower variance)
        return (torch.arange(b, device=device)
                + torch.rand(b, device=device)) / b

    def _update_u_scale(self, u_1):
        if self._u_initialised:
            return
        with torch.no_grad():
            batch_std = u_1.reshape(u_1.shape[0], -1).std(dim=1).mean().item()
        self._u_steps += 1
        if self._u_ema is None:
            self._u_ema = float(batch_std)
        else:
            self._u_ema = (self._u_ema_decay * self._u_ema
                           + (1 - self._u_ema_decay) * float(batch_std))
        self._u_scale.fill_(max(self._u_ema, 1e-3))
        if self._u_steps >= self._u_warmup_batches:
            self._u_initialised = True

    # ------------------------------------------------------------------
    def forward(self, cond, u_gt, **kwargs):
        """Velocity-matching loss.

        Args:
            cond: [B, cond_ch, H, W] phase-preserving DAS condition.
            u_gt: [B, 1, H, W] normalised log-SoS target.
        """
        b = cond.shape[0]
        device = cond.device
        self._update_u_scale(u_gt)

        u_0 = torch.randn_like(u_gt) * self._u_scale
        t = self._sample_time(b, device)
        t_e = t.view(b, 1, 1, 1)

        u_t = (1.0 - t_e) * u_0 + t_e * u_gt
        v_pred = self.net(torch.cat([cond, u_t], dim=1), t)
        valid_mask = kwargs.get("valid_mask")
        if valid_mask is None:
            loss = F.mse_loss(v_pred, u_gt - u_0)
        else:
            if valid_mask.shape != u_gt.shape or not valid_mask.bool().any():
                raise ValueError("valid_mask must match target and contain valid pixels")
            error = (v_pred.float() - (u_gt - u_0).float()).square()
            loss = error.masked_select(valid_mask.bool()).mean()
            background_weight = float(kwargs.get("background_weight", 0.0))
            if background_weight < 0:
                raise ValueError("background_weight must be nonnegative")
            background = ~valid_mask.bool()
            if background_weight and background.any():
                # The entire ODE state must have a learned trajectory at inference.
                loss = loss + background_weight * error.masked_select(background).mean()

        self.last_loss_dict = {
            "loss": float(loss.detach()),
            "u_std": float(u_gt.detach().std()),
            "sigma_u": self.sigma_u,
        }
        return loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, cond, n_steps=20, n_samples=1, noise_scale=1.0,
               return_u=False):
        """Euler ODE sampling in log-SoS space.

        Returns:
            c: [n_samples, B, 1, H, W] sound speed [m/s] (or u if return_u).
        """
        b = cond.shape[0]
        device = cond.device
        dt = 1.0 / max(int(n_steps), 1)
        outs = []
        for _ in range(max(int(n_samples), 1)):
            u = torch.randn(cond.shape[0], 1, cond.shape[-2], cond.shape[-1],
                            device=device, dtype=cond.dtype)
            u = u * (self._u_scale * noise_scale)
            for i in range(max(int(n_steps), 1)):
                t = torch.full((b,), i * dt, device=device)
                v = self.net(torch.cat([cond, u], dim=1), t)
                if self.velocity_clamp > 0:
                    v = v.clamp(-self.velocity_clamp, self.velocity_clamp)
                u = u + dt * v
            if self.u_clamp > 0:
                u = u.clamp(-self.u_clamp, self.u_clamp)
            outs.append(u)
        u = torch.stack(outs, dim=0)                       # [S, B, 1, H, W]
        if return_u:
            return u
        return G.C_REF * torch.exp(G.RHO_SCALE * u)

    # ------------------------------------------------------------------
    def checkpoint(self, extra=None):
        blob = {
            "cfg": self.cfg,
            "state_dict": self.state_dict(),
            "sigma_u": self.sigma_u,
            "u_clamp": self.u_clamp,
            "velocity_clamp": self.velocity_clamp,
        }
        if extra:
            blob.update(extra)
        return blob

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        blob = torch.load(path, map_location=map_location, weights_only=False)
        net = cls(unet=blob["cfg"],
                  u_clamp=blob.get("u_clamp", 4.0),
                  velocity_clamp=blob.get("velocity_clamp", 8.0))
        net.load_state_dict(blob["state_dict"])
        return net
