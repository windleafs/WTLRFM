"""Deterministic geometry-aware SoS regression (flow ablation baseline).

Capacity-matched counterpart of :class:`GeometryAwareSoSFlow`: the exact same
acquisition encoder and WTLR-UNet backbone, but a single deterministic
regression pass instead of conditional flow matching.  Comparing the two
isolates how much of the geometry-robustness comes from the encoder (geometry)
versus the flow decoder (generative posterior).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import data.geometry as G
from wtlrfm import WTLRUNet

from .acquisition_encoder import AcquisitionConditionEncoder


class GeometryDeterministicSoS(nn.Module):
    """(y, g) -> encoder -> WTLR-UNet -> u_hat -> c = C_REF * exp(RHO_SCALE * u).

    Args:
        unet: WTLR-UNet config; ``in_channel``/``cond_channels`` are overridden.
        encoder: acquisition-encoder config (same keys as the flow's).
        u_clamp: clamp of the predicted log-SoS state, mirroring the flow.
        loss: 'l1' or 'mse' on the normalised log-SoS target.
    """

    def __init__(self, unet, encoder=None, u_clamp=4.0, loss='l1'):
        super().__init__()
        self.encoder = AcquisitionConditionEncoder(**(encoder or {}))
        cfg = dict(unet)
        cfg['cond_channels'] = self.encoder.out_channels
        cfg['in_channel'] = self.encoder.out_channels  # no ODE state channel
        self.net = WTLRUNet(**cfg)
        self.cfg = cfg
        self.u_clamp = float(u_clamp)
        self.loss_type = str(loss)
        self.last_loss_dict = {}

    def predict_u(self, condition):
        encoded = self.encoder(condition)
        t = torch.zeros(encoded.shape[0], device=encoded.device,
                        dtype=encoded.dtype)
        return self.net(encoded, t).clamp(-self.u_clamp, self.u_clamp)

    def forward(self, condition, u_gt=None):
        """Return the training loss when ``u_gt`` is given, else the SoS map."""
        u_hat = self.predict_u(condition)
        if u_gt is None:
            return G.C_REF * torch.exp(G.RHO_SCALE * u_hat)
        if u_gt.shape != u_hat.shape:
            raise ValueError(f'u_gt {tuple(u_gt.shape)} does not match '
                             f'prediction {tuple(u_hat.shape)}')
        loss = (F.l1_loss(u_hat, u_gt) if self.loss_type == 'l1'
                else F.mse_loss(u_hat, u_gt))
        self.last_loss_dict = {'loss': float(loss.detach()),
                               'loss_type': self.loss_type}
        return loss

    def checkpoint(self, extra=None):
        blob = {
            'kind': 'geometry_deterministic_sos',
            'encoder_cfg': dict(self.encoder.cfg),
            'unet': dict(self.cfg),
            'u_clamp': self.u_clamp,
            'loss': self.loss_type,
            'state_dict': self.state_dict(),
        }
        if extra:
            blob.update(extra)
        return blob

    @classmethod
    def from_checkpoint(cls, path, map_location='cpu'):
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if blob.get('kind') != 'geometry_deterministic_sos':
            raise ValueError('Not a geometry-deterministic SoS checkpoint')
        model = cls(unet=blob['unet'], encoder=blob['encoder_cfg'],
                    u_clamp=blob.get('u_clamp', 4.0),
                    loss=blob.get('loss', 'l1'))
        model.load_state_dict(blob['state_dict'])
        return model
