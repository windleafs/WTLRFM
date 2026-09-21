"""Geometry-aware wrapper around the existing multiplicative SoS flow."""

import torch
import torch.nn as nn

from .acquisition_encoder import AcquisitionConditionEncoder
from .sos_mult_flow import SoSMultiplicativeFlowNetwork


class GeometryAwareSoSFlow(nn.Module):
    """Structured acquisition encoder + unchanged log-SoS flow backbone."""

    def __init__(self, unet, encoder=None, u_clamp=4.0, u_source_scale=-1.0,
                 velocity_clamp=8.0, reflow_t_schedule='stratified'):
        super().__init__()
        self.encoder = AcquisitionConditionEncoder(**(encoder or {}))
        cfg = dict(unet)
        cfg['cond_channels'] = self.encoder.out_channels
        cfg['in_channel'] = self.encoder.out_channels + 1
        self.flow = SoSMultiplicativeFlowNetwork(
            unet=cfg, u_clamp=u_clamp, u_source_scale=u_source_scale,
            velocity_clamp=velocity_clamp,
            reflow_t_schedule=reflow_t_schedule)

    @property
    def sigma_u(self):
        return self.flow.sigma_u

    @property
    def last_loss_dict(self):
        return self.flow.last_loss_dict

    def encode_condition(self, condition, return_aux=False):
        return self.encoder(condition, return_aux=return_aux)

    def forward(self, condition, u_gt, **kwargs):
        return self.flow(self.encoder(condition), u_gt, **kwargs)

    @torch.no_grad()
    def sample(self, condition, n_steps=20, n_samples=1, noise_scale=1.0,
               return_u=False, return_cond=False):
        encoded, aux = self.encoder(condition, return_aux=True)
        out = self.flow.sample(encoded, n_steps=n_steps, n_samples=n_samples,
                               noise_scale=noise_scale, return_u=return_u)
        return (out, encoded, aux) if return_cond else out

    def load_flow_checkpoint(self, path, map_location='cpu'):
        """Warm-start the inner flow from an existing SoS checkpoint."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if blob.get('kind') == 'geometry_aware_sos_flow':
            self.load_state_dict(blob['state_dict'])
            return {'kind': 'geometry_aware_sos_flow', 'loaded': 'encoder+flow'}
        cfg = blob.get('cfg', {})
        if cfg and cfg.get('cond_channels') != self.flow.cfg.get('cond_channels'):
            raise ValueError('Pretrained condition layout does not match encoder output')
        self.flow.load_state_dict(blob['state_dict'], strict=True)
        if 'sigma_u' in blob:
            self.flow._u_scale.fill_(float(blob['sigma_u']))
            self.flow._u_initialised = True
        return {'kind': 'flow', 'loaded': 'flow'}

    def checkpoint(self, extra=None):
        blob = {
            'kind': 'geometry_aware_sos_flow',
            'encoder_cfg': dict(self.encoder.cfg),
            'unet': dict(self.flow.cfg),
            'state_dict': self.state_dict(),
            'sigma_u': self.sigma_u,
            'u_clamp': self.flow.u_clamp,
            'velocity_clamp': self.flow.velocity_clamp,
            'reflow_t_schedule': self.flow.reflow_t_schedule,
        }
        if extra:
            blob.update(extra)
        return blob

    @classmethod
    def from_checkpoint(cls, path, map_location='cpu'):
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if blob.get('kind') != 'geometry_aware_sos_flow':
            raise ValueError('Not a geometry-aware SoS flow checkpoint')
        model = cls(unet=blob['unet'], encoder=blob['encoder_cfg'],
                    u_clamp=blob.get('u_clamp', 4.0),
                    u_source_scale=blob.get('sigma_u', -1.0),
                    velocity_clamp=blob.get('velocity_clamp', 8.0),
                    reflow_t_schedule=blob.get('reflow_t_schedule', 'stratified'))
        model.load_state_dict(blob['state_dict'])
        return model
