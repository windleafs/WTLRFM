"""Acquisition-aware encoder for structured phase-preserving DAS inputs.

It maps a variable set of transmit events to a fixed number of canonical event
slots using each event's physical angle and launch-delay metadata.  Every event
is described by its compressed complex DAS image, a normalised complex
correlation with a near-broadside reference event, and explicit geometry.  The
output has the same ``full / angle / sub-aperture`` channel layout as the
existing 36-channel L11 condition, so the original flow backbone can be reused.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_complex(x, name):
    if torch.is_complex(x):
        return x
    if x.shape[-1] == 2:
        return torch.view_as_complex(x.contiguous())
    raise ValueError(f'{name} must be complex or end in a real/imaginary pair')


def _mask(value, shape, device):
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    out = torch.as_tensor(value, dtype=torch.bool, device=device)
    if out.shape != torch.Size(shape):
        raise ValueError('Condition mask has the wrong shape')
    return out


def _group_scale(z, valid=None):
    rms = z.abs().square().mean(dim=(-2, -1)).sqrt()
    if valid is not None:
        denom = valid.sum().clamp_min(1)
        scale = (rms*valid).sum()/denom
    else:
        scale = rms.mean()
    return scale.clamp_min(1e-30)


def _polar(z, scale):
    mag = torch.asinh(z.abs()/scale)/np.arcsinh(3.)
    # Use a smooth unit phasor instead of atan2: atan2(0, 0) has an undefined
    # gradient in exactly the zero-echo regions produced by invalid delays.
    unit = z/(z.abs() + scale*1e-6)
    return torch.stack([mag*unit.real, mag*unit.imag], dim=-3)


def _zero_last(net):
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class AcquisitionConditionEncoder(nn.Module):
    """Encode structured RF-condition groups into fixed flow channels.

    Args:
        speeds: assumed homogeneous speeds used to build ``speed_events``.
        ref_speed_index: index in ``speeds`` used for per-event/subap maps.
        n_event_slots: fixed canonical event slots consumed by the flow.
        n_subap: fixed canonical receive sub-aperture outputs.
        canonical_angle_deg: bound of canonical slot angles.
        event_geom_dim/global_geom_dim/subap_geom_dim: dimensions produced by
            ``data.rf_geometry`` (4, 10 and 4 respectively).
    """

    def __init__(self, speeds=(1450., 1500., 1550.), ref_speed_index=1,
                 n_event_slots=11, n_subap=4, canonical_angle_deg=8.,
                 event_bandwidth_deg=2., hidden_channels=16,
                 event_geom_dim=4, global_geom_dim=10, subap_geom_dim=4):
        super().__init__()
        self.speeds = tuple(float(c) for c in speeds)
        self.ref_speed_index = int(ref_speed_index)
        self.n_event_slots = int(n_event_slots)
        self.n_subap = int(n_subap)
        self.canonical_angle_deg = float(canonical_angle_deg)
        self.event_geom_dim = int(event_geom_dim)
        self.global_geom_dim = int(global_geom_dim)
        self.subap_geom_dim = int(subap_geom_dim)
        if len(self.speeds) < 1 or not (0 <= self.ref_speed_index < len(self.speeds)):
            raise ValueError('Invalid speed group/ref_speed_index')
        if self.n_event_slots < 2 or self.n_subap < 1 or self.canonical_angle_deg <= 0:
            raise ValueError('Invalid canonical acquisition layout')
        self.out_channels = 2*len(self.speeds) + 2*self.n_event_slots + 2*self.n_subap
        self.cfg = dict(speeds=self.speeds, ref_speed_index=self.ref_speed_index,
                        n_event_slots=self.n_event_slots, n_subap=self.n_subap,
                        canonical_angle_deg=self.canonical_angle_deg,
                        event_bandwidth_deg=event_bandwidth_deg,
                        hidden_channels=hidden_channels,
                        event_geom_dim=self.event_geom_dim,
                        global_geom_dim=self.global_geom_dim,
                        subap_geom_dim=self.subap_geom_dim)

        canonical = np.deg2rad(np.linspace(-self.canonical_angle_deg,
                                           self.canonical_angle_deg,
                                           self.n_event_slots))
        self.slot_angles = nn.Parameter(torch.tensor(canonical, dtype=torch.float32))
        bw0 = max(float(event_bandwidth_deg), .1)*np.pi/180.
        self.log_bandwidth = nn.Parameter(torch.tensor(float(np.log(np.expm1(bw0)))))
        self.null_score = nn.Parameter(torch.tensor(-5.))
        self.register_buffer('speed_geom', torch.tensor(
            [(c-1500.)/100. for c in self.speeds], dtype=torch.float32).view(-1, 1))

        self.event_net = _zero_last(nn.Sequential(
            nn.Conv2d(2+3+self.event_geom_dim, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1)))
        self.event_film = nn.Linear(self.global_geom_dim, 4)
        self.full_residual = _zero_last(nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1)))
        self.subap_residual = _zero_last(nn.Sequential(
            nn.Conv2d(2+self.subap_geom_dim, hidden_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1)))
        self.full_gain = nn.Linear(self.global_geom_dim, len(self.speeds))
        self.subap_gain = nn.Linear(self.global_geom_dim, self.n_subap)
        for layer in (self.event_film, self.full_gain, self.subap_gain):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _validate(self, cond):
        required = ('speed_events', 'subap', 'event_geom', 'subap_geom', 'global_geom')
        missing = [k for k in required if k not in cond]
        if missing:
            raise ValueError('Missing structured condition fields: ' + ', '.join(missing))
        speed_events = _as_complex(cond['speed_events'], 'speed_events')
        subap = _as_complex(cond['subap'], 'subap')
        if speed_events.ndim != 5 or subap.ndim != 4:
            raise ValueError('Expected speed_events [B,S,A,H,W] and subap [B,K,H,W]')
        b, s, a, h, w = speed_events.shape
        if (subap.shape[0], subap.shape[-2], subap.shape[-1]) != (b, h, w):
            raise ValueError('Condition groups must share batch and image shape')
        if s != len(self.speeds) or subap.shape[1] != self.n_subap:
            raise ValueError('Condition group count does not match encoder configuration')
        geom = cond['event_geom']
        subgeom = cond['subap_geom']
        glob = cond['global_geom']
        if geom.shape != (b, a, self.event_geom_dim):
            raise ValueError('event_geom must be [B,A,%d]' % self.event_geom_dim)
        if subgeom.shape != (b, self.n_subap, self.subap_geom_dim):
            raise ValueError('subap_geom must be [B,K,%d]' % self.subap_geom_dim)
        if glob.shape != (b, self.global_geom_dim):
            raise ValueError('global_geom must be [B,%d]' % self.global_geom_dim)
        return speed_events, subap, geom.float(), subgeom.float(), glob.float(), (b, s, a, h, w)

    def forward(self, cond, return_aux=False):
        speed_events, subap, event_geom, subap_geom, global_geom, shape = self._validate(cond)
        b, s_count, n_event, h, w = shape
        device = speed_events.device
        event_mask = _mask(cond.get('event_mask'), (b, n_event), device)
        subap_mask = _mask(cond.get('subap_mask'), (b, self.n_subap), device)
        if not bool(event_mask.any(dim=1).all()):
            raise ValueError('Each sample needs at least one active event')
        if not bool(subap_mask.any(dim=1).all()):
            raise ValueError('Each sample needs at least one active sub-aperture')

        # Full-aperture images are recomputed from the active event set, so
        # angle dropout cannot leak through a precomputed full image.
        active = event_mask[:, None, :, None, None].to(speed_events.real.dtype)
        full = (speed_events*active).sum(dim=2)
        gain = .1*torch.tanh(self.full_gain(global_geom))[:, :, None, None]
        full = full*(1+gain)
        full_scale = _group_scale(full)
        full_polar = _polar(full, full_scale)                    # [B,S,2,H,W]
        full_feats = []
        for si in range(s_count):
            geom = self.speed_geom[si].view(1, 1, 1, 1).expand(b, 1, h, w)
            residual = self.full_residual(torch.cat([full_polar[:, si], geom], dim=1))
            full_feats.append(full_polar[:, si] + residual)
        full_out = torch.stack(full_feats, dim=1).flatten(1, 2)   # [B,2S,H,W]

        # Per-event polar features plus a physically meaningful relative-phase
        # correlation against the available event nearest to broadside.
        events = speed_events[:, self.ref_speed_index]
        event_scale = _group_scale(events, event_mask.float())
        event_polar = _polar(events, event_scale)                # [B,A,2,H,W]
        ref_score = event_geom[:, :, 2].abs() + (~event_mask).float()*1e6
        ref_index = ref_score.argmin(dim=1)
        gather_index = ref_index.view(b, 1, 1, 1).expand(b, 1, h, w)
        reference = events.gather(1, gather_index).squeeze(1)
        q = events*reference[:, None].conj()
        q = q/(events.abs()*reference[:, None].abs() + 1e-12)
        q = torch.nan_to_num(q, nan=0., posinf=0., neginf=0.)
        geom_maps = event_geom[:, :, :, None, None].expand(b, n_event, -1, h, w)
        event_inputs = torch.cat([
            event_polar,
            torch.stack([q.real, q.imag, q.abs()], dim=2),
            geom_maps,
        ], dim=2)
        flat = event_inputs.reshape(b*n_event, -1, h, w)
        residual = self.event_net(flat).reshape(b, n_event, 2, h, w)
        features = event_polar + residual
        gamma, beta = self.event_film(global_geom).chunk(2, dim=1)
        features = features*(1+gamma[:, None, :, None, None]) + beta[:, None, :, None, None]
        features = features*event_mask[:, :, None, None, None].to(features.dtype)

        theta = torch.atan2(event_geom[:, :, 0], event_geom[:, :, 1])
        canonical = self.slot_angles.clamp(-np.deg2rad(self.canonical_angle_deg),
                                         np.deg2rad(self.canonical_angle_deg))
        bandwidth = F.softplus(self.log_bandwidth) + np.deg2rad(.1)
        score = -.5*((theta[:, :, None]-canonical[None, None, :])/bandwidth)**2
        score = score.masked_fill(~event_mask[:, :, None], -1e4)
        null = self.null_score.view(1, 1, 1).expand(b, 1, self.n_event_slots)
        weights = torch.softmax(torch.cat([score, null], dim=1), dim=1)[:, :n_event]
        slots = torch.einsum('bak,bachw->bkchw', weights, features)
        event_out = slots.flatten(1, 2)

        sub_gain = .1*torch.tanh(self.subap_gain(global_geom))[:, :, None, None]
        subap = subap*(1+sub_gain)*subap_mask[:, :, None, None]
        sub_scale = _group_scale(subap, subap_mask.float())
        sub_polar = _polar(subap, sub_scale)                    # [B,K,2,H,W]
        sub_geom_maps = subap_geom[:, :, :, None, None].expand(b, self.n_subap, -1, h, w)
        sub_inputs = torch.cat([sub_polar, sub_geom_maps], dim=2)
        sub_res = self.subap_residual(sub_inputs.reshape(b*self.n_subap, -1, h, w))
        sub_feats = (sub_polar + sub_res.reshape(b, self.n_subap, 2, h, w))
        sub_out = (sub_feats*subap_mask[:, :, None, None, None]).flatten(1, 2)

        encoded = torch.cat([full_out, event_out, sub_out], dim=1)
        if encoded.shape[1] != self.out_channels or not torch.isfinite(encoded).all():
            raise FloatingPointError('Invalid encoded condition tensor')
        if return_aux:
            return encoded, {'event_weights': weights, 'reference_event': ref_index,
                             'event_features': features}
        return encoded
