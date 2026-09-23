"""Verify the depth-response fix on identical weights (no retrain needed).

Reproduces the reviewer's per-depth measurements on one val record with the
v3 encoder, toggling only the new input-construction flags:

  * single-angle echo RMS at 10/30/40 mm (input attenuation, unchanged)
  * mean |q| (relative-phase correlation) at each band, legacy vs relative floor
  * asinh polar-magnitude deep/shallow ratio, group vs TGC row scale
  * echo-induced encoder response at 40 mm: ||E(y) - E(y_zero)|| restricted
    to the 40 mm band, plus the zero-echo baseline RMS (must not grow)
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.geometry_dataset import StructuredSoSDataset, geometry_collate  # noqa: E402
from data import geometry as G  # noqa: E402
from models.geometry_flow import GeometryAwareSoSFlow  # noqa: E402
from models.acquisition_encoder import _depth_scale, _group_scale  # noqa: E402

BANDS = {'10 mm': .010, '30 mm': .030, '40 mm': .040}
HALF = .003
DEVICE = 'cuda:1'


def band_mask(depth_m):
    z = G.z_grid()
    return (torch.tensor(z, device=DEVICE) >= depth_m-HALF) \
        & (torch.tensor(z, device=DEVICE) < depth_m+HALF)


def run(tag, model, batch):
    with torch.no_grad():
        enc, aux = model.encoder(batch['condition'], return_aux=True)
        # measure on the ref-speed event stack exactly as the encoder does
        events_full = batch['condition']['speed_events'].to(DEVICE)
        events = events_full[:, model.encoder.ref_speed_index]  # [B,A,H,W]
        scale = _group_scale(
            events, batch['condition']['event_mask'].to(DEVICE).float())
        if model.encoder.depth_tgc:
            scale = _depth_scale(events, scale, model.encoder.tgc_smooth,
                                 model.encoder.tgc_floor)
        ref_idx = aux['reference_event']
        b = torch.arange(events.shape[0], device=DEVICE)
        reference = events[b, ref_idx]
        num = events*reference[:, None].conj()
        floor = (model.encoder.q_rel_eps*scale**2 if model.encoder.q_rel_eps > 0
                 else torch.full_like(scale, 1e-12))
        q = num/(events.abs()*reference[:, None].abs() + floor)
        polar = torch.asinh(events.abs()/scale)/np.arcsinh(3.)
        out = {}
        for name, depth in BANDS.items():
            zm = band_mask(depth)
            out[name] = dict(
                echo_rms=float(events[0, 1, :, zm].abs().mean()),
                absq=float(q[0, :, :, zm].abs().mean()),
                polar=float(polar[0, :, :, zm].mean()))
        # zero-echo baseline and echo-induced response in the 40 mm band
        zero = {k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in batch['condition'].items()}
        for k in ('speed_events', 'subap', 'subap_events'):
            zero[k] = torch.zeros_like(zero[k])
        enc0, _ = model.encoder(zero, return_aux=True)
        band40 = band_mask(.040)
        out['response@40mm'] = float((enc[:, :, :, band40]
                                      - enc0[:, :, :, band40]).norm())
        out['baseline@40mm'] = float(enc0[:, :, :, band40].norm())
        out['response@10mm'] = float((enc[:, :, :, band_mask(.010)]
                                      - enc0[:, :, :, band_mask(.010)]).norm())
    print(f'--- {tag} ---')
    for name in BANDS:
        d = out[name]
        print(f"{name}: echo_rms={d['echo_rms']:.3e}  |q|={d['absq']:.3f}  "
              f"polar={d['polar']:.3f}")
    print(f"echo-induced response: 10mm {out['response@10mm']:.3f}  "
          f"40mm {out['response@40mm']:.3f}  (ratio "
          f"{out['response@40mm']/max(out['response@10mm'],1e-12):.3f}); "
          f"zero-echo baseline@40mm {out['baseline@40mm']:.3f}")
    return out


def to_device(value):
    if torch.is_tensor(value):
        return value.to(DEVICE)
    if isinstance(value, dict):
        return {k: to_device(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v) for v in value]
    return value


def main():
    ds = StructuredSoSDataset('/data/zhuangyang/geometry_flow_v2_cache', 'val')
    ds.records = [r for r in ds.records if r['id'] == 'val_000']
    batch = to_device(geometry_collate([ds[0]]))
    blob = torch.load('out/geometry_flow_v3_decoupled_20260922/best.pth',
                      map_location='cpu', weights_only=False)
    model = GeometryAwareSoSFlow(unet=blob['unet'], encoder=blob['encoder_cfg'],
                                 u_source_scale=-1.).to(DEVICE).eval()
    model.load_state_dict(blob['state_dict'])
    model.encoder.depth_tgc = False
    model.encoder.q_rel_eps = 0.
    run('legacy (group scale, absolute 1e-12 q floor)', model, batch)
    model.encoder.q_rel_eps = 0.05
    run('relative q floor only (eps=0.05)', model, batch)
    model.encoder.depth_tgc = True
    run('relative q floor + depth TGC (eps=0.05, floor=0.08)', model, batch)


if __name__ == '__main__':
    main()
