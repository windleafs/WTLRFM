"""Train/adapter-tune the geometry-aware structured-condition SoS flow."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.geometry_dataset import StructuredSoSDataset, geometry_collate
from engine import EMA, Logger, load_config, save_ckpt, set_seed
from models.geometry_flow import GeometryAwareSoSFlow


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/sos_flow.json'))
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--extra-train-cache', type=Path,
                   help='additional structured cache whose train split is '
                        'concatenated for training (val/test stay primary)')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--pretrained', type=Path)
    p.add_argument('--adapter-only', action='store_true',
                   help='freeze the pretrained flow and train only the acquisition encoder')
    p.add_argument('--eval-only', action='store_true',
                   help='evaluate a pretrained model without target-domain optimisation')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--event-slots', type=int, default=11)
    p.add_argument('--canonical-angle', type=float, default=8.)
    p.add_argument('--event-bandwidth', type=float, default=2.)
    p.add_argument('--hidden-channels', type=int, default=16)
    p.add_argument('--event-dropout', type=float, default=0.)
    p.add_argument('--min-events', type=int, default=6)
    p.add_argument('--lateral-mirror', action='store_true')
    p.add_argument('--val-every', type=int, default=5)
    p.add_argument('--ode-steps', type=int, default=10)
    p.add_argument('--n-samples', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=20260921)
    return p.parse_args()


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def adapt_condition_weight(src, target, canonical_angle_deg, n_event_slots):
    """Interpolate legacy 13-angle condition kernels to canonical event slots."""
    old_angles = np.arange(-12., 13., 2.)
    new_angles = np.linspace(-canonical_angle_deg, canonical_angle_deg, n_event_slots)
    parts = [src[:, :6]]
    for angle in new_angles:
        pos = np.interp(angle, old_angles, np.arange(len(old_angles)))
        lo = int(np.floor(pos))
        hi = min(lo+1, len(old_angles)-1)
        w = pos-lo
        parts.append(src[:, 6+2*lo:8+2*lo]*(1-w)
                     + src[:, 6+2*hi:8+2*hi]*w)
    parts.append(src[:, 32:40])
    if src.shape[1] == 41:
        parts.append(src[:, 40:41])
    out = torch.cat(parts, dim=1)
    if out.shape != target.shape:
        raise ValueError(f'Adapted {tuple(out.shape)} does not match {tuple(target.shape)}')
    return out


def load_pretrained(model, path, canonical_angle_deg, n_event_slots):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if blob.get('kind') == 'geometry_aware_sos_flow':
        model.load_state_dict(blob['state_dict'])
        return {'loaded': 'encoder+flow', 'adapted_layers': []}
    expected = model.flow.cfg['cond_channels']
    source = blob.get('cfg', {}).get('cond_channels')
    state = model.flow.state_dict()
    changed = []
    if source == expected:
        model.flow.load_state_dict(blob['state_dict'], strict=True)
    elif source == 40 and expected == 36:
        for key, target in state.items():
            src = blob['state_dict'][key]
            if src.shape == target.shape:
                state[key] = src
            elif src.ndim == 4 and src.shape[1] in (40, 41) and target.shape[1] in (36, 37):
                state[key] = adapt_condition_weight(src, target, canonical_angle_deg,
                                                    n_event_slots)
                changed.append(key)
            else:
                raise ValueError(f'Cannot adapt pretrained tensor {key}: '
                                 f'{tuple(src.shape)} -> {tuple(target.shape)}')
        model.flow.load_state_dict(state, strict=True)
    else:
        raise ValueError(f'Unsupported pretrained condition layout {source} -> {expected}')
    if 'sigma_u' in blob:
        model.flow._u_scale.fill_(float(blob['sigma_u']))
        model.flow._u_initialised = True
    return {'loaded': 'flow', 'source_cond_channels': source,
            'adapted_layers': changed}


def grid_metrics(c_pred, c_gt, grid):
    """Metrics on the structured cache's physical grid, not legacy geometry."""
    c_pred = np.asarray(c_pred, np.float64)
    c_gt = np.asarray(c_gt, np.float64)
    nx, nz = c_pred.shape[-2:]
    if (nx, nz) != (int(grid['nx']), int(grid['nz'])):
        raise ValueError('Prediction shape does not match structured-cache grid')
    x = np.linspace(float(grid['x0_m']), float(grid['x1_m']), nx)
    z = np.linspace(float(grid['z0_m']), float(grid['z1_m']), nz)
    roi = ((x[:, None] >= x.min()) & (x[:, None] <= x.max())
           & (z[None, :] >= 3e-3) & (z[None, :] <= 45e-3))
    if not roi.any():
        raise ValueError('Structured-cache grid has no pixels in the 3-45 mm ROI')
    err = c_pred - c_gt
    mae = float(np.abs(err).mean())
    roi_mae = float(np.abs(err[..., roi]).mean())
    roi_mean_err = float(np.abs(err[..., roi].mean(axis=-1)).mean())
    a, b = c_pred[..., roi], c_gt[..., roi]
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denom = np.sqrt((a*a).sum(-1)*(b*b).sum(-1)) + 1e-12
    corr = float(((a*b).sum(-1)/denom).mean())
    return {'mae': mae, 'roi_mae': roi_mae,
            'roi_mean_err': roi_mean_err, 'corr': corr}


@torch.no_grad()
def evaluate(model, loader, device, n_steps, n_samples, grid):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        b = move(batch, device)
        c = model.sample(b['condition'], n_steps=n_steps, n_samples=n_samples)
        preds.append(c.mean(0)[:, 0].cpu().numpy())
        gts.append(b['c_gt'][:, 0].cpu().numpy())
    return grid_metrics(np.concatenate(preds), np.concatenate(gts), grid)


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; existing runs are never overwritten')
    cfg = load_config(args.config)
    args.out.mkdir(parents=True)
    log = Logger(str(args.out/'train.log'))
    device = torch.device(args.device)
    set_seed(args.seed)

    train_set = StructuredSoSDataset(
        args.cache, 'train', event_dropout_p=args.event_dropout,
        min_events=args.min_events, lateral_mirror=args.lateral_mirror,
        seed=args.seed, limit=args.limit)
    val_limit = max(1, args.limit//4) if args.limit else 0
    val_set = StructuredSoSDataset(args.cache, 'val', limit=val_limit)
    train_dataset = train_set
    if args.extra_train_cache:
        extra = StructuredSoSDataset(
            args.extra_train_cache, 'train', event_dropout_p=args.event_dropout,
            min_events=args.min_events, lateral_mirror=args.lateral_mirror,
            seed=args.seed + 7919, limit=args.limit)
        train_dataset = torch.utils.data.ConcatDataset([train_set, extra])
    batch_size = int(args.batch_size or cfg['data'].get('batch_size', 4))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=geometry_collate, drop_last=len(train_dataset) > batch_size,
        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=geometry_collate, pin_memory=True,
        persistent_workers=args.workers > 0)
    manifest = train_set.manifest
    cond = manifest['condition']
    speeds = cond.get('speeds', [1450., 1500., 1550.])
    ref_speed = cond.get('ref_speed', speeds[1])
    ref_index = int(np.flatnonzero(np.isclose(speeds, ref_speed))[0])
    encoder_cfg = dict(speeds=speeds, ref_speed_index=ref_index,
                       n_event_slots=args.event_slots, n_subap=cond.get('n_subap', 4),
                       canonical_angle_deg=args.canonical_angle,
                       event_bandwidth_deg=args.event_bandwidth,
                       hidden_channels=args.hidden_channels,
                       event_geom_dim=cond.get('event_geom_dim', 4),
                       global_geom_dim=cond.get('global_geom_dim', 10),
                       subap_geom_dim=cond.get('subap_geom_dim', 4))

    pretrained_blob = None
    unet = dict(cfg['unet'])
    if args.pretrained:
        pretrained_blob = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        source_cfg = pretrained_blob.get('unet', pretrained_blob.get('cfg'))
        if source_cfg and source_cfg.get('cond_channels') == 36:
            unet = dict(source_cfg)
    model = GeometryAwareSoSFlow(
        unet=unet, encoder=encoder_cfg,
        u_clamp=cfg['flow'].get('u_clamp', 4.),
        u_source_scale=cfg['flow'].get('u_source_scale', -1.),
        velocity_clamp=cfg['flow'].get('velocity_clamp', 8.),
        reflow_t_schedule=cfg['flow'].get('reflow_t_schedule', 'stratified')).to(device)
    pretrained_info = None
    if args.pretrained:
        pretrained_info = load_pretrained(model, args.pretrained,
                                          args.canonical_angle, args.event_slots)
    if args.adapter_only:
        if not args.pretrained:
            raise ValueError('--adapter-only requires --pretrained')
        model.flow.requires_grad_(False)
        parameters = list(model.encoder.parameters())
    else:
        parameters = list(model.parameters())
    opt = torch.optim.Adam(parameters, lr=float(args.lr or cfg['train']['lr']),
                           weight_decay=cfg['train'].get('weight_decay', 0.))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(args.epochs or cfg['train']['n_epoch']), eta_min=1e-6)
    ema = EMA(model, decay=cfg['train'].get('ema_decay', .999))
    log(f'device={device} train={len(train_dataset)} '
        f'(primary {len(train_set)} + extra '
        f'{len(train_dataset)-len(train_set)}) val={len(val_set)} '
        f'encoder_channels={model.encoder.out_channels} '
        f'trainable={sum(p.numel() for p in parameters)/1e6:.3f}M '
        f'adapter_only={args.adapter_only}')

    config = {'args': vars(args), 'encoder': encoder_cfg, 'unet': model.flow.cfg,
              'n_train': len(train_dataset),
              'pretrained': str(args.pretrained) if args.pretrained else None,
              'pretrained_info': pretrained_info, 'manifest': manifest,
              'note': 'Structured event aggregation; angle semantics are physical values, not channel order.'}
    (args.out/'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n')
    if args.eval_only:
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(args.seed + 17)
            metrics = evaluate(model, val_loader, device, args.ode_steps, args.n_samples,
                               train_set.manifest['grid'])
        result = {'mode': 'eval_only', 'target_labelled_examples': 0,
                  'val': metrics, 'pretrained': pretrained_info}
        (args.out/'eval.json').write_text(json.dumps(result, indent=2) + '\n')
        log(json.dumps(result))
        log.close()
        return
    best = float('inf')
    epochs = int(args.epochs or cfg['train']['n_epoch'])
    start = time.time()
    history = args.out/'history.jsonl'
    for epoch in range(1, epochs+1):
        if args.adapter_only:
            model.encoder.train()
            model.flow.eval()
        else:
            model.train()
        total = count = 0
        for batch in train_loader:
            b = move(batch, device)
            loss = model(b['condition'], b['u_gt'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite geometry-flow training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg['train'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(parameters, cfg['train']['grad_clip'])
            opt.step()
            ema.update(model)
            total += float(loss.detach())*len(b['u_gt'])
            count += len(b['u_gt'])
        sched.step()
        row = {'epoch': epoch, 'loss': total/max(count, 1), 'lr': sched.get_last_lr()[0],
               'seconds': time.time()-start, 'sigma_u': model.sigma_u}
        if epoch % args.val_every == 0 or epoch == epochs:
            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                torch.manual_seed(args.seed + 17)
                metrics = evaluate(ema.module, val_loader, device,
                                   args.ode_steps, args.n_samples,
                                   train_set.manifest['grid'])
            row['val'] = metrics
            if metrics['roi_mae'] < best:
                best = metrics['roi_mae']
                save_ckpt(ema.module, str(args.out/'best.pth'),
                          extra={'epoch': epoch, 'val': metrics, 'config': config})
                row['best'] = True
        with history.open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        log(json.dumps(row))
        save_ckpt(ema.module, str(args.out/'last.pth'),
                  extra={'epoch': epoch, 'config': config})
    log(f'done best val roi_mae={best:.3f} m/s')
    log.close()


if __name__ == '__main__':
    main()
