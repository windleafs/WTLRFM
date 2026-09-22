"""Train the deterministic geometry-aware SoS baseline (encoder + WTLR-UNet).

Mirrors scripts/train_geometry_flow.py exactly (data, augmentation, schedule,
EMA, best-by-roi_mae) except that decoding is a single deterministic
regression pass instead of flow-matching sampling.  This isolates the
contribution of the acquisition encoder from that of the flow decoder.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from data.geometry_dataset import StructuredSoSDataset, geometry_collate
from engine import EMA, Logger, load_config, save_ckpt, set_seed
from models.geometry_deterministic import GeometryDeterministicSoS
from train_geometry_flow import grid_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/sos_flow.json'))
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
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
    p.add_argument('--u-clamp', type=float, default=4.)
    p.add_argument('--loss', choices=('l1', 'mse'), default='l1')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--resume', type=Path,
                   help='checkpoint (EMA weights) to continue from; requires '
                        'the run directory to already exist')
    return p.parse_args()


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


@torch.no_grad()
def evaluate(model, loader, device, grid):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        b = move(batch, device)
        preds.append(model(b['condition'])[:, 0].cpu().numpy())
        gts.append(b['c_gt'][:, 0].cpu().numpy())
    return grid_metrics(np.concatenate(preds), np.concatenate(gts), grid)


def main():
    args = parse_args()
    if args.out.exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory; existing runs are never overwritten')
    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=bool(args.resume))
    log = Logger(str(args.out/'train.log'))
    device = torch.device(args.device)
    set_seed(args.seed)

    train_set = StructuredSoSDataset(
        args.cache, 'train', event_dropout_p=args.event_dropout,
        min_events=args.min_events, lateral_mirror=args.lateral_mirror,
        seed=args.seed, limit=args.limit)
    val_limit = max(1, args.limit//4) if args.limit else 0
    val_set = StructuredSoSDataset(args.cache, 'val', limit=val_limit)
    batch_size = int(args.batch_size or cfg['data'].get('batch_size', 4))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=geometry_collate, drop_last=len(train_set) > batch_size,
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
    model = GeometryDeterministicSoS(
        unet=dict(cfg['unet']), encoder=encoder_cfg,
        u_clamp=args.u_clamp, loss=args.loss).to(device)
    parameters = list(model.parameters())
    opt = torch.optim.Adam(parameters, lr=float(args.lr or cfg['train']['lr']),
                           weight_decay=cfg['train'].get('weight_decay', 0.))
    epochs = int(args.epochs or cfg['train']['n_epoch'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6)
    start_epoch = 0
    if args.resume:
        blob = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(blob['state_dict'])
        start_epoch = int(blob.get('epoch', 0))
        for _ in range(start_epoch):
            sched.step()
        log(f'resumed from {args.resume} at epoch {start_epoch} '
            f'lr={sched.get_last_lr()[0]:.3e}')
    ema = EMA(model, decay=cfg['train'].get('ema_decay', .999))
    log(f'device={device} train={len(train_set)} val={len(val_set)} '
        f'encoder_channels={model.encoder.out_channels} '
        f'trainable={sum(p.numel() for p in parameters)/1e6:.3f}M '
        f'loss={args.loss}')

    config = {'args': vars(args), 'encoder': encoder_cfg, 'unet': model.cfg,
              'manifest': manifest,
              'note': 'Deterministic ablation of GeometryAwareSoSFlow: '
                      'same encoder and UNet, single-pass regression.'}
    if not args.resume:
        (args.out/'config.json').write_text(json.dumps(config, indent=2, default=str)+'\n')

    best = float('inf')
    history = args.out/'history.jsonl'
    if args.resume and history.exists():
        for line in history.read_text().splitlines():
            row = json.loads(line)
            if 'val' in row:
                best = min(best, row['val']['roi_mae'])
    start = time.time()
    for epoch in range(start_epoch+1, epochs+1):
        model.train()
        total = count = 0
        for batch in train_loader:
            b = move(batch, device)
            loss = model(b['condition'], b['u_gt'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite deterministic training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg['train'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(parameters, cfg['train']['grad_clip'])
            opt.step()
            ema.update(model)
            total += float(loss.detach())*len(b['u_gt'])
            count += len(b['u_gt'])
        sched.step()
        row = {'epoch': epoch, 'loss': total/max(count, 1),
               'lr': sched.get_last_lr()[0], 'seconds': time.time()-start}
        if epoch % args.val_every == 0 or epoch == epochs:
            metrics = evaluate(ema.module, val_loader, device,
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
