#!/usr/bin/env python3
"""Train the multiplicative flow matching model for 2-D sound-speed maps.

    python train.py --config configs/sos_flow.json --cache cache --out out/flow

The flow is trained directly on the normalised log-SoS map
``u = log(c / 1500) / 0.05`` (single stage, no frozen baseline).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data.geometry as G
from engine import (EMA, Logger, build_loaders, fmt_metrics, load_config,
                    map_metrics, save_ckpt, set_seed, to_device)
from models import SoSMultiplicativeFlowNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sos_flow.json")
    p.add_argument("--cache", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, n_steps, n_samples):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        b = to_device(batch, device)
        c = model.sample(b["cond"], n_steps=n_steps, n_samples=n_samples)
        c = c.mean(dim=0)                      # ensemble mean as point estimate
        preds.append(c[:, 0].cpu().numpy())
        gts.append(b["c_gt"][:, 0].cpu().numpy())
    return map_metrics(np.concatenate(preds), np.concatenate(gts))


def main():
    args = parse_args()
    overrides = {"data.cache": args.cache, "data.num_workers": args.workers,
                 "train.n_epoch": args.epochs, "data.batch_size": args.batch,
                 "train.lr": args.lr}
    cfg = load_config(args.config, overrides)
    out_dir = args.out or os.path.join("out", cfg["name"] + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(os.path.join(out_dir, "train.log"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    set_seed(cfg.get("seed", 0))
    log(f"device={device}  out={out_dir}\ncache={cfg['data']['cache']}")

    tr_loader, va_loader, tr_set, va_set = build_loaders(
        cfg, limit=args.limit or None)
    log(f"train={len(tr_set)}  val={len(va_set)}  "
        f"cond_ch={cfg['unet']['cond_channels']}  "
        f"geom_aug={'on' if tr_set.geom_aug else 'off'}")

    model = SoSMultiplicativeFlowNetwork(
        unet=cfg["unet"],
        u_clamp=cfg["flow"].get("u_clamp", 4.0),
        u_source_scale=cfg["flow"].get("u_source_scale", -1.0),
        velocity_clamp=cfg["flow"].get("velocity_clamp", 8.0),
        reflow_t_schedule=cfg["flow"].get("reflow_t_schedule", "stratified"),
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n_par/1e6:.2f} M")

    ema = EMA(model, decay=cfg["train"].get("ema_decay", 0.999))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"].get("weight_decay", 0.0))
    sched = None
    if cfg["train"].get("lr_schedule") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["train"]["n_epoch"], eta_min=cfg["train"]["lr"] * 0.02)
    n_steps_val = int(cfg["train"].get("ode_steps_val", 10))
    n_samp_val = int(cfg["train"].get("n_samples_val", 4))

    best = float("inf")
    for epoch in range(1, cfg["train"]["n_epoch"] + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss = 0.0
        for it, batch in enumerate(tr_loader, 1):
            b = to_device(batch, device)
            loss = model(b["cond"], b["u_gt"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg["train"]["grad_clip"])
            opt.step()
            ema.update(model)
            run_loss += float(loss.detach())
            if it % cfg["train"].get("log_every", 20) == 0:
                log(f"ep {epoch:3d} it {it:4d}/{len(tr_loader)}  "
                    f"L={run_loss/it:.4f}  sigma_u={model.sigma_u:.3f}  "
                    f"({time.perf_counter()-t0:.0f}s)")
        train_msg = (f"ep {epoch:3d} train L={run_loss/max(1,it):.4f} "
                     f"sigma_u={model.sigma_u:.3f}")

        if epoch % cfg["train"].get("val_every", 5) == 0:
            metrics = evaluate(ema.module, va_loader, device, n_steps_val,
                               n_samp_val)
            log(train_msg + "  val " + fmt_metrics(metrics))
            score = metrics["roi_mae"]
            if score < best:
                best = score
                save_ckpt(ema.module, os.path.join(out_dir, "best.pth"),
                          extra={"epoch": epoch, "val": metrics})
                log(f"  * new best val roi_mae={best:.3f} m/s -> best.pth")
        else:
            log(train_msg)

        if epoch % cfg["train"].get("save_every", 25) == 0:
            save_ckpt(ema.module, os.path.join(out_dir, f"ep{epoch:04d}.pth"),
                      extra={"epoch": epoch})
        save_ckpt(ema.module, os.path.join(out_dir, "last.pth"),
                  extra={"epoch": epoch})
        if sched is not None:
            sched.step()

    log(f"done. best val roi_mae = {best:.3f} m/s")
    log.close()


if __name__ == "__main__":
    main()
