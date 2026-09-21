"""Shared training utilities: config, seeding, EMA, metrics, logging.

Deliberately lightweight (no tensorboardX / Palette framework dependency):
the SoS-map task is regression on cached tensors, so a plain PyTorch loop is
enough and keeps the two-stage (baseline -> flow) contract explicit.
"""

import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data.geometry as G
from data.obpw_dataset import OpenBreastSoSDataset


# ---------------------------------------------------------------- config
def load_config(path, overrides=None):
    with open(path) as f:
        cfg = json.load(f)
    for key, val in (overrides or {}).items():
        if val is None:
            continue
        _set_nested(cfg, key, val)
    return cfg


def _set_nested(cfg, dotted, val):
    parts = dotted.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- logging
class Logger:
    def __init__(self, path=None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.fh = open(path, "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self):
        if self.path:
            self.fh.close()


# ---------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average of parameters; buffers are copied (not
    averaged) so the auto-estimated sigma_rho stays in sync with the online
    network -- the 260702 WTLRFM fix."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for cur, ema in zip(model.parameters(), self.module.parameters()):
            ema.data.mul_(self.decay).add_(cur.data, alpha=1.0 - self.decay)
        for cur, ema in zip(model.buffers(), self.module.buffers()):
            ema.data.copy_(cur.data)

    def state_dict(self):
        return self.module.state_dict()


# ---------------------------------------------------------------- metrics
def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2)


def map_metrics(c_pred, c_gt):
    """Sound-speed map metrics in m/s.

    Args:
        c_pred, c_gt: array-like [..., H, W] (last two dims are the map).
    Returns:
        dict with mae (whole map), roi_mae (3-45 mm), roi_mean_err
        (|mean SoS in ROI| error) and corr (per-sample map correlation mean).
    """
    c_pred = np.asarray(c_pred, np.float64)
    c_gt = np.asarray(c_gt, np.float64)
    roi = G.roi_mask(c_pred.shape[-2], c_pred.shape[-1])
    err = c_pred - c_gt
    mae = float(np.abs(err).mean())
    roi_mae = float(np.abs(err[..., roi]).mean())
    roi_mean_err = float(np.abs(err[..., roi].mean(axis=-1)).mean())
    # per-sample Pearson correlation over the ROI
    a = c_pred[..., roi]
    b = c_gt[..., roi]
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denom = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1)) + 1e-12
    corr = float(((a * b).sum(-1) / denom).mean())
    return {"mae": mae, "roi_mae": roi_mae,
            "roi_mean_err": roi_mean_err, "corr": corr}


def fmt_metrics(m):
    return " ".join(f"{k}={v:.3f}" for k, v in m.items())


# ---------------------------------------------------------------- data
def build_loaders(cfg, batch_size=None, num_workers=None, limit=None,
                  return_name=False):
    dcfg = cfg["data"]
    cache = dcfg["cache"]
    bs = int(batch_size or dcfg.get("batch_size", 8))
    nw = int(num_workers if num_workers is not None else
             dcfg.get("num_workers", 8))
    tr = OpenBreastSoSDataset(cache, "train", augment=dcfg.get("augment", True),
                              geom_aug=dcfg.get("geom_aug"),
                              return_name=return_name)
    va = OpenBreastSoSDataset(cache, "val", augment=False,
                              return_name=return_name)
    if limit:
        tr.names = tr.names[:limit]
        va.names = va.names[:max(1, limit // 4)]
    tr_loader = torch.utils.data.DataLoader(
        tr, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
        drop_last=len(tr) > bs, persistent_workers=nw > 0)
    va_loader = torch.utils.data.DataLoader(
        va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0)
    return tr_loader, va_loader, tr, va


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def save_ckpt(model, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(model, "checkpoint"):
        blob = model.checkpoint(extra)
    else:
        blob = {"state_dict": model.state_dict()}
        if extra:
            blob.update(extra)
    torch.save(blob, path)


def load_ckpt(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
