#!/usr/bin/env python3
"""Standalone masked abdominal flow training with epoch-boundary resume."""
import argparse
import copy
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.abdominal_dataset import (AbdominalSoSDataset, CAVEAT, aggregate_metrics,
                                    cache_fingerprint, sample_metrics, training_mean_map)
from data import geometry as G
from engine import EMA, set_seed, to_device
from models import SoSMultiplicativeFlowNetwork

PRIOR_ATTRS = ("_u_initialised", "_u_ema", "_u_ema_decay", "_u_warmup_batches", "_u_steps",
               "u_source_scale", "reflow_t_schedule")


def atomic_save(blob, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, temp)
    temp.replace(path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def prior_state(model):
    return {key: getattr(model, key) for key in PRIOR_ATTRS if hasattr(model, key)}


def restore_prior(model, state):
    for key, value in state.items():
        setattr(model, key, value)


def save_training_state(path, model, ema, optimizer, scheduler, epoch, best, cfg, data_meta):
    atomic_save({"model": model.state_dict(), "ema": ema.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "best": best, "config": cfg, "data_meta": data_meta,
                 "rng": rng_state(), "prior": prior_state(model),
                 "ema_prior": prior_state(ema.module), "ema_decay": ema.decay}, path)


def load_training_state(path, model, ema, optimizer, scheduler, cfg, data_meta):
    state = torch.load(path, map_location="cpu", weights_only=False)
    old, new = copy.deepcopy(state["config"]), copy.deepcopy(cfg)
    old["train"].pop("n_epoch", None)
    new["train"].pop("n_epoch", None)
    if old != new or state["data_meta"] != data_meta:
        raise ValueError("Resume config/cache fingerprint or training names changed")
    if cfg["train"]["n_epoch"] < state["epoch"]:
        raise ValueError("Requested epochs precede completed resume epoch")
    model.load_state_dict(state["model"])
    ema.module.load_state_dict(state["ema"])
    ema.decay = state["ema_decay"]
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    # Extending a completed run is supported, but changes its cosine horizon.
    if hasattr(scheduler, "T_max"):
        scheduler.T_max = cfg["train"]["n_epoch"]
    restore_prior(model, state["prior"])
    restore_prior(ema.module, state["ema_prior"])
    restore_rng(state["rng"])
    return state["epoch"], state["best"]


def autocast_context(device, enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled and device.type == "cuda" else nullcontext()


@torch.no_grad()
def prediction_batches(model, loader, device, n_steps, n_samples, seed=12345, bf16=False):
    """A fork protects training RNG, including loader iterator construction."""
    model.eval()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in loader:
            b = to_device(batch, device)
            with autocast_context(device, bf16):
                samples = model.sample(b["cond"], n_steps=n_steps, n_samples=n_samples).float()
            if not torch.isfinite(samples).all():
                raise FloatingPointError("Nonfinite prediction samples")
            pred = samples.mean(0)[:, 0].cpu().numpy()
            std = samples.std(0, unbiased=False)[:, 0].cpu().numpy()
            if not np.isfinite(pred).all() or not np.isfinite(std).all():
                raise FloatingPointError("Nonfinite prediction mean/std")
            yield batch, pred, std


def evaluate(model, loader, device, n_steps, n_samples, seed=12345, bf16=False, mean_map=None):
    rows = {"model": []}
    if mean_map is not None:
        rows.update({"constant1500": [], "constant1540": [], "train_mean": []})
    for batch, preds, _ in prediction_batches(model, loader, device, n_steps, n_samples, seed, bf16):
        for i, pred in enumerate(preds):
            truth = batch["c_gt"][i, 0].numpy()
            valid, wall = (batch[key][i, 0].numpy() for key in ("valid_mask", "wall_mask"))
            estimates = {"model": pred}
            if mean_map is not None:
                estimates.update(constant1500=np.full_like(pred, 1500),
                                 constant1540=np.full_like(pred, 1540), train_mean=mean_map)
            for key, estimate in estimates.items():
                rows[key].append(sample_metrics(estimate, truth, valid, wall))
    return {key: aggregate_metrics(value) for key, value in rows.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache")
    parser.add_argument("--config", default=str(ROOT / "configs/sos_abdominal.json"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", nargs="?", const="auto", help="training_state.pth path; defaults to --out")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    for section, key, value in (("data", "cache", args.cache), ("data", "batch_size", args.batch),
                                ("data", "num_workers", args.workers), ("train", "n_epoch", args.epochs)):
        if value is not None:
            cfg[section][key] = value
    cfg["data"]["cache"] = str(Path(cfg["data"]["cache"]).resolve())
    cfg["data"]["limit"] = args.limit
    tcfg, dcfg = cfg["train"], cfg["data"]
    if min(tcfg["n_epoch"], dcfg["batch_size"], tcfg["val_every"], tcfg["ode_steps_val"], tcfg["n_samples_val"]) < 1 or dcfg["num_workers"] < 0 or args.limit < 0:
        raise ValueError("Epochs/batch/evaluation counts must be positive; workers/limit nonnegative")
    if cfg["unet"]["cond_channels"] != 20 or cfg["unet"]["in_channel"] != 21 or cfg["unet"]["out_channel"] != 1:
        raise ValueError("Abdominal model requires 20 condition / 21 input / 1 output channels")
    if dcfg.get("augment") or cfg["flow"].get("u_source_scale") != 1.0:
        raise ValueError("Abdominal training requires no augmentation and fixed u_source_scale=1.0")
    out = Path(args.out).resolve()
    resume = out / "training_state.pth" if args.resume == "auto" else Path(args.resume).resolve() if args.resume else None
    if out.exists() and not resume:
        raise FileExistsError("Output already exists; use --resume or a new --out")
    if resume and (not resume.is_file() or not (out / "best.pth").is_file()):
        raise ValueError("Resume requires training_state.pth and the existing output best.pth")
    device = torch.device(args.device)
    torch.set_num_threads(min(4, max(1, int(cfg.get("torch_threads", 4)))))
    set_seed(cfg.get("seed", 0))
    train = AbdominalSoSDataset(dcfg["cache"], "train", args.limit)
    val = AbdominalSoSDataset(dcfg["cache"], "val", args.limit)
    data_meta = {"cache_meta": train.meta, "splits": train.splits, "train_names": train.names,
                 "val_names": val.names, "cache_fingerprint": cache_fingerprint(train.root, train.meta, train.splits),
                 "normalization": {"c_ref": G.C_REF, "rho_scale": G.RHO_SCALE}, "caveat": CAVEAT}
    loaders = [DataLoader(ds, batch_size=dcfg["batch_size"], shuffle=shuffle,
                          num_workers=dcfg["num_workers"], pin_memory=device.type == "cuda")
               for ds, shuffle in ((train, True), (val, False))]
    model = SoSMultiplicativeFlowNetwork(unet=cfg["unet"], **cfg["flow"]).to(device)
    ema = EMA(model, decay=tcfg["ema_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0))
    if tcfg.get("lr_schedule", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["n_epoch"], eta_min=tcfg["lr"] * .02)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    start, best = 0, float("inf")
    if resume:
        start, best = load_training_state(resume, model, ema, optimizer, scheduler, cfg, data_meta)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "data_meta.json").write_text(json.dumps(data_meta, indent=2) + "\n")
    use_bf16 = bool(tcfg.get("bf16", True)) and device.type == "cuda"
    if use_bf16:
        with torch.cuda.device(device):
            use_bf16 = torch.cuda.is_bf16_supported()
    print(f"device={device} bf16={use_bf16} train={len(train)} val={len(val)} out={out}", flush=True)
    for epoch in range(start + 1, tcfg["n_epoch"] + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loaders[0]:
            b = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_bf16):
                loss = model(b["cond"], b["u_gt"], valid_mask=b["valid_mask"],
                             background_weight=tcfg.get("background_weight", 0.1))
            if loss.ndim or not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite/nonscalar loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0), error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            restore_prior(ema.module, prior_state(model))
            total += float(loss.detach()) * len(batch["name"])
            count += len(batch["name"])
        metrics = None
        if epoch == 1 or epoch % tcfg["val_every"] == 0 or epoch == tcfg["n_epoch"]:
            metrics = evaluate(ema.module, loaders[1], device, tcfg["ode_steps_val"],
                               tcfg["n_samples_val"], tcfg.get("eval_seed", 12345), use_bf16)["model"]
            if metrics["tissue_mae"] < best:
                best = metrics["tissue_mae"]
                atomic_save(ema.module.checkpoint({"epoch": epoch, "val": metrics, "data_meta": data_meta, "config": cfg}), out / "best.pth")
        scheduler.step()
        atomic_save(ema.module.checkpoint({"epoch": epoch, "val": metrics, "data_meta": data_meta, "config": cfg}), out / "last.pth")
        save_training_state(out / "training_state.pth", model, ema, optimizer, scheduler, epoch, best, cfg, data_meta)
        record = {"epoch": epoch, "loss": total / count, "lr": optimizer.param_groups[0]["lr"], "val": metrics}
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    # Final report is validation only. Test is accessed exclusively by prediction.
    best_blob = torch.load(out / "best.pth", map_location="cpu", weights_only=False)
    ema.module.load_state_dict(best_blob["state_dict"])
    mean_map, _ = training_mean_map(train)
    final = evaluate(ema.module, loaders[1], device, tcfg["ode_steps_val"], tcfg["n_samples_val"],
                     tcfg.get("eval_seed", 12345), use_bf16, mean_map)
    (out / "validation_summary.json").write_text(json.dumps({"split": "val", "checkpoint_epoch": best_blob["epoch"],
        "metrics": final, "train_mean_names": train.names, "caveat": CAVEAT}, indent=2, allow_nan=False) + "\n")
    print(f"done best validation tissue_mae={best:.4f} m/s", flush=True)


if __name__ == "__main__":
    main()
