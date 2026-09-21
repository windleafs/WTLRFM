"""Validated, unaugmented abdominal cache and shared evaluation helpers."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data import geometry as G

SCHEMA = "abdominal_sos_v1"
SHAPE = (128, 160)
CAVEAT = ("Synthetic abdominal plane-wave domain only; results do not establish "
          "clinical or cross-acquisition generalization. Metrics apply to the "
          "cached probe-relative FOV (default depth 0.15-47.85 mm), not the "
          "entire source anatomy. Whole/ROI metrics include low-weight background; "
          "tissue and wall metrics use valid in-grid tissue only.")


def read_cache(cache):
    root = Path(cache).resolve()
    meta = json.loads((root / "meta.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
        raise ValueError(f"Expected cache schema {SCHEMA}")
    if not isinstance(meta.get("config"), dict):
        raise ValueError("Cache metadata must contain config object")
    for key, size in zip(("cx", "cz"), SHAPE):
        grid = np.asarray(meta.get(key), dtype=np.float64)
        if grid.shape != (size,) or not np.isfinite(grid).all() or not (np.diff(grid) > 0).all():
            raise ValueError(f"Invalid metadata {key} coordinates")
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("splits.json must contain train/val/test lists")
    seen = set()
    for split, names in splits.items():
        if not isinstance(names, list):
            raise ValueError(f"Invalid split {split}")
        for name in names:
            if not isinstance(name, str) or not re.fullmatch(r"liver_pw_[0-9]{6,}", name):
                raise ValueError(f"Invalid sample name: {name!r}")
            if name in seen:
                raise ValueError(f"Duplicate or overlapping split name: {name}")
            seen.add(name)
    return root, meta, splits


def cache_fingerprint(root, meta, splits, selected=("train", "val")):
    """Metadata plus file identity, without opening held-out test samples."""
    digest = hashlib.sha256(json.dumps({"meta": meta, "splits": splits},
                                      sort_keys=True).encode())
    for split in selected:
        for name in splits[split]:
            stat = (Path(root) / f"{name}.npz").stat()
            digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


class AbdominalSoSDataset(Dataset):
    def __init__(self, cache, split="train", limit=0):
        self.root, self.meta, self.splits = read_cache(cache)
        if split not in self.splits or limit < 0:
            raise ValueError("Invalid split or negative limit")
        self.split = split
        self.names = self.splits[split][:limit or None]
        if not self.names:
            raise ValueError(f"Empty {split} split")

    def __len__(self):
        return len(self.names)

    def load_arrays(self, name):
        with np.load(self.root / f"{name}.npz", allow_pickle=False) as archive:
            required = ("cond", "c_gt", "valid_mask", "wall_mask", "segmentation", "cx", "cz", "meta")
            if not set(required).issubset(archive.files):
                raise ValueError(f"{name}: missing cache fields")
            a = {key: archive[key] for key in required}
        for key, dtype, shape in (("cond", np.float16, (20, *SHAPE)),
                                  ("c_gt", np.float32, SHAPE),
                                  ("valid_mask", np.bool_, SHAPE),
                                  ("wall_mask", np.bool_, SHAPE),
                                  ("segmentation", np.uint8, SHAPE)):
            if a[key].shape != shape or a[key].dtype != dtype:
                raise ValueError(f"{name}: invalid {key} shape/dtype")
        if not np.isfinite(a["cond"]).all() or not np.isfinite(a["c_gt"]).all() or (a["c_gt"] <= 0).any():
            raise ValueError(f"{name}: nonfinite condition or invalid sound speed")
        if not a["valid_mask"].any():
            raise ValueError(f"{name}: empty valid tissue mask")
        if (a["wall_mask"] & ~a["valid_mask"]).any():
            raise ValueError(f"{name}: wall outside valid tissue")
        expected_wall = a["valid_mask"] & np.isin(a["segmentation"], [2, 3, 4, 5, 6])
        if not np.array_equal(a["wall_mask"], expected_wall):
            raise ValueError(f"{name}: wall mask disagrees with segmentation")
        for key, size in zip(("cx", "cz"), SHAPE):
            if a[key].shape != (size,) or not np.allclose(a[key], self.meta[key], rtol=0, atol=1e-8):
                raise ValueError(f"{name}: {key} differs from cache metadata")
        try:
            sample_meta = json.loads(str(a["meta"].item()))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}: invalid sample metadata") from exc
        if not isinstance(sample_meta, dict) or not sample_meta:
            raise ValueError(f"{name}: sample metadata must be a nonempty object")
        if sample_meta.get("schema", SCHEMA) != SCHEMA or sample_meta.get("name", name) != name:
            raise ValueError(f"{name}: inconsistent sample metadata")
        a["meta"] = sample_meta
        return a

    def __getitem__(self, index):
        name = self.names[index]
        a = self.load_arrays(name)
        return {"name": name, "cond": torch.from_numpy(a["cond"].astype(np.float32)),
                "c_gt": torch.from_numpy(a["c_gt"][None]),
                "u_gt": torch.from_numpy(G.c_to_u(a["c_gt"]).astype(np.float32)[None]),
                "valid_mask": torch.from_numpy(a["valid_mask"][None]),
                "wall_mask": torch.from_numpy(a["wall_mask"][None])}


def masked_metrics(pred, truth, mask):
    error = (np.asarray(pred, np.float64) - np.asarray(truth, np.float64))[mask]
    if not error.size:
        return {"mae": None, "rmse": None, "pixels": 0}
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())), "pixels": int(error.size)}


def sample_metrics(pred, truth, valid, wall):
    from engine import map_metrics
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions")
    metrics = map_metrics(pred, truth)
    metrics["rmse"] = float(np.sqrt(np.mean((pred.astype(np.float64) - truth) ** 2)))
    for label, mask in (("tissue", valid), ("wall", wall & valid)):
        metrics.update({f"{label}_{k}": v for k, v in masked_metrics(pred, truth, mask).items()})
    return metrics


def aggregate_metrics(rows):
    """Legacy metrics are sample means; masked errors are pixel pooled."""
    if not rows:
        raise ValueError("Cannot aggregate empty evaluation")
    result = {key: float(np.mean([r[key] for r in rows]))
              for key in ("mae", "roi_mae", "roi_mean_err", "corr")}
    result["rmse"] = float(np.sqrt(np.mean([r["rmse"] ** 2 for r in rows])))
    for label in ("tissue", "wall"):
        count = sum(r[f"{label}_pixels"] for r in rows)
        result[f"{label}_pixels"] = count
        for metric in ("mae", "rmse"):
            power = 2 if metric == "rmse" else 1
            total = sum(r[f"{label}_{metric}"] ** power * r[f"{label}_pixels"]
                        for r in rows if r[f"{label}_pixels"])
            result[f"{label}_{metric}"] = (total / count) ** (1 / power) if count else None
    return result


def training_mean_map(dataset):
    if dataset.split != "train" or not set(dataset.names).issubset(dataset.splits["train"]):
        raise ValueError("Mean map baseline must use train names only")
    total, count = np.zeros(SHAPE, np.float64), np.zeros(SHAPE, np.int64)
    for name in dataset.names:
        a = dataset.load_arrays(name)
        total += a["c_gt"] * a["valid_mask"]
        count += a["valid_mask"]
    fallback = total.sum() / count.sum()
    mean = np.full(SHAPE, fallback, np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean.astype(np.float32), count
