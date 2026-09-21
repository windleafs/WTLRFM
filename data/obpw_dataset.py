"""Torch dataset over the preprocessed OpenBreastUS plane-wave SoS cache.

Cache produced by ``scripts/prepare_cache.py``: one npz per sample with
    cond   float16 [2*len(speeds), NX, NZ]   asinh-compressed DAS I/Q
    c_gt   float32 [NX, NZ]                  ground-truth sound speed [m/s]
    meta   scalars                           dataset labels (for metrics)
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from . import geometry as G


def load_split(cache_dir, split="train", split_file=None):
    """Return the list of sample names for a split.

    If ``splits.json`` exists in the cache directory it is used; otherwise a
    deterministic 80/10/10 split over the sorted sample list is created.
    """
    names = sorted(f[:-4] for f in os.listdir(cache_dir)
                   if f.startswith("sample_") and f.endswith(".npz"))
    if not names:
        raise FileNotFoundError(f"no cached samples in {cache_dir}")
    split_file = split_file or os.path.join(cache_dir, "splits.json")
    if os.path.isfile(split_file):
        with open(split_file) as f:
            splits = json.load(f)
    else:
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(names))
        n_val = max(1, int(round(0.1 * len(names))))
        n_test = max(1, int(round(0.1 * len(names))))
        splits = {
            "train": [names[i] for i in idx[:len(names) - n_val - n_test]],
            "val": [names[i] for i in idx[len(names) - n_val - n_test:
                                           len(names) - n_test]],
            "test": [names[i] for i in idx[len(names) - n_test:]],
        }
        with open(split_file, "w") as f:
            json.dump(splits, f, indent=1)
    if split == "all":
        return names
    return splits[split]


class OpenBreastSoSDataset(Dataset):
    """(condition DAS images, normalised log-SoS map) pairs."""

    def __init__(self, cache_dir, split="train", split_file=None,
                 augment=False, geom_aug=None, return_name=False):
        self.cache_dir = cache_dir
        self.names = load_split(cache_dir, split, split_file)
        self.augment = bool(augment) and split == "train"
        self.geom_aug = dict(geom_aug) if (geom_aug and self.augment) else None
        self.return_name = return_name

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        with np.load(os.path.join(self.cache_dir, name + ".npz")) as d:
            cond = np.asarray(d["cond"], np.float32)
            c_gt = np.asarray(d["c_gt"], np.float32)
        u_gt = G.c_to_u(c_gt).astype(np.float32)[None]      # [1, NX, NZ]
        if self.augment:
            if np.random.rand() < 0.5:
                # lateral mirror: probe/grid is left-right symmetric
                cond = cond[:, ::-1].copy()
                u_gt = u_gt[:, ::-1].copy()
                c_gt = c_gt[::-1].copy()
            if self.geom_aug is not None:
                from .geom_aug import perturb
                cond, u_gt, c_gt = perturb(cond, u_gt, c_gt, self.geom_aug)
        out = {
            "cond": torch.from_numpy(np.ascontiguousarray(cond)),
            "u_gt": torch.from_numpy(np.ascontiguousarray(u_gt)),
            "c_gt": torch.from_numpy(np.ascontiguousarray(c_gt[None])),
        }
        if self.return_name:
            out["name"] = name
        return out


def load_cached_sample(cache_dir, name):
    """Load one cached sample as a dict of numpy arrays (inference helper)."""
    with np.load(os.path.join(cache_dir, name + ".npz")) as d:
        return {"cond": np.asarray(d["cond"], np.float32),
                "c_gt": np.asarray(d["c_gt"], np.float32),
                "meta": {k: d[k] for k in d.files if k not in ("cond", "c_gt")}}
