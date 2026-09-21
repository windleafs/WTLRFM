"""Focused cache, masked metrics, deterministic evaluation and resume tests."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import geometry as G
from data.abdominal_dataset import (AbdominalSoSDataset, SCHEMA, SHAPE, aggregate_metrics,
                                    cache_fingerprint, masked_metrics, sample_metrics, training_mean_map)
from engine import EMA, set_seed
from scripts import train_abdominal as train


class TinyFlow(torch.nn.Module):
    def __init__(self, unet=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self._u_steps = 7
        self._u_initialised = True
        self._u_ema = .4
        self.cfg = unet or {}

    def forward(self, cond, target, valid_mask=None, background_weight=0.0):
        assert valid_mask is not None and valid_mask.dtype == torch.bool
        error = (self.weight + torch.randn_like(target) - target) ** 2
        loss = error[valid_mask].mean()
        if (~valid_mask).any():
            loss = loss + background_weight * error[~valid_mask].mean()
        return loss

    def sample(self, cond, n_steps=1, n_samples=1):
        return 1500 + self.weight + torch.randn(n_samples, len(cond), 1, *SHAPE, device=cond.device)

    def checkpoint(self, extra=None):
        return {"state_dict": self.state_dict(), "cfg": self.cfg, **(extra or {})}


class AbdominalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "cache"
        self.root.mkdir()
        self.meta = {"schema": SCHEMA, "config": {}, "cx": G.x_grid().tolist(), "cz": G.z_grid().tolist()}
        self.splits = {"train": ["liver_pw_000001"], "val": ["liver_pw_000002"], "test": ["liver_pw_000003"]}
        (self.root / "meta.json").write_text(json.dumps(self.meta))
        (self.root / "splits.json").write_text(json.dumps(self.splits))
        valid = np.zeros(SHAPE, bool)
        valid[10:100, 10:120] = True
        self.arrays = {"cond": np.zeros((20, *SHAPE), np.float16), "c_gt": np.full(SHAPE, 1520, np.float32),
                       "valid_mask": valid, "wall_mask": valid.copy(), "segmentation": (valid * 2).astype(np.uint8),
                       "cx": G.x_grid(), "cz": G.z_grid(), "meta": json.dumps({"sample_id": 1})}
        self.write_sample("liver_pw_000001")
        self.write_sample("liver_pw_000002")
        # Deliberately no test NPZ: training must never open it.

    def tearDown(self):
        self.temp.cleanup()

    def write_sample(self, name, **overrides):
        np.savez(self.root / f"{name}.npz", **{**self.arrays, **overrides})

    def test_dataset_normalization_and_contract(self):
        dataset = AbdominalSoSDataset(self.root)
        batch = dataset[0]
        self.assertEqual(tuple(batch["cond"].shape), (20, *SHAPE))
        self.assertEqual(batch["cond"].dtype, torch.float32)
        self.assertEqual(batch["valid_mask"].dtype, torch.bool)
        np.testing.assert_allclose(batch["u_gt"], G.c_to_u(self.arrays["c_gt"])[None], rtol=1e-6)
        self.assertEqual(dataset.names, self.splits["train"])
        for changes in ({"cond": np.zeros((19, *SHAPE), np.float16)},
                        {"valid_mask": np.zeros(SHAPE, bool)},
                        {"wall_mask": np.zeros(SHAPE, bool)}, {"cx": G.x_grid() + .001},
                        {"c_gt": np.full(SHAPE, np.nan, np.float32)}, {"meta": "{}"}):
            with self.subTest(changes=list(changes)):
                self.write_sample(dataset.names[0], **changes)
                with self.assertRaises(ValueError):
                    dataset[0]

    def test_split_and_root_validation(self):
        self.splits["test"] = self.splits["train"]
        (self.root / "splits.json").write_text(json.dumps(self.splits))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            AbdominalSoSDataset(self.root)
        self.meta["schema"] = "wrong"
        (self.root / "meta.json").write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "schema"):
            AbdominalSoSDataset(self.root)

    def test_masked_metrics_and_pooling(self):
        truth = np.zeros(SHAPE, np.float32)
        pred = np.full(SHAPE, 100, np.float32)
        valid = np.zeros(SHAPE, bool)
        valid[0, :2] = True
        pred[0, :2] = [3, 4]
        values = masked_metrics(pred, truth, valid)
        self.assertEqual(values["mae"], 3.5)
        self.assertAlmostEqual(values["rmse"], np.sqrt(12.5))
        self.assertIsNone(masked_metrics(pred, truth, np.zeros(SHAPE, bool))["mae"])
        row = sample_metrics(pred, truth, valid, valid)
        pooled = aggregate_metrics([row, row])
        self.assertEqual(pooled["tissue_pixels"], 4)
        self.assertEqual(pooled["tissue_mae"], 3.5)

    def test_train_only_baseline_and_fingerprint(self):
        dataset = AbdominalSoSDataset(self.root)
        mean, count = training_mean_map(dataset)
        np.testing.assert_allclose(mean, 1520)
        self.assertEqual(count.max(), 1)
        with self.assertRaisesRegex(ValueError, "train names"):
            training_mean_map(AbdominalSoSDataset(self.root, "val"))
        first = cache_fingerprint(dataset.root, dataset.meta, dataset.splits)
        self.write_sample(dataset.names[0], c_gt=np.full(SHAPE, 1540, np.float32))
        self.assertNotEqual(first, cache_fingerprint(dataset.root, dataset.meta, dataset.splits))

    def test_eval_rng_and_population_std(self):
        loader = DataLoader(AbdominalSoSDataset(self.root, "val"), batch_size=1)
        set_seed(12)
        rng = torch.get_rng_state().clone()
        first = list(train.prediction_batches(TinyFlow(), loader, torch.device("cpu"), 1, 1))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        second = list(train.prediction_batches(TinyFlow(), loader, torch.device("cpu"), 1, 1))
        np.testing.assert_array_equal(first[0][1], second[0][1])
        np.testing.assert_array_equal(first[0][2], 0)

    def test_resume_restores_optimizer_rng_and_prior(self):
        model = TinyFlow()
        ema = EMA(model, .99)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        model.weight.square().backward()
        optimizer.step()
        ema.update(model)
        scheduler.step()
        cfg, meta = {"train": {"n_epoch": 5}}, {"cache_fingerprint": "abc"}
        path = Path(self.temp.name) / "state.pth"
        train.save_training_state(path, model, ema, optimizer, scheduler, 1, 3.5, cfg, meta)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        original = model.weight.detach().clone()
        model.weight.data.fill_(9)
        model._u_steps = 99
        epoch, best = train.load_training_state(path, model, ema, optimizer, scheduler, cfg, meta)
        self.assertEqual((epoch, best, model._u_steps), (1, 3.5, 7))
        self.assertTrue(torch.equal(original, model.weight))
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            train.load_training_state(path, model, ema, optimizer, scheduler, cfg, {"cache_fingerprint": "changed"})

    def test_interrupted_resume_matches_uninterrupted(self):
        full = Path(self.temp.name) / "full"
        interrupted = Path(self.temp.name) / "interrupted"
        args = ["--cache", str(self.root), "--device", "cpu", "--epochs", "3", "--batch", "1"]
        real_save = train.save_training_state

        def interrupt_after_save(*values, **kwargs):
            real_save(*values, **kwargs)
            raise RuntimeError("simulated interruption")

        with mock.patch.object(train, "SoSMultiplicativeFlowNetwork", TinyFlow):
            train.main(args + ["--out", str(full)])
            with mock.patch.object(train, "save_training_state", side_effect=interrupt_after_save):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    train.main(args + ["--out", str(interrupted)])
            train.main(args + ["--out", str(interrupted), "--resume"])
        a = torch.load(full / "training_state.pth", weights_only=False)
        b = torch.load(interrupted / "training_state.pth", weights_only=False)
        for key in ("model", "ema"):
            for name in a[key]:
                self.assertTrue(torch.equal(a[key][name], b[key][name]))
        self.assertEqual(a["scheduler"], b["scheduler"])
        self.assertEqual(a["best"], b["best"])
        self.assertTrue(torch.equal(a["rng"]["torch"], b["rng"]["torch"]))

    def test_prediction_cli_exports(self):
        from scripts import predict_abdominal as predict
        self.write_sample("liver_pw_000003")
        dataset = AbdominalSoSDataset(self.root)
        ckpt = Path(self.temp.name) / "best.pth"
        meta = {"cache_meta": self.meta, "train_names": self.splits["train"],
                "normalization": {"c_ref": G.C_REF, "rho_scale": G.RHO_SCALE},
                "cache_fingerprint": cache_fingerprint(dataset.root, dataset.meta, dataset.splits)}
        torch.save(TinyFlow().checkpoint({"data_meta": meta, "epoch": 1}), ckpt)
        out = Path(self.temp.name) / "predictions"
        with mock.patch.object(predict.SoSMultiplicativeFlowNetwork, "from_checkpoint", return_value=TinyFlow()):
            predict.main(["--ckpt", str(ckpt), "--cache", str(self.root), "--out", str(out),
                          "--device", "cpu", "--n-samples", "1", "--ode-steps", "1", "--plot-limit", "0"])
        with np.load(out / "cmaps/liver_pw_000003.npz") as arrays:
            self.assertTrue(np.isfinite(arrays["pred"]).all())
            np.testing.assert_array_equal(arrays["std"], 0)
            np.testing.assert_array_equal(arrays["c_pred"], arrays["pred"])
            np.testing.assert_array_equal(arrays["c_std"], arrays["std"])
            self.assertEqual(arrays["truth"].shape, SHAPE)
            self.assertEqual(json.loads(str(arrays["meta"].item()))["std_ddof"], 0)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["train_mean_names"], self.splits["train"])
        self.assertEqual(set(summary["metrics"]), {"model", "constant1500", "constant1540", "train_mean"})

    def test_training_cli_resume_no_test_access_or_overwrite(self):
        out = Path(self.temp.name) / "run"
        args = ["--cache", str(self.root), "--out", str(out), "--device", "cpu", "--epochs", "1", "--batch", "1"]
        with mock.patch.object(train, "SoSMultiplicativeFlowNetwork", TinyFlow):
            train.main(args)
            with self.assertRaises(FileExistsError):
                train.main(args)
            train.main(args + ["--epochs", "2", "--resume"])
        self.assertEqual({p.name for p in out.glob("*.pth")}, {"best.pth", "last.pth", "training_state.pth"})
        state = torch.load(out / "training_state.pth", weights_only=False)
        self.assertEqual(state["epoch"], 2)
        summary = json.loads((out / "validation_summary.json").read_text())
        self.assertEqual(summary["split"], "val")
        self.assertEqual(summary["train_mean_names"], self.splits["train"])
        self.assertIn("train_mean", summary["metrics"])


if __name__ == "__main__":
    unittest.main()
