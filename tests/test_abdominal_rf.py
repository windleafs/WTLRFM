"""Physics adapter checks without external dataset or GPU requirements."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.abdominal_rf import analytic_iq, build_condition, targets
from data import geometry as G
from models.sos_mult_flow import SoSMultiplicativeFlowNetwork


class RFTests(unittest.TestCase):
    def test_absolute_clock_demodulation(self):
        fc, fs = 3e6, 12e6
        t = 2.3e-6 + np.arange(128) / fs
        rf = np.cos(2 * np.pi * fc * t)[:, None, None]
        iq = analytic_iq(rf, t, fc)
        np.testing.assert_allclose(iq, 1, atol=2e-6)

    def test_target_physical_axes(self):
        dx = .001
        x = (np.arange(64) - 31.5) * dx
        z = np.arange(64) * dx
        xx, zz = np.meshgrid(x - dx / 2, z - dx)
        s = dict(x=x, z=z, c=(1500 + 100 * xx + 200 * zz).astype(np.float32),
                 seg=np.full((64, 64), 5, np.uint8))
        result = targets(s)
        xx, zz = np.meshgrid(G.x_grid(), G.z_grid(), indexing="ij")
        np.testing.assert_allclose(result["c_gt"], 1500 + 100 * xx + 200 * zz, atol=.0002)
        self.assertFalse(result["valid_mask"][:, :10].any())
        self.assertTrue(result["wall_mask"][:, 10:].all())

    def test_point_target_and_no_label_leakage(self):
        torch.set_num_threads(2)
        fs, fc = 12e6, 3e6
        t = np.arange(800) / fs
        xe = np.linspace(-.01, .01, 16)
        angles = np.array([-6., 0., 6.])
        launch = -xe[:, None] * np.sin(np.deg2rad(angles))[None] / 1540
        launch -= launch.min(axis=0)
        distance = np.sqrt(xe ** 2 + .02 ** 2)
        tx = (launch + distance[:, None] / 1500).min(axis=0)
        arrival = distance[:, None] / 1500 + tx[None] + 1 / fc
        dt = t[:, None, None] - arrival[None]
        rf = (np.cos(2 * np.pi * fc * dt) * np.exp(-.5 * (dt / .15e-6) ** 2)).astype(np.float32)
        sample = dict(rf=rf, time=t, fs=fs, fc=fc, xe=xe, angles=angles, launch=launch,
                      meta={"config": {"probe": {"source_cycles": 2}}})
        xi, zi = np.linspace(-.004, .004, 9), np.linspace(.016, .024, 17)
        with patch.object(G, "x_grid", return_value=xi), patch.object(G, "z_grid", return_value=zi):
            cond = build_condition(sample, chunk=64)
            altered = copy.deepcopy(sample)
            altered["c"] = np.array([[9999.]])
            altered["seg"] = np.array([[0]])
            np.testing.assert_array_equal(cond, build_condition(altered, chunk=64))
        mag = np.hypot(cond[2], cond[3])
        peak = np.unravel_index(mag.argmax(), mag.shape)
        self.assertLessEqual(abs(xi[peak[0]]), .001)
        self.assertLessEqual(abs(zi[peak[1]] - .02), .0005)

    def test_masked_loss_ignores_invalid_predictions(self):
        class Dummy(SoSMultiplicativeFlowNetwork):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.register_buffer("_u_scale", torch.tensor(1.))
                self._u_initialised = True
                self.reflow_t_schedule = "uniform"
                self.net = lambda x, t: torch.zeros_like(x[:, :1])
        model = Dummy()
        gt = torch.ones(1, 1, 2, 2)
        cond = torch.zeros_like(gt)
        mask = torch.tensor([[[[True, False], [False, False]]]])
        torch.manual_seed(4)
        first = model(cond, gt, valid_mask=mask)
        gt[~mask] = 100
        torch.manual_seed(4)
        second = model(cond, gt, valid_mask=mask)
        self.assertEqual(first.item(), second.item())
        torch.manual_seed(4)
        weighted = model(cond, gt, valid_mask=mask, background_weight=0.1)
        self.assertGreater(weighted.item(), second.item())
        with self.assertRaises(ValueError):
            model(cond, gt, valid_mask=torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
