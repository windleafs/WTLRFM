import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.acquisition import audit_acquisition, validate_acquisition, fit_pulse_echo_timing
from data.sos_generalization import candidate_maps, event_split, score_images


class GeneralizationTests(unittest.TestCase):
    def acquisition(self):
        return dict(rf=np.ones((7, 8, 128), np.float32), xe=np.linspace(-.001, .001, 8),
                    angles_deg=np.arange(-3, 4.), tx_t_ref_s=np.zeros(7), fs_hz=20e6,
                    fc_hz=5e6, t0_s=0., c_steer=1540., bandwidth_fraction=.6,
                    metadata={})

    def test_audit_detects_same_shape_different_angles(self):
        a = self.acquisition()
        report = audit_acquisition(a, dict(angles_deg=np.linspace(-8, 8, 7), fc_hz=7.5e6))
        self.assertIn('angles_deg', report['mismatches'])
        self.assertIn('fc_hz', report['mismatches'])
        self.assertFalse(report['hardware_calibrated'])
        with self.assertRaises(ValueError):
            validate_acquisition({**a, 'tx_t_ref_s': np.zeros(6)})
        with self.assertRaises(ValueError):
            validate_acquisition({**a, 'fc_hz': 11e6})

    def test_multidepth_timing(self):
        z = np.array([.01, .02, .035, .04])
        fit = fit_pulse_echo_timing(z, .7e-6 + 2*z/1480)
        self.assertAlmostEqual(fit['speed_m_s'], 1480, places=6)
        self.assertAlmostEqual(fit['time_offset_s'], .7e-6, places=12)
        with self.assertRaises(ValueError):
            fit_pulse_echo_timing(np.ones(4), np.arange(4.))

    def test_candidates_have_exact_endpoints_and_physical_smoothing(self):
        x, z = np.arange(12)*.001, np.arange(16)*.0005
        c = np.full((12, 16), 1500.)
        c[6] = 1600
        maps = candidate_maps(c, x, z, 1520., sigmas_mm=[1.], alphas=[0., 1.])
        np.testing.assert_array_equal(maps['flow'], c)
        np.testing.assert_allclose(maps['smooth1_alpha0'], 1520)
        self.assertLess(maps['smooth1_alpha1'].max(), c.max())
        self.assertTrue(all(np.isfinite(v).all() and (v > 0).all() for v in maps.values()))
        with self.assertRaises(ValueError):
            candidate_maps(-c, x, z, 1520.)

    def test_event_split_and_coherence(self):
        split = event_split(11)
        sets = [set(v) for v in split.values()]
        self.assertEqual(set.union(*sets), set(range(11)))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertTrue(all(len(s) >= 2 for s in sets))
        image = np.ones((11, 4, 6), np.complex64)
        metrics = score_images(image, np.ones((4, 6), bool), split)
        self.assertEqual(metrics['test']['coherence'], 1.)
        with self.assertRaises(ValueError):
            score_images(image*0, np.ones((4, 6), bool), split)
        with self.assertRaises(ValueError):
            event_split(5)


if __name__ == '__main__':
    unittest.main()
