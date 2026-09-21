import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch environment required')
class GeometryAcquisitionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(7)

    def structured(self, n_event=7, h=8, w=10):
        torch = self.torch
        from data.rf_geometry import event_geometry, global_geometry, subap_geometry
        rng = np.random.default_rng(3)
        speed_events = torch.tensor(
            rng.normal(size=(3, n_event, h, w)) + 1j*rng.normal(size=(3, n_event, h, w)),
            dtype=torch.complex64)
        subap = torch.tensor(rng.normal(size=(4, h, w)) + 1j*rng.normal(size=(4, h, w)),
                             dtype=torch.complex64)
        angles = np.linspace(-6, 6, n_event)
        refs = np.linspace(0, .5e-6, n_event)
        xe = np.linspace(-.002, .002, 16)
        return {
            'speed_events': speed_events[None],
            'subap': subap[None],
            'event_geom': torch.tensor(event_geometry(angles, refs)[None]),
            'subap_geom': torch.tensor(subap_geometry(xe)[None]),
            'global_geom': torch.tensor(global_geometry(40e6, 7e6, 1500., .5, xe, angles)[None]),
            'event_mask': torch.ones(1, n_event, dtype=torch.bool),
            'subap_mask': torch.ones(1, 4, dtype=torch.bool),
        }

    def test_structured_das_shapes_and_masks(self):
        torch = self.torch
        from data.rf_geometry import das_speed_events, event_geometry
        rf = torch.randn(4, 8, 96)
        xe = np.linspace(-.001, .001, 8)
        angles = np.array([-3., -1., 1., 3.])
        refs = np.array([.2, .1, .1, .2])*1e-6
        xi = np.array([-.001, 0., .001])
        zi = np.array([.001, .002])
        mask = np.array([True, False, True, True])
        out = das_speed_events(rf, xe, angles, refs, xi, zi, 20e6, 5e6,
                               speeds=(1500., 1550.), ref_speed=1500.,
                               n_subap=2, event_mask=mask, chunk=5)
        self.assertEqual(tuple(out['speed_events'].shape), (2, 4, 3, 2))
        self.assertEqual(tuple(out['subap'].shape), (2, 3, 2))
        self.assertTrue(torch.isfinite(out['speed_events'].real).all())
        self.assertTrue((out['speed_events'][:, 1] == 0).all())
        self.assertFalse(bool(out['event_mask'][1]))
        self.assertEqual(event_geometry(angles, refs).shape, (4, 4))
        with self.assertRaises(ValueError):
            das_speed_events(torch.randn(4, 8, 96), xe, angles, refs, xi, zi,
                             20e6, 5e6, event_mask=np.zeros(4, bool))

    def test_encoder_geometry_masks_and_gradients(self):
        torch = self.torch
        from models.acquisition_encoder import AcquisitionConditionEncoder
        encoder = AcquisitionConditionEncoder(n_event_slots=5, canonical_angle_deg=8.,
                                              hidden_channels=8)
        cond = self.structured()
        cond['event_mask'][:, -1] = False
        out, aux = encoder(cond, return_aux=True)
        self.assertEqual(tuple(out.shape), (1, 24, 8, 10))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(tuple(aux['event_weights'].shape), (1, 7, 5))
        changed = dict(cond)
        geom = changed['event_geom'].clone()
        geom[:, :2, 0] = 0.
        geom[:, :2, 1] = 1.
        changed['event_geom'] = geom
        self.assertFalse(torch.allclose(out, encoder(changed)))
        loss = out.square().mean()
        loss.backward()
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        with self.assertRaises(ValueError):
            encoder({**cond, 'event_mask': torch.zeros(1, 7, dtype=torch.bool)})

    def test_rf_augmentation_masks_and_bounded_timing(self):
        torch = self.torch
        from data.rf_augment import _time_shift, augment_rf, validate_augmentation
        rf = torch.zeros(6, 16, 64)
        rf[:, :, 10] = 2.
        shifted = _time_shift(rf, torch.full((6, 1), 2.))
        self.assertEqual(tuple(shifted.shape), tuple(rf.shape))
        self.assertAlmostEqual(float(shifted[0, 0, 12]), 2.)
        self.assertAlmostEqual(float(shifted[0, 0, 10]), 0.)
        cfg = validate_augmentation({
            'time_shift_s': (0., 0.), 'event_jitter_s': (0., 0.),
            'global_gain_db': (1., 1.), 'channel_gain_db': (0., 0.),
            'spectral_gain_db': None, 'noise_snr_db': None,
            'event_dropout_p': .8, 'min_events': 3,
            'receiver_dropout_p': .8, 'min_elements': 8})
        out, meta = augment_rf(torch.randn(6, 16, 64), 20e6,
                               rng=np.random.default_rng(4), cfg=cfg)
        self.assertEqual(tuple(out.shape), (6, 16, 64))
        self.assertEqual(meta['active_events'], 3)
        self.assertEqual(meta['active_elements'], 8)
        self.assertEqual(sum(meta['event_mask']), 3)
        self.assertEqual(sum(meta['element_mask']), 8)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out[~torch.tensor(meta['event_mask'])] == 0).all())
        with self.assertRaises(ValueError):
            validate_augmentation({'event_dropout_p': 1.})

    def test_geometry_flow_smoke_and_checkpoint(self):
        torch = self.torch
        from models.geometry_flow import GeometryAwareSoSFlow
        unet = dict(in_channel=0, out_channel=1, inner_channel=8, norm_groups=4,
                    channel_mults=[1, 2], attn_res=[], res_blocks=1, dropout=0.,
                    image_size=8, cond_channels=0, wtlr_channels=8, wtlr_levels=2,
                    wtlr_wavelet='haar', wtlr_res_blocks=1, wtlr_fusion='gate',
                    use_dwt_resample=False, dwt_wavelet='haar', use_speckle_layer=False)
        model = GeometryAwareSoSFlow(
            unet=unet,
            encoder=dict(n_event_slots=3, n_subap=4, canonical_angle_deg=8.,
                         hidden_channels=4),
            u_source_scale=1., u_clamp=2., velocity_clamp=4.)
        cond = self.structured(n_event=4, h=8, w=8)
        u_gt = torch.randn(1, 1, 8, 8)*.1
        loss = model(cond, u_gt)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
        sample = model.sample(cond, n_steps=2, n_samples=1)
        self.assertEqual(tuple(sample.shape), (1, 1, 1, 8, 8))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'m.pth'
            torch.save(model.checkpoint({'epoch': 1}), path)
            restored = GeometryAwareSoSFlow.from_checkpoint(path)
            self.assertAlmostEqual(restored.sigma_u, model.sigma_u)
            self.assertEqual(restored.encoder.out_channels, model.encoder.out_channels)


if __name__ == '__main__':
    unittest.main()
