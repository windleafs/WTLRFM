import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec('jax') and importlib.util.find_spec('optax'), 'JAX environment required')
class WFCGeneralizationTests(unittest.TestCase):
    def test_lowdim_bounds_identity_and_gradient(self):
        import jax
        import jax.numpy as jnp
        from wfc_integration.generalization import corrected_map
        base = jnp.full((8, 12), 1500.)
        u = jnp.zeros((2, 3))
        np.testing.assert_allclose(corrected_map(base, u), base)
        self.assertTrue(np.isfinite(jax.grad(lambda v: corrected_map(base, v).sum())(u)).all())
        self.assertLessEqual(float(corrected_map(base, u+100).max()), 1800)
        self.assertGreaterEqual(float(corrected_map(base, u-100).min()), 1350)

    def test_refinement_improves_unseen_toy_events(self):
        import jax.numpy as jnp
        from data.sos_generalization import event_split
        from wfc_integration.generalization import refine_lowdim
        angles = jnp.linspace(-1, 1, 11)
        def images(c, f):
            phase = (jnp.mean(c)-1520)*.04*f[:, None, None]
            return jnp.broadcast_to(jnp.exp(1j*phase), (len(f), 4, 6))
        splits = event_split(11)
        fields = {key: angles[jnp.asarray(idx)] for key, idx in splits.items()}
        result = refine_lowdim(images, fields, np.ones((4, 6), bool), np.full((4, 6), 1500.),
                               controls_shape=(1, 2), steps=12, learning_rate=.1)
        self.assertGreater(result['metrics']['test']['coherence'], result['initial_metrics']['test']['coherence'])
        self.assertGreater(result['selected_step'], 0)
        self.assertTrue(np.isfinite(result['c_pred']).all())
        with self.assertRaises(ValueError):
            refine_lowdim(images, fields, np.ones((4, 6), bool), np.full((4, 6), 1500.), steps=-1)


if __name__ == '__main__':
    unittest.main()
