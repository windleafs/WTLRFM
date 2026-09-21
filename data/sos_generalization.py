import numpy as np
from scipy.ndimage import gaussian_filter


def validate_map(c, x, z):
    c, x, z = np.asarray(c), np.asarray(x), np.asarray(z)
    if c.shape != (len(x), len(z)) or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError('Speed must be finite, positive [x,z] matching its coordinates')
    for axis in (x, z):
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
            raise ValueError('Map axes must be finite and increasing')
        if not np.allclose(np.diff(axis), np.diff(axis).mean(), rtol=1e-4, atol=1e-9):
            raise ValueError('Smoothing requires uniformly spaced physical axes')


def candidate_maps(c, x, z, base_speed, sigmas_mm=(1., 2., 4.), alphas=(.25, .5, 1.)):
    validate_map(c, x, z)
    if not np.isfinite(base_speed) or base_speed <= 0:
        raise ValueError('Base speed must be finite and positive')
    c = np.asarray(c, np.float32)
    spacing = np.array([np.diff(x).mean(), np.diff(z).mean()])*1000
    maps = {'flow': c.copy(), 'flow_mean': np.full_like(c, c.mean()),
            'constant1540': np.full_like(c, 1540.), 'scalar_fit': np.full_like(c, base_speed)}
    for sigma in sigmas_mm:
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError('Smoothing scale must be nonnegative millimetres')
        residual = gaussian_filter(np.log(c/base_speed), sigma/spacing, mode='nearest')
        for alpha in alphas:
            if not np.isfinite(alpha) or not 0 <= alpha <= 1:
                raise ValueError('Shrinkage must be in [0,1]')
            maps[f'smooth{sigma:g}_alpha{alpha:g}'] = (base_speed*np.exp(alpha*residual)).astype(np.float32)
    return maps


def event_split(n):
    if n < 6:
        raise ValueError('At least six events required for optimization/validation/test coherence')
    indices = np.arange(n)
    test = indices[::4]
    remaining = np.setdiff1d(indices, test)
    val = remaining[::3]
    train = np.setdiff1d(remaining, val)
    if min(len(train), len(val), len(test)) < 2:
        raise ValueError('Each event partition requires at least two angles')
    return {'train': train.tolist(), 'val': val.tolist(), 'test': test.tolist()}


def score_images(images, roi, splits):
    images, roi = np.asarray(images), np.asarray(roi, bool)
    if images.ndim != 3 or roi.shape != images.shape[1:] or not roi.any() or not np.isfinite(images).all():
        raise ValueError('Invalid complex images or ROI')
    result = {}
    for key, indices in splits.items():
        selected = images[np.asarray(indices)][:, roi]
        inc = np.abs(selected).sum(0).mean()
        if not np.isfinite(inc) or inc <= 0:
            raise ValueError('No measured energy in evaluation ROI')
        env = np.abs(selected.sum(0))
        result[key] = {'coherence': float(env.mean()/inc), 'mean_envelope': float(env.mean())}
    return result
