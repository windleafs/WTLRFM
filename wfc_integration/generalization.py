import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from data.sos_generalization import event_split


def build_model(acquisition, x, z, backend, dz_mm=.4, depth_mm=40., tx_window='flat'):
    backend = Path(backend).resolve()
    sys.path.insert(0, str(backend))
    from rf_imaging import create_rf_model, sample_speed
    from wfc import WFCConfig

    a = acquisition
    fov = (max(float(x[0]), float(a['xe'][0])), min(float(x[-1]), float(a['xe'][-1])),
           min(float(z[-1]), depth_mm*1e-3))
    if dz_mm <= 0 or fov[0] >= fov[1] or fov[2] <= .005:
        raise ValueError('Invalid imaging resolution or common field of view')
    cfg = WFCConfig(c0=float(a['c_steer']), nf=0, bw_frac=float(a['bandwidth_fraction'])/2,
                    pad_x=.012, dz=dz_mm*1e-3, dz_img=dz_mm*1e-3, ncx=64, ncz=80)
    rf = a['rf'] * np.asarray(a.get('tgc_gain', 1.))
    model = create_rf_model(rf, a['xe'], a['angles_deg'], float(a['fs_hz']), float(a['fc_hz']),
                            a['tx_t_ref_s'], cfg, fov, t0=float(a['t0_s']), aperture='interpolated')
    splits = event_split(len(a['angles_deg']))
    if tx_window == 'flat':
        model.Wfull = jnp.full((model.ne,), np.hanning(model.ne).mean(), dtype=jnp.float32)
    spectrum = np.abs(np.asarray(model.D)[:, splits['train']]).mean(axis=(0, 1))
    model.set_source_spectrum(spectrum)
    xx, zz = np.meshgrid(np.asarray(model.xg), np.asarray(model.zg), indexing='ij')
    roi = (xx >= fov[0]) & (xx <= fov[1]) & (zz >= .005) & (zz <= fov[2])
    if not roi.any():
        raise ValueError('Empty physical evaluation ROI')
    protocol = dict(backend=str(backend), dz_mm=dz_mm, fov_m=list(fov),
                    speed_grid=[64, 80], nf=int(model.nf),
                    frequency_band_hz=[float(model.fphys.min()), float(model.fphys.max())],
                    transmit_window_assumed=tx_window, source_spectrum='magnitude estimated from optimization events only',
                    source_waveform_calibrated=False, outside_map='nearest boundary extension',
                    objective='angle coherence proxy, not quantitative sound-speed accuracy')
    project = lambda c: sample_speed(model, np.asarray(c).T, x, z)
    return model, project, roi, splits, protocol


def subset_fields(model, indices):
    fields = model.f0_img.reshape(model.nang, 2, model.nf, model.nx)
    return fields[jnp.asarray(indices)].reshape(-1, model.nf, model.nx)


def coherence(images, roi):
    mask = jnp.asarray(roi, jnp.float32)
    num = jnp.sum(jnp.abs(jnp.sum(images, axis=0))*mask)
    den = jnp.sum(jnp.sum(jnp.abs(images), axis=0)*mask)
    return num / jnp.maximum(den, 1e-20), den / jnp.sum(mask)


def corrected_map(base, controls, max_log_change=.05):
    residual = jax.image.resize(jnp.tanh(controls), base.shape, method='linear')
    return jnp.clip(base*jnp.exp(max_log_change*residual), 1350., 1800.)


def refine_lowdim(image_fn, fields, roi, base, controls_shape=(2, 6), steps=30,
                  learning_rate=.05, prior_weight=.002, smooth_weight=.002,
                  max_log_change=.05, callback=None):
    import optax

    base = jnp.asarray(base, jnp.float32)
    roi = np.asarray(roi, bool)
    values = [learning_rate, prior_weight, smooth_weight, max_log_change]
    if (steps < 0 or len(controls_shape) != 2 or min(controls_shape) < 1 or
            not np.isfinite(values).all() or min(prior_weight, smooth_weight) < 0 or
            learning_rate <= 0 or max_log_change <= 0):
        raise ValueError('Invalid low-dimensional optimization settings')
    if not np.isfinite(base).all() or np.min(base) < 1350 or np.max(base) > 1800 or not roi.any():
        raise ValueError('Invalid initial speed map or ROI')
    if set(fields) != {'train', 'val', 'test'}:
        raise ValueError('Need separate optimization, validation and test fields')
    controls = jnp.zeros(controls_shape, dtype=jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(learning_rate))
    state = optimizer.init(controls)

    @jax.jit
    def metrics(c, f):
        return coherence(image_fn(c, f), roi)

    initial = {key: tuple(float(v) for v in metrics(base, fields[key])) for key in ('train', 'val')}
    if any(not np.isfinite(v).all() or v[1] <= 0 for v in initial.values()):
        raise ValueError('Initial map has no finite measured energy')

    def loss(u):
        c = corrected_map(base, u, max_log_change)
        cf, energy = metrics(c, fields['train'])
        r = jnp.tanh(u)
        smooth = sum(jnp.mean(jnp.diff(r, axis=i)**2) for i in (0, 1) if r.shape[i] > 1)
        penalty = prior_weight*jnp.mean(r*r) + smooth_weight*smooth
        energy_penalty = jnp.maximum(.1-energy/initial['train'][1], 0.)**2
        return -cf+penalty+energy_penalty, (cf, energy)

    value_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    best, best_score, best_step = controls, initial['val'][0], 0
    history = [{'step': 0, 'train_coherence': initial['train'][0], 'val_coherence': best_score}]
    for step in range(1, steps+1):
        (value, _), grad = value_grad(controls)
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(grad)).all():
            raise FloatingPointError('Nonfinite physical loss or gradient')
        updates, state = optimizer.update(grad, state, controls)
        controls = optax.apply_updates(controls, updates)
        current = corrected_map(base, controls, max_log_change)
        tr, te = (float(v) for v in metrics(current, fields['train']))
        va, ve = (float(v) for v in metrics(current, fields['val']))
        if not np.isfinite([tr, te, va, ve]).all():
            raise FloatingPointError('Nonfinite optimized image metrics')
        if va > best_score and ve >= .1*initial['val'][1] and te >= .1*initial['train'][1]:
            best, best_score, best_step = controls, va, step
        row = {'step': step, 'loss_before_update': float(value), 'train_coherence': tr,
               'val_coherence': va, 'selected_step': best_step}
        history.append(row)
        if callback:
            callback(row)
    chosen = corrected_map(base, best, max_log_change)
    initial['test'] = tuple(float(v) for v in metrics(base, fields['test']))
    final = {key: tuple(float(v) for v in metrics(chosen, f)) for key, f in fields.items()}
    pack = lambda d: {k: {'coherence': v[0], 'incoherent_energy': v[1]} for k, v in d.items()}
    return dict(c_pred=np.asarray(chosen), controls=np.asarray(best), selected_step=best_step,
                initial_metrics=pack(initial), metrics=pack(final), history=history)
