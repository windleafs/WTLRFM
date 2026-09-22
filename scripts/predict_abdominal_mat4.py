"""Zero-shot GeometryFlow prediction on AbdominalMap4 full-matrix capture.

Synthesises 11-angle plane-wave RF (the training grid, +/-8 deg) from the FMC
`scat` array by coherent delayed summation, builds the structured acquisition
condition with the exact conventions of the geometry-flow cache, and runs the
breast-trained GeometryFlow checkpoint.  Produces a B-mode / SoS-overlay /
posterior-std / ground-truth figure in the reference-panel style.

Timing convention (self-consistent by construction): element i fires at
launch_i = x_i*sin(theta)/1500 - min_j(...), so the equivalent plane wave has
t_ref(theta) = -min_i(x_i*sin(theta))/1500 in the DAS model
t_tx = t_ref + (x*sin(theta) + z*cos(theta))/1500.  The steering sign of the
stored FMC is resolved empirically by picking the sign whose compounded
B-mode is sharpest.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import geometry as G
from data.rf_geometry import structured_condition
from models.geometry_flow import GeometryAwareSoSFlow
from scripts.train_geometry_flow import grid_metrics

C_DELAY = 1500.          # assumed speed for synthesis delays and DAS steering


def delayed_sum(rf, launch, fs):
    """y[t] = sum_tx rf[rx, tx, t - launch_tx] via FFT fractional delays."""
    nt = rf.shape[-1]
    length = next_fast_len(nt + int(np.ceil(launch.max()*fs)) + 32)
    freq = rfftfreq(length, 1/fs)
    spectrum = rfft(rf, n=length, axis=-1)
    out = np.empty((rf.shape[0], launch.shape[1], length), np.float32)
    for a in range(launch.shape[1]):
        ramp = np.exp(-2j*np.pi*launch[:, a, None]*freq[None]).astype(np.complex64)
        out[:, a] = irfft(np.einsum('rtf,tf->rf', spectrum, ramp), n=length)
    return out


def synthesize(f_scat, xe, angles, fs, sign, block=32):
    launch = sign*xe[:, None]*np.sin(np.deg2rad(angles))[None]/C_DELAY
    launch -= launch.min(axis=0, keepdims=True)
    n_rx = f_scat.shape[1]
    pw = None
    for start in range(0, n_rx, block):             # scat stored [tx, rx, t]
        rf = f_scat[:, start:start+block, :].astype(np.float32).transpose(1, 0, 2)
        out = delayed_sum(rf, launch, fs)           # [rx-block, angle, time]
        if pw is None:
            pw = np.empty((n_rx, out.shape[1], out.shape[-1]), np.float32)
        pw[start:start+block] = out
    return pw, launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default=str(ROOT/'dataAbdominal/AbdominalMap4.mat'))
    p.add_argument('--scat-key', default='scat',
                   help='FMC dataset name inside the MAT (e.g. '
                        'fsr_dataset_fund for the L12-5 capture)')
    p.add_argument('--ckpt', default=str(ROOT/'out/geometry_flow_v2_20260921/best.pth'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--fc', type=float, default=0.,
                   help='carrier frequency; 0 = estimate from the FMC spectrum')
    p.add_argument('--n-samples', type=int, default=16)
    p.add_argument('--burst-us', type=float, default=0.,
                   help='constant added to t_ref (use 4/fc to match the '
                        'training burst-centre convention)')
    p.add_argument('--ode-steps', type=int, default=10)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory')
    args.out.mkdir(parents=True)
    started = time.monotonic()
    angles = np.linspace(-8., 8., 11)

    with h5py.File(args.data, 'r') as f:
        t = np.array(f['time']).ravel()
        fs = float(1/np.mean(np.diff(t)))
        xe = np.array(f['rxAptPos'])[0]
        if not np.all(np.diff(xe) > 0):
            raise ValueError('Receiver coordinates must be ascending')
        truth = xt = zt = None
        if 'C' in f:
            truth = np.array(f['C'], dtype=np.float64)
            xt = np.array(f['x']).ravel()
            zt = np.array(f['z']).ravel()
            if truth.shape != (xt.size, zt.size):
                raise ValueError('Expected C [x, z]')
        # carrier + fractional bandwidth from the mean amplitude spectrum
        probe = np.array(f[args.scat_key][0, :16, :], dtype=np.float64).ravel()
        spec = np.abs(np.fft.rfft(probe))
        fr = np.fft.rfftfreq(probe.size, 1/fs)
        fc = args.fc if args.fc > 0 else float(fr[np.argmax(spec)])
        band = fr[spec > spec.max()/2.]
        bw_fraction = float((band[-1]-band[0])/(2*fc))
        print(f'[acq] fc={fc/1e6:.3f} MHz bw_fraction={bw_fraction:.3f} '
              f'pitch={np.mean(np.diff(xe))*1e3:.4f} mm n_el={len(xe)}', flush=True)
        # steering-sign check: an off-axis event focused with the correct
        # convention keeps the speckle peak-to-mean ratio of the 0-deg event;
        # the wrong sign smears the two-way PSF and drops the ratio.
        def focus(env):
            return float(env.max()/env.mean())
        sharp = {}
        for sign in (+1., -1.):
            pw, _ = synthesize(f[args.scat_key], xe, np.array([0., 7.2]), fs, sign)
            rf0 = np.ascontiguousarray(pw.transpose(1, 0, 2))  # [angle, rx, t]
            t_ref_check = np.array(
                [0., -np.min(sign*xe*np.sin(np.deg2rad(7.2))/C_DELAY)])
            cond0 = structured_condition(
                rf0, xe, np.array([0., 7.2]), t_ref_check,
                G.x_grid(), G.z_grid(),
                fs, fc, C_DELAY, bw_fraction, t0_s=float(t[0]), chunk=8192)
            ev = np.abs(cond0['speed_events'][1].numpy())
            sharp[sign] = focus(ev[1])/focus(ev[0])
        sign = max(sharp, key=sharp.get)
        print(f'[sign] sharpness +:{sharp[1.]:.3f} -:{sharp[-1.]:.3f} -> using sign {sign:+.0f}',
              flush=True)
        pw, launch = synthesize(f[args.scat_key], xe, angles, fs, sign)
    rf = np.ascontiguousarray(pw.transpose(1, 0, 2))          # [11, rx, time]
    # plane-wave launch anchor: t_ref = -min_i(sign * x_i * sin(theta) / c)
    t_ref = np.asarray([-np.min(sign*xe*np.sin(np.deg2rad(a))/C_DELAY)
                        for a in angles]) + args.burst_us*1e-6
    print(f'[rf] {rf.shape} fs={fs:.4e} t_ref_us={np.round(t_ref*1e6, 3).tolist()}',
          flush=True)

    device = torch.device(args.device)
    cond = structured_condition(
        rf, xe, angles, t_ref, G.x_grid(), G.z_grid(), fs, fc,
        C_DELAY, bw_fraction, t0_s=float(t[0]), chunk=4096)
    condition = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in cond.items()}
    condition = {k: v[None] if torch.is_tensor(v) else v for k, v in condition.items()}
    cond_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in condition.items()}

    model = GeometryAwareSoSFlow(
        unet=torch.load(args.ckpt, map_location='cpu', weights_only=False)['unet'],
        encoder=torch.load(args.ckpt, map_location='cpu',
                           weights_only=False)['encoder_cfg'],
        u_source_scale=-1.).to(device).eval()
    blob = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(blob['state_dict'])
    torch.manual_seed(20260922)
    with torch.no_grad():
        draws = model.sample(condition, n_steps=args.ode_steps,
                             n_samples=args.n_samples)
    pred = draws.mean(0)[0, 0].cpu().numpy()
    std = draws.std(0, unbiased=False)[0, 0].cpu().numpy()
    if not (np.isfinite(pred).all() and np.isfinite(std).all()):
        raise ValueError('Nonfinite model output')

    xi, zi = G.x_grid().astype(np.float64), G.z_grid().astype(np.float64)
    xx, zz = np.meshgrid(xi, zi, indexing='ij')
    if truth is not None:
        # truth is C[ix, iz]; interpolate z first per x-column, then x.
        step_z = np.stack([np.interp(zi, zt, truth[ix, :])
                           for ix in range(truth.shape[0])], axis=0)
        gt = np.stack([np.interp(xi, xt, step_z[:, iz])
                       for iz in range(len(zi))], axis=1)
        support = ((xx >= xt.min()) & (xx <= xt.max())
                   & (zz >= zt.min()) & (zz <= zt.max()))
        roi = support & (zz >= 3e-3) & (zz <= 45e-3)
        metrics = grid_metrics(pred, gt, dict(x0_m=xi[0], x1_m=xi[-1],
                                              z0_m=zi[0], z1_m=zi[-1],
                                              nx=len(xi), nz=len(zi)))
        masked = {}
        for name, est in (('model', pred), ('constant_1540', np.full_like(gt, 1540.))):
            a, b = est[roi], gt[roi]
            ac, bc = a-a.mean(), b-b.mean()
            masked[name] = dict(mae=float(np.abs(a-b).mean()),
                                bias=float((a-b).mean()),
                                corr=float((ac*bc).sum()
                                           / (np.linalg.norm(ac)*np.linalg.norm(bc)+1e-12)),
                                mean_pred=float(a.mean()), mean_gt=float(b.mean()))
        print(json.dumps({'grid_metrics': metrics, 'roi_masked': masked}, indent=2),
              flush=True)
    else:
        gt = None
        roi = (zz >= 3e-3) & (zz <= 45e-3)
        metrics = masked = None
        print('[gt] none stored; prediction-only run', flush=True)

    # --- figures
    env = np.abs(cond_cpu['speed_events'].numpy()[0, 1].sum(axis=0))  # compound
    bmode = 20*np.log10(np.maximum(env/np.percentile(env[env > 0], 98), 1e-3))
    if gt is not None:
        lo, hi = int(np.floor(min(pred[roi].min(), gt[roi].min())/10.)*10), \
                 int(np.ceil(max(pred[roi].max(), gt[roi].max())/10.)*10)
    else:
        lo = int(np.floor(np.percentile(pred[roi], 0.5)/10.)*10)
        hi = int(np.ceil(np.percentile(pred[roi], 99.5)/10.)*10)
    extent = [xi[0]*1e3, xi[-1]*1e3, zi[-1]*1e3, zi[0]*1e3]

    panels = [
        ('B-mode (11-angle compound)', bmode, 'gray', None, None),
        ('SoS prediction', pred, 'jet', lo, hi),
        ('SoS overlay on B-mode', None, 'jet', lo, hi),
    ]
    if gt is not None:
        panels.append(('Ground truth', np.where(support, gt, np.nan), 'jet', lo, hi))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8*len(panels), 4.4))
    for ax, (title, img, cmap, vlo, vhi) in zip(axes, panels):
        if img is not None:
            im = ax.imshow(img.T, cmap=cmap, vmin=vlo, vmax=vhi,
                           extent=extent, aspect='equal')
        else:
            ax.imshow(bmode.T, cmap='gray', vmin=-45, vmax=0,
                      extent=extent, aspect='equal')
            ov = np.ma.masked_where(~roi, pred)
            im = ax.imshow(ov.T, cmap='jet', vmin=vlo, vmax=vhi,
                           extent=extent, aspect='equal', alpha=.45)
        if cmap == 'jet':
            cb = fig.colorbar(im, ax=ax, shrink=.85)
            cb.set_label('Speed of sound [m/s]')
        ax.set(title=title, xlabel='Lateral [mm]', ylabel='Depth [mm]')
    fig.tight_layout()
    fig.savefig(args.out/'abdominalmap4_geometryflow.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(std.T, cmap='inferno', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='posterior std [m/s]')
    ax.set(title='Posterior std (16 samples)', xlabel='Lateral [mm]',
           ylabel='Depth [mm]')
    fig.tight_layout()
    fig.savefig(args.out/'abdominalmap4_uncertainty.png', dpi=160)
    plt.close(fig)

    np.savez_compressed(args.out/'prediction.npz', c_pred=pred, c_std=std,
                        **({} if gt is None else dict(c_gt=gt, support=support)),
                        roi=roi, bmode_db=bmode,
                        cx=xi, cz=zi, angles_deg=angles,
                        rf_plane_wave=rf.astype(np.float32))
    meta = dict(source=str(Path(args.data).resolve()), ckpt=str(Path(args.ckpt).resolve()),
                fs_hz=fs, fc_hz=fc, scat_key=args.scat_key,
                bw_fraction=bw_fraction,
                c_delay_mps=C_DELAY, steering_sign=float(sign),
                burst_offset_us=args.burst_us,
                sharpness_check=sharp, t0_s=float(t[0]),
                angles_deg=angles.tolist(), n_samples=args.n_samples,
                ode_steps=args.ode_steps, grid_metrics=metrics,
                roi_masked=masked,
                note=('Zero-shot cross-phantom transfer: model trained on '
                      'numerical breast phantoms (192 el / 38 mm), applied to '
                      'abdominal FMC (128 el / 25.4 mm) with synthesized '
                      'plane waves. GT support limits x to +/-17.8 mm.'))
    (args.out/'summary.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(f"[done] {time.monotonic()-started:.1f}s -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
