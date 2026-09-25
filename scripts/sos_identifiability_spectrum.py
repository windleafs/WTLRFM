"""Estimate a low-dimensional SoS identifiability spectrum for Geometry-Flow.

Two-stage workflow:
1) --stage sim rebuilds selected dataset media with the same scatterer
   realization and acquisition, perturbs sound speed by +/- delta_c along a
   compact physical basis, and re-runs UltraWave.
2) --stage analyze converts paired RF files into the exact structured
   conditions used by Geometry-Flow and accumulates dataset-averaged Gram
   matrices for RF, full, event, subap and optionally encoder features.

With --ckpt, the analysis also finite-differences the model posterior mean.
This is a low-dimensional local analysis, not the full image Jacobian nullspace.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def basis_kwargs(args):
    return dict(z_min=args.z_min_mm * 1e-3,
                z_max=args.z_max_mm * 1e-3,
                depth_slabs=args.depth_slabs,
                lateral_modes=args.lateral_modes,
                axial_modes=args.axial_modes,
                gaussian_x=args.gaussian_x,
                gaussian_z=args.gaussian_z,
                gaussian_sigma_mm=args.gaussian_sigma_mm)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("sim", "analyze"))
    p.add_argument("--root", type=Path,
                   default=Path("/data/zhuangyang/NumerialBreastPhantoms/"
                                "l11_ultrawave_500_11angle"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ids", default="val_000,val_004,val_008")
    p.add_argument("--delta-c", type=float, default=7.5)
    p.add_argument("--scheme", choices=("central", "forward"), default="central")
    p.add_argument("--depth-slabs", type=int, default=5)
    p.add_argument("--lateral-modes", type=int, default=4)
    p.add_argument("--axial-modes", type=int, default=4)
    p.add_argument("--gaussian-x", type=int, default=3)
    p.add_argument("--gaussian-z", type=int, default=2)
    p.add_argument("--gaussian-sigma-mm", type=float, default=4.)
    p.add_argument("--z-min-mm", type=float, default=3.)
    p.add_argument("--z-max-mm", type=float, default=43.)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--top-modes", type=int, default=5)
    return p.parse_args()


def _load_dataset_index(root):
    index = json.loads((root / "index.json").read_text())
    return {r["id"]: r for r in index["samples"]}


def _metadata_from_raw(path):
    d = np.load(path, allow_pickle=False)
    key = "metadata_json" if "metadata_json" in d.files else "meta_json"
    if key not in d.files:
        raise ValueError(f"{path}: missing metadata json")
    return d, json.loads(str(d[key].item()))


def stage_sim(args):
    sys.path.insert(0, "/home/zhuangyang/fmmodel/neural_asp")
    sys.path.insert(0, "/home/zhuangyang/fmmodel/UltraWave/benchmarks")
    sys.path.insert(0, "/data/zhuangyang/NumerialBreastPhantoms")
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported in the sim process")

    import h5py
    import scripts.generate_l11_ultrawave_raw as gen
    from analysis.sos_identifiability import build_physical_basis

    gen.configure_gpu()
    args.out.mkdir(parents=True, exist_ok=True)
    by_id = _load_dataset_index(args.root)
    ids = [s for s in args.ids.split(",") if s]
    missing = [s for s in ids if s not in by_id]
    if missing:
        raise KeyError(f"unknown dataset ids: {missing}")

    ref_stored = np.load(args.root / "reference_native.npz")["rf_native"]
    records = []
    basis_names = None
    start_all = time.monotonic()

    for sid in ids:
        rec = by_id[sid]
        with h5py.File(rec["h5"], "r") as f:
            plane = np.asarray(f["phan"][rec["z_index"]])

        case = gen.geometry_case()
        maps, *_ = gen.medium_builder.build_medium(
            plane, case["x"], case["z"], seed=rec["scatter_seed"],
            preset="dual_scale")
        base_c = maps["sound_speed"].astype(np.float32, copy=True)

        basis, names, _ = build_physical_basis(
            case["x"], case["z"], **basis_kwargs(args))
        if basis_names is None:
            basis_names = names
        elif basis_names != names:
            raise RuntimeError("basis names changed across records")

        raw_path = args.root / "raw" / f"{sid}.npz"
        raw_npz, raw_meta = _metadata_from_raw(raw_path)
        base_rf = raw_npz["rf"].astype(np.float32)
        angles = np.asarray(raw_meta["angles_deg"], np.float64)

        sample_dir = args.out / sid
        sample_dir.mkdir(exist_ok=True)
        np.savez_compressed(sample_dir / "baseline.npz", rf=base_rf,
                            c=base_c, angles=angles,
                            basis_names=np.asarray(names))

        signs = (1,) if args.scheme == "forward" else (-1, 1)
        for k, name in enumerate(names):
            for sign in signs:
                suffix = "plus" if sign > 0 else "minus"
                out_path = sample_dir / f"{k:03d}_{name}_{suffix}.npz"
                if out_path.exists():
                    continue
                pert_maps = {key: np.array(value, copy=True)
                             for key, value in maps.items()}
                delta = float(sign) * float(args.delta_c) * basis[k].T
                pert_maps["sound_speed"] = (base_c + delta).astype(np.float32)
                if (pert_maps["sound_speed"] <= 0).any():
                    raise ValueError("SoS perturbation produced non-positive speed")
                case_k = gen.geometry_case()
                case_k["maps"] = pert_maps
                gen.bench.validate_case(case_k)
                fit = gen.absorption_model(case_k)
                solver = gen.solver_for(case_k, pert_maps, fit)
                rf_list, trefs = [], []
                for ai, angle in enumerate(angles):
                    tref = gen.set_angle(solver, case_k, float(angle))
                    total, _ = solver.run()
                    rf, _ = gen.sim.analytic_channels(
                        total - ref_stored[:, :, ai], gen.DT,
                        band=raw_meta.get("band_hz", [4e6, 7.5e6]))
                    rf_list.append(rf.T.astype(np.float32))
                    trefs.append(float(tref))
                rf_out = np.stack(rf_list)
                if rf_out.shape != base_rf.shape or not np.isfinite(rf_out).all():
                    raise RuntimeError(f"{sid}/{name}: invalid perturbed RF")
                np.savez_compressed(out_path, rf=rf_out,
                                    trefs=np.asarray(trefs, np.float64))
                print(json.dumps({"id": sid, "mode": name, "sign": sign,
                                  "elapsed_s": round(time.monotonic()-start_all, 1)}),
                      flush=True)

        records.append({"id": sid, "source": str(raw_path),
                        "base_anatomy_id": rec.get("base_anatomy_id")})

    manifest = {
        "kind": "sos_identifiability_perturbations",
        "scheme": args.scheme,
        "delta_c_m_s": args.delta_c,
        "basis": {"names": basis_names, **basis_kwargs(args)},
        "records": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] simulated {len(records)} records x {len(basis_names)} modes",
          flush=True)


def _condition_from_rf(rf, metadata, args, device):
    import torch
    from data import geometry as G
    from data.rf_geometry import structured_condition

    angles = np.asarray(metadata["angles_deg"], np.float64)
    refs = np.asarray(metadata["source_tref_s"], np.float64)
    fs = float(metadata["fs_hz"])
    fc = float(metadata.get("source_f0_hz", metadata.get("fc_hz")))
    pitch = float(metadata.get("pitch_m", 2e-4))
    xe = (np.arange(rf.shape[1]) - (rf.shape[1]-1)/2) * pitch
    band = np.asarray(metadata.get("band_hz", []), np.float64)
    bw = (float((band[1]-band[0])/(2*fc)) if band.shape == (2,)
          else float(metadata.get("bandwidth_fraction", .6)))
    cond = structured_condition(
        torch.as_tensor(rf, device=device), xe, angles, refs,
        G.x_grid(), G.z_grid(), fs, fc,
        float(metadata.get("c_steer_m_s", metadata.get("c_steer", 1500.))),
        bw, speeds=(1450., 1500., 1550.), ref_speed=1500., n_subap=4,
        t0_s=float(metadata.get("t0_s", 0.)), chunk=args.chunk)
    return cond


def _numpy_group(cond, name):
    if name == "full":
        active = cond["event_mask"][None, :, None, None].to(
            cond["speed_events"].real.dtype)
        x = (cond["speed_events"] * active).sum(dim=1)
    elif name == "event":
        x = cond["speed_events"][1]
    elif name == "subap":
        x = cond["subap"]
    else:
        raise KeyError(name)
    return x.detach().cpu().numpy()


def _batched_condition(cond, device):
    import torch
    return {k: (v.to(device)[None] if torch.is_tensor(v) else v)
            for k, v in cond.items()}


def _finite_difference(plus, minus, baseline, delta, scheme):
    from analysis.sos_identifiability import relative_response
    if scheme == "central":
        return relative_response(plus, minus, baseline, delta)
    mirror = 2 * np.asarray(baseline) - np.asarray(plus)
    return relative_response(plus, mirror, baseline, delta)


def stage_analyze(args):
    import torch
    sys.path.insert(0, str(ROOT))
    from analysis.sos_identifiability import (
        build_physical_basis, network_mode_diagnostics, real_gram,
        solve_generalized_spectrum, state_gram, synthesize_modes)
    from data import geometry as G
    from models.geometry_flow import GeometryAwareSoSFlow

    manifest = json.loads((args.out / "manifest.json").read_text())
    scheme = manifest["scheme"]
    delta = float(manifest["delta_c_m_s"])
    names = list(manifest["basis"]["names"])
    x, z = G.x_grid().astype(np.float64), G.z_grid().astype(np.float64)
    basis, names_now, roi = build_physical_basis(x, z, **basis_kwargs(args))
    if names_now != names:
        raise ValueError("CLI basis configuration does not match simulation manifest")
    gc = state_gram(basis, roi)

    device = torch.device(args.device)
    model = None
    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = GeometryAwareSoSFlow(
            unet=blob["unet"], encoder=blob["encoder_cfg"],
            u_source_scale=-1.).to(device).eval()
        model.load_state_dict(blob["state_dict"])

    gram_sum = {k: np.zeros((len(names), len(names)), np.float64)
                for k in ("rf", "full", "event", "subap")}
    if model is not None:
        gram_sum["encoder"] = np.zeros_like(gc)
        net_gram_sum = np.zeros_like(gc)
        state_net_cross_sum = np.zeros_like(gc)

    for rec in manifest["records"]:
        sid = rec["id"]
        sample_dir = args.out / sid
        raw_npz, metadata = _metadata_from_raw(Path(rec["source"]))
        base_rf = raw_npz["rf"].astype(np.float32)
        base_cond = _condition_from_rf(base_rf, metadata, args, device)
        base_groups = {"rf": base_rf}
        for group in ("full", "event", "subap"):
            base_groups[group] = _numpy_group(base_cond, group)
        if model is not None:
            base_batched = _batched_condition(base_cond, device)
            with torch.no_grad():
                base_enc = model.encode_condition(base_batched)[0].cpu().numpy()

        responses = {k: [] for k in gram_sum}
        net_derivs = []
        for k, name in enumerate(names):
            plus = np.load(sample_dir / f"{k:03d}_{name}_plus.npz")["rf"]
            minus = (np.load(sample_dir / f"{k:03d}_{name}_minus.npz")["rf"]
                     if scheme == "central" else None)
            plus_cond = _condition_from_rf(plus, metadata, args, device)
            minus_cond = (_condition_from_rf(minus, metadata, args, device)
                          if minus is not None else None)

            endpoint = {
                "rf": (plus, minus),
                "full": (_numpy_group(plus_cond, "full"),
                         _numpy_group(minus_cond, "full") if minus_cond else None),
                "event": (_numpy_group(plus_cond, "event"),
                          _numpy_group(minus_cond, "event") if minus_cond else None),
                "subap": (_numpy_group(plus_cond, "subap"),
                          _numpy_group(minus_cond, "subap") if minus_cond else None),
            }
            for group in ("rf", "full", "event", "subap"):
                p, m = endpoint[group]
                responses[group].append(
                    _finite_difference(p, m, base_groups[group], delta, scheme))

            if model is not None:
                pbat = _batched_condition(plus_cond, device)
                mbat = (_batched_condition(minus_cond, device)
                        if minus_cond is not None else None)
                with torch.no_grad():
                    penc = model.encode_condition(pbat)[0].cpu().numpy()
                    menc = (model.encode_condition(mbat)[0].cpu().numpy()
                            if mbat is not None else None)
                responses["encoder"].append(
                    _finite_difference(penc, menc, base_enc, delta, scheme))

                seed = 20260925
                torch.manual_seed(seed)
                with torch.no_grad():
                    pp = model.sample(pbat, n_steps=args.ode_steps,
                                      n_samples=args.n_samples).mean(0)[0, 0]
                if mbat is not None:
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        pm = model.sample(mbat, n_steps=args.ode_steps,
                                          n_samples=args.n_samples).mean(0)[0, 0]
                    dnet = (pp - pm) / (2. * delta)
                else:
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        pb = model.sample(base_batched, n_steps=args.ode_steps,
                                          n_samples=args.n_samples).mean(0)[0, 0]
                    dnet = (pp - pb) / delta
                net_derivs.append(dnet.cpu().numpy())

        for group, rs in responses.items():
            gram_sum[group] += real_gram(rs)

        if model is not None:
            nd = np.stack(net_derivs).astype(np.float64)
            nf = nd[:, roi]
            bf = basis[:, roi].astype(np.float64)
            net_gram_sum += (nf @ nf.T) / max(nf.shape[1], 1)
            state_net_cross_sum += (bf @ nf.T) / max(nf.shape[1], 1)

        print(f"[analyze] {sid}", flush=True)

    nrec = max(len(manifest["records"]), 1)
    spectra = {}
    for group, g in gram_sum.items():
        avg = g / nrec
        sp = solve_generalized_spectrum(avg, gc)
        spectra[group] = {
            "gram": avg,
            "singular_values": sp["singular_values"],
            "coefficients": sp["coefficients"],
            "modes": synthesize_modes(sp["coefficients"], basis),
            "state_rank": sp["state_rank"],
        }

    if model is not None:
        gn = net_gram_sum / nrec
        cross = state_net_cross_sum / nrec
        for group in spectra:
            spectra[group]["network_mode_diagnostics"] = network_mode_diagnostics(
                spectra[group]["coefficients"], gc, gn, cross)

    npz = {"basis": basis, "state_metric": gc, "x": x, "z": z,
           "basis_names": np.asarray(names)}
    report = {"n_records": nrec, "delta_c_m_s": delta, "scheme": scheme,
              "basis_names": names, "groups": {}}
    top = max(int(args.top_modes), 1)
    for group, item in spectra.items():
        npz[f"{group}_gram"] = item["gram"]
        npz[f"{group}_singular_values"] = item["singular_values"]
        npz[f"{group}_coefficients"] = item["coefficients"]
        npz[f"{group}_modes"] = item["modes"]
        if "network_mode_diagnostics" in item:
            npz[f"{group}_network_diag"] = item["network_mode_diagnostics"]
        s = item["singular_values"]
        weakest = list(range(min(top, len(s))))
        strongest = list(range(max(0, len(s)-top), len(s)))[::-1]
        report["groups"][group] = {
            "state_rank": int(item["state_rank"]),
            "singular_values": [float(v) for v in s],
            "weakest_indices": weakest,
            "strongest_indices": strongest,
        }
    np.savez_compressed(args.out / "identifiability_spectrum.npz", **npz)
    (args.out / "identifiability_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    _plot_summary(args.out, spectra, x, z, top)
    print(f"[done] -> {args.out/'identifiability_report.json'}", flush=True)


def _plot_summary(out, spectra, x, z, top):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, item in spectra.items():
        s = item["singular_values"]
        ax.semilogy(np.arange(1, len(s)+1), np.maximum(s, 1e-14),
                    marker="o", label=group)
    ax.set(xlabel="generalized mode index (weak -> strong)",
           ylabel="relative observation sensitivity / (m/s)",
           title="SoS identifiability spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "identifiability_spectrum.png", dpi=170)
    plt.close(fig)

    key = "encoder" if "encoder" in spectra else "event"
    item = spectra[key]
    count = min(int(top), len(item["singular_values"]))
    extent = [x[0]*1e3, x[-1]*1e3, z[-1]*1e3, z[0]*1e3]
    fig, axes = plt.subplots(2, count, figsize=(3.2*count, 6), squeeze=False)
    ids = list(range(count)) + list(range(len(item["singular_values"])-1,
                                          len(item["singular_values"])-1-count, -1))
    for j, idx in enumerate(ids):
        row, col = divmod(j, count)
        mode = item["modes"][idx]
        lim = max(float(np.max(np.abs(mode))), 1e-9)
        im = axes[row, col].imshow(mode.T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                                   extent=extent, aspect="auto")
        label = "weak" if row == 0 else "strong"
        axes[row, col].set_title(
            f"{label} {idx}: s={item['singular_values'][idx]:.2e}")
        axes[row, col].set(xlabel="x [mm]", ylabel="depth [mm]")
        fig.colorbar(im, ax=axes[row, col], shrink=.75)
    fig.suptitle(f"{key} identifiability modes")
    fig.tight_layout()
    fig.savefig(out / f"{key}_identifiability_modes.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    if args.stage == "sim":
        stage_sim(args)
    else:
        stage_analyze(args)


if __name__ == "__main__":
    main()
