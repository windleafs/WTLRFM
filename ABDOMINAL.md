# Abdominal 2-D Sound-Speed Prediction

This pipeline trains the existing WTLRFM multiplicative flow architecture on the
k-Wave abdominal-wall dataset. It does not reuse the incompatible 40-channel
OpenBreast checkpoint. Original OpenBreast files and outputs are preserved.

## Data and Geometry

- Source: `/data/zhuangyang/SIMUABERV1/simu_kwave_aber_v2/output/dataset`.
- 1000 HDF5 samples, manifest train/validation/test counts 800/100/100.
- Splits are checked for source-file overlap. The Visible Human variants are
  not established independent subjects; this is not a clinical subject-held-out benchmark.
- Real pressure RF: time x receiver x angle = 2113 x 128 x 3 for the checked
  sample, 12 MHz sampling, 3 MHz centre frequency, angles -6/0/+6 degrees.
- Cache/model order is **[x,z]**. Prediction grid: 128 x 160, x=-19.2..18.9 mm,
  z=0.15..47.85 mm, relative to probe. Plots transpose arrays to display depth
  vertically. This output is a cropped abdominal-wall map, not the full
  110 mm deep source medium.
- Tissue supervision uses segmentation IDs 2..9, in-grid pixels, depth >=3 mm.
  Abdominal wall metrics use IDs 2..6. Coupling and excluded pixels are not
  part of the primary loss. They receive a 0.1-weight auxiliary velocity loss to
  constrain the full ODE state; their predictions are excluded from tissue metrics.

## RF Timing and Conditions

The adapter imports the dataset's MATLAB-aware HDF5 reader, but only reads
RF, geometry, sound speed and segmentation, not the large aberration delay labels.
Hilbert transform creates analytic RF; mixing uses absolute `rf/time_s` and the
sample carrier. DAS restores carrier phase at the same absolute query time.

For each assumed speed c, source i, receiver e and pixel p:

```
t_tx(p) = min_i(launch_i + distance(i,p)/c) + source_cycles/(2*fc)
t_query(e,p) = t_tx(p) + distance(e,p)/c
sample_index = (t_query - time_s[0]) * fs
```

This finite-aperture earliest-arrival approximation uses stored launch delays
and therefore respects k-Wave's opposite steering-sign convention. The dataset
stores nominal rather than native-time-rounded launch offsets. The adapter
uses nominal burst-centre timing, and does not reconstruct native dt from the
true sound-speed map: targets never determine conditioning. This is not an
exact nonlinear/refracting wave propagation model. Synthetic point-target
unit tests verify the implementation; no acquired-data phase calibration is claimed.

Source and receiver are at the second axial row, so labels are shifted by one
axial grid spacing. The stored half-cell-centred lateral coordinates are shifted
by -dx/2 on even grids to match the actual k-Wave/array coordinate convention.

20 real-valued channels preserve DAS complex phase: real/imaginary pairs from
3 assumed full-aperture speeds (1450/1500/1550), 3 individual angles at 1500,
and 4 receive subapertures at 1500. Group RMS normalization uses depths >=3 mm
without consulting target/segmentation. The dataset retains direct pressure;
no unavailable homogeneous-reference subtraction is pretended. Conditions
near the transducer may therefore contain transmit/direct-wave effects.

## Commands

Run from the `wtlrfm_sos_wfc` directory. Use a free GPU; other users' jobs must
not be stopped. Caches and outputs below are separate from OpenBreast.

```bash
PY=/home/zhuangyang/miniconda3/envs/py310/bin/python
CACHE=/data/zhuangyang/SIMUABERV1/simu_kwave_aber_v2/output/sos_cache_v1
OUT=/home/zhuangyang/fmmodel/dbua_test/wtlrfm_sos_wfc/out/abdominal_flow_v2

$PY scripts/prepare_abdominal_cache.py --cache "$CACHE" --device cuda:1
$PY scripts/train_abdominal.py --cache "$CACHE" --out "$OUT" --device cuda:1 --workers 4
$PY scripts/predict_abdominal.py --cache "$CACHE" --ckpt "$OUT/best.pth" --out "${OUT}_test" --device cuda:1

# Resume in the same output with matching configuration/cache.
$PY scripts/train_abdominal.py --cache "$CACHE" --out "$OUT" --device cuda:1 --workers 4 --resume

# Six-sample preprocessing and short training plumbing check.
$PY scripts/prepare_abdominal_cache.py --cache /tmp/abdominal_cache_smoke --device cuda:1 --limit-per-split 2
$PY scripts/train_abdominal.py --cache /tmp/abdominal_cache_smoke --out /tmp/abdominal_training_smoke --device cuda:1 --epochs 2 --limit 2

$PY -m unittest discover -s tests -p 'test_abdominal*.py'
```

The default initial experiment trains 30 epochs with the original 64-wide
backbone, batch size 8, fixed log-speed prior standard deviation 1, masked
flow velocity MSE, BF16 on supported CUDA hardware, Adam and cosine decay.
No synthetic geometric augmentation is applied. This is a finite initial
training run, not an assertion of convergence or clinical accuracy.

Training selects `best.pth` by validation tissue MAE. `last.pth` holds final
EMA inference weights; `training_state.pth` also stores online weights,
optimizer/scheduler, EMA, prior lifecycle, RNG and cache identity for resume.
Validation sampling has a fixed seed isolated from the training RNG.
Changing the epoch horizon on resume changes the cosine schedule.

Prediction saves NPZ maps with physical coordinates, masks and metadata,
per-sample CSV, summary JSON, and a bounded set of comparison PNGs. Ensemble
standard deviation uses population variance so a single trajectory is finite;
it measures sampling variability, not calibrated confidence. Whole/ROI scores
include low-weight background; tissue/wall metrics are the relevant scores.
Constant 1500/1540 m/s and a **training-only** masked mean map provide baselines.
Test samples are not used to select a checkpoint or fit these baselines.

See the output `validation_summary.json` and test `summary.json` for actual
measured performance. Strong training fit alone does not demonstrate RF-based
reconstruction; compare held-out results with the mean-map baseline.
