"""Shared geometry / normalisation constants for the SoS-map pipeline.

Everything (preprocessing, dataset, model, WFC integration) must agree on:
  * the prediction grid (x, z) of the 2-D sound-speed map,
  * the reference sound speed and the log-SoS normalisation scale,
  * the tissue ROI used for metrics.

Grid choice
-----------
    x: 128 samples, 0.30 mm pitch, spanning exactly the 38.1 mm aperture
       (x = -19.05 .. +19.05 mm, relative to the probe centre).
    z: 160 samples, 0.30 mm pitch, z = 0.15 .. 47.85 mm below the probe face
       (covers the 3-45 mm tissue ROI of the OpenBreastUS dataset).

The WTLR UNet keys its wavelet features by the feature-map *width*, so the
lateral size must be the power-of-two-friendly 128 (levels 3 -> widths
64/32/16 match the UNet decoder); the axial size may be arbitrary.
"""

import numpy as np

# ---- probe / data geometry (from openbreast_pw_iq README) ----------------
PITCH = 0.3e-3              # element pitch (m)
APERTURE = 128 * PITCH      # 38.1 mm
N_ELEM = 128
N_ANGLE = 13
FS = 20e6                   # IQ sampling rate (Hz)
FC = 5e6                    # centre frequency (Hz)

# ---- prediction grid ------------------------------------------------------
NX = 128
NZ = 160
DX = 0.3e-3
DZ = 0.3e-3
X0 = -APERTURE / 2.0        # first x sample  = -19.05 mm
Z0 = 0.15e-3                # first z sample (pixel centre) = 0.15 mm

# ---- SoS parameterisation ------------------------------------------------
C_REF = 1500.0              # reference sound speed (water background, m/s)
RHO_SCALE = 0.05            # u = log(c / C_REF) / RHO_SCALE  (O(1) variable)

# ---- ROI for metrics (geometric; dataset label uses the tissue mask) -----
ROI_Z = (3e-3, 45e-3)
ROI_X = (-APERTURE / 2.0, APERTURE / 2.0)


def x_grid(nx: int = NX) -> np.ndarray:
    """Lateral coordinates of the prediction grid [m]."""
    if nx == NX:
        return (X0 + DX * np.arange(NX)).astype(np.float32)
    return np.linspace(X0, -X0, nx, dtype=np.float32)


def z_grid(nz: int = NZ) -> np.ndarray:
    """Axial coordinates of the prediction grid [m] (pixel centres)."""
    if nz == NZ:
        return (Z0 + DZ * np.arange(NZ)).astype(np.float32)
    return np.linspace(Z0, Z0 + DZ * (NZ - 1), nz, dtype=np.float32)


def c_to_u(c):
    """sound speed [m/s] -> normalised log-SoS u (network variable)."""
    return np.log(np.asarray(c, np.float64) / C_REF) / RHO_SCALE


def u_to_c(u):
    """normalised log-SoS u -> sound speed [m/s]."""
    return C_REF * np.exp(RHO_SCALE * np.asarray(u, np.float64))


def roi_mask(nx: int = NX, nz: int = NZ) -> np.ndarray:
    """Boolean [nx, nz] geometric ROI mask (3-45 mm depth, inside aperture)."""
    x = x_grid(nx)
    z = z_grid(nz)
    mx = (x >= ROI_X[0]) & (x <= ROI_X[1])
    mz = (z >= ROI_Z[0]) & (z <= ROI_Z[1])
    return mx[:, None] & mz[None, :]
