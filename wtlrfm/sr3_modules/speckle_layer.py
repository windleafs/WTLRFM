"""
Fourier-Optics Speckle Layer (SpeckleGAN-style)

Implements the speckle generation operator

    I_sp(x, y) = | F^{-1} { F { I(x, y) * exp(j * phi(x, y)) } * rect_d(x, y) } |

where
    I(x, y)        input feature map (one channel at a time, real-valued),
    phi(x, y)      per-pixel random phase ~ U[0, 2*pi),
    rect_d(x, y)   square low-pass window (side length d, learnable),
    F, F^{-1}      2-D FFT / IFFT (O(n log n)).

For each input feature channel the layer produces ``num_scales`` speckle
variants (different d values), so the intermediate tensor has shape
``[B, num_scales * C, H, W]``.  A channel-attention block (global pool + two
1x1 convs, sigmoid gate) then re-weights every (feature x scale) pair, and a
1x1 conv fuses the result back to ``C`` channels.

An optional polar <-> cartesian wrapping (``polar_transform=True``) is provided
for IVUS / OCT-style central-radial speckle.  By default this is off, since
the input here are already cartesian IQ patches.

The whole module is wrapped in a learnable scalar gate (initialised to 0), so
that at the very beginning of training the layer is a no-op (``y = x``) and
gradually starts contributing as the gate grows.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class SpeckleLayer(nn.Module):
    """Fourier-optics speckle layer with multi-scale fusion.

    Args:
        channels:      number of input feature channels C.
        num_scales:    number of different speckle sizes K (default 4).
        image_size:    expected spatial size of the input (used to seed d).
        d_init:        list of ``num_scales`` initial fractional cut-off
                       frequencies in (0, 1) (e.g. [0.25, 0.5, 0.75, 1.0]).
                       The actual cut-off d is parameterised as
                       ``d = N * sigmoid(log_d)`` so it stays in (0, N].
        sharpness:     softness of the rectangular window in the frequency
                       domain (higher = closer to a hard rect).
        attn_reduction: channel-reduction ratio of the SE-style attention.
        residual_gate: if True, output = x + gate * speckle(x) with
                       ``gate`` a learnable scalar initialised to 0
                       (identity at init).
        polar_transform: if True, wrap FFT in polar <-> cartesian transforms
                         (suitable for IVUS-like radial speckle).
        polar_size:    (R, T) resolution of the polar representation.
    """

    def __init__(self, channels, num_scales=4, image_size=128,
                 d_init=None, sharpness=4.0,
                 attn_reduction=8, residual_gate=True,
                 polar_transform=False, polar_size=None):
        super().__init__()

        self.channels = channels
        self.num_scales = num_scales
        self.image_size = image_size
        self.sharpness = sharpness
        self.residual_gate = residual_gate
        self.polar_transform = polar_transform
        self.polar_size = polar_size or (image_size, image_size)

        # --- learnable cut-off frequency d for each scale ---
        if d_init is None:
            d_init = [(i + 1) / num_scales for i in range(num_scales)]
        assert len(d_init) == num_scales, \
            f"d_init must have length {num_scales}"
        d_init_t = torch.tensor(d_init, dtype=torch.float32).clamp(1e-3,
                                                                   1 - 1e-3)
        # logit so that d = N * sigmoid(log_d)
        log_d_init = torch.log(d_init_t / (1.0 - d_init_t))
        self.log_d = nn.Parameter(log_d_init)  # [num_scales]

        # --- channel-attention fusion (K*C -> K*C, then 1x1 -> C) ---
        fused = channels * num_scales
        hidden = max(fused // attn_reduction, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, fused, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(fused, channels, 1)

        # --- residual gate (init 0 so layer starts as identity) ---
        if residual_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gate", None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_lowpass(self, H, W, device, dtype):
        """Build a soft rectangular low-pass window in the (centred)
        frequency domain.

        Returns a real-valued tensor of shape [num_scales, H, W].
        """
        # centred frequency coordinates (in pixel units, [-N/2, N/2))
        ky = torch.arange(H, device=device, dtype=dtype) - H // 2
        kx = torch.arange(W, device=device, dtype=dtype) - W // 2
        ky = ky.view(1, H, 1)
        kx = kx.view(1, 1, W)

        # cut-off d in pixel units, per scale
        N = float(min(H, W))
        d = torch.sigmoid(self.log_d) * N  # [num_scales]
        d = d.view(self.num_scales, 1, 1)

        # soft rectangular window via two sigmoids (separable)
        win_y = torch.sigmoid(self.sharpness * (d / 2.0 - ky.abs()))
        win_x = torch.sigmoid(self.sharpness * (d / 2.0 - kx.abs()))
        return win_y * win_x  # [num_scales, H, W]

    def _cart_to_polar(self, x):
        """Cartesian -> polar via ``grid_sample``.  Assumes the image centre
        is the catheter centre.  Output spatial size is ``self.polar_size``."""
        B, C, H, W = x.shape
        R, T = self.polar_size
        device, dtype = x.device, x.dtype

        rho = torch.linspace(0.0, 1.0, R, device=device, dtype=dtype)
        theta = torch.linspace(0.0, 2 * math.pi, T + 1, device=device,
                               dtype=dtype)[:-1]
        rho_g, theta_g = torch.meshgrid(rho, theta, indexing='ij')  # [R, T]

        # max radius fits inside the image: half of the shortest side
        max_r = (min(H, W) - 1) / 2.0
        xs = rho_g * max_r * torch.cos(theta_g)
        ys = rho_g * max_r * torch.sin(theta_g)

        # normalise to [-1, 1] for grid_sample
        xs = xs / ((W - 1) / 2.0)
        ys = ys / ((H - 1) / 2.0)
        grid = torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    def _polar_to_cart(self, x, out_hw):
        """Polar -> cartesian inverse of ``_cart_to_polar``."""
        B, C, R, T = x.shape
        H, W = out_hw
        device, dtype = x.device, x.dtype

        ys = torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2.0
        xs = torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

        max_r = (min(H, W) - 1) / 2.0
        rho = torch.sqrt(xs * xs + ys * ys) / max_r            # in [0, ~1]
        theta = torch.atan2(ys, xs) % (2 * math.pi)             # in [0, 2pi)

        # normalise to grid_sample coords:
        #   rho axis (R) corresponds to the y-axis of the polar image,
        #   theta axis (T) corresponds to the x-axis.
        grid_y = rho * 2.0 - 1.0           # [-1, 1] (clipped by padding_mode)
        grid_x = theta / math.pi - 1.0     # [-1, 1)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)\
                    .expand(B, -1, -1, -1)
        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        x_in = x
        if self.polar_transform:
            cart_hw = (x.shape[-2], x.shape[-1])
            x = self._cart_to_polar(x)

        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # --- per-pixel random phase (shared across channels per sample) ---
        phi = 2.0 * math.pi * torch.rand(B, 1, H, W, device=device, dtype=dtype)
        cos_p, sin_p = torch.cos(phi), torch.sin(phi)

        # x * exp(j*phi) (FFT is done in float32 for numerical stability,
        # then cast back at the end)
        x32 = x.to(torch.float32)
        real = x32 * cos_p.to(torch.float32)
        imag = x32 * sin_p.to(torch.float32)
        x_c = torch.complex(real, imag)                       # [B, C, H, W]

        # forward FFT, centred
        X = torch.fft.fft2(x_c, norm='ortho')
        X = torch.fft.fftshift(X, dim=(-2, -1))

        # multi-scale low-pass windows  [K, H, W]
        win = self._build_lowpass(H, W, device, torch.float32)
        win_c = torch.complex(win, torch.zeros_like(win))

        # broadcast multiply: [B, 1, C, H, W] * [1, K, 1, H, W] -> [B, K, C, H, W]
        Y = X.unsqueeze(1) * win_c.view(1, self.num_scales, 1, H, W)

        # inverse FFT and take magnitude
        Y = torch.fft.ifftshift(Y, dim=(-2, -1))
        y = torch.fft.ifft2(Y, norm='ortho').abs()            # real
        y = y.to(dtype)

        # reshape (B, K, C, H, W) -> (B, K*C, H, W)
        y = y.reshape(B, self.num_scales * C, H, W)

        # channel attention + fusion to C
        y = y * self.attn(y)
        y = self.fuse(y)

        if self.polar_transform:
            y = self._polar_to_cart(y, cart_hw)

        if self.residual_gate:
            return x_in + self.gate * y
        return y
