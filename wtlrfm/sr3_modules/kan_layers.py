"""Kolmogorov-Arnold Network (KAN) layers for the WTLR UNet bottleneck.

Background / provenance
------------------------
KANs (Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024) replace a linear
layer's fixed scalar weights with per-edge *learnable 1D functions*
(parameterised as a B-spline plus a base activation term). Applying this
inside a convolutional UNet backbone is not something we invented here: it
follows the "U-KAN" design of Li et al., "U-KAN Makes Strong Backbone for
Medical Image Segmentation and Generation" (AAAI 2025, arXiv:2406.02918),
which inserts a *Tokenized KAN* block near the bottleneck of a U-Net
(their Eq. 7), and additionally proposes a diffusion/generative variant
("Diffusion U-KAN") that injects a time embedding into the KAN block
(their Eq. 11).

The B-spline math in ``KANLinear`` below follows the standard formulation
used by most public KAN implementations (recursive Cox-de Boor basis on a
fixed uniform grid, e.g. "efficient-kan"). It is re-implemented from
scratch here (no code copied) so it can be unit-tested and audited
independently.

IMPORTANT / NOT YET VALIDATED
------------------------------
Swapping bottleneck layers for KAN layers is an *architectural hypothesis*,
not a validated improvement for ultrasound SR flow-matching. Per this
project's research standards (see AGENTS.md):
  - No claim of improved PSNR/SSIM/hallucination behaviour is made here.
  - This must be compared against the unmodified baseline
    (``sr_struct_residual_flow_B_narrowgap_4_6mhz``) using the same data
    split, seed, and training budget before any conclusion is drawn.
  - KAN layers are known (see Convolutional-KAN literature) to increase
    parameter count and can be less numerically stable / slower to train
    than plain convolutions; watch for NaNs, slower convergence, and
    inspect ``kan_grid_range`` if activations saturate at the grid
    boundary (spline term degrades to ~0 outside the grid, falling back
    to the base-activation linear term only).
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """B-spline based Kolmogorov-Arnold linear layer.

    Each of the ``in_features x out_features`` connections is a learnable
    1D function ``phi(x) = base_activation(x) * base_weight + spline(x)``
    where ``spline`` is a linear combination of B-spline basis functions
    defined on a fixed uniform grid over ``grid_range``. This is a drop-in
    replacement for ``nn.Linear`` that operates on the *last* dimension of
    its input (shape ``(..., in_features) -> (..., out_features)``).

    Args:
        in_features: input feature dimension.
        out_features: output feature dimension.
        grid_size: number of grid intervals spanning ``grid_range``.
        spline_order: B-spline order (3 = cubic, standard choice).
        base_activation: nonlinearity applied before the base linear term.
        scale_base: init scale for the base (linear) weight.
        scale_spline: init scale for the spline weight.
        grid_range: (min, max) range the spline basis is defined over.
            Inputs should roughly fall in this range (e.g. features that
            have passed through GroupNorm/LayerNorm); values far outside
            it fall back to the base-activation term only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation=nn.SiLU,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range=(-2.0, 2.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()  # (in_features, grid_size + 2*order + 1)
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))

        self.base_activation = base_activation()
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.base_weight.mul_(self.scale_base)
        # Small random init for spline coefficients: the network starts out
        # close to a plain gated-linear layer and gradually learns
        # non-linear per-edge corrections.
        nn.init.uniform_(self.spline_weight, -0.1, 0.1)
        with torch.no_grad():
            self.spline_weight.mul_(self.scale_spline / max(self.grid_size, 1) ** 0.5)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline basis functions at ``x``.

        Args:
            x: (..., in_features)
        Returns:
            (..., in_features, grid_size + spline_order)
        """
        grid = self.grid.to(dtype=x.dtype, device=x.device)  # (in_features, G)
        x = x.unsqueeze(-1)  # (..., in_features, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, : -(k + 1)]
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_num = grid[:, k + 1 :] - x
            right_den = grid[:, k + 1 :] - grid[:, 1:-k]
            bases = (left_num / left_den) * bases[..., :-1] + (right_num / right_den) * bases[..., 1:]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_out = F.linear(self.base_activation(x), self.base_weight)
        bspline_basis = self.b_splines(x)  # (N, in_features, G+order)
        spline_out = torch.einsum(
            "nik,oik->no", bspline_basis, self.spline_weight
        )
        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)


class TokKANLayer(nn.Module):
    """Single Tokenized-KAN sub-block, following Eq. 7 of U-KAN (Li et al.
    2024): ``Z_k = LN(Z_{k-1} + DwConv(BN(KAN(Z_{k-1}))))``.

    Operates on channel-last tokens ``(B, N, C)`` but takes/returns
    channel-first feature maps ``(B, C, H, W)`` so it can be dropped into
    a conv UNet; the depth-wise conv step needs the spatial layout to
    restore local structure that the purely per-token KAN layer cannot
    capture.
    """

    def __init__(self, dim: int, grid_size: int = 5, spline_order: int = 3,
                 grid_range=(-2.0, 2.0), dropout: float = 0.0):
        super().__init__()
        self.kan = KANLinear(dim, dim, grid_size=grid_size, spline_order=spline_order,
                              grid_range=grid_range)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, N, C)

        kan_out = self.dropout(self.kan(tokens))
        kan_img = kan_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        kan_img = F.relu(self.bn(self.dwconv(kan_img)))
        kan_tokens = kan_img.permute(0, 2, 3, 1).reshape(b, h * w, c)

        out_tokens = self.norm(tokens + kan_tokens)
        return out_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)


class TokKANMidBlock(nn.Module):
    """Time-conditioned stack of :class:`TokKANLayer`, used as an (optional)
    replacement for one of the ResNet+Attention bottleneck blocks in
    ``WTLRUNet``.

    Design notes (adapted, not a literal reproduction of either published
    variant -- see module docstring):
      - Time/noise-level conditioning is injected once, at the block's
        entry, via the same ``FeatureWiseAffine`` FiLM mechanism every
        ``ResnetBlock`` in this codebase already uses (for consistency),
        rather than the additive-only injection of U-KAN Eq. 11.
      - The depth-wise conv + residual + LayerNorm path (U-KAN Eq. 7) is
        kept for all layers (unlike "Diffusion U-KAN", which drops it) to
        preserve a strong identity path through this deep, multi-stage,
        conditional flow-matching UNet.
    """

    def __init__(self, dim: int, noise_level_emb_dim=None, num_layers: int = 3,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range=(-2.0, 2.0), dropout: float = 0.0):
        super().__init__()
        # Local import to avoid a circular import at module load time
        # (wtlr_unet.py imports this module).
        from .wtlr_unet import FeatureWiseAffine

        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim, use_affine_level=False)
        self.layers = nn.ModuleList([
            TokKANLayer(dim, grid_size=grid_size, spline_order=spline_order,
                        grid_range=grid_range, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.noise_func(x, time_emb)
        for layer in self.layers:
            x = layer(x)
        return x
