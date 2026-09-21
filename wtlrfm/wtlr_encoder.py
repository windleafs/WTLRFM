"""
WTLR Encoder - Wavelet-Transform based Low-Resolution image Encoder

Inspired by DMUISR (Tianyu Liu et al., 2024, Physics in Medicine & Biology)
"Super-resolution reconstruction of ultrasound image using a modified diffusion model"

The WTLR encoder decomposes the LR image via multi-level DWT, processes each
sub-band independently, applies channel attention for adaptive sub-band weighting,
and produces multi-scale frequency-aware features for UNet decoder injection.

Architecture:
    LR image → Multi-level DWT decomposition
    Each level: LL, LH, HL, HH → independent SubBandEncoders
    → Channel Attention for sub-band importance weighting
    → Fused multi-scale feature maps aligned with UNet decoder resolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class HaarDWT2D(nn.Module):
    """
    Hardware-friendly 2D Haar DWT using fixed convolution filters.
    Decomposes input into LL, LH, HL, HH sub-bands at half resolution.
    """

    def __init__(self, wavelet='haar'):
        super().__init__()
        w = pywt.Wavelet(wavelet)
        self.wavelet_name = wavelet
        dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)

        ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        # [4, 1, k, k] - one filter per sub-band
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)
        self.pad = (len(dec_lo) - 2) // 2

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            ll, lh, hl, hh: each [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape
        filters = self.filters.repeat(C, 1, 1, 1)  # [C*4, 1, k, k]
        
        # Dual-mode hybrid wavelet padding:
        if self.wavelet_name == 'haar':
            # Haar achieves mathematically perfect reconstruction with 0 padding
            out = F.conv2d(x, filters, stride=2, padding=0, groups=C)
        else:
            # Other wavelets like bior2.2 use standard padding to maintain power-of-2 size without size expansion
            out = F.conv2d(x, filters, stride=2, padding=self.pad, groups=C)
            
        # out: [B, C*4, H/2, W/2] → split into 4 sub-bands
        out = out.reshape(B, C, 4, H // 2, W // 2)
        ll = out[:, :, 0]
        lh = out[:, :, 1]
        hl = out[:, :, 2]
        hh = out[:, :, 3]
        return ll, lh, hl, hh


class ResidualConvBlock(nn.Module):
    """Residual convolution block with GroupNorm for sub-band feature extraction."""

    def __init__(self, channels, norm_groups=None):
        super().__init__()
        if norm_groups is None:
            norm_groups = min(32, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(norm_groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(norm_groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class SubBandEncoder(nn.Module):
    """
    Encoder for a single wavelet sub-band.
    Projects sub-band from in_channels to feature_channels,
    then applies residual conv blocks for feature extraction.
    """

    def __init__(self, in_channels, feature_channels, num_res_blocks=2):
        super().__init__()
        norm_groups = min(32, feature_channels)
        layers = [nn.Conv2d(in_channels, feature_channels, 3, padding=1)]
        for _ in range(num_res_blocks):
            layers.append(ResidualConvBlock(feature_channels, norm_groups))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ChannelAttention(nn.Module):
    """
    SE-style channel attention for weighting sub-band features.
    Operates on the concatenated features from all 4 sub-bands,
    producing per-channel importance weights.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.squeeze(x).view(B, C)
        w = self.excitation(w).view(B, C, 1, 1)
        return x * w


class WTLRLevelEncoder(nn.Module):
    """
    Encoder for one DWT level: processes all 4 sub-bands independently,
    applies channel attention on concatenated features, then fuses to output.
    """

    def __init__(self, in_channels, feature_channels, num_res_blocks=2):
        super().__init__()
        self.ll_enc = SubBandEncoder(in_channels, feature_channels, num_res_blocks)
        self.lh_enc = SubBandEncoder(in_channels, feature_channels, num_res_blocks)
        self.hl_enc = SubBandEncoder(in_channels, feature_channels, num_res_blocks)
        self.hh_enc = SubBandEncoder(in_channels, feature_channels, num_res_blocks)

        concat_ch = feature_channels * 4
        self.channel_attn = ChannelAttention(concat_ch)

        norm_groups = min(32, feature_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_ch, feature_channels, 1),
            nn.GroupNorm(norm_groups, feature_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, ll, lh, hl, hh):
        """
        Args:
            ll, lh, hl, hh: each [B, C_in, H, W]
        Returns:
            fused: [B, feature_channels, H, W]
        """
        f_ll = self.ll_enc(ll)
        f_lh = self.lh_enc(lh)
        f_hl = self.hl_enc(hl)
        f_hh = self.hh_enc(hh)

        cat = torch.cat([f_ll, f_lh, f_hl, f_hh], dim=1)
        cat = self.channel_attn(cat)
        return self.fusion(cat)


class WTLREncoder(nn.Module):
    """
    Wavelet-Transform based Low-Resolution image (WTLR) Encoder.

    Performs multi-level DWT on the LR image and extracts frequency-aware
    features at each scale. Each level produces a feature map at half the
    resolution of the previous, directly matching UNet decoder resolutions.

    For image_size=128 and num_levels=3:
        Level 1: 64x64   (matches UNet after 1st downsample)
        Level 2: 32x32   (matches UNet after 2nd downsample)
        Level 3: 16x16   (matches UNet bottleneck)

    Args:
        in_channels: Input image channels (e.g. 3 for RGB)
        feature_channels: Output feature channels per level
        num_levels: Number of DWT decomposition levels
        wavelet: Wavelet type (default 'haar')
        num_res_blocks: Residual blocks per sub-band encoder
    """

    def __init__(
        self,
        in_channels=3,
        feature_channels=64,
        num_levels=3,
        wavelet='haar',
        num_res_blocks=2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.feature_channels = feature_channels
        self.dwt = HaarDWT2D(wavelet)

        self.level_encoders = nn.ModuleList()
        for _ in range(num_levels):
            self.level_encoders.append(
                WTLRLevelEncoder(in_channels, feature_channels, num_res_blocks)
            )

        norm_groups = min(32, feature_channels)
        self.output_norms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 1),
                nn.GroupNorm(norm_groups, feature_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_levels)
        ])

    def forward(self, x):
        """
        Args:
            x: LR image [B, C, H, W]
        Returns:
            features: dict mapping resolution → feature tensor
                {64: [B, F, 64, 64], 32: [B, F, 32, 32], 16: [B, F, 16, 16]}
        """
        features = {}
        current = x

        for level in range(self.num_levels):
            ll, lh, hl, hh = self.dwt(current)
            feat = self.level_encoders[level](ll, lh, hl, hh)
            feat = self.output_norms[level](feat)

            res = feat.shape[-1]
            features[res] = feat

            current = ll  # next level operates on LL sub-band

        return features


# --------------- DWT / IDWT based resampling primitives ---------------
#
# These modules are designed as drop-in replacements for the SR3 UNet's
# Downsample (stride-2 Conv3x3) and Upsample (nearest-neighbour x2 + Conv3x3)
# layers.  They keep input/output channel counts identical so that the rest
# of the UNet wiring (skip connections, ResBlock channel arithmetic, ...) is
# unchanged.
#
# Design choice (MWCNN-style):
#   Down:  DWT  -> [B, 4C, H/2, W/2] -> 1x1 Conv -> [B, C, H/2, W/2]
#   Up:    1x1 Conv [B, C, H, W] -> [B, 4C, H, W] -> IDWT -> [B, C, 2H, 2W]
#
# Both operations are information-preserving in the wavelet domain and avoid
# the aliasing of stride-2 convolutions / staircase artefacts of nearest-up.


class HaarIDWT2D(nn.Module):
    """Inverse 2D DWT (any orthogonal or biorthogonal wavelets) implemented via a single
    grouped strided transposed convolution.

    NOTE on filter choice: ``torch.nn.functional.conv2d`` /
    ``conv_transpose2d`` perform cross-correlation (no kernel flip), whereas
    ``pywt``'s ``rec_*`` filters are designed for *proper* convolution
    (with flip). To achieve perfect reconstruction with ``conv_transpose2d``, 
    we construct the synthesis filters as the tensor product of the flipped 
    reconstruction filters (rec_lo_f, rec_hi_f). This yields bit-exact 
    reconstruction (up to fp rounding precision) for any wavelet family.
    """

    def __init__(self, wavelet='haar'):
        super().__init__()
        w = pywt.Wavelet(wavelet)
        self.wavelet_name = wavelet
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)

        # Flip reconstruction filters to align with transpose convolution expectations
        rec_lo_f = torch.flip(rec_lo, dims=[0])
        rec_hi_f = torch.flip(rec_hi, dims=[0])

        ll = rec_lo_f.unsqueeze(0) * rec_lo_f.unsqueeze(1)
        lh = rec_lo_f.unsqueeze(0) * rec_hi_f.unsqueeze(1)
        hl = rec_hi_f.unsqueeze(0) * rec_lo_f.unsqueeze(1)
        hh = rec_hi_f.unsqueeze(0) * rec_hi_f.unsqueeze(1)

        # [4, 1, k, k] - one filter per sub-band
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)
        self.pad = (len(rec_lo) - 2) // 2

    def forward(self, ll, lh, hl, hh):
        """
        Args:
            ll, lh, hl, hh: each [B, C, H, W]
        Returns:
            x: [B, C, 2H, 2W]
        """
        B, C, H, W = ll.shape
        # Interleave sub-bands per channel so layout matches the grouped
        # transposed conv expectation: channel c contributes 4 contiguous
        # input channels [ll_c, lh_c, hl_c, hh_c].
        x = torch.stack([ll, lh, hl, hh], dim=2)        # [B, C, 4, H, W]
        x = x.reshape(B, C * 4, H, W)
        filters = self.filters.repeat(C, 1, 1, 1)        # [C*4, 1, k, k]
        
        # Dual-mode hybrid wavelet transpose padding:
        if self.wavelet_name == 'haar':
            out = F.conv_transpose2d(
                x, filters, stride=2, padding=0, groups=C
            )
        else:
            out = F.conv_transpose2d(
                x, filters, stride=2, padding=self.pad, groups=C
            )
        return out


class DWTDownsample(nn.Module):
    """DWT-based 2x downsampling.

    Replaces the SR3 stride-2 Conv3x3 downsampler.  The Haar DWT halves
    spatial resolution losslessly into 4 sub-bands; a 1x1 conv then mixes
    the 4 sub-bands and projects 4*C channels back down to ``dim_out``.
    """

    def __init__(self, dim_in, dim_out=None, wavelet='haar', norm_groups=32):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim_in
        self.dwt = HaarDWT2D(wavelet)
        # Per-subband normalisation stabilises training because LH/HL/HH have
        # near-zero mean and much smaller variance than LL.
        ng = min(norm_groups, dim_in * 4)
        self.norm = nn.GroupNorm(ng, dim_in * 4)
        self.proj = nn.Conv2d(dim_in * 4, dim_out, 1)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        cat = torch.cat([ll, lh, hl, hh], dim=1)         # [B, 4C, H/2, W/2]
        return self.proj(self.norm(cat))


class IDWTUpsample(nn.Module):
    """IDWT-based 2x upsampling.

    Replaces the SR3 (nearest-x2 + Conv3x3) upsampler.  A 1x1 conv expands
    the feature map to 4*C channels representing learned wavelet sub-bands;
    the inverse Haar transform produces a 2x larger feature map.

    Checkerboard fix (v61 analysis): learned sub-bands + IDWT are
    mathematically a stride-2/kernel-2 transposed conv -- the classic
    checkerboard generator.  The previous post-expand GroupNorm made it
    worse by normalising LH/HL/HH up to the same variance as LL, injecting
    full-amplitude Nyquist energy at every upsample while the (heavily
    low-passed) losses provide no gradient to suppress it.  Therefore:
      * the GroupNorm now normalises the INPUT features, not the sub-bands;
      * LH/HL/HH are gated by ``hf_gain``, a raw nn.Parameter initialised
        to ZERO (untouched by BaseNetwork.init_weights(), same trick as
        log_gain elsewhere).  At init the module degenerates to a smooth
        LL-only upsampler (ICNR-style); high frequencies are only produced
        once the loss actually asks for them.
    """

    def __init__(self, dim_in, dim_out=None, wavelet='haar', norm_groups=32,
                 legacy_norm=False):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim_in
        # ``legacy_norm`` restores the PRE-checkerboard-fix architecture
        # (GroupNorm on the 4*dim_out sub-bands AFTER expand, no hf_gain gate)
        # so checkpoints trained before the v61 fix (e.g. the ep100 envelope
        # regression baseline) can be loaded for use as a frozen module.
        self.legacy_norm = bool(legacy_norm)
        self.expand = nn.Conv2d(dim_in, dim_out * 4, 1)
        self.idwt = HaarIDWT2D(wavelet)
        if self.legacy_norm:
            ng = min(norm_groups, dim_out * 4)
            self.norm = nn.GroupNorm(ng, dim_out * 4)
        else:
            ng = min(norm_groups, dim_in)
            self.norm = nn.GroupNorm(ng, dim_in)
            self.hf_gain = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        if self.legacy_norm:
            sub = self.norm(self.expand(x))              # [B, 4*dim_out, H, W]
            B, C4, H, W = sub.shape
            C = C4 // 4
            sub = sub.reshape(B, C, 4, H, W)
            return self.idwt(sub[:, :, 0], sub[:, :, 1],
                             sub[:, :, 2], sub[:, :, 3])
        sub = self.expand(self.norm(x))                  # [B, 4*dim_out, H, W]
        B, C4, H, W = sub.shape
        C = C4 // 4
        sub = sub.reshape(B, C, 4, H, W)
        ll = sub[:, :, 0]
        lh = sub[:, :, 1] * self.hf_gain[0]
        hl = sub[:, :, 2] * self.hf_gain[1]
        hh = sub[:, :, 3] * self.hf_gain[2]
        return self.idwt(ll, lh, hl, hh)                 # [B, dim_out, 2H, 2W]
