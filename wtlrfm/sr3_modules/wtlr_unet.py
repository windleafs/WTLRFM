"""
WTLR-Enhanced SR3 UNet

Integrates the WTLR (Wavelet-Transform based Low-Resolution image) encoder
from DMUISR into the SR3 UNet backbone for flow-matching super-resolution.

The WTLR encoder extracts multi-scale frequency-aware features from the LR
condition image, which are injected into the UNet decoder via adaptive fusion
at matching resolution levels. This provides the velocity network with rich
frequency-domain guidance, improving recovery of fine structures.

Architecture:
    Encoder: standard SR3 UNet encoder (downsampling path)
    Bottleneck: ResBlock + SelfAttention
    Decoder: SR3 UNet decoder with WTLR feature injection at each upsampling stage
    WTLR: parallel branch processing LR image via multi-level DWT
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from ..wtlr_encoder import WTLREncoder, DWTDownsample, IDWTUpsample
from ..liif_encoder import LIIFConditionEncoder
from .speckle_layer import SpeckleLayer
from .kan_layers import TokKANMidBlock


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class PhysicsParamEmbed(nn.Module):
    """Physical-parameter embedding for ultrasound imaging conditions.

    Inspired by PgD-ULM (Qiang et al. 2024): each physical scalar is first
    independently projected to emb_dim//input_dim dimensions, then the
    concatenated result is projected to emb_dim through a 2-layer MLP with
    LayerNorm.  This allows the network to learn nonlinear interactions between
    physical parameters (e.g. depth × pixel-size coupling).

    A learnable null_embed is returned when physics_params is None, implementing
    the "null parameter strategy" from PgD-ULM that allows graceful degradation
    on samples lacking metadata.

    Args:
        input_dim:  Number of physical scalar inputs (default 4:
                    z_start_norm, z_end_norm, axial_ps_norm, lateral_ps_norm).
        emb_dim:    Output embedding dimension (should match noise_level_channel
                    in WTLRUNet, i.e. inner_channel).
    """

    def __init__(self, input_dim: int = 4, emb_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        per_feat_dim = max(emb_dim // input_dim, 4)

        # Independent per-feature projections (like PgD-ULM FeatureModulationEmbed)
        # Last linear in each branch is zero-init for training stability.
        def _make_feat_proj():
            fc1 = nn.Linear(1, per_feat_dim)
            fc2 = nn.Linear(per_feat_dim, per_feat_dim)
            nn.init.zeros_(fc2.weight)
            nn.init.zeros_(fc2.bias)
            return nn.Sequential(fc1, nn.GELU(), fc2)

        self.feat_projs = nn.ModuleList([
            _make_feat_proj() for _ in range(input_dim)
        ])

        # Fusion MLP: [per_feat_dim * input_dim] -> emb_dim
        # The final linear is zero-initialised so physics_emb outputs exactly
        # zero at the start of training.  This means the physics branch adds
        # no perturbation to the pre-trained time embedding, preserving the
        # original network behaviour and avoiding the gradient explosion that
        # occurs when a randomly-initialised offset disrupts the FeatureWiseAffine
        # conditioning signal in every ResBlock simultaneously.
        _fc1 = nn.Linear(per_feat_dim * input_dim, emb_dim)
        _ln  = nn.LayerNorm(emb_dim)
        _act = nn.GELU()
        _fc2 = nn.Linear(emb_dim, emb_dim)
        nn.init.zeros_(_fc2.weight)
        nn.init.zeros_(_fc2.bias)
        self.fusion = nn.Sequential(_fc1, _ln, _act, _fc2)

        # Learnable null embedding (used when physics_params is None)
        # Also zero-init for the same stability reason.
        self.null_embed = nn.Parameter(torch.zeros(1, emb_dim))

    def forward(self, physics_params):
        """
        Args:
            physics_params: (B, input_dim) float tensor in [0, 1], or None.

        Returns:
            emb: (B, emb_dim) physics embedding.
        """
        if physics_params is None:
            # Null-parameter strategy: return learned null embedding
            # (broadcast over batch at call site)
            return self.null_embed  # (1, emb_dim)

        B = physics_params.shape[0]
        feats = []
        for i, proj in enumerate(self.feat_projs):
            fi = physics_params[:, i].unsqueeze(1)  # (B, 1)
            feats.append(proj(fi))                   # (B, per_feat_dim)
        cat = torch.cat(feats, dim=1)                # (B, per_feat_dim * input_dim)
        return self.fusion(cat)                      # (B, emb_dim)


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32,
                 dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(dim, dim_out, noise_level_emb_dim,
                                     norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


# --------------- WTLR Fusion Modules ---------------

class WTLRCrossAttention(nn.Module):
    """
    Cross-attention between UNet features (query) and WTLR features (key/value).
    Allows the decoder to selectively attend to frequency-domain information.
    """

    def __init__(self, unet_channels, wtlr_channels, n_head=4, norm_groups=32):
        super().__init__()
        self.n_head = n_head
        head_dim = unet_channels // n_head
        self.scale = head_dim ** -0.5

        self.norm_q = nn.GroupNorm(min(norm_groups, unet_channels), unet_channels)
        self.norm_k = nn.GroupNorm(min(norm_groups, wtlr_channels), wtlr_channels)

        self.q_proj = nn.Conv2d(unet_channels, unet_channels, 1)
        self.k_proj = nn.Conv2d(wtlr_channels, unet_channels, 1)
        self.v_proj = nn.Conv2d(wtlr_channels, unet_channels, 1)
        self.out_proj = nn.Conv2d(unet_channels, unet_channels, 1)

    def forward(self, unet_feat, wtlr_feat):
        B, C, H, W = unet_feat.shape

        q = self.q_proj(self.norm_q(unet_feat))
        k = self.k_proj(self.norm_k(wtlr_feat))
        v = self.v_proj(wtlr_feat)

        head_dim = C // self.n_head
        q = q.view(B, self.n_head, head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.n_head, head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.n_head, head_dim, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return unet_feat + self.out_proj(out)


class AdaptiveWTLRFusion(nn.Module):
    """
    Adaptive fusion of UNet skip-connection features with WTLR features.
    Uses a learned gating mechanism to control how much wavelet information
    flows into the decoder at each resolution level.

    fusion_type:
        'gate':      learned sigmoid gate blending UNet and WTLR features
        'cross_attn': cross-attention from UNet features to WTLR features
        'add':       project + add with learnable scale
    """

    def __init__(self, unet_channels, wtlr_channels, fusion_type='gate', norm_groups=32):
        super().__init__()
        self.fusion_type = fusion_type
        ng = min(norm_groups, unet_channels)

        if fusion_type == 'gate':
            self.wtlr_proj = nn.Sequential(
                nn.Conv2d(wtlr_channels, unet_channels, 1),
                nn.GroupNorm(ng, unet_channels),
                Swish(),
            )
            self.gate = nn.Sequential(
                nn.Conv2d(unet_channels * 2, unet_channels, 1),
                nn.Sigmoid(),
            )
        elif fusion_type == 'cross_attn':
            self.cross_attn = WTLRCrossAttention(unet_channels, wtlr_channels,
                                                  norm_groups=norm_groups)
        elif fusion_type == 'add':
            self.wtlr_proj = nn.Sequential(
                nn.Conv2d(wtlr_channels, unet_channels, 1),
                nn.GroupNorm(ng, unet_channels),
                Swish(),
            )
            self.scale = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, unet_feat, wtlr_feat):
        """
        Args:
            unet_feat: [B, C_u, H, W] from UNet skip connection
            wtlr_feat: [B, C_w, H, W] from WTLR encoder (may need resize)
        Returns:
            fused: [B, C_u, H, W]
        """
        if wtlr_feat.shape[-2:] != unet_feat.shape[-2:]:
            wtlr_feat = F.interpolate(wtlr_feat, size=unet_feat.shape[-2:],
                                       mode='bilinear', align_corners=False)

        if self.fusion_type == 'gate':
            w_proj = self.wtlr_proj(wtlr_feat)
            gate = self.gate(torch.cat([unet_feat, w_proj], dim=1))
            return unet_feat + gate * w_proj

        elif self.fusion_type == 'cross_attn':
            return self.cross_attn(unet_feat, wtlr_feat)

        elif self.fusion_type == 'add':
            w_proj = self.wtlr_proj(wtlr_feat)
            return unet_feat + self.scale * w_proj


# --------------- LR Conditioning Modules ---------------

class ZeroConv2d(nn.Module):
    """Convolution layer initialized to zero, used by ControlNet for
    gradual residual injection (outputs start at zero)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class LRConditionProjector(nn.Module):
    """Projects LR condition image to match UNet inner_channel dimension
    for element-wise addition after the first encoder convolution."""

    def __init__(self, cond_channels, target_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            Swish(),
            nn.Conv2d(hidden_channels, target_channels, 3, padding=1),
        )

    def forward(self, cond):
        return self.net(cond)


class ControlNetEncoder(nn.Module):
    """ControlNet encoder that mirrors the main UNet encoder.

    Processes the LR condition image through the same architecture as the
    main encoder, producing zero-initialised residuals at every skip level
    that are added to the main encoder's skip features.
    """

    def __init__(self, hint_channels, inner_channel, channel_mults, attn_res,
                 res_blocks, dropout, noise_level_channel, norm_groups,
                 image_size, use_zero_conv=True):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(hint_channels, inner_channel, 3, padding=1),
            Swish(),
            nn.Conv2d(inner_channel, inner_channel, 3, padding=1),
        )

        _zc = (lambda c_in, c_out: ZeroConv2d(c_in, c_out)) if use_zero_conv \
              else (lambda c_in, c_out: nn.Conv2d(c_in, c_out, 1))

        self.zero_conv_input = _zc(inner_channel, inner_channel)

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        now_res = image_size

        blocks, zero_convs = [], []
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                blocks.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout,
                    with_attn=use_attn))
                zero_convs.append(_zc(channel_mult, channel_mult))
                pre_channel = channel_mult
            if not is_last:
                blocks.append(Downsample(pre_channel))
                zero_convs.append(_zc(pre_channel, pre_channel))
                now_res //= 2

        self.blocks = nn.ModuleList(blocks)
        self.zero_convs = nn.ModuleList(zero_convs)

    def forward(self, hint, time_emb):
        """Returns a list of control features aligned with the main encoder's
        ``feats`` list (same length, same shapes)."""
        x = self.input_conv(hint)
        outputs = [self.zero_conv_input(x)]
        for block, zc in zip(self.blocks, self.zero_convs):
            if isinstance(block, ResnetBlocWithAttn):
                x = block(x, time_emb)
            else:
                x = block(x)
            outputs.append(zc(x))
        return outputs


class LRCrossAttention(nn.Module):
    """Cross-attention where UNet decoder features (Q) attend to
    LR context features (K, V).  Spatial sizes are matched via
    bilinear interpolation when they differ."""

    def __init__(self, query_dim, context_dim, n_head=4, norm_groups=32):
        super().__init__()
        self.n_head = n_head
        head_dim = query_dim // n_head
        self.scale = head_dim ** -0.5

        self.norm_q = nn.GroupNorm(min(norm_groups, query_dim), query_dim)
        self.norm_kv = nn.GroupNorm(min(norm_groups, context_dim), context_dim)

        self.q_proj = nn.Conv2d(query_dim, query_dim, 1)
        self.k_proj = nn.Conv2d(context_dim, query_dim, 1)
        self.v_proj = nn.Conv2d(context_dim, query_dim, 1)
        self.out_proj = nn.Conv2d(query_dim, query_dim, 1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        if context.shape[-2:] != (H, W):
            context = F.interpolate(context, size=(H, W),
                                    mode='bilinear', align_corners=False)

        q = self.q_proj(self.norm_q(x))
        k = self.k_proj(self.norm_kv(context))
        v = self.v_proj(context)

        hd = C // self.n_head
        N = H * W
        q = q.view(B, self.n_head, hd, N).transpose(2, 3)
        k = k.view(B, self.n_head, hd, N).transpose(2, 3)
        v = v.view(B, self.n_head, hd, N).transpose(2, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(2, 3).reshape(B, C, H, W)
        return x + self.out_proj(out)


class LRContextEncoder(nn.Module):
    """Lightweight encoder that produces multi-scale context features from
    the LR condition image, used as K/V source for :class:`LRCrossAttention`."""

    def __init__(self, in_channels, context_dim, inner_channel=64,
                 channel_mults=(1, 2, 4, 8), num_res_blocks=2,
                 norm_groups=32):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, inner_channel, 3, padding=1)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.out_projs = nn.ModuleList()

        pre_ch = inner_channel
        for i, mult in enumerate(channel_mults):
            ch = inner_channel * mult
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(nn.Sequential(
                    nn.GroupNorm(min(norm_groups, pre_ch), pre_ch),
                    nn.SiLU(),
                    nn.Conv2d(pre_ch, ch, 3, padding=1),
                ))
                pre_ch = ch
            self.stages.append(nn.ModuleList(blocks))
            self.out_projs.append(nn.Conv2d(ch, context_dim, 1))

            if i < len(channel_mults) - 1:
                self.downsamples.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.downsamples.append(None)

    def forward(self, cond):
        """Returns ``{resolution: tensor}`` mapping."""
        x = self.input_conv(cond)
        features = {}
        for stage, proj, ds in zip(self.stages, self.out_projs,
                                   self.downsamples):
            for block in stage:
                x = block(x)
            features[x.shape[-1]] = proj(x)
            if ds is not None:
                x = ds(x)
        return features


# --------------- Main WTLRUNet ---------------

class WTLRUNet(nn.Module):
    """
    SR3 UNet enhanced with WTLR encoder for flow-matching super-resolution.

    The WTLR encoder runs in parallel on the LR condition image (extracted
    from the first `cond_channels` of the input). Its multi-scale features
    are fused into the UNet decoder via AdaptiveWTLRFusion at each resolution
    level where a matching WTLR feature exists.

    Args:
        in_channel: Total input channels (cond + noisy, typically 6)
        out_channel: Output channels (typically 3)
        inner_channel: Base UNet channel count
        norm_groups: GroupNorm groups
        channel_mults: Channel multiplier per encoder stage
        attn_res: Resolutions at which to apply self-attention
        res_blocks: ResBlocks per stage
        dropout: Dropout rate
        with_noise_level_emb: Use time/noise embedding
        image_size: Input spatial size (e.g. 128)
        cond_channels: Channels of the condition image (LR)
        wtlr_channels: WTLR encoder feature channels
        wtlr_levels: Number of DWT decomposition levels
        wtlr_wavelet: Wavelet family for DWT
        wtlr_res_blocks: ResBlocks per sub-band in WTLR encoder
        wtlr_fusion: Fusion type ('gate', 'cross_attn', 'add')
    """

    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8),
        attn_res=(16,),
        res_blocks=2,
        dropout=0.2,
        with_noise_level_emb=True,
        image_size=128,
        cond_channels=3,
        wtlr_channels=64,
        wtlr_levels=3,
        wtlr_wavelet='haar',
        wtlr_res_blocks=2,
        wtlr_fusion='gate',
        lr_condition_mode='concat',
        lr_condition_config=None,
        use_dwt_resample=False,
        dwt_wavelet='haar',
        legacy_idwt_norm=False,
        use_speckle_layer=False,
        speckle_config=None,
        use_physics_params=False,
        physics_param_dim=4,
        use_kan_bottleneck=False,
        kan_num_layers=3,
        kan_grid_size=5,
        kan_spline_order=3,
        kan_grid_range=(-2.0, 2.0),
        kan_dropout=0.0,
    ):
        super().__init__()

        self.cond_channels = cond_channels
        self.wtlr_levels = wtlr_levels
        self.lr_condition_mode = lr_condition_mode
        self.use_dwt_resample = use_dwt_resample
        self.dwt_wavelet = dwt_wavelet
        self.use_speckle_layer = use_speckle_layer
        self.speckle_config = speckle_config or {}

        # Pick spatial-resampling primitives.  Default keeps the original SR3
        # behaviour (stride-2 Conv3x3 / nearest-x2 + Conv3x3); enabling
        # ``use_dwt_resample`` swaps in DWT/IDWT versions defined in
        # ``wtlr_encoder.py`` for an aliasing-free, frequency-aware UNet
        # backbone that pairs naturally with the WTLR side branch.
        if use_dwt_resample:
            _make_down = lambda c: DWTDownsample(c, c, wavelet=dwt_wavelet,
                                                 norm_groups=norm_groups)
            _make_up = lambda c: IDWTUpsample(c, c, wavelet=dwt_wavelet,
                                              norm_groups=norm_groups,
                                              legacy_norm=legacy_idwt_norm)
        else:
            _make_down = lambda c: Downsample(c)
            _make_up = lambda c: Upsample(c)
        self._make_down = _make_down
        self._make_up = _make_up

        # ---- WTLR Encoder ----
        self.wtlr_encoder = WTLREncoder(
            in_channels=cond_channels,
            feature_channels=wtlr_channels,
            num_levels=wtlr_levels,
            wavelet=wtlr_wavelet,
            num_res_blocks=wtlr_res_blocks,
        )

        # ---- Time Embedding ----
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel),
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        # ---- Physics Parameter Embedding ----
        # When enabled, a PhysicsParamEmbed projects the 4 normalised physical
        # scalars (z_start, z_end, axial_pixel_size, lateral_pixel_size) to the
        # same dimension as noise_level_channel and adds them to the time
        # embedding before every ResBlock.  This is the approach used in
        # PgD-ULM (Qiang et al. 2024) adapted from FeatureInteractionEmbed.
        self.use_physics_params = bool(use_physics_params)
        if use_physics_params and noise_level_channel is not None:
            self.physics_emb = PhysicsParamEmbed(
                input_dim=int(physics_param_dim),
                emb_dim=noise_level_channel,
            )
        else:
            self.physics_emb = None

        # ---- LR Condition Mode Modules ----
        lr_cfg = lr_condition_config or {}

        if lr_condition_mode == 'add':
            proj_ch = lr_cfg.get('add_proj_channels', 64)
            self.cond_proj = LRConditionProjector(
                cond_channels, inner_channel, proj_ch)

        elif lr_condition_mode == 'controlnet':
            ctrl_cfg = lr_cfg.get('controlnet', {})
            self.controlnet_encoder = ControlNetEncoder(
                hint_channels=ctrl_cfg.get('hint_channels', cond_channels),
                inner_channel=inner_channel,
                channel_mults=ctrl_cfg.get('channel_mults', channel_mults),
                attn_res=set(attn_res),
                res_blocks=ctrl_cfg.get('res_blocks', res_blocks),
                dropout=dropout,
                noise_level_channel=noise_level_channel,
                norm_groups=norm_groups,
                image_size=image_size,
                use_zero_conv=ctrl_cfg.get('use_zero_conv', True),
            )

        elif lr_condition_mode == 'cross_attention':
            cross_cfg = lr_cfg.get('cross_attn', {})
            self._cross_context_dim = cross_cfg.get('context_dim', 256)
            self._cross_num_heads = cross_cfg.get('num_heads', 4)
            self.lr_context_encoder = LRContextEncoder(
                in_channels=cross_cfg.get('encoder_channels', cond_channels),
                context_dim=self._cross_context_dim,
                inner_channel=cross_cfg.get('encoder_inner_channel', 64),
                channel_mults=cross_cfg.get(
                    'encoder_channel_mults', (1, 2, 4, 8)),
                num_res_blocks=cross_cfg.get('encoder_res_blocks', 2),
                norm_groups=norm_groups,
            )

        elif lr_condition_mode == 'liif':
            liif_cfg = lr_cfg.get('liif', {})
            self._cross_context_dim = liif_cfg.get('context_dim', 256)
            self._cross_num_heads = liif_cfg.get('num_heads', 8)
            self.lr_context_encoder = LIIFConditionEncoder(
                in_channels=liif_cfg.get('encoder_channels', cond_channels),
                feature_dim=liif_cfg.get('feature_dim', 256),
                context_dim=self._cross_context_dim,
                hidden_dim=liif_cfg.get('hidden_dim', 256),
                encoder_inner_channel=liif_cfg.get(
                    'encoder_inner_channel', 64),
                encoder_channel_mults=liif_cfg.get(
                    'encoder_channel_mults', (1, 2, 4)),
                encoder_res_blocks=liif_cfg.get('encoder_res_blocks', 2),
                use_cell_decode=liif_cfg.get('use_cell_decode', True),
                local_ensemble=liif_cfg.get('local_ensemble', True),
                norm_groups=norm_groups,
            )

        # ---- Encoder ----
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size

        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(self._make_down(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        # ---- Middle ----
        # First bottleneck block always keeps the standard Conv+SelfAttention
        # ResnetBlocWithAttn design (spatial locality + global attention at
        # the lowest resolution). When ``use_kan_bottleneck`` is enabled,
        # the second block is replaced by a Tokenized-KAN block (adapted
        # from U-KAN, Li et al. 2024/2025 -- see kan_layers.py docstring)
        # that models nonlinear per-channel interactions via learnable
        # spline functions instead of a plain ResBlock. This is an
        # opt-in, backward-compatible change: ``use_kan_bottleneck``
        # defaults to False, so every existing config/checkpoint is
        # bit-for-bit unaffected.
        self.use_kan_bottleneck = bool(use_kan_bottleneck)
        mid_blocks = [
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups, dropout=dropout, with_attn=True),
        ]
        if self.use_kan_bottleneck:
            mid_blocks.append(TokKANMidBlock(
                pre_channel,
                noise_level_emb_dim=noise_level_channel,
                num_layers=kan_num_layers,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                grid_range=tuple(kan_grid_range),
                dropout=kan_dropout,
            ))
        else:
            mid_blocks.append(ResnetBlocWithAttn(pre_channel, pre_channel,
                               noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups, dropout=dropout, with_attn=False))
        self.mid = nn.ModuleList(mid_blocks)

        # ---- WTLR Fusion modules (one per decoder ResBlock) ----
        self.wtlr_resolutions = set()
        r = image_size
        for level in range(wtlr_levels):
            r = r // 2
            self.wtlr_resolutions.add(r)

        # ---- Decoder ----
        ups = []
        fusion_modules = []
        cross_attn_modules = []
        cross_attn_active = []

        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks + 1):
                skip_ch = feat_channels.pop()
                ups.append(ResnetBlocWithAttn(
                    pre_channel + skip_ch, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult

                if now_res in self.wtlr_resolutions:
                    fusion_modules.append(
                        AdaptiveWTLRFusion(skip_ch, wtlr_channels,
                                           fusion_type=wtlr_fusion,
                                           norm_groups=norm_groups))
                else:
                    fusion_modules.append(None)

                if lr_condition_mode in ('cross_attention', 'liif') and use_attn:
                    cross_attn_modules.append(
                        LRCrossAttention(
                            channel_mult, self._cross_context_dim,
                            n_head=self._cross_num_heads,
                            norm_groups=norm_groups))
                    cross_attn_active.append(True)
                else:
                    cross_attn_modules.append(None)
                    cross_attn_active.append(False)

            if not is_last:
                ups.append(self._make_up(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)
        self.fusion_modules = nn.ModuleList(
            [m if m is not None else nn.Identity() for m in fusion_modules]
        )
        self._fusion_active = [m is not None for m in fusion_modules]

        if lr_condition_mode in ('cross_attention', 'liif'):
            self.cross_attn_modules = nn.ModuleList(
                [m if m is not None else nn.Identity()
                 for m in cross_attn_modules])
            self._cross_attn_active = cross_attn_active

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        # ---- Speckle layer (optional) ----
        # Inserted *after* the second-to-last decoder ResBlock so that the
        # feature map already lives at the full output resolution (see
        # SpeckleGAN, Fig. 3).  The layer applies a Fourier-optics speckle
        # operator with ``num_scales`` learnable cut-off frequencies, fused
        # back to the same channel count via channel attention.
        if self.use_speckle_layer:
            total_decoder_resblocks = len(channel_mults) * (res_blocks + 1)
            # ``fusion_idx`` is post-incremented in forward; trigger the
            # speckle call once after the (N-1)-th ResBlock has been run,
            # i.e. when fusion_idx has just reached ``total - 1``.
            self._speckle_after_idx = total_decoder_resblocks - 1

            sp_cfg = dict(self.speckle_config)
            sp_cfg.setdefault('num_scales', 4)
            sp_cfg.setdefault('image_size', image_size)
            self.speckle_layer = SpeckleLayer(
                channels=pre_channel,
                num_scales=sp_cfg['num_scales'],
                image_size=sp_cfg['image_size'],
                d_init=sp_cfg.get('d_init', None),
                sharpness=sp_cfg.get('sharpness', 4.0),
                attn_reduction=sp_cfg.get('attn_reduction', 8),
                residual_gate=sp_cfg.get('residual_gate', True),
                polar_transform=sp_cfg.get('polar_transform', False),
                polar_size=sp_cfg.get('polar_size', None),
            )
        else:
            self._speckle_after_idx = -1
            self.speckle_layer = None

    def forward(self, x, time, cond=None, lr_image=None, lr_size=None,
                physics_params=None, **kwargs):
        """
        Args:
            x: [B, in_channel, H, W]
                - concat mode: concatenated (y_cond, y_noisy), in_channel=6
                - other modes: y_noisy only, in_channel=3
            time: [B] noise / time level
            cond: [B, cond_channels, H, W] LR condition image
                  (required for add / controlnet / cross_attention modes,
                   ignored in concat mode)
            lr_image: [B, C, max_lr_H, max_lr_W] padded native LR image
                      (optional, used by LIIF mode for true arbitrary-scale)
            lr_size: [B] actual spatial size per sample before padding
                     (optional, used together with lr_image)
            physics_params: [B, physics_param_dim] normalised physical
                            parameters in [0, 1], or None (null parameter).
        Returns:
            [B, out_channel, H, W]
        """
        if self.lr_condition_mode == 'concat':
            cond_img = x[:, :self.cond_channels, :, :]
        else:
            cond_img = cond

        wtlr_features = self.wtlr_encoder(cond_img)

        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        # Inject physics parameter embedding into the time embedding.
        # Following PgD-ULM: physics_emb is added to t so every ResBlock
        # automatically conditions on the physical imaging parameters via the
        # existing FeatureWiseAffine pathway — no structural change required.
        if self.physics_emb is not None and t is not None:
            p_emb = self.physics_emb(physics_params)  # (B, C) or (1, C)
            if p_emb.shape[0] == 1:
                # null_embed broadcast — expand to batch
                p_emb = p_emb.expand(t.shape[0], -1)
            t = t + p_emb  # additive fusion into time token

        # Mode-specific pre-processing
        ctrl_feats = None
        lr_context = None
        if self.lr_condition_mode == 'controlnet':
            ctrl_feats = self.controlnet_encoder(cond_img, t)
        elif self.lr_condition_mode in ('cross_attention', 'liif'):
            if self.lr_condition_mode == 'liif' and lr_image is not None:
                target_h, target_w = x.shape[-2], x.shape[-1]
                lr_context = self.lr_context_encoder(
                    lr_image, target_h=target_h, target_w=target_w,
                    lr_sizes=lr_size)
            else:
                lr_context = self.lr_context_encoder(cond_img)

        # Encoder
        feats = []
        for i, layer in enumerate(self.downs):
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

            if self.lr_condition_mode == 'add' and i == 0:
                x = x + self.cond_proj(cond_img)

            feats.append(x)

        # Middle
        for layer in self.mid:
            if isinstance(layer, (ResnetBlocWithAttn, TokKANMidBlock)):
                x = layer(x, t)
            else:
                x = layer(x)

        # Decoder with WTLR fusion + mode-specific injection
        fusion_idx = 0
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                skip = feats.pop()
                skip_res = skip.shape[-1]

                # ControlNet residual injection
                if ctrl_feats is not None:
                    skip = skip + ctrl_feats.pop()

                # WTLR fusion
                if self._fusion_active[fusion_idx]:
                    wtlr_feat = wtlr_features.get(skip_res)
                    if wtlr_feat is not None:
                        skip = self.fusion_modules[fusion_idx](skip, wtlr_feat)

                x = layer(torch.cat([x, skip], dim=1), t)

                # Cross-attention with LR context
                if (self.lr_condition_mode in ('cross_attention', 'liif')
                        and self._cross_attn_active[fusion_idx]):
                    cur_res = x.shape[-1]
                    ctx = lr_context.get(cur_res)
                    if ctx is None:
                        closest = min(lr_context.keys(),
                                      key=lambda r: abs(r - cur_res))
                        ctx = lr_context[closest]
                    x = self.cross_attn_modules[fusion_idx](x, ctx)

                fusion_idx += 1

                # Speckle layer: inserted *after* the 2nd-to-last decoder
                # ResBlock so that the feature map already lives at the
                # full output resolution.
                if (self.speckle_layer is not None
                        and fusion_idx == self._speckle_after_idx):
                    x = self.speckle_layer(x)
            else:
                x = layer(x)

        return self.final_conv(x)
