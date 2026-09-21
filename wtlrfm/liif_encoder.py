"""
LIIF (Local Implicit Image Function) Condition Encoder

Implements a LIIF-inspired conditioning module for super-resolution
diffusion / flow-matching models.

Instead of simple bicubic upsampling or basic feature extraction, this
encoder:
  1. Extracts a deep feature map from the LR condition image at reduced
     resolution via a lightweight CNN backbone.
  2. For each query position (at any target resolution), computes:
       - Local feature: sampled from the feature map
       - Relative coordinate: sub-pixel offset from the nearest feature
         grid point, giving explicit positional awareness
       - Cell size: area each target pixel covers in the feature
         coordinate system, encoding scale information
  3. Combines these via an MLP (the "implicit function") to produce
     position-aware conditioning features.

The resulting multi-scale feature dict has the same interface as
`LRContextEncoder` and can be injected via cross-attention.

Reference:
    Chen et al., "Learning Continuous Image Representation with Local
    Implicit Image Function", CVPR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIIFConditionEncoder(nn.Module):
    """LIIF-inspired condition encoder producing coordinate-aware multi-scale
    features from the LR condition image.

    Args:
        in_channels: Input image channels.
        feature_dim: Channel count of the backbone feature map.
        context_dim: Output channel count (consumed by cross-attention).
        hidden_dim: Hidden dimension of the implicit function MLP.
        encoder_inner_channel: Base channel count of the backbone.
        encoder_channel_mults: Per-stage channel multipliers.
            ``len(mults) - 1`` gives the number of 2x spatial downsamples.
        encoder_res_blocks: Conv blocks per encoder stage.
        use_cell_decode: Include cell-size input to the MLP for scale
            awareness.
        local_ensemble: Average predictions from the 4 nearest feature
            grid points (more robust but 4x MLP cost).
        norm_groups: GroupNorm group count.
    """

    def __init__(
        self,
        in_channels=3,
        feature_dim=256,
        context_dim=256,
        hidden_dim=256,
        encoder_inner_channel=64,
        encoder_channel_mults=(1, 2, 4),
        encoder_res_blocks=2,
        use_cell_decode=True,
        local_ensemble=True,
        norm_groups=32,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.use_cell_decode = use_cell_decode
        self.local_ensemble = local_ensemble

        # ---- Feature Backbone ----
        self.input_conv = nn.Conv2d(in_channels, encoder_inner_channel,
                                    3, padding=1)

        stages, downsamples = [], []
        ch = encoder_inner_channel
        for i, mult in enumerate(encoder_channel_mults):
            target_ch = encoder_inner_channel * mult
            blocks = []
            for _ in range(encoder_res_blocks):
                blocks.append(nn.Sequential(
                    nn.GroupNorm(min(norm_groups, ch), ch),
                    nn.SiLU(),
                    nn.Conv2d(ch, target_ch, 3, padding=1),
                ))
                ch = target_ch
            stages.append(nn.ModuleList(blocks))
            if i < len(encoder_channel_mults) - 1:
                downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                downsamples.append(None)

        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)

        self.feat_proj = nn.Sequential(
            nn.GroupNorm(min(norm_groups, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, feature_dim, 1),
        )

        # ---- Implicit Function (MLP) ----
        mlp_in = feature_dim + 2          # local feature + relative coord
        if use_cell_decode:
            mlp_in += 2                   # + cell size
        self.imnet = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, context_dim),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_coord(h, w, device):
        """Pixel-center coordinates in [-1, 1]."""
        ch = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
        cw = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
        grid_h, grid_w = torch.meshgrid(ch, cw, indexing='ij')
        return torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def extract_features(self, x):
        """Run the backbone and return a compact feature map.

        Returns shape ``[B, feature_dim, H', W']`` where
        ``H' = H / 2^num_downsamples``.
        """
        x = self.input_conv(x)
        for stage, ds in zip(self.stages, self.downsamples):
            for block in stage:
                x = block(x)
            if ds is not None:
                x = ds(x)
        return self.feat_proj(x)

    # ------------------------------------------------------------------
    # LIIF Querying
    # ------------------------------------------------------------------

    def _apply_imnet(self, parts, B, H, W):
        """Concatenate ``parts``, run the implicit MLP, reshape back."""
        inp = torch.cat(parts, dim=1)                    # [B, C_in, H, W]
        C_in = inp.shape[1]
        out = self.imnet(inp.permute(0, 2, 3, 1).reshape(-1, C_in))
        return out.reshape(B, H, W, self.context_dim).permute(0, 3, 1, 2)

    def _query_simple(self, feat, feat_coord_map, coord_batch,
                      target_h, target_w, feat_h, feat_w):
        """Bilinear feature sampling + relative coord + MLP."""
        B = feat.shape[0]
        device = feat.device

        grid = coord_batch.flip(-1)                       # (h,w) → (x,y)

        q_feat = F.grid_sample(feat, grid, mode='bilinear',
                               padding_mode='border', align_corners=False)

        q_coord = F.grid_sample(feat_coord_map, grid, mode='nearest',
                                padding_mode='border', align_corners=False)

        rel_coord = coord_batch.permute(0, 3, 1, 2) - q_coord
        rel_coord[:, 0, :, :] *= feat_h
        rel_coord[:, 1, :, :] *= feat_w

        parts = [q_feat, rel_coord]
        if self.use_cell_decode:
            cell_h = (2.0 / target_h) * feat_h
            cell_w = (2.0 / target_w) * feat_w
            cell = torch.tensor([cell_h, cell_w], device=device,
                                dtype=feat.dtype).view(1, 2, 1, 1)
            cell = cell.expand(B, -1, target_h, target_w)
            parts.append(cell)

        return self._apply_imnet(parts, B, target_h, target_w)

    def _query_ensemble(self, feat, feat_coord_map, coord_batch,
                        target_h, target_w, feat_h, feat_w):
        """Local ensemble over 4 nearest grid points (LIIF paper)."""
        B = feat.shape[0]
        device = feat.device
        eps = 1e-6

        rx = 1.0 / feat_h
        ry = 1.0 / feat_w

        preds, areas = [], []
        for vx in (-1, 1):
            for vy in (-1, 1):
                coord_ = coord_batch.clone()
                coord_[..., 0] += vx * rx + eps
                coord_[..., 1] += vy * ry + eps

                grid = coord_.flip(-1)

                q_feat = F.grid_sample(feat, grid, mode='nearest',
                                       padding_mode='border',
                                       align_corners=False)
                q_coord = F.grid_sample(feat_coord_map, grid, mode='nearest',
                                        padding_mode='border',
                                        align_corners=False)

                rel_coord = coord_batch.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= feat_h
                rel_coord[:, 1, :, :] *= feat_w

                parts = [q_feat, rel_coord]
                if self.use_cell_decode:
                    cell_h = (2.0 / target_h) * feat_h
                    cell_w = (2.0 / target_w) * feat_w
                    cell = torch.tensor([cell_h, cell_w], device=device,
                                        dtype=feat.dtype).view(1, 2, 1, 1)
                    cell = cell.expand(B, -1, target_h, target_w)
                    parts.append(cell)

                preds.append(self._apply_imnet(parts, B, target_h, target_w))
                area = torch.abs(
                    rel_coord[:, 0:1, :, :] * rel_coord[:, 1:2, :, :])
                areas.append(area + eps)

        total = sum(areas)
        ret = sum(p * (a / total)
                  for p, a in zip(preds, reversed(areas)))
        return ret

    def query_features(self, feat, target_h, target_w):
        """LIIF-style feature query at ``(target_h, target_w)``."""
        B, C, feat_h, feat_w = feat.shape
        device = feat.device

        coord = self.make_coord(target_h, target_w, device)
        coord_batch = coord.unsqueeze(0).expand(B, -1, -1, -1)

        feat_coord = self.make_coord(feat_h, feat_w, device)
        feat_coord_map = (feat_coord.permute(2, 0, 1)
                          .unsqueeze(0).expand(B, -1, -1, -1))

        if self.local_ensemble:
            return self._query_ensemble(
                feat, feat_coord_map, coord_batch,
                target_h, target_w, feat_h, feat_w)
        return self._query_simple(
            feat, feat_coord_map, coord_batch,
            target_h, target_w, feat_h, feat_w)

    # ------------------------------------------------------------------
    # Per-sample processing (for native LR with variable sizes)
    # ------------------------------------------------------------------

    def _forward_native_lr(self, lr_images, lr_sizes, target_h, target_w):
        """Process per-sample native LR images with variable sizes.

        Each sample is cropped from its padding, run through the backbone
        individually, then queried at the common target resolutions.  This
        gives correct cell-size / coordinate information for each scale.

        Args:
            lr_images: ``[B, C, max_H, max_W]`` padded native LR batch.
            lr_sizes:  ``[B]`` actual spatial size per sample (before pad).
            target_h:  HR height (determines output resolution ladder).
            target_w:  HR width.

        Returns:
            ``{resolution: [B, context_dim, res, res]}``
        """
        B = lr_images.shape[0]

        resolutions = []
        r = max(target_h, target_w)
        while r >= 8:
            resolutions.append(r)
            r //= 2

        accum = {res: [] for res in resolutions}

        for i in range(B):
            s = int(lr_sizes[i].item()) if torch.is_tensor(lr_sizes[i]) \
                else int(lr_sizes[i])
            lr_i = lr_images[i:i + 1, :, :s, :s]
            feat_i = self.extract_features(lr_i)          # [1, F, h_f, w_f]
            for res in resolutions:
                accum[res].append(self.query_features(feat_i, res, res))

        return {res: torch.cat(accum[res], dim=0) for res in resolutions}

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def forward(self, lr_input, target_h=None, target_w=None,
                lr_sizes=None):
        """Produce multi-scale coordinate-aware features.

        Supports two modes:

        **Legacy mode** (``target_h is None``):
            ``lr_input`` is assumed to be at HR resolution (the upsampled
            condition image).  Features are generated at resolutions
            ``{H, H/2, …, 8}``.

        **Native-LR mode** (``target_h`` given):
            ``lr_input`` is the **padded** native LR batch.
            ``lr_sizes`` gives the actual (unpadded) spatial size per sample.
            Each sample is processed individually so that coordinates and
            cell sizes correctly reflect the true LR→HR scale.

        Returns:
            ``{resolution: [B, context_dim, res, res]}``
        """
        if target_h is not None and lr_sizes is not None:
            return self._forward_native_lr(
                lr_input, lr_sizes, target_h, target_w)

        feat = self.extract_features(lr_input)
        H, W = lr_input.shape[-2:]
        if target_h is not None:
            H, W = target_h, target_w
        features = {}
        res = max(H, W)
        while res >= 8:
            features[res] = self.query_features(feat, res, res)
            res //= 2
        return features
