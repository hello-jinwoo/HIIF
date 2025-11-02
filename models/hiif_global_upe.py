"""
HIIF with Global Decoder and User Preference Embedding (UPE)

Key differences from HIIF-PPS:
- Single global decoder (no user-specific decoders)
- UPE injection into decoder input
- Increased capacity (384 hidden_dim, 20 blocks)
- Simplified training (no decoder load/offload)
- RGB-only support (input_type='rgb', output_type='rgb_residual')

Architecture:
- Shared encoder (EDSR/SwinIR/RDN) for all users
- UPE Processor (shallow transformer) to process preference embeddings
- Global decoder with UPE conditioning
- No test-time training required (direct inference)

Dimension Flow:
    Input RGB: (B, 3, H, W)
    ↓ Encoder
    Features: (B, 384, H', W')
    ↓ Query + UPE
    Raw UPE: (B, 16, 1280)
    ↓ UPE Processor
    Processed UPE: (B, 512)
    ↓ Spatial Expansion
    UPE Expanded: (B, H*W, 512)
    ↓ Concat with grid + hi_coord
    Decoder Input: (B, H*W, 2052) = 1536 + 2 + 2 + 512
    ↓ Global Decoder
    Output RGB: (B, 3, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models.hiif import MLP_with_shortcut, qkv_attn, compute_hi_coord
from models.upe_processor import UPEProcessor
from utils import make_coord


@register('hiif-global-upe')
class HIIF_Global_UPE(nn.Module):
    """
    HIIF with Global Decoder and User Preference Embedding.

    Key Features:
    - Single decoder for all users (no load/offload)
    - UPE injection for user-specific conditioning
    - Increased decoder capacity (384 hidden_dim, 20 blocks)
    - Direct inference (no test-time training)
    """

    def __init__(self, encoder_spec, hidden_dim=384, blocks=24,
                 upe_dim=512, upe_processor_config=None):
        """
        Args:
            encoder_spec: Encoder configuration dict
            hidden_dim: Hidden dimension for decoder (default: 384, increased from 256)
            blocks: Number of attention heads (default: 24, increased from 16)
                    Note: Must be divisible by hidden_dim (384/24=16)
            upe_dim: UPE output dimension (default: 512)
            upe_processor_config: UPE processor config dict (optional)
                If None, uses default: {
                    'input_dim': 1280,  # DINO (768) + CLIP (512)
                    'hidden_dim': 512,
                    'output_dim': 512,
                    'num_layers': 3,
                    'num_heads': 8,
                }
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.blocks = blocks
        self.upe_dim = upe_dim
        self.n_hi_layers = 6

        # RGB-only for now (no color space variations)
        self.input_type = 'rgb'
        self.output_type = 'rgb_residual'

        # Update encoder spec with n_colors=3 (RGB only)
        encoder_spec_copy = encoder_spec.copy()
        if 'args' not in encoder_spec_copy:
            encoder_spec_copy['args'] = {}
        encoder_spec_copy['args']['n_colors'] = 3

        # Shared encoder
        self.encoder = models.make(encoder_spec_copy)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)

        # UPE Processor
        if upe_processor_config is None:
            upe_processor_config = {
                'input_dim': 1280,      # Default: DINO (768) + CLIP (512)
                'hidden_dim': 512,
                'output_dim': upe_dim,
                'num_layers': 3,
                'num_heads': 8,
            }
        self.upe_processor = UPEProcessor(**upe_processor_config)

        # Global decoder (single, always in memory)
        self.decoder = self._create_decoder(hidden_dim, blocks, upe_dim)

        print(f"[HIIF_Global_UPE] Initialized:")
        print(f"  Encoder: {encoder_spec['name']}")
        print(f"  Encoder out_dim: {self.encoder.out_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Attention blocks: {blocks}")
        print(f"  UPE dim: {upe_dim}")
        print(f"  Total params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def _create_decoder(self, hidden_dim, blocks, upe_dim):
        """
        Create single global decoder with UPE support.

        First FC layer input:
            - 4 neighbors: 4 × hidden_dim = 4 × 384 = 1536
            - rel_cell: 2
            - hi_coord: 2
            - UPE: upe_dim (512)
            - Total: 1536 + 2 + 2 + 512 = 2052

        Args:
            hidden_dim: Hidden dimension (384)
            blocks: Number of attention blocks (20)
            upe_dim: UPE embedding dimension (512)

        Returns:
            decoder: ModuleDict containing FC layers and attention blocks
        """
        # FC layers with variable input dimensions
        fc_layers = nn.ModuleList()

        for layer_idx in range(self.n_hi_layers):
            if layer_idx == 0:
                # First layer: neighbors (1536) + cell (2) + hi_coord (2) + UPE (512) = 2052
                in_dim = hidden_dim * 4 + 2 + 2 + upe_dim
                out_dim = hidden_dim
            elif layer_idx < self.n_hi_layers - 1:
                # Middle layers: hidden + hi_coord
                in_dim = hidden_dim + 2
                out_dim = hidden_dim
            else:
                # Last layer: hidden + hi_coord → RGB
                in_dim = hidden_dim + 2
                out_dim = 3  # RGB output

            fc_layers.append(
                MLP_with_shortcut(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    hidden_dim=256  # Internal MLP hidden dim
                )
            )

        # Attention blocks (INCREASED capacity)
        conv0 = qkv_attn(hidden_dim, blocks)  # 24 heads (was 16 in baseline)
        conv1 = qkv_attn(hidden_dim, blocks)  # 24 heads

        return nn.ModuleDict({
            'fc_layers': fc_layers,
            'conv0': conv0,
            'conv1': conv1,
        })

    def gen_feat(self, inp):
        """
        Generate features from input (shared encoder).

        Args:
            inp: Input RGB (B, 3, H, W) in [0, 1]

        Returns:
            Feature map (B, hidden_dim, H', W')
        """
        self.inp = inp  # Store original RGB for residual connection
        self.feat = self.encoder(inp)
        self.feat = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell, upe):
        """
        Query RGB values using global decoder with UPE injection.

        This follows the exact same structure as hiif.py query_rgb(),
        but with UPE added to the first FC layer input.

        Args:
            coord: Coordinate tensor (B, H, W, 2)
            cell: Cell size tensor (B, 2)
            upe: Processed UPE (B, upe_dim) from UPE processor

        Returns:
            RGB predictions (B, 3, H, W)
        """
        # Get features (already computed in gen_feat)
        feat = self.feat

        # Same as hiif.py: Create position grid
        pos_lr = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        preds = []
        areas = []

        # Sample 4 neighbors (same as hiif.py)
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # Grid sample features
                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest',
                                      align_corners=False)

                # Compute relative coordinate
                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest',
                                          align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2] / 2
                rel_coord[:, 1, :, :] *= feat.shape[-1] / 2
                rel_coord_n = rel_coord.permute(0, 2, 3, 1).reshape(
                    rel_coord.shape[0], -1, rel_coord.shape[1]
                )

                # Compute area for weighting
                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                preds.append(feat_)

                if vx == -1 and vy == -1:
                    # Local coord (for hierarchical coordinate computation)
                    rel_coord_mask = (rel_coord_n > 0).float()
                    rxry = torch.tensor([rx, ry], device=coord.device)[None, None, :]
                    local_coord = rel_coord_mask * rel_coord_n + \
                                  (1. - rel_coord_mask) * (rxry - rel_coord_n)

        # Compute relative cell
        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        # Area-based weighting (swap for correct interpolation)
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            preds[index] = preds[index] * (area / tot_area).unsqueeze(1)

        # Concatenate: 4 neighbors + rel_cell → grid
        grid = torch.cat([
            *preds,
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])
        ], dim=1)

        B, C_g, H, W = grid.shape
        grid = grid.permute(0, 2, 3, 1).reshape(B, H * W, C_g)

        # Expand UPE to spatial dimensions
        upe_expanded = upe.unsqueeze(1).expand(-1, H * W, -1)  # (B, H*W, upe_dim)

        # Apply decoder layers
        for n in range(self.n_hi_layers):
            hi_coord = compute_hi_coord(local_coord, n)

            if n == 0:
                # First layer: concatenate grid + hi_coord + UPE
                x = torch.cat([grid, hi_coord, upe_expanded], dim=-1)
            else:
                # Other layers: concatenate x + hi_coord (no UPE)
                x = torch.cat([x, hi_coord], dim=-1)

            x = self.decoder['fc_layers'][n](x)

            if n == 0:
                # Apply attention after first FC layer
                x = self.decoder['conv0'](x)
                x = self.decoder['conv1'](x)

        # Reshape to (B, 3, H, W)
        result = x.permute(0, 2, 1).reshape(B, 3, H, W)

        # Post-process: RGB residual
        # self.inp is in [0, 1], result is residual
        base = F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
                             padding_mode='border', align_corners=False)
        ret = base + result  # Residual connection

        return ret

    def forward(self, inp, coord, cell, upe):
        """
        Forward pass with UPE.

        Args:
            inp: Input RGB (B, 3, H, W) in [0, 1]
            coord: Coordinates (B, H, W, 2)
            cell: Cell size (B, 2)
            upe: Raw UPE (B, 16, 1280) or (B, A, D_raw)
                 Will be processed by UPE processor

        Returns:
            Output RGB (B, 3, H, W) in [0, 1]
        """
        # Generate features with shared encoder
        self.gen_feat(inp)

        # Process UPE
        if upe.dim() == 3:
            # Batch of UPEs: (B, A, D_raw) → (B, upe_dim)
            upe_processed = self.upe_processor(upe)  # UPEProcessor handles batched input
        elif upe.dim() == 2:
            # Single UPE: (A, D_raw) → (upe_dim,)
            upe_processed = self.upe_processor(upe).unsqueeze(0)  # (1, upe_dim)
        else:
            raise ValueError(
                f"Invalid UPE shape: {upe.shape}. "
                f"Expected (A, D_raw) or (B, A, D_raw)"
            )

        # Query RGB using global decoder with UPE
        return self.query_rgb(coord, cell, upe_processed)
