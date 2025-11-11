"""
HIIF with User-Specific Decoders for PPS Learning

Architecture:
- Shared encoder (EDSR/SwinIR/RDN) for all users
- User-specific HIIF decoders (100 decoders, one per user)
- Decoder load/offload: Only current decoder in GPU, others on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models.hiif import MLP_with_shortcut, qkv_attn, compute_hi_coord
from models.color_utils import rgb_to_hsv, hsv_to_rgb, rgb_to_oklab, oklab_to_rgb, rgb_to_yuv, yuv_to_rgb
from utils import make_coord


@register('hiif-pps')
class HIIF_PPS(nn.Module):
    """HIIF with user-specific decoders for personalized photographic style

    Uses disk-based decoder load/offload:
    - Only ONE decoder is kept in memory at a time
    - Decoders are loaded from disk when needed (or created if new)
    - Decoders are saved to disk when switching users
    """

    def __init__(self, encoder_spec, hidden_dim=256, blocks=16, user_ids=None,
                 input_type='rgb', output_type='rgb_residual'):
        """
        Args:
            encoder_spec: Encoder configuration dict
            hidden_dim: Hidden dimension for decoders
            blocks: Number of attention blocks
            user_ids: List of available user IDs (for reference, not created in memory)
            input_type: Input color space ('rgb', 'hsv', 'oklab', 'yuv', 'all')
            output_type: Output type ('rgb_residual', 'hsv_residual',
                        'oklab_residual', 'yuv_residual', 'affine_coef')
        """
        super().__init__()

        # Validate input_type and output_type
        valid_inputs = ['rgb', 'hsv', 'oklab', 'yuv', 'all']
        valid_outputs = ['rgb_residual', 'hsv_residual', 'oklab_residual', 'yuv_residual', 'affine_coef']

        if input_type not in valid_inputs:
            raise ValueError(f"Invalid input_type: {input_type}. Must be one of {valid_inputs}")
        if output_type not in valid_outputs:
            raise ValueError(f"Invalid output_type: {output_type}. Must be one of {valid_outputs}")

        self.input_type = input_type
        self.output_type = output_type
        self.hidden_dim = hidden_dim
        self.blocks = blocks
        self.n_hi_layers = 6

        # Update encoder spec with correct n_colors
        encoder_spec_copy = encoder_spec.copy()
        if 'args' not in encoder_spec_copy:
            encoder_spec_copy['args'] = {}
        encoder_spec_copy['args']['n_colors'] = self._get_input_channels()

        # Shared encoder
        self.encoder = models.make(encoder_spec_copy)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)

        # Decoder configuration (for creating decoders on-demand)
        self.decoder_config = {
            'hidden_dim': hidden_dim,
            'blocks': blocks,
            'output_channels': self._get_output_channels(),
        }

        # Available user IDs (for validation, not created in memory)
        self.available_user_ids = user_ids if user_ids is not None else []

        # Current decoder (only ONE in memory at a time)
        self.current_decoder = None
        self.current_user_id = None

    def _get_input_channels(self):
        """Get number of input channels based on input_type."""
        if self.input_type == 'all':
            return 12  # RGB + HSV + OKLab + YUV
        else:
            return 3

    def _get_output_channels(self):
        """Get number of output channels based on output_type."""
        if self.output_type == 'affine_coef':
            return 12  # 3x3 matrix A (9) + 3x1 vector d (3)
        else:
            return 3

    def _preprocess_input(self, inp):
        """Convert input RGB to specified color space(s).

        Args:
            inp: RGB input tensor (B, 3, H, W) with values in [0, 1]

        Returns:
            Preprocessed tensor (B, C, H, W)
        """
        if self.input_type == 'rgb':
            return inp
        elif self.input_type == 'hsv':
            return rgb_to_hsv(inp)
        elif self.input_type == 'oklab':
            return rgb_to_oklab(inp)
        elif self.input_type == 'yuv':
            return rgb_to_yuv(inp)
        elif self.input_type == 'all':
            hsv = rgb_to_hsv(inp)
            oklab = rgb_to_oklab(inp)
            yuv = rgb_to_yuv(inp)
            return torch.cat([inp, hsv, oklab, yuv], dim=1)  # (B, 12, H, W)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

    def _create_decoder(self, hidden_dim, blocks, output_channels):
        """
        Create a single HIIF decoder (same structure as original hiif.py)

        Args:
            hidden_dim: Hidden dimension
            blocks: Number of attention blocks
            output_channels: Number of output channels (3 or 12)

        Returns:
            nn.ModuleDict with fc_layers, conv0, conv1
        """
        fc_layers = nn.ModuleList([
            MLP_with_shortcut(
                hidden_dim * 4 + 2 + 2 if d == 0 else hidden_dim + 2,
                output_channels if d == self.n_hi_layers - 1 else hidden_dim,
                256
            ) for d in range(self.n_hi_layers)
        ])

        conv0 = qkv_attn(hidden_dim, blocks)
        conv1 = qkv_attn(hidden_dim, blocks)

        return nn.ModuleDict({
            'fc_layers': fc_layers,
            'conv0': conv0,
            'conv1': conv1,
        })

    def load_user_decoder(self, user_id, checkpoint_manager):
        """
        Load user decoder from disk (or create new if not exists)

        This method:
        1. Creates a new decoder module
        2. Tries to load weights from checkpoint
        3. Moves decoder to GPU
        4. Updates current_user_id

        Args:
            user_id: User ID to load
            checkpoint_manager: CheckpointManager instance for loading weights

        Note: Previous decoder should be offloaded before calling this
        """
        if self.current_user_id == user_id and self.current_decoder is not None:
            return  # Already loaded

        # Get encoder device
        device = next(self.encoder.parameters()).device

        # Create new decoder
        decoder = self._create_decoder(
            self.decoder_config['hidden_dim'],
            self.decoder_config['blocks'],
            self.decoder_config['output_channels']
        )

        # Try to load weights from checkpoint
        if checkpoint_manager is not None:
            if checkpoint_manager.load_user_decoder_weights(decoder, user_id):
                print(f"Loaded decoder for {user_id} from checkpoint")
            else:
                print(f"Initialized new decoder for {user_id}")
        else:
            print(f"Initialized new decoder for {user_id} (no checkpoint manager)")

        # Move to GPU
        decoder.to(device)

        # Update current decoder
        self.current_decoder = decoder
        self.current_user_id = user_id

    def offload_user_decoder(self, checkpoint_manager):
        """
        Save current decoder to disk and free memory

        This method:
        1. Saves current decoder weights to disk
        2. Removes decoder from memory
        3. Clears current_user_id

        Args:
            checkpoint_manager: CheckpointManager instance for saving weights
        """
        if self.current_decoder is None:
            return  # Nothing to offload

        # Save to disk
        if checkpoint_manager is not None:
            checkpoint_manager.save_user_decoder(self, self.current_user_id)
            print(f"Saved decoder for {self.current_user_id}")
        else:
            print(f"Discarded decoder for {self.current_user_id} (no checkpoint manager)")

        # Free memory
        self.current_decoder = None
        self.current_user_id = None

    def gen_feat(self, inp):
        """
        Generate features from input (shared encoder)

        Args:
            inp: Input RGB (B, 3, H, W)

        Returns:
            Feature map (B, hidden_dim, H', W')
        """
        self.inp = inp  # Store original RGB for postprocessing
        inp_processed = self._preprocess_input(inp)
        self.feat = self.encoder(inp_processed)
        self.feat = self.freq(self.feat)
        return self.feat

    def _postprocess_output(self, output, coord):
        """Post-process model output.

        Args:
            output: Raw model output (B, C, H, W) where C depends on output_type
            coord: Coordinate tensor for grid sampling

        Returns:
            Processed output (B, 3, H, W) in normalized space [-1, 1] for rgb_residual.
            Must be denormalized externally to get final RGB in [0, 1].
        """
        if self.output_type == 'rgb_residual':
            # RGB residual in normalized space
            # Note: self.inp is in normalized space [-1, 1] during training/validation
            # Output stays in normalized space, will be denormalized externally
            base = F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
                               padding_mode='border', align_corners=False)
            return base + output  # No clamp - keep in normalized space

        elif self.output_type == 'hsv_residual':
            # HSV residual: inp_rgb → HSV → + residual → RGB
            base = F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
                               padding_mode='border', align_corners=False)
            base_hsv = rgb_to_hsv(base)
            output_hsv = base_hsv + output
            return hsv_to_rgb(output_hsv)

        elif self.output_type == 'oklab_residual':
            # OKLab residual: inp_rgb → OKLab → + residual → RGB
            base = F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
                               padding_mode='border', align_corners=False)
            base_oklab = rgb_to_oklab(base)
            output_oklab = base_oklab + output
            return oklab_to_rgb(output_oklab)

        elif self.output_type == 'yuv_residual':
            # YUV residual: inp_rgb → YUV → + residual → RGB
            base = F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
                               padding_mode='border', align_corners=False)
            base_yuv = rgb_to_yuv(base)
            output_yuv = base_yuv + output
            return yuv_to_rgb(output_yuv)

        elif self.output_type == 'affine_coef':
            # Affine transform: A * rgb + d
            return self._apply_affine_transform(output, coord)

        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")

    def _apply_affine_transform(self, coef, coord):
        """Apply per-pixel affine transform.

        Args:
            coef: Affine coefficients (B, 12, H, W)
                  First 9 channels: 3x3 matrix A (row-major)
                  Last 3 channels: 3x1 vector d
            coord: Coordinate tensor

        Returns:
            Transformed RGB (B, 3, H, W) with values in [0, 1]
        """
        B, _, H, W = coef.shape

        # Extract A and d
        A = coef[:, :9, :, :].reshape(B, 3, 3, H, W)  # (B, 3, 3, H, W)
        d = coef[:, 9:12, :, :]  # (B, 3, H, W)

        # Get base RGB
        base = F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
                           padding_mode='border', align_corners=False)

        # Vectorized batched matrix multiplication
        A_flat = A.permute(0, 3, 4, 1, 2).reshape(B * H * W, 3, 3)
        base_flat = base.permute(0, 2, 3, 1).reshape(B * H * W, 3, 1)
        d_flat = d.permute(0, 2, 3, 1).reshape(B * H * W, 3, 1)

        # Apply: A @ base + d
        transformed_flat = torch.bmm(A_flat, base_flat) + d_flat
        transformed = transformed_flat.squeeze(-1).reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return transformed.clamp(0, 1)

    def query_rgb(self, coord, cell, user_id):
        """
        Query RGB values using user-specific decoder

        Args:
            coord: Coordinate tensor (B, H, W, 2)
            cell: Cell size tensor (B, 2)
            user_id: User ID (string)

        Returns:
            RGB predictions (B, 3, H, W)

        Note: Decoder must be loaded before calling this method
              (use load_user_decoder in training loop)
        """
        # Verify correct decoder is loaded
        if self.current_user_id != user_id or self.current_decoder is None:
            raise RuntimeError(
                f"Decoder for {user_id} is not loaded. "
                f"Current decoder: {self.current_user_id}. "
                f"Call load_user_decoder() first."
            )

        # Get current decoder
        decoder = self.current_decoder

        # Apply HIIF decoder logic (same as original hiif.py)
        feat = self.feat

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

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest',
                                      align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest',
                                          align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2] / 2
                rel_coord[:, 1, :, :] *= feat.shape[-1] / 2
                rel_coord_n = rel_coord.permute(0, 2, 3, 1).reshape(
                    rel_coord.shape[0], -1, rel_coord.shape[1]
                )

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                preds.append(feat_)

                if vx == -1 and vy == -1:
                    # Local coord
                    rel_coord_mask = (rel_coord_n > 0).float()
                    rxry = torch.tensor([rx, ry], device=coord.device)[None, None, :]
                    local_coord = rel_coord_mask * rel_coord_n + \
                                  (1. - rel_coord_mask) * (rxry - rel_coord_n)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        # Swap areas for correct weighting
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            preds[index] = preds[index] * (area / tot_area).unsqueeze(1)

        grid = torch.cat([
            *preds,
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])
        ], dim=1)

        B, C_g, H, W = grid.shape
        grid = grid.permute(0, 2, 3, 1).reshape(B, H * W, C_g)

        # Apply decoder layers
        for n in range(self.n_hi_layers):
            hi_coord = compute_hi_coord(local_coord, n)
            if n == 0:
                x = torch.cat([grid, hi_coord], dim=-1)
            else:
                x = torch.cat([x, hi_coord], dim=-1)

            x = decoder['fc_layers'][n](x)

            if n == 0:
                x = decoder['conv0'](x)
                x = decoder['conv1'](x)

        # Get output channels from model
        output_channels = self._get_output_channels()
        result = x.permute(0, 2, 1).reshape(B, output_channels, H, W)

        # Post-process to final RGB
        ret = self._postprocess_output(result, coord)

        return ret

    def forward(self, inp, coord, cell, user_indices):
        """
        Forward pass

        Args:
            inp: Input RGB (B, 3, H, W)
            coord: Coordinates (B, H, W, 2)
            cell: Cell size (B, 2)
            user_indices: User IDs (list of B strings, all same due to same-user batching)

        Returns:
            Output RGB (B, 3, H, W)
        """
        # Generate features with shared encoder
        self.gen_feat(inp)

        # All samples in batch are from same user (due to PPSUserBatchSampler)
        user_id = user_indices[0]

        # Query RGB using user-specific decoder
        return self.query_rgb(coord, cell, user_id)
