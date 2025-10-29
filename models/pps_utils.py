"""
PPS utilities for implicit neural models.

Provides a mixin class for easy integration of configurable input/output
color spaces into implicit neural models (HIIF, LIIF, LTE, etc.).
"""

import torch
import torch.nn.functional as F
from models.color_utils import rgb_to_hsv, hsv_to_rgb, rgb_to_oklab, oklab_to_rgb


class PPSMixin:
    """Mixin class for PPS support in implicit neural models.

    This mixin provides configurable input color spaces and output types
    for photographic style transfer tasks. It can be used with any implicit
    neural model that processes images.

    Example usage:
        ```python
        from models.pps_utils import PPSMixin

        class MyModel(nn.Module, PPSMixin):
            def __init__(self, ..., input_type='rgb', output_type='rgb_residual'):
                super().__init__()
                PPSMixin.__init__(self, input_type, output_type)

                # Configure encoder with correct input channels
                encoder_spec['args']['n_colors'] = self.get_input_channels()
                self.encoder = models.make(encoder_spec)

                # Configure decoder with correct output channels
                output_channels = self.get_output_channels()
                self.decoder = create_decoder(output_channels)

            def forward(self, inp, ...):
                # Preprocess input
                inp_processed = self.preprocess_input(inp)

                # Model processing...
                output = self.model_forward(inp_processed, ...)

                # Postprocess output
                final_rgb = self.postprocess_output(output, inp, coord)
                return final_rgb
        ```
    """

    def __init__(self, input_type='rgb', output_type='rgb_residual'):
        """Initialize PPSMixin.

        Args:
            input_type: Input color space ('rgb', 'hsv', 'oklab', 'all')
            output_type: Output type ('rgb_residual', 'hsv_residual',
                        'oklab_residual', 'affine_coef')
        """
        self.input_type = input_type
        self.output_type = output_type

    def get_input_channels(self):
        """Get number of input channels based on input_type.

        Returns:
            int: 3 for single color space, 9 for 'all'
        """
        if self.input_type == 'all':
            return 9  # RGB + HSV + OKLab
        else:
            return 3

    def get_output_channels(self):
        """Get number of output channels based on output_type.

        Returns:
            int: 3 for residuals, 12 for affine coefficients
        """
        if self.output_type == 'affine_coef':
            return 12  # 3x3 matrix A (9) + 3x1 vector d (3)
        else:
            return 3

    def preprocess_input(self, inp):
        """Convert RGB input to specified color space(s).

        Args:
            inp: RGB input tensor (B, 3, H, W) with values in [0, 1]

        Returns:
            Preprocessed tensor (B, C, H, W) where C = get_input_channels()
        """
        if self.input_type == 'rgb':
            return inp
        elif self.input_type == 'hsv':
            return rgb_to_hsv(inp)
        elif self.input_type == 'oklab':
            return rgb_to_oklab(inp)
        elif self.input_type == 'all':
            hsv = rgb_to_hsv(inp)
            oklab = rgb_to_oklab(inp)
            return torch.cat([inp, hsv, oklab], dim=1)  # (B, 9, H, W)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

    def postprocess_output(self, output, inp_rgb, coord):
        """Convert model output to final RGB.

        Args:
            output: Raw model output (B, C, H, W) where C = get_output_channels()
            inp_rgb: Original RGB input (B, 3, H, W) for base values
            coord: Coordinate tensor for grid sampling

        Returns:
            Final RGB output (B, 3, H, W) with values in [0, 1]
        """
        if self.output_type == 'rgb_residual':
            # RGB residual: base + residual
            base = F.grid_sample(inp_rgb, coord.flip(-1), mode='bilinear',
                               padding_mode='border', align_corners=False)
            return base + output

        elif self.output_type == 'hsv_residual':
            # HSV residual: RGB → HSV → + residual → RGB
            base = F.grid_sample(inp_rgb, coord.flip(-1), mode='bilinear',
                               padding_mode='border', align_corners=False)
            base_hsv = rgb_to_hsv(base)
            output_hsv = base_hsv + output
            return hsv_to_rgb(output_hsv)

        elif self.output_type == 'oklab_residual':
            # OKLab residual: RGB → OKLab → + residual → RGB
            base = F.grid_sample(inp_rgb, coord.flip(-1), mode='bilinear',
                               padding_mode='border', align_corners=False)
            base_oklab = rgb_to_oklab(base)
            output_oklab = base_oklab + output
            return oklab_to_rgb(output_oklab)

        elif self.output_type == 'affine_coef':
            # Affine transform: A * rgb + d
            return self.apply_affine_transform(output, inp_rgb, coord)

        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")

    def apply_affine_transform(self, coef, inp_rgb, coord):
        """Apply per-pixel affine transform.

        Applies the transformation: output = A @ base + d
        where A is a 3x3 matrix and d is a 3x1 vector, both per-pixel.

        Args:
            coef: Affine coefficients (B, 12, H, W)
                  First 9 channels: 3x3 matrix A (row-major)
                  Last 3 channels: 3x1 vector d
            inp_rgb: Original RGB input (B, 3, H, W)
            coord: Coordinate tensor for grid sampling

        Returns:
            Transformed RGB (B, 3, H, W) with values in [0, 1]
        """
        B, _, H, W = coef.shape

        # Extract A (9 channels) and d (3 channels)
        A = coef[:, :9, :, :].reshape(B, 3, 3, H, W)  # (B, 3, 3, H, W)
        d = coef[:, 9:12, :, :]  # (B, 3, H, W)

        # Get base RGB values
        base = F.grid_sample(inp_rgb, coord.flip(-1), mode='bilinear',
                           padding_mode='border', align_corners=False)

        # Reshape for batched matrix multiplication
        # A: (B, 3, 3, H, W) -> (B*H*W, 3, 3)
        # base: (B, 3, H, W) -> (B*H*W, 3, 1)
        # d: (B, 3, H, W) -> (B*H*W, 3, 1)
        A_flat = A.permute(0, 3, 4, 1, 2).reshape(B * H * W, 3, 3)
        base_flat = base.permute(0, 2, 3, 1).reshape(B * H * W, 3, 1)
        d_flat = d.permute(0, 2, 3, 1).reshape(B * H * W, 3, 1)

        # Apply transformation: A @ base + d
        transformed_flat = torch.bmm(A_flat, base_flat) + d_flat

        # Reshape back to image format
        transformed = transformed_flat.squeeze(-1).reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return transformed.clamp(0, 1)
