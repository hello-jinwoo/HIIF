"""
Data Processing Utilities for PPS Training and Validation

This module provides centralized data processing functions to ensure consistency
between training (train_pps.py) and validation (validation_core.py) pipelines.

Key Components:
- DataNormalizer: Handles input/GT normalization with cached tensors
- preprocess_pps_batch: Reshapes PPS batch from (B, N, 3, H, W) to (B*N, 3, H, W)

Usage:
    # Initialize normalizer once
    normalizer = DataNormalizer(config['data_norm'])

    # In training loop
    inp, coord, cell, gt_prefer, gt_non_prefer, user_id = preprocess_pps_batch(batch)
    inp = normalizer.normalize_input(inp)
    gt_prefer = normalizer.normalize_gt(gt_prefer)
"""

from typing import Dict, Tuple, List, Union
import torch


class DataNormalizer:
    """
    Handles data normalization and denormalization with cached tensors.

    This class creates normalization tensors once during initialization and reuses
    them for all subsequent normalize/denormalize operations, improving performance.

    Attributes:
        inp_sub: Input subtraction tensor (1, C, 1, 1)
        inp_div: Input division tensor (1, C, 1, 1)
        gt_sub: GT subtraction tensor (1, C, 1, 1)
        gt_div: GT division tensor (1, C, 1, 1)
        device: Device where tensors are stored
    """

    def __init__(self, data_norm_config: Dict, device: str = 'cuda'):
        """
        Initialize normalizer with config.

        Args:
            data_norm_config: Dict with structure:
                {
                    'inp': {'sub': [0.5], 'div': [0.5]},
                    'gt': {'sub': [0.5], 'div': [0.5]}
                }
            device: Device to create tensors on ('cuda' or 'cpu')
        """
        self.device = device

        # Create and cache normalization tensors
        self.inp_sub = torch.FloatTensor(data_norm_config['inp']['sub']).view(1, -1, 1, 1).to(device)
        self.inp_div = torch.FloatTensor(data_norm_config['inp']['div']).view(1, -1, 1, 1).to(device)
        self.gt_sub = torch.FloatTensor(data_norm_config['gt']['sub']).view(1, -1, 1, 1).to(device)
        self.gt_div = torch.FloatTensor(data_norm_config['gt']['div']).view(1, -1, 1, 1).to(device)

    def normalize_input(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor.

        Args:
            inp: Input tensor (B, 3, H, W) in [0, 1]

        Returns:
            Normalized tensor
        """
        return (inp - self.inp_sub) / self.inp_div

    def normalize_gt(self, gt: torch.Tensor) -> torch.Tensor:
        """
        Normalize ground truth tensor.

        Args:
            gt: GT tensor (B, 3, H, W) in [0, 1]

        Returns:
            Normalized tensor
        """
        return (gt - self.gt_sub) / self.gt_div

    def denormalize_gt(self, gt_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize GT tensor back to [0, 1] range.

        Args:
            gt_normalized: Normalized GT tensor

        Returns:
            Denormalized tensor in [0, 1]
        """
        return gt_normalized * self.gt_div + self.gt_sub


def preprocess_pps_batch(
    batch: Dict[str, torch.Tensor],
    num_augmentations: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[List, torch.Tensor]]:
    """
    Preprocess PPS batch by reshaping from (B, N, ...) to (B*N, ...).

    PPS batches contain multiple augmented versions of each sample:
    - batch['inp']: (B, N, 3, H, W) where N=4 (prefer_orig, prefer_aug, non_prefer_orig, non_prefer_aug)
    - batch['coord']: (B, H, W, 2) - shared across all augmentations
    - batch['cell']: (B, 2) - shared across all augmentations
    - batch['gt_prefer']: (B, 3, H, W) - prefer ground truth
    - batch['gt_non_prefer']: (B, 3, H, W) - non-prefer ground truth

    This function reshapes tensors to treat each augmentation as a separate sample,
    which is required for the model's batch processing.

    Args:
        batch: Dict containing batch data from PPS dataloader
        num_augmentations: Number of augmentations per sample (default: 4)

    Returns:
        Tuple of (inp, coord, cell, gt_prefer, gt_non_prefer, user_id):
            - inp: (B*N, 3, H, W) reshaped input
            - coord: (B*N, H, W, 2) expanded coordinates
            - cell: (B*N, 2) expanded cell sizes
            - gt_prefer: (B*N, 3, H, W) expanded prefer GT
            - gt_non_prefer: (B*N, 3, H, W) expanded non-prefer GT
            - user_id: User indices (unchanged)

    Example:
        >>> batch = dataloader.next()
        >>> inp, coord, cell, gt_p, gt_np, user_id = preprocess_pps_batch(batch)
        >>> # inp.shape: (8, 3, 128, 128) if B=2, N=4
        >>> # coord.shape: (8, 128, 128, 2)
        >>> # cell.shape: (8, 2)
    """
    # Get batch dimensions
    B = batch['inp'].shape[0]
    H, W = batch['inp'].shape[-2:]

    # Reshape inputs: (B, N, 3, H, W) -> (B*N, 3, H, W)
    inp = batch['inp'].reshape(B * num_augmentations, 3, H, W)

    # Expand coord: (B, H, W, 2) -> (B, N, H, W, 2) -> (B*N, H, W, 2)
    # Each augmentation uses the same coordinate grid
    coord = batch['coord'].unsqueeze(1).repeat(1, num_augmentations, 1, 1, 1).reshape(
        B * num_augmentations, H, W, 2
    )

    # Expand cell: (B, 2) -> (B, N, 2) -> (B*N, 2)
    # Each augmentation uses the same cell size
    cell = batch['cell'].unsqueeze(1).repeat(1, num_augmentations, 1).reshape(
        B * num_augmentations, 2
    )

    # Expand prefer GT: (B, 3, H, W) -> (B, N, 3, H, W) -> (B*N, 3, H, W)
    # All augmentations share the same prefer GT
    gt_prefer = batch['gt_prefer'].unsqueeze(1).repeat(1, num_augmentations, 1, 1, 1).reshape(
        B * num_augmentations, 3, H, W
    )

    # Expand non-prefer GT: (B, 3, H, W) -> (B, N, 3, H, W) -> (B*N, 3, H, W)
    # All augmentations share the same non-prefer GT
    gt_non_prefer = batch['gt_non_prefer'].unsqueeze(1).repeat(1, num_augmentations, 1, 1, 1).reshape(
        B * num_augmentations, 3, H, W
    )

    # User indices remain unchanged
    user_id = batch['user_id']

    return inp, coord, cell, gt_prefer, gt_non_prefer, user_id
