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

    def __init__(self, data_norm_config: Dict, device: Union[str, torch.device] = 'cuda'):
        """
        Initialize normalizer with config.

        Args:
            data_norm_config: Dict with structure:
                {
                    'inp': {'sub': [0.5], 'div': [0.5]},
                    'gt': {'sub': [0.5], 'div': [0.5]}
                }
            device: Device to create tensors on ('cuda', 'cpu', or torch.device)
        """
        # Convert to torch.device if string
        self.device = device if isinstance(device, torch.device) else torch.device(device)

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


def expand_to_BN(tensor: torch.Tensor, N: int = 4) -> torch.Tensor:
    """
    Expand tensor for N augmentation versions.

    Used for color augmentation: each sample has N versions (original + augmented).
    This function repeats each sample N times along the batch dimension.

    Args:
        tensor: Input tensor with batch dimension B
                Can be any shape: (B, ...), (B, H, W, 2), (B, 2A, D), etc.
        N: Number of augmentation versions (default: 4)

    Returns:
        expanded: (B*N, ...) tensor with each sample repeated N times

    Examples:
        >>> coord = torch.randn(8, 128, 128, 2)  # (B, H, W, 2)
        >>> coord_exp = expand_to_BN(coord, N=4)
        >>> coord_exp.shape
        torch.Size([32, 128, 128, 2])  # (B*4, H, W, 2)

        >>> upe = torch.randn(8, 16, 1280)  # (B, 16, 1280)
        >>> upe_exp = expand_to_BN(upe, N=4)
        >>> upe_exp.shape
        torch.Size([32, 16, 1280])  # (B*4, 16, 1280)

        >>> cell = torch.randn(8, 2)  # (B, 2)
        >>> cell_exp = expand_to_BN(cell, N=4)
        >>> cell_exp.shape
        torch.Size([32, 2])  # (B*4, 2)

    Implementation:
        Uses `repeat_interleave` to repeat each sample N times:
        [s0, s0, s0, s0, s1, s1, s1, s1, ...]

        Alternative (tile):
        [s0, s1, s2, ..., s0, s1, s2, ..., ...]
    """
    # Repeat each sample N times along batch dimension
    # [s0, s0, s0, s0, s1, s1, s1, s1, ...]
    expanded = tensor.repeat_interleave(N, dim=0)
    return expanded


def preprocess_pps_batch_with_upe(
    batch: Dict[str, torch.Tensor],
    num_augmentations: int = 4
) -> tuple:
    """
    Preprocess PPS batch with UPE by reshaping from (B, N, ...) to (B*N, ...).

    Similar to preprocess_pps_batch, but also handles UPE expansion.

    Input batch structure:
        - batch['inp']: (B, 4, 3, H, W) - 4 augmented versions already stacked
        - batch['coord']: (B, H, W, 2) - shared across all augmentations
        - batch['cell']: (B, 2) - shared across all augmentations
        - batch['gt_prefer']: (B, 3, H, W) - prefer ground truth
        - batch['gt_non_prefer']: (B, 3, H, W) - non-prefer ground truth
        - batch['upe']: (B, 16, 1280) - Raw UPE per user (NEW)
        - batch['user_id']: List[str] - User IDs

    Output structure:
        All tensors expanded to (B*4, ...) to treat each augmentation as separate sample:
        - inp: (B*4, 3, H, W)
        - coord: (B*4, H, W, 2)
        - cell: (B*4, 2)
        - gt_prefer: (B*4, 3, H, W)
        - gt_non_prefer: (B*4, 3, H, W)
        - upe: (B*4, 16, 1280) - Each augmentation gets same UPE (NEW)
        - user_id: List[str] - Unchanged

    Args:
        batch: Dict containing batch data from PPS dataloader with UPE
        num_augmentations: Number of augmentations per sample (default: 4)

    Returns:
        Tuple of (inp, coord, cell, gt_prefer, gt_non_prefer, upe, user_id):
            - inp: (B*N, 3, H, W) reshaped input
            - coord: (B*N, H, W, 2) expanded coordinates
            - cell: (B*N, 2) expanded cell sizes
            - gt_prefer: (B*N, 3, H, W) expanded prefer GT
            - gt_non_prefer: (B*N, 3, H, W) expanded non-prefer GT
            - upe: (B*N, 16, 1280) expanded UPE (NEW)
            - user_id: User IDs (unchanged)

    Example:
        >>> batch = dataloader.next()
        >>> inp, coord, cell, gt_p, gt_np, upe, user_id = preprocess_pps_batch_with_upe(batch)
        >>> # inp.shape: (32, 3, 128, 128) if B=8, N=4
        >>> # upe.shape: (32, 16, 1280)
        >>> # coord.shape: (32, 128, 128, 2)
    """
    # Get batch dimensions
    B = batch['inp'].shape[0]
    H, W = batch['inp'].shape[-2:]

    # Reshape inputs: (B, N, 3, H, W) -> (B*N, 3, H, W)
    inp = batch['inp'].reshape(B * num_augmentations, 3, H, W)

    # Expand coord: (B, H, W, 2) -> (B*N, H, W, 2)
    coord = expand_to_BN(batch['coord'], N=num_augmentations)

    # Expand cell: (B, 2) -> (B*N, 2)
    cell = expand_to_BN(batch['cell'], N=num_augmentations)

    # DEBUG: Check if batch GTs are identical BEFORE expansion
    import torch.nn.functional as F
    if torch.equal(batch['gt_prefer'], batch['gt_non_prefer']):
        print("[BUG] preprocess_pps_batch_with_upe: batch['gt_prefer'] == batch['gt_non_prefer'] BEFORE expand!")
        print(f"  User IDs: {batch['user_id']}")
        mse_before = F.mse_loss(batch['gt_prefer'], batch['gt_non_prefer']).item()
        print(f"  MSE: {mse_before}")

    # Expand prefer GT: (B, 3, H, W) -> (B*N, 3, H, W)
    gt_prefer = expand_to_BN(batch['gt_prefer'], N=num_augmentations)

    # Expand non-prefer GT: (B, 3, H, W) -> (B*N, 3, H, W)
    gt_non_prefer = expand_to_BN(batch['gt_non_prefer'], N=num_augmentations)

    # DEBUG: Check if expanded GTs are identical AFTER expansion
    if torch.equal(gt_prefer, gt_non_prefer):
        print("[BUG] preprocess_pps_batch_with_upe: gt_prefer == gt_non_prefer AFTER expand!")
        print(f"  User IDs: {batch['user_id']}")
        mse_after = F.mse_loss(gt_prefer, gt_non_prefer).item()
        print(f"  MSE: {mse_after}")

    # NEW: Expand UPE: (B, 16, 1280) -> (B*N, 16, 1280)
    # Each augmentation of a sample gets the same UPE
    upe = expand_to_BN(batch['upe'], N=num_augmentations)

    # User indices remain unchanged
    user_id = batch['user_id']

    return inp, coord, cell, gt_prefer, gt_non_prefer, upe, user_id
