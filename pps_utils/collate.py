"""
Custom Collate Functions for PPS Training

This module provides collate functions for creating batches with UPE support.
"""

from typing import List, Dict

import torch


def collate_with_upe(batch: List[Dict]) -> Dict:
    """
    Collate function for PPS batches with UPE support.

    Handles:
    - Mixed-user batches (different users in same batch)
    - Each sample has its own UPE
    - Proper tensor stacking
    - None filtering (if cache_policy='skip')

    Args:
        batch: List of sample dicts from PPSUPEWrapper, each containing:
            - 'user_id': str
            - 'image_id': str
            - 'inp': (4, 3, H, W) - 4 augmented versions
            - 'gt_prefer': (3, H, W)
            - 'gt_non_prefer': (3, H, W)
            - 'coord': (H, W, 2)
            - 'cell': (2,)
            - 'upe': (16, 1280) - Raw UPE

    Returns:
        Batched dict:
            - 'user_id': List[str] (length B)
            - 'image_id': List[str] (length B)
            - 'inp': (B, 4, 3, H, W)
            - 'gt_prefer': (B, 3, H, W)
            - 'gt_non_prefer': (B, 3, H, W)
            - 'coord': (B, H, W, 2)
            - 'cell': (B, 2)
            - 'upe': (B, 16, 1280)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=8, collate_fn=collate_with_upe)
        >>> for batch in loader:
        >>>     print(batch['inp'].shape)  # (8, 4, 3, 128, 128)
        >>>     print(batch['upe'].shape)  # (8, 16, 1280)
    """
    # Filter out None samples (if cache_policy='skip')
    batch = [s for s in batch if s is not None]

    if len(batch) == 0:
        raise RuntimeError(
            "Empty batch after filtering None samples. "
            "This likely means all samples in batch had missing UPEs. "
            "Please run UPE extraction or check cache_policy."
        )

    # Stack tensors
    return {
        'user_id': [s['user_id'] for s in batch],  # List of strings
        'image_id': [s['image_id'] for s in batch],  # List of strings
        'inp': torch.stack([s['inp'] for s in batch]),  # (B, 4, 3, H, W)
        'gt_prefer': torch.stack([s['gt_prefer'] for s in batch]),  # (B, 3, H, W)
        'gt_non_prefer': torch.stack([s['gt_non_prefer'] for s in batch]),  # (B, 3, H, W)
        'coord': torch.stack([s['coord'] for s in batch]),  # (B, H, W, 2)
        'cell': torch.stack([s['cell'] for s in batch]),  # (B, 2)
        'upe': torch.stack([s['upe'] for s in batch]),  # (B, 16, 1280)
    }


def collate_baseline(batch: List[Dict]) -> Dict:
    """
    Collate function for baseline PPS (without UPE).

    This is for compatibility with baseline training (hiif_pps.py).

    Args:
        batch: List of sample dicts from PPSColorAugmentedWrapper

    Returns:
        Batched dict without UPE
    """
    # Filter out None samples
    batch = [s for s in batch if s is not None]

    if len(batch) == 0:
        raise RuntimeError("Empty batch after filtering None samples")

    return {
        'user_id': [s['user_id'] for s in batch],
        'image_id': [s['image_id'] for s in batch],
        'inp': torch.stack([s['inp'] for s in batch]),
        'gt_prefer': torch.stack([s['gt_prefer'] for s in batch]),
        'gt_non_prefer': torch.stack([s['gt_non_prefer'] for s in batch]),
        'coord': torch.stack([s['coord'] for s in batch]),
        'cell': torch.stack([s['cell'] for s in batch]),
    }
