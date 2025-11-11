"""
Validation Core Functions for PPS (Personalized Photographic Style)

This module provides pure, stateless functions for validation pipeline:
- Data loading with augmentation
- Decoder test-time training
- Full-resolution patch-based evaluation
- Per-user validation orchestration

All functions are stateless (no global variables) and can be reused by both
train_pps.py (integrated validation) and validate_pps.py (standalone validation).

Key Features:
- Full-resolution patch-based processing (128×128 patches, stride 64)
- Batched patch processing (10-20x speedup)
- All 5 metrics (PSNR, SSIM, LPIPS, CIEDE2000, NIQE)
- Test-time training for validation users
"""

import os
import random
import time
from typing import Dict, List, Optional, Callable
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets
import utils
from pps_utils.loss_scheduler import LossWeightScheduler, compute_dual_gt_loss
from pps_utils.data_processing import DataNormalizer, preprocess_pps_batch


# Global coordinate cache for patch processing
# Key: (patch_size, device_str), Value: coord tensor
_COORD_CACHE = {}


def get_cached_coord(patch_size: int, device: torch.device) -> torch.Tensor:
    """
    Get cached coordinate tensor for a given patch size.

    Coordinates are identical for all patches of the same size, so we cache them
    to avoid redundant computation (make_coord involves meshgrid, which is expensive).

    Args:
        patch_size: int, size of the patch
        device: torch.device, device to create tensor on

    Returns:
        torch.Tensor, shape (patch_size, patch_size, 2), cached coordinate tensor
    """
    cache_key = (patch_size, str(device))

    if cache_key not in _COORD_CACHE:
        # Create coordinate tensor once
        coord = utils.make_coord([patch_size, patch_size], flatten=False).to(device)
        _COORD_CACHE[cache_key] = coord

    return _COORD_CACHE[cache_key]


def save_image_tensor(tensor: torch.Tensor, path: str) -> None:
    """
    Save tensor as image file.

    Args:
        tensor: torch.Tensor (3, H, W) in [0, 1]
        path: str, output path

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert to PIL image
    img_np = (tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(path, quality=95)


def create_grid_image(prefer_gt: torch.Tensor, prefer_pred: torch.Tensor,
                     non_prefer_gt: torch.Tensor, non_prefer_pred: torch.Tensor,
                     grid_size: int = 256) -> Image.Image:
    """
    Create a 2×2 grid image from 4 tensors.

    Grid layout:
        Row 1: prefer_gt | prefer_pred
        Row 2: non_prefer_gt | non_prefer_pred

    Args:
        prefer_gt: torch.Tensor (3, H, W) in [0, 1]
        prefer_pred: torch.Tensor (3, H, W) in [0, 1]
        non_prefer_gt: torch.Tensor (3, H, W) in [0, 1]
        non_prefer_pred: torch.Tensor (3, H, W) in [0, 1]
        grid_size: int, target size for longest dimension (default: 256)

    Returns:
        PIL.Image, 2×2 grid image
    """
    # Convert tensors to PIL images
    images = []
    for tensor in [prefer_gt, prefer_pred, non_prefer_gt, non_prefer_pred]:
        img_np = (tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        images.append(img)

    # Resize all images to grid_size on longest side
    resized_images = []
    for img in images:
        w, h = img.size
        if w >= h:
            new_w = grid_size
            new_h = int(h * grid_size / w)
        else:
            new_h = grid_size
            new_w = int(w * grid_size / h)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        resized_images.append(resized)

    # Find max dimensions for uniform grid
    max_w = max(img.size[0] for img in resized_images)
    max_h = max(img.size[1] for img in resized_images)

    # Create grid (2×2) with white background
    grid_w = max_w * 2
    grid_h = max_h * 2
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    # Paste images into grid
    positions = [(0, 0), (max_w, 0), (0, max_h), (max_w, max_h)]
    for img, (x, y) in zip(resized_images, positions):
        grid.paste(img, (x, y))

    return grid


def denormalize(tensor: torch.Tensor, data_norm: Dict) -> torch.Tensor:
    """
    Denormalize tensor back to [0, 1].

    Args:
        tensor: torch.Tensor (B, 3, H, W) normalized
        data_norm: dict with 'sub' and 'div' keys

    Returns:
        torch.Tensor in [0, 1]
    """
    sub = torch.FloatTensor(data_norm['sub']).view(1, -1, 1, 1).to(tensor.device)
    div = torch.FloatTensor(data_norm['div']).view(1, -1, 1, 1).to(tensor.device)
    return tensor * div + sub


def create_dataloader_with_aug(base_dataset, user_samples: List[int],
                               aug_config: Dict, batch_size: int,
                               num_workers: int = 8, batch_image: Optional[int] = None) -> DataLoader:
    """
    Create dataloader with specific augmentation config.

    Uses PPSColorAugmentedWrapper for cropped patches (used in training).

    Args:
        base_dataset: PPSPreferencePairDataset
        user_samples: list of sample indices
        aug_config: dict, augmentation configuration
        batch_size: int, total number of patches per batch
        num_workers: int, number of DataLoader worker processes (default: 8)
        batch_image: int or None, number of images to load per batch (default: None)
            - If None, uses batch_size images (1 patch per image, current behavior)
            - If specified, loads batch_image images and extracts (batch_size // batch_image)
              patches from each image for better I/O efficiency

    Returns:
        DataLoader
    """
    from datasets.pps_wrapper import pps_multi_crop_collate_fn

    # Validate batch_image parameter
    if batch_image is not None:
        if batch_image <= 0:
            raise ValueError(f"batch_image must be positive, got {batch_image}")
        if batch_size % batch_image != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by batch_image ({batch_image}). "
                f"Got remainder: {batch_size % batch_image}"
            )

        # Calculate patches per image
        patches_per_image = batch_size // batch_image
        dataloader_batch_size = batch_image

        # Add patches_per_image to augmentation config
        aug_config_with_patches = aug_config.copy()
        aug_config_with_patches['patches_per_image'] = patches_per_image
    else:
        # Original behavior: 1 patch per image
        aug_config_with_patches = aug_config
        dataloader_batch_size = batch_size

    # Create subset
    subset = Subset(base_dataset, user_samples)

    # Wrap with augmentation
    wrapper_config = {
        'name': 'pps-color-augmented',
        'args': aug_config_with_patches
    }
    wrapped_dataset = datasets.make(wrapper_config, args={'dataset': subset})

    # Create loader with custom collate function
    # num_workers: configurable via config file for parallel data loading
    loader = DataLoader(
        wrapped_dataset,
        batch_size=dataloader_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pps_multi_crop_collate_fn
    )

    return loader


def create_fullimage_dataloader(base_dataset, user_samples: List[int],
                                aug_config: Dict, batch_size: int = 1,
                                num_workers: int = 0) -> DataLoader:
    """
    Create dataloader for full-resolution image evaluation.

    Uses PPSFullImageWrapper for full-resolution images (no crop/resize).

    Args:
        base_dataset: PPSPreferencePairDataset
        user_samples: list of sample indices
        aug_config: dict, augmentation configuration
        batch_size: int (default: 1, recommended for full-res images)
        num_workers: int (default: 0, to avoid CPU contention during metrics)

    Returns:
        DataLoader
    """
    # Create subset
    subset = Subset(base_dataset, user_samples)

    # Wrap with full-image wrapper (no resize, no crop)
    wrapper_config = {
        'name': 'pps-full-image',
        'args': aug_config
    }
    wrapped_dataset = datasets.make(wrapper_config, args={'dataset': subset})

    # Create loader
    loader = DataLoader(wrapped_dataset, batch_size=batch_size,
                       shuffle=False, num_workers=num_workers, pin_memory=True)

    return loader


def train_decoder_one_step(model, batch: Dict, encoder_optimizer,
                          decoder_optimizer, loss_scheduler: LossWeightScheduler,
                          data_norm: Dict) -> float:
    """
    Train decoder for one step.

    Encoder is frozen (eval mode), only decoder is trained.

    Args:
        model: HIIF_PPS model
        batch: dict, data batch
        encoder_optimizer: optimizer for encoder (not used, frozen)
        decoder_optimizer: optimizer for decoder
        loss_scheduler: LossWeightScheduler
        data_norm: dict, normalization config

    Returns:
        float: loss value
    """
    # Keep encoder frozen in eval mode, only train decoder
    model.encoder.eval()
    model.freq.eval()
    if hasattr(model, 'current_decoder') and model.current_decoder is not None:
        model.current_decoder.train()

    # Move to GPU
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()

    # Preprocess batch and normalize
    normalizer = DataNormalizer(data_norm, device='cuda')
    inp, coord, cell, gt_prefer, gt_non_prefer, user_indices = preprocess_pps_batch(batch)
    inp = normalizer.normalize_input(inp)
    gt_prefer = normalizer.normalize_gt(gt_prefer)
    gt_non_prefer = normalizer.normalize_gt(gt_non_prefer)

    # Forward pass
    pred = model(inp, coord, cell, user_indices)

    # Dual GT loss with w_p scheduling
    w_p = loss_scheduler.get_weight()
    loss = compute_dual_gt_loss(pred, gt_prefer, gt_non_prefer, w_p)

    # Backward (encoder frozen, only decoder)
    decoder_optimizer.zero_grad()
    loss.backward()
    decoder_optimizer.step()
    loss_scheduler.step()

    return loss.item()


def evaluate_decoder(model, eval_loader: DataLoader, metrics_calc,
                    data_norm: Dict, max_samples: Optional[int] = None,
                    save_images_dir: Optional[str] = None,
                    patch_size: int = 128, stride: int = 64,
                    aug_type: str = 'clean',
                    max_patches_per_batch: int = 16,
                    log_fn: Optional[Callable] = None,
                    enable_profiling: bool = False) -> Dict:
    """
    Evaluate decoder on evaluation set with full-resolution patch-based processing.

    Args:
        model: HIIF_PPS model
        eval_loader: DataLoader for evaluation (PPSFullImageWrapper)
        metrics_calc: PPSMetrics instance
        data_norm: dict, normalization config
        max_samples: int or None, maximum samples to evaluate
        save_images_dir: str or None, directory to save images
        patch_size: int, patch size for processing (default: 128)
        stride: int, sliding window stride (default: 64)
        aug_type: str, augmentation type ('clean', 'different_aug')
        max_patches_per_batch: int, maximum patches per forward pass to avoid OOM (default: 16)
        log_fn: function or None, logging function for debug output
        enable_profiling: bool, whether to enable timing profiling (default: False)

    Returns:
        dict: aggregated metrics
    """
    model.eval()

    # Profiling accumulators
    if enable_profiling:
        profiling_times = {
            'extract_patches': 0.0,
            'batch_prepare': 0.0,
            'forward_pass': 0.0,
            'stitch_patches': 0.0,
            'metrics_compute': 0.0,
            'total_per_image': []
        }

    # Metrics storage - use all available metrics from metrics_calc
    # New structure: measure both predictions against both GTs
    available_metrics = metrics_calc.available_metrics
    metrics_storage = {
        'prefer_gt': {
            'from_prefer_input': {m: [] for m in available_metrics},
            'from_non_prefer_input': {m: [] for m in available_metrics}
        },
        'non_prefer_gt': {
            'from_prefer_input': {m: [] for m in available_metrics},
            'from_non_prefer_input': {m: [] for m in available_metrics}
        },
        'gt_vs_gt': {m: [] for m in available_metrics}  # GT comparison (prefer vs non-prefer)
    }

    # Data normalization
    normalizer = DataNormalizer(data_norm, device='cuda')

    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc='Evaluating', leave=False)):
            if max_samples is not None and sample_count >= max_samples:
                break

            # Move to GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            # Get batch size
            B = batch['inp'].shape[0]

            # Determine input indices based on aug_type
            # inp shape: (B, 4, 3, H, W)
            # 0: prefer_orig, 1: prefer_aug, 2: non_prefer_orig, 3: non_prefer_aug
            if aug_type == 'clean':
                prefer_idx = 0  # prefer_orig (clean, no augmentation)
                non_prefer_idx = 2  # non_prefer_orig (clean, no augmentation)
            elif aug_type == 'aug':
                prefer_idx = 1  # prefer_aug (with color augmentation)
                non_prefer_idx = 3  # non_prefer_aug (with color augmentation)
            else:
                raise ValueError(f"Unknown aug_type: {aug_type}. Must be 'clean' or 'aug'.")

            for i in range(B):
                if max_samples is not None and sample_count >= max_samples:
                    break

                if enable_profiling:
                    t_image_start = time.time()

                user_id = batch['user_id'][i] if isinstance(batch['user_id'], list) else batch['user_id'][i].item()
                image_id = batch['image_id'][i] if isinstance(batch['image_id'], list) else batch['image_id'][i]

                # Process both prefer and non_prefer inputs
                predictions = {}

                for input_type, inp_idx in [('prefer', prefer_idx), ('non_prefer', non_prefer_idx)]:
                    # Get single sample input
                    inp = batch['inp'][i, inp_idx, :, :, :]  # (3, H, W)
                    H, W = inp.shape[-2:]

                    # Extract patches
                    if enable_profiling:
                        t_extract_start = time.time()
                    patches, positions, original_shape = utils.extract_patches(
                        inp, patch_size=patch_size, stride=stride
                    )
                    num_patches = len(patches)
                    if enable_profiling:
                        profiling_times['extract_patches'] += time.time() - t_extract_start

                    try:
                        # OPTIMIZED: Batch all operations instead of looping
                        if enable_profiling:
                            t_batch_start = time.time()

                        # Stack all patches first
                        patches_batch = torch.stack(patches)  # (num_patches, 3, patch_h, patch_w)
                        patch_h, patch_w = patches_batch.shape[-2:]

                        # Normalize all patches at once (single call)
                        patches_batch = normalizer.normalize_input(patches_batch)

                        # Get cached coordinates (created once, reused for all patches)
                        coord_template = get_cached_coord(patch_h, patches_batch.device)

                        # Expand coordinates for all patches (single expand, not a loop!)
                        coords_batch = coord_template.unsqueeze(0).expand(num_patches, -1, -1, -1)
                        # coords_batch: (num_patches, patch_h, patch_w, 2)

                        # Create cells as batched tensor (single tensor creation)
                        cell_value = torch.tensor([2 / patch_h, 2 / patch_w], device=patches_batch.device)
                        cells_batch = cell_value.unsqueeze(0).expand(num_patches, -1)
                        # cells_batch: (num_patches, 2)

                        user_indices_batch = [user_id] * num_patches

                        if enable_profiling:
                            profiling_times['batch_prepare'] += time.time() - t_batch_start

                        # MINI-BATCHED FORWARD PASS (to avoid OOM)
                        if enable_profiling:
                            t_forward_start = time.time()

                        pred_patches_list_batched = []

                        for start_idx in range(0, num_patches, max_patches_per_batch):
                            end_idx = min(start_idx + max_patches_per_batch, num_patches)

                            # Mini-batch forward pass
                            mini_pred = model(
                                patches_batch[start_idx:end_idx],
                                coords_batch[start_idx:end_idx],
                                cells_batch[start_idx:end_idx],
                                user_indices_batch[start_idx:end_idx]
                            )
                            pred_patches_list_batched.append(mini_pred)

                        # Concatenate results
                        pred_patches_batch = torch.cat(pred_patches_list_batched, dim=0)
                        # pred_patches_batch: (num_patches, 3, patch_h, patch_w)

                        # Denormalize all patches and clamp to valid range
                        pred_patches_denorm = normalizer.denormalize_gt(pred_patches_batch).clamp(0, 1)

                        # Convert to list
                        pred_patches_list = [pred_patches_denorm[j] for j in range(num_patches)]

                        if enable_profiling:
                            profiling_times['forward_pass'] += time.time() - t_forward_start

                        # Stitch patches back to full resolution
                        if enable_profiling:
                            t_stitch_start = time.time()
                        pred_full = utils.stitch_patches(pred_patches_list, positions, original_shape)
                        # pred_full: (3, H, W)
                        if enable_profiling:
                            profiling_times['stitch_patches'] += time.time() - t_stitch_start

                        # Expand to batch dimension and store
                        predictions[input_type] = pred_full.unsqueeze(0)  # (1, 3, H, W)

                        # Log success
                        if log_fn:
                            log_fn(f"      ✓ Generated prediction for {input_type}: "
                                  f"shape={pred_full.shape}, range=[{pred_full.min():.3f}, {pred_full.max():.3f}]")

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Log OOM to file instead of stderr
                            if log_fn:
                                log_fn(f"      [WARNING] GPU OOM during mini-batch processing "
                                      f"({num_patches} patches, {max_patches_per_batch} per batch, "
                                      f"image {H}×{W}, {input_type}). Trying sequential fallback...")

                            torch.cuda.empty_cache()

                            # FALLBACK: Sequential processing (one patch at a time)
                            # Note: Sequential fallback already uses optimized batching from above
                            try:
                                pred_patches_list_seq = []
                                for j in range(num_patches):
                                    pred_patch = model(
                                        patches_batch[j:j+1],
                                        coords_batch[j:j+1],
                                        cells_batch[j:j+1],
                                        [user_indices_batch[j]]
                                    )
                                    pred_patches_list_seq.append(pred_patch)

                                pred_patches_batch = torch.cat(pred_patches_list_seq, dim=0)
                                pred_patches_denorm = normalizer.denormalize_gt(pred_patches_batch).clamp(0, 1)
                                pred_patches_list = [pred_patches_denorm[j] for j in range(num_patches)]
                                pred_full = utils.stitch_patches(pred_patches_list, positions, original_shape)
                                predictions[input_type] = pred_full.unsqueeze(0)

                                if log_fn:
                                    log_fn(f"      ✓ Sequential fallback succeeded for {input_type}")

                            except Exception as fallback_error:
                                if log_fn:
                                    log_fn(f"      ✗ Sequential fallback failed for {input_type}: {fallback_error}")
                                # Skip this image entirely
                                sample_count += 1
                                continue
                        else:
                            raise

                # Get ground truths based on aug_type
                if aug_type == 'clean':
                    # Clean mode: use augmented GT (but since input is also clean, they match)
                    gt_prefer = batch['gt_prefer'][i:i+1]
                    gt_non_prefer = batch['gt_non_prefer'][i:i+1]
                elif aug_type == 'aug':
                    # Aug mode: use ORIGINAL (clean) GT, not augmented GT
                    # This measures how well augmented input can match the original clean GT
                    gt_prefer = batch['inp'][i, 0, :, :, :].unsqueeze(0)  # prefer_orig
                    gt_non_prefer = batch['inp'][i, 2, :, :, :].unsqueeze(0)  # non_prefer_orig
                else:
                    raise ValueError(f"Unknown aug_type: {aug_type}")

                # Compute GT-vs-GT metrics (baseline comparison)
                if enable_profiling:
                    t_metrics_start = time.time()

                metrics_gt_vs_gt = metrics_calc.compute_all(gt_prefer, gt_non_prefer)
                for k, v in metrics_gt_vs_gt.items():
                    if k in metrics_storage['gt_vs_gt']:
                        metrics_storage['gt_vs_gt'][k].append(v)

                # Debug logging for predictions
                if log_fn:
                    if 'prefer' in predictions and 'non_prefer' in predictions:
                        log_fn(f"    ✓ Both predictions exist for sample {sample_count + 1}, computing metrics...")
                    else:
                        log_fn(f"    ✗ Missing predictions for sample {sample_count + 1}: "
                              f"prefer={('prefer' in predictions)}, non_prefer={('non_prefer' in predictions)}")

                # Compute metrics: both predictions against both GTs
                # Note: Using vectorized metrics (calc_delta_e_cie76, calc_ciede2000, calc_delta_e_cam16_ucs)
                if 'prefer' in predictions and 'non_prefer' in predictions:
                    # Prefer GT: measure both predictions
                    metrics_prefer_from_prefer = metrics_calc.compute_all(predictions['prefer'], gt_prefer)
                    metrics_prefer_from_nonprefer = metrics_calc.compute_all(predictions['non_prefer'], gt_prefer)

                    for k, v in metrics_prefer_from_prefer.items():
                        if k in metrics_storage['prefer_gt']['from_prefer_input']:
                            metrics_storage['prefer_gt']['from_prefer_input'][k].append(v)

                    for k, v in metrics_prefer_from_nonprefer.items():
                        if k in metrics_storage['prefer_gt']['from_non_prefer_input']:
                            metrics_storage['prefer_gt']['from_non_prefer_input'][k].append(v)

                    # Non-Prefer GT: measure both predictions
                    metrics_nonprefer_from_prefer = metrics_calc.compute_all(predictions['prefer'], gt_non_prefer)
                    metrics_nonprefer_from_nonprefer = metrics_calc.compute_all(predictions['non_prefer'], gt_non_prefer)

                    for k, v in metrics_nonprefer_from_prefer.items():
                        if k in metrics_storage['non_prefer_gt']['from_prefer_input']:
                            metrics_storage['non_prefer_gt']['from_prefer_input'][k].append(v)

                    for k, v in metrics_nonprefer_from_nonprefer.items():
                        if k in metrics_storage['non_prefer_gt']['from_non_prefer_input']:
                            metrics_storage['non_prefer_gt']['from_non_prefer_input'][k].append(v)

                if enable_profiling:
                    profiling_times['metrics_compute'] += time.time() - t_metrics_start
                    profiling_times['total_per_image'].append(time.time() - t_image_start)

                # Save images if requested
                if save_images_dir is not None and sample_count < 5:  # Save first 5
                    # Save individual images with new naming convention (prefer/non_prefer before gt/pred)
                    if 'prefer' in predictions:
                        save_image_tensor(predictions['prefer'][0],
                                         os.path.join(save_images_dir, f'{user_id}_{image_id}_prefer_pred.jpg'))
                    if 'non_prefer' in predictions:
                        save_image_tensor(predictions['non_prefer'][0],
                                         os.path.join(save_images_dir, f'{user_id}_{image_id}_non_prefer_pred.jpg'))
                    save_image_tensor(gt_prefer[0],
                                     os.path.join(save_images_dir, f'{user_id}_{image_id}_prefer_gt.jpg'))
                    save_image_tensor(gt_non_prefer[0],
                                     os.path.join(save_images_dir, f'{user_id}_{image_id}_non_prefer_gt.jpg'))

                    # Create and save 2×2 grid image
                    if 'prefer' in predictions and 'non_prefer' in predictions:
                        grid = create_grid_image(
                            gt_prefer[0], predictions['prefer'][0],
                            gt_non_prefer[0], predictions['non_prefer'][0],
                            grid_size=256
                        )
                        grid.save(os.path.join(save_images_dir, f'{user_id}_{image_id}_results_grid.jpg'),
                                 quality=95)

                sample_count += 1

    # Aggregate metrics
    aggregated = {
        'gt_vs_gt': {k: float(np.mean(v)) if len(v) > 0 else 0.0
                     for k, v in metrics_storage['gt_vs_gt'].items()},
        'prefer_gt': {
            'from_prefer_input': {k: float(np.mean(v)) if len(v) > 0 else 0.0
                                 for k, v in metrics_storage['prefer_gt']['from_prefer_input'].items()},
            'from_non_prefer_input': {k: float(np.mean(v)) if len(v) > 0 else 0.0
                                     for k, v in metrics_storage['prefer_gt']['from_non_prefer_input'].items()}
        },
        'non_prefer_gt': {
            'from_prefer_input': {k: float(np.mean(v)) if len(v) > 0 else 0.0
                                 for k, v in metrics_storage['non_prefer_gt']['from_prefer_input'].items()},
            'from_non_prefer_input': {k: float(np.mean(v)) if len(v) > 0 else 0.0
                                     for k, v in metrics_storage['non_prefer_gt']['from_non_prefer_input'].items()}
        },
        'num_samples': sample_count
    }

    # Output profiling results if enabled
    if enable_profiling and log_fn:
        log_fn(f"\n{'='*60}")
        log_fn(f"PROFILING RESULTS (sample_count={sample_count})")
        log_fn(f"{'='*60}")

        # Calculate averages
        total_images = len(profiling_times['total_per_image'])
        if total_images > 0:
            avg_per_image = np.mean(profiling_times['total_per_image'])
            log_fn(f"Average time per image: {avg_per_image:.3f}s")

            # Breakdown of time per operation (accumulated)
            log_fn(f"\nTime breakdown (accumulated):")
            log_fn(f"  Extract patches:  {profiling_times['extract_patches']:7.3f}s ({profiling_times['extract_patches']/total_images:.3f}s/img)")
            log_fn(f"  Batch prepare:    {profiling_times['batch_prepare']:7.3f}s ({profiling_times['batch_prepare']/total_images:.3f}s/img)")
            log_fn(f"  Forward pass:     {profiling_times['forward_pass']:7.3f}s ({profiling_times['forward_pass']/total_images:.3f}s/img)")
            log_fn(f"  Stitch patches:   {profiling_times['stitch_patches']:7.3f}s ({profiling_times['stitch_patches']/total_images:.3f}s/img)")
            log_fn(f"  Metrics compute:  {profiling_times['metrics_compute']:7.3f}s ({profiling_times['metrics_compute']/total_images:.3f}s/img)")

            # Calculate percentage breakdown
            total_tracked = sum([
                profiling_times['extract_patches'],
                profiling_times['batch_prepare'],
                profiling_times['forward_pass'],
                profiling_times['stitch_patches'],
                profiling_times['metrics_compute']
            ])

            log_fn(f"\nPercentage breakdown:")
            log_fn(f"  Extract patches:  {100.0 * profiling_times['extract_patches'] / total_tracked:6.2f}%")
            log_fn(f"  Batch prepare:    {100.0 * profiling_times['batch_prepare'] / total_tracked:6.2f}%")
            log_fn(f"  Forward pass:     {100.0 * profiling_times['forward_pass'] / total_tracked:6.2f}%")
            log_fn(f"  Stitch patches:   {100.0 * profiling_times['stitch_patches'] / total_tracked:6.2f}%")
            log_fn(f"  Metrics compute:  {100.0 * profiling_times['metrics_compute'] / total_tracked:6.2f}%")

        log_fn(f"{'='*60}\n")

    return aggregated


def validate_user(user_id: str,
                 eval_samples: List[int], eval_dataset,
                 train_val_samples: List[int], train_val_dataset,
                 model, config: Dict, metrics_calc, save_dir: str,
                 log_fn: Callable) -> Dict:
    """
    Validate one user with few-shot learning (test-time training).

    Process:
    1. Use fixed evaluation samples from evaluation dataset
    2. Sample N training samples from train_val dataset (excluding eval image_ids)
    3. Train decoder from scratch for max_iterations
    4. Evaluate at specified checkpoints (e.g., [100, 500])
    5. Test with multiple augmentation types (clean, different_aug)
    6. Compute all metrics on full-resolution outputs

    Args:
        user_id: str, user identifier
        eval_samples: list of int, sample indices from eval_dataset (fixed 16 samples)
        eval_dataset: PPSPreferencePairDataset for evaluation
        train_val_samples: list of int, sample indices from train_val_dataset
        train_val_dataset: PPSPreferencePairDataset for training
        model: HIIF_PPS model (encoder only, decoder will be created)
        config: dict, validation config
        metrics_calc: PPSMetrics instance
        save_dir: str, save directory
        log_fn: function, logging function

    Returns:
        dict: validation results for this user
    """
    log_fn(f"\n{'='*80}")
    log_fn(f"Validating user: {user_id}")
    log_fn(f"Evaluation samples: {len(eval_samples)} (fixed from evaluation dataset)")
    log_fn(f"Training/validation samples: {len(train_val_samples)} (before exclusion)")

    # Get configuration
    samples_per_user = config['decoder_training']['samples_per_user']
    max_iterations = config['decoder_training']['max_iterations']

    # Random seed control for sample selection
    sample_seed = config['decoder_training'].get('sample_selection_seed', -1)
    if sample_seed >= 0:
        random.seed(sample_seed)
        log_fn(f"Sample selection seed: {sample_seed} (reproducible mode)")
    else:
        log_fn(f"Sample selection seed: random mode (seed={sample_seed})")

    # 1. Use evaluation samples directly (fixed from evaluation dataset)
    log_fn(f"\n[Evaluation Samples]")
    log_fn(f"Using {len(eval_samples)} fixed evaluation samples from evaluation dataset")

    # 2. Build evaluation image_id set for exclusion
    eval_image_ids = set()
    for sample_idx in eval_samples:
        sample = eval_dataset.samples[sample_idx]
        eval_image_ids.add(sample['image_id'])

    log_fn(f"Evaluation image_ids: {sorted(eval_image_ids)}")

    # 3. Filter training samples to exclude evaluation image_ids
    log_fn(f"\n[Training Sample Selection]")
    train_candidates = []
    excluded_count = 0
    for sample_idx in train_val_samples:
        sample = train_val_dataset.samples[sample_idx]
        if sample['image_id'] not in eval_image_ids:
            train_candidates.append(sample_idx)
        else:
            log_fn(f"  Excluding sample {sample_idx} (image_id={sample['image_id']}) - matches evaluation set")
            excluded_count += 1

    log_fn(f"Training candidates after exclusion: {len(train_candidates)} (excluded {excluded_count} matching eval image_ids)")

    # 4. Check if we have enough training samples
    if len(train_candidates) == 0:
        log_fn(f"ERROR: No training samples available after excluding evaluation samples!")
        log_fn(f"       SKIPPING validation for this user")
        return {
            'user_id': user_id,
            'status': 'skipped',
            'reason': 'no_training_samples_after_exclusion',
            'eval_samples': len(eval_samples),
            'train_val_samples_before_exclusion': len(train_val_samples),
            'excluded_samples': excluded_count
        }

    # 5. Sample training set (up to samples_per_user, or all if fewer available)
    train_count = min(samples_per_user, len(train_candidates))

    if len(train_candidates) <= train_count:
        # Use all available samples
        train_samples = train_candidates
        log_fn(f"Using all {len(train_samples)} available training samples (requested {samples_per_user}, adaptive)")
    else:
        # Random sample
        train_samples = random.sample(train_candidates, train_count)
        log_fn(f"Sampled {train_count} training samples from {len(train_candidates)} candidates")

    log_fn(f"\n[Sample Split Summary]")
    log_fn(f"Final split: {len(train_samples)} train + {len(eval_samples)} eval (exclusive by image_id)")
    log_fn(f"  Train samples: {len(train_samples)} (from train_val_dataset)")
    log_fn(f"  Eval samples: {len(eval_samples)} (from eval_dataset, fixed)")
    log_fn(f"  Exclusion: {excluded_count} train_val samples matched eval image_ids")

    # Create training dataloader (cropped patches) - uses train_val_dataset
    train_aug_config = config['augmentation']['train_aug']
    batch_size = config['decoder_training']['batch_size']
    num_workers = config['decoder_training'].get('num_workers', 8)  # Default: 8
    batch_image = config['decoder_training'].get('batch_image', None)  # Optional: batch_image for efficient I/O
    train_loader = create_dataloader_with_aug(train_val_dataset, train_samples,
                                             train_aug_config, batch_size, num_workers,
                                             batch_image=batch_image)

    # Get patch processing parameters
    patch_size = config['decoder_training'].get('patch_size', 128)
    stride = config['decoder_training'].get('stride', 64)

    # Get evaluation dataloader parameters
    eval_batch_size = config['decoder_training'].get('eval_batch_size', 1)
    eval_num_workers = config['decoder_training'].get('eval_num_workers', 0)

    # Create evaluation dataloaders (full resolution) - uses eval_dataset
    eval_loaders = {
        'clean': create_fullimage_dataloader(
            eval_dataset, eval_samples,
            config['augmentation']['clean'], batch_size=eval_batch_size,
            num_workers=eval_num_workers
        ),
    }

    # Only add aug if it exists in config
    if 'aug' in config['augmentation']:
        eval_loaders['aug'] = create_fullimage_dataloader(
            eval_dataset, eval_samples,
            config['augmentation']['aug'], batch_size=eval_batch_size,
            num_workers=eval_num_workers
        )

    # Load decoder for this user (create new from scratch)
    model.load_user_decoder(user_id, checkpoint_manager=None)  # No checkpoint = create new

    # Create decoder optimizer
    import utils as utils_module
    decoder_optimizer = utils_module.make_optimizer(
        model.current_decoder.parameters(),
        config['decoder_training']['optimizer']
    )

    # Create loss scheduler for this user
    max_iterations = config['decoder_training']['max_iterations']
    loss_config = config['decoder_training']['loss_schedule']
    loss_scheduler = LossWeightScheduler(
        schedule_type=loss_config['schedule_type'],
        w_p_start=loss_config['w_p_start'],
        w_p_end=loss_config['w_p_end'],
        total_iterations=max_iterations
    )

    # Training and evaluation
    eval_checkpoints = config['decoder_training']['eval_checkpoints']
    results = {
        'user_id': user_id,
        'train_samples': len(train_samples),
        'eval_samples': len(eval_samples),
        'checkpoints': {}
    }

    iteration = 0
    train_loader_iter = iter(train_loader)

    encoder_optimizer = None  # Not used, encoder is frozen

    # Profiling for decoder training
    training_times = {
        'data_loading': [],
        'forward_backward': [],
        'total_step': []
    }
    enable_training_profiling = True  # Profile first checkpoint

    pbar = tqdm(total=max_iterations, desc=f'Training {user_id}', leave=False)

    while iteration < max_iterations:
        if enable_training_profiling:
            t_step_start = time.time()

        # Get batch (cycle through dataloader)
        if enable_training_profiling:
            t_data_start = time.time()

        try:
            batch = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        if enable_training_profiling:
            training_times['data_loading'].append(time.time() - t_data_start)

        # Train one step
        if enable_training_profiling:
            t_train_start = time.time()

        loss = train_decoder_one_step(model, batch, encoder_optimizer, decoder_optimizer,
                                     loss_scheduler, config['data_norm'])

        if enable_training_profiling:
            training_times['forward_backward'].append(time.time() - t_train_start)
            training_times['total_step'].append(time.time() - t_step_start)

        iteration += 1
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss:.4f}', 'w_p': f'{loss_scheduler.get_weight():.3f}'})

        # Evaluate at checkpoints
        if iteration in eval_checkpoints:
            log_fn(f"\n[Checkpoint {iteration}] Evaluating...")

            checkpoint_results = {'iteration': iteration, 'aug_types': {}}

            for aug_type, eval_loader in eval_loaders.items():
                log_fn(f"  Aug type: {aug_type}")

                # Create save dir for images
                if config.get('save_images', True):
                    images_dir = os.path.join(save_dir, 'images', user_id,
                                            f'iter_{iteration:04d}', aug_type)
                else:
                    images_dir = None

                # Get max_patches_per_batch from config (default: 16)
                max_patches_per_batch = config['decoder_training'].get('max_patches_per_batch', 16)

                # Evaluate (enable profiling for first checkpoint only to reduce log spam)
                enable_prof = (iteration == eval_checkpoints[0])
                metrics = evaluate_decoder(model, eval_loader, metrics_calc,
                                         config['data_norm'], max_samples=20,
                                         save_images_dir=images_dir,
                                         patch_size=patch_size, stride=stride,
                                         aug_type=aug_type,
                                         max_patches_per_batch=max_patches_per_batch,
                                         log_fn=log_fn,
                                         enable_profiling=enable_prof)

                checkpoint_results['aug_types'][aug_type] = metrics

                # Log GT-vs-GT comparison first (baseline)
                if 'gt_vs_gt' in metrics:
                    parts = []
                    for metric_name in sorted(metrics['gt_vs_gt'].keys()):
                        value = metrics['gt_vs_gt'][metric_name]
                        parts.append(f"{metric_name.upper()}: {value:.4f}")
                    log_fn(f"    GT Comparison (Prefer vs Non-Prefer):")
                    log_fn(f"      {', '.join(parts)}")

                # Log metrics - new format with GT-based organization
                log_fn(f"    Prefer GT:")
                for input_type in ['from_prefer_input', 'from_non_prefer_input']:
                    parts = []
                    for metric_name in sorted(metrics['prefer_gt'][input_type].keys()):
                        value = metrics['prefer_gt'][input_type][metric_name]
                        parts.append(f"{metric_name.upper()}: {value:.4f}")
                    input_label = "prefer input" if input_type == 'from_prefer_input' else "non-prefer input"
                    log_fn(f"      {input_label:18s} - {', '.join(parts)}")

                # Prefer GT average
                avg_metrics_prefer = {}
                for metric_name in sorted(metrics['prefer_gt']['from_prefer_input'].keys()):
                    val1 = metrics['prefer_gt']['from_prefer_input'][metric_name]
                    val2 = metrics['prefer_gt']['from_non_prefer_input'][metric_name]
                    avg_metrics_prefer[metric_name] = (val1 + val2) / 2.0
                parts = [f"{m.upper()}: {v:.4f}" for m, v in sorted(avg_metrics_prefer.items())]
                log_fn(f"      {'[AVERAGE]':18s} - {', '.join(parts)}")

                log_fn(f"    Non-Prefer GT:")
                for input_type in ['from_prefer_input', 'from_non_prefer_input']:
                    parts = []
                    for metric_name in sorted(metrics['non_prefer_gt'][input_type].keys()):
                        value = metrics['non_prefer_gt'][input_type][metric_name]
                        parts.append(f"{metric_name.upper()}: {value:.4f}")
                    input_label = "prefer input" if input_type == 'from_prefer_input' else "non-prefer input"
                    log_fn(f"      {input_label:18s} - {', '.join(parts)}")

                # Non-Prefer GT average
                avg_metrics_non_prefer = {}
                for metric_name in sorted(metrics['non_prefer_gt']['from_prefer_input'].keys()):
                    val1 = metrics['non_prefer_gt']['from_prefer_input'][metric_name]
                    val2 = metrics['non_prefer_gt']['from_non_prefer_input'][metric_name]
                    avg_metrics_non_prefer[metric_name] = (val1 + val2) / 2.0
                parts = [f"{m.upper()}: {v:.4f}" for m, v in sorted(avg_metrics_non_prefer.items())]
                log_fn(f"      {'[AVERAGE]':18s} - {', '.join(parts)}")

            results['checkpoints'][f'iter_{iteration:04d}'] = checkpoint_results

            # Output decoder training profiling (first checkpoint only)
            if iteration == eval_checkpoints[0] and len(training_times['total_step']) > 0:
                avg_total = np.mean(training_times['total_step'])
                avg_data = np.mean(training_times['data_loading'])
                avg_train = np.mean(training_times['forward_backward'])

                log_fn("")
                log_fn("=" * 60)
                log_fn(f"DECODER TRAINING PROFILING (first {len(training_times['total_step'])} iterations)")
                log_fn("=" * 60)
                log_fn(f"Average time per iteration: {avg_total:.3f}s")
                log_fn("")
                log_fn("Time breakdown:")
                log_fn(f"  Data loading:     {avg_data:.3f}s ({100.0 * avg_data / avg_total:6.2f}%)")
                log_fn(f"  Forward/Backward: {avg_train:.3f}s ({100.0 * avg_train / avg_total:6.2f}%)")
                log_fn("=" * 60)
                log_fn("")

                # Clear profiling data after first checkpoint
                training_times['data_loading'].clear()
                training_times['forward_backward'].clear()
                training_times['total_step'].clear()

    pbar.close()

    # Offload decoder
    model.offload_user_decoder(checkpoint_manager=None)

    return results
