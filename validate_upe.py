"""
Validation Script for UPE (User Preference Embedding) System

Performs direct inference validation (NO test-time training):
1. Load trained model from checkpoint
2. For each validation user:
   - Extract UPE from first N preference pairs (default: 16)
   - Evaluate on remaining samples with direct inference
3. Compute all metrics: PSNR, SSIM, LPIPS, Delta E variants, etc.
4. Save results as JSON + optional visualizations

Configuration Options:
- random_seed (optional): Integer seed for reproducible sample selection.
  When specified, samples are shuffled deterministically before splitting
  into UPE extraction and evaluation sets. Each user gets a unique but
  reproducible shuffle based on the seed and their user_id.
"""

import argparse
import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from PIL import Image

import datasets
import models
import utils
from metrics import PPSMetrics
from models.upe_extractor import UPEExtractor
from pps_utils.upe_cache import UPECache
from pps_utils.data_processing import DataNormalizer, preprocess_pps_batch_with_upe
from utils import extract_patches, stitch_patches


def tensor_to_pil(tensor):
    """
    Convert a (3, H, W) tensor in [0, 1] range to PIL Image.

    Args:
        tensor: (3, H, W) tensor in [0, 1] range

    Returns:
        PIL Image in RGB format
    """
    # Clamp to [0, 1] and move to CPU
    tensor = torch.clamp(tensor, 0, 1).cpu()

    # Convert to [0, 255] uint8
    tensor = (tensor * 255).byte()

    # Permute from (3, H, W) to (H, W, 3)
    array = tensor.permute(1, 2, 0).numpy()

    return Image.fromarray(array, mode='RGB')


def create_grid_2x2(images_list):
    """
    Create a 2x2 grid from 4 PIL images.

    Grid layout:
        [images_list[0]]  [images_list[1]]
        [images_list[2]]  [images_list[3]]

    Args:
        images_list: List of 4 PIL Images

    Returns:
        PIL Image with 2x2 grid (size depends on aspect ratios)
    """
    if len(images_list) != 4:
        raise ValueError(f"Expected 4 images, got {len(images_list)}")

    # Resize each image proportionally with longest side at 256 pixels
    resized = []
    for img in images_list:
        w, h = img.size
        if w >= h:
            new_w = 256
            new_h = int(h * 256 / w)
        else:
            new_h = 256
            new_w = int(w * 256 / h)
        resized.append(img.resize((new_w, new_h), Image.LANCZOS))

    # Find max dimensions for uniform grid
    max_w = max(img.size[0] for img in resized)
    max_h = max(img.size[1] for img in resized)

    # Create grid canvas with white background
    grid_w = max_w * 2
    grid_h = max_h * 2
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    # Paste images in 2x2 layout
    grid.paste(resized[0], (0, 0))          # Top-left
    grid.paste(resized[1], (max_w, 0))      # Top-right
    grid.paste(resized[2], (0, max_h))      # Bottom-left
    grid.paste(resized[3], (max_w, max_h))  # Bottom-right

    return grid


def save_validation_images(sample, pred_denorm, gt_prefer_denorm, gt_non_prefer_denorm, save_dir, eval_mode='patch', aug_mode='clean', aug_idx=0):
    """
    Save validation images for a single sample.

    Saves:
    1. Individual ground truth images (prefer, non_prefer)
    2. Individual prediction images (pred_from_prefer, pred_from_non_prefer)
    3. 2x2 grid: [[prefer_gt, pred_from_prefer], [non_prefer_gt, pred_from_non_prefer]]

    Args:
        sample: Dict with 'user_id' and 'image_id' keys
        pred_denorm: (B*4, 3, H, W) predictions in [0, 1] range (or (4, 3, H, W) for full mode)
        gt_prefer_denorm: (B*4, 3, H, W) prefer GT in [0, 1] range (or (1, 3, H, W) for full mode)
        gt_non_prefer_denorm: (B*4, 3, H, W) non-prefer GT in [0, 1] range (or (1, 3, H, W) for full mode)
        save_dir: Directory to save images
        eval_mode: str, 'patch' or 'full' - determines filename prefix
        aug_mode: str, 'clean' or 'aug' - augmentation mode
        aug_idx: int, augmentation index (for num_aug_imgs > 1)
    """
    user_id = sample['user_id']
    image_id = sample['image_id']

    # Extract indices: 0=prefer_orig, 2=non_prefer_orig
    # (indices 1 and 3 are duplicates when augment=False)
    if eval_mode == 'full':
        # In full mode, pred_denorm is (4, 3, H, W), gt is (1, 3, H, W)
        prefer_idx = 0
        non_prefer_idx = 2
        prefer_gt = gt_prefer_denorm[0] if gt_prefer_denorm.shape[0] == 1 else gt_prefer_denorm[prefer_idx]
        non_prefer_gt = gt_non_prefer_denorm[0] if gt_non_prefer_denorm.shape[0] == 1 else gt_non_prefer_denorm[non_prefer_idx]
        pred_from_prefer = pred_denorm[prefer_idx]
        pred_from_non_prefer = pred_denorm[non_prefer_idx]
    else:
        # In patch mode, all are (B*4, 3, H, W)
        prefer_idx = 0
        non_prefer_idx = 2
        prefer_gt = gt_prefer_denorm[prefer_idx]
        non_prefer_gt = gt_non_prefer_denorm[non_prefer_idx]
        pred_from_prefer = pred_denorm[prefer_idx]
        pred_from_non_prefer = pred_denorm[non_prefer_idx]

    # Convert to PIL Images
    prefer_gt_img = tensor_to_pil(prefer_gt)
    non_prefer_gt_img = tensor_to_pil(non_prefer_gt)
    pred_from_prefer_img = tensor_to_pil(pred_from_prefer)
    pred_from_non_prefer_img = tensor_to_pil(pred_from_non_prefer)

    # Create naming prefix with eval_mode and aug_mode indicators
    mode_suffix = "full" if eval_mode == "full" else "patch"
    aug_suffix = f"aug{aug_idx}" if aug_idx > 0 or aug_mode == 'aug' else ""

    if aug_suffix:
        prefix = f"{user_id}_{image_id}_{mode_suffix}_{aug_mode}_{aug_suffix}"
    else:
        prefix = f"{user_id}_{image_id}_{mode_suffix}_{aug_mode}"

    # Save individual images
    prefer_gt_img.save(os.path.join(save_dir, f"{prefix}_prefer_gt.png"))
    non_prefer_gt_img.save(os.path.join(save_dir, f"{prefix}_non_prefer_gt.png"))
    pred_from_prefer_img.save(os.path.join(save_dir, f"{prefix}_pred_from_prefer.png"))
    pred_from_non_prefer_img.save(os.path.join(save_dir, f"{prefix}_pred_from_non_prefer.png"))

    # Create and save 2x2 grid
    # Layout: [[prefer_gt, pred_from_prefer], [non_prefer_gt, pred_from_non_prefer]]
    grid_images = [prefer_gt_img, pred_from_prefer_img, non_prefer_gt_img, pred_from_non_prefer_img]
    grid = create_grid_2x2(grid_images)
    grid.save(os.path.join(save_dir, f"{prefix}_grid.png"))


def auto_configure_upe_input_dim(config):
    """
    Automatically calculate and inject UPE processor input_dim from upe_config.

    This function:
    1. Reads content_model and color_model from upe_config
    2. Calculates the expected input_dim = content_dim + color_dim
    3. Injects input_dim into upe_processor_config if not present
    4. Injects content_dim and color_dim into upe_config for cache validation
    5. Validates input_dim if manually specified

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If input_dim is specified but doesn't match expected value
    """
    from models.feature_extractors import get_feature_extractor_dim

    upe_config = config.get('upe_config', {})
    content_model = upe_config.get('content_model')
    color_model = upe_config.get('color_model')

    if not content_model or not color_model:
        # If upe_config is not fully specified, skip auto-configuration
        return

    # Calculate expected dimensions (uses fixed variants per model)
    content_dim = get_feature_extractor_dim(content_model)
    color_dim = get_feature_extractor_dim(color_model)
    expected_input_dim = content_dim + color_dim

    # Inject dimensions into upe_config for cache validation
    upe_config['content_dim'] = content_dim
    upe_config['color_dim'] = color_dim

    # Get or create upe_processor_config
    if 'model' not in config or 'args' not in config['model']:
        return  # No model config, skip

    model_args = config['model']['args']
    if 'upe_processor_config' not in model_args:
        model_args['upe_processor_config'] = {}

    upe_processor_config = model_args['upe_processor_config']

    if 'input_dim' in upe_processor_config:
        # Validate manually specified input_dim
        actual_input_dim = upe_processor_config['input_dim']
        if actual_input_dim != expected_input_dim:
            print(f"⚠️  WARNING: UPE processor input_dim mismatch!")
            print(f"   Expected: {expected_input_dim} ({content_model}={content_dim} + {color_model}={color_dim})")
            print(f"   Got: {actual_input_dim}")
            print(f"   Using auto-calculated value: {expected_input_dim}")
            upe_processor_config['input_dim'] = expected_input_dim
    else:
        # Auto-inject calculated input_dim
        upe_processor_config['input_dim'] = expected_input_dim
        print(f"✅ Auto-configured UPE processor input_dim: {expected_input_dim}")
        print(f"   Content: {content_model} ({content_dim})")
        print(f"   Color: {color_model} ({color_dim})")


def validate_config(config):
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required keys are missing or values are invalid
    """
    # Check required top-level keys
    required_keys = ['model', 'val_dataset', 'upe_config']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    # Validate upe_config
    upe_cfg = config['upe_config']
    if 'num_pairs' not in upe_cfg:
        raise ValueError("Missing 'upe_config.num_pairs'")
    if upe_cfg['num_pairs'] <= 0:
        raise ValueError(f"upe_config.num_pairs must be > 0, got {upe_cfg['num_pairs']}")

    # Validate metrics config
    if 'metrics' in config:
        if not isinstance(config['metrics'], list):
            raise ValueError("'metrics' must be a list of metric names")


def create_eval_dataset(base_dataset, eval_sample_indices, config, aug_mode='clean'):
    """
    Create evaluation dataset from specific sample indices.

    Args:
        base_dataset: Original PPSPreferencePairDataset
        eval_sample_indices: List of sample indices to include
        config: Validation config dict
        aug_mode: str, 'clean' or 'aug' - whether to apply augmentation

    Returns:
        eval_dataset: Dataset ready for evaluation (with transforms applied)
    """
    from torch.utils.data import Subset

    # Create subset with only eval samples
    eval_subset = Subset(base_dataset, eval_sample_indices)

    # Check eval_mode from validation config
    eval_mode = config.get('validation', {}).get('eval_mode', 'patch')

    # Determine augmentation setting based on aug_mode
    use_augment = (aug_mode == 'aug')

    # Apply wrapper based on eval_mode
    if 'wrapper' in config['val_dataset']:
        wrapper_config = config['val_dataset']['wrapper']

        if eval_mode == 'full':
            # Use full-image wrapper (no crop, no resize)
            from datasets.pps_fullimage_wrapper import PPSFullImageWrapper

            eval_dataset = PPSFullImageWrapper(
                dataset=eval_subset,
                augment=use_augment,
            )
        elif wrapper_config['name'] == 'pps-color-augmented':
            # Use crop-based wrapper (current behavior for patch mode)
            from datasets.pps_wrapper import PPSColorAugmentedWrapper

            # Create wrapper with augmentation based on aug_mode
            eval_dataset = PPSColorAugmentedWrapper(
                dataset=eval_subset,
                crop_size=wrapper_config['args'].get('crop_size', 128),
                augment=use_augment,
            )
        else:
            eval_dataset = eval_subset
    else:
        eval_dataset = eval_subset

    return eval_dataset


def validate_user_with_upe(
    user_id: str,
    user_samples: List[int],
    base_dataset,
    model: nn.Module,
    upe_extractor: UPEExtractor,
    upe_cache: UPECache,
    metrics_calc: PPSMetrics,
    config: Dict,
    normalizer: DataNormalizer,
    device: torch.device,
    save_dir: Optional[Path] = None,
    aug_mode: str = 'clean',
    num_aug_imgs: int = 1,
) -> Optional[Dict]:
    """
    Validate one user with direct inference (NO training).

    Steps:
    1. Check if user has sufficient samples
    2. Extract UPE from first N samples (or load from cache)
    3. Evaluate on remaining samples with direct inference
    4. Return metrics

    Args:
        user_id: User identifier
        user_samples: List of sample indices for this user
        base_dataset: Base dataset
        model: Trained HIIF_Global_UPE model
        upe_extractor: UPE extractor
        upe_cache: UPE cache
        metrics_calc: Metrics calculator
        config: Validation config
        normalizer: Data normalizer
        device: Device to use
        save_dir: Directory to save results
        aug_mode: str, 'clean' or 'aug' - augmentation mode
        num_aug_imgs: int, number of augmentations per image (for aug mode)

    Returns:
        Dict with metrics and metadata, or None if user has insufficient samples
    """
    num_upe = config['upe_config']['num_pairs']

    # Apply random seed for reproducible sample selection if specified
    random_seed = config.get('random_seed', None)
    if random_seed is not None:
        # Create a user-specific seed by combining global seed with user_id hash
        # This ensures different users get different shuffles but reproducibly
        import hashlib
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % (2**32)
        user_seed = random_seed + user_hash

        # Shuffle samples deterministically
        rng = np.random.RandomState(user_seed)
        shuffled_samples = user_samples.copy()
        rng.shuffle(shuffled_samples)
        user_samples = shuffled_samples

    # Load evaluation samples (UPE is already cached separately)
    num_eval_pairs = config.get('validation', {}).get('num_eval_pairs', -1)
    if num_eval_pairs > 0:
        eval_sample_indices = user_samples[:num_eval_pairs]
    else:
        # -1 means use all available samples
        eval_sample_indices = user_samples

    # Load UPE (already extracted and cached)
    upe_raw = upe_cache.load(user_id, config['upe_config'])
    upe_cached = True

    if upe_raw is None:
        # UPE should be pre-extracted for validation, but extract if missing
        # Use first num_upe pairs for UPE extraction
        upe_sample_indices = user_samples[:num_upe]
        upe_samples = [base_dataset[i] for i in upe_sample_indices]
        upe_raw = upe_extractor(upe_samples)
        upe_cache.save(user_id, upe_raw, config['upe_config'])
        upe_cached = False

    # Move raw UPE to GPU (keep as raw for now)
    upe_raw = upe_raw.to(device)  # (16, 1280)

    # Create eval dataset with specified aug_mode
    eval_dataset = create_eval_dataset(base_dataset, eval_sample_indices, config, aug_mode=aug_mode)

    # Create eval loader (batch_size=1 since images have different sizes)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Create images directory if save_dir is provided (mode-specific subdirectory)
    images_dir = None
    if save_dir is not None:
        images_dir = Path(save_dir) / 'images' / aug_mode
        images_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate with direct inference
    # Store metrics for both prefer and non_prefer GTs separately
    # Now track metrics for each of 4 input types: prefer_orig, prefer_aug, non_prefer_orig, non_prefer_aug
    metric_names = config.get('metrics', ['psnr', 'ssim', 'lpips'])

    # Metrics structure: {gt_type: {input_type: {metric_name: [values]}}}
    # For num_aug_imgs > 1, we collect metrics per augmentation, then average
    metrics_prefer_gt = {
        'prefer_orig': {name: [] for name in metric_names},
        'prefer_aug': {name: [] for name in metric_names},
        'non_prefer_orig': {name: [] for name in metric_names},
        'non_prefer_aug': {name: [] for name in metric_names},
    }
    metrics_non_prefer_gt = {
        'prefer_orig': {name: [] for name in metric_names},
        'prefer_aug': {name: [] for name in metric_names},
        'non_prefer_orig': {name: [] for name in metric_names},
        'non_prefer_aug': {name: [] for name in metric_names},
    }

    # GT comparison metrics (prefer_gt vs non_prefer_gt)
    gt_comparison_metrics = {name: [] for name in metric_names}

    # Get eval_mode for conditional processing
    eval_mode = config.get('validation', {}).get('eval_mode', 'patch')

    model.eval()

    # Loop over augmentations (for num_aug_imgs > 1)
    for aug_idx in range(num_aug_imgs):
        # Set deterministic seed for this augmentation iteration
        if aug_mode == 'aug' and num_aug_imgs > 1:
            import random
            import hashlib
            # Create a deterministic seed based on user_id and aug_idx
            seed_str = f"{user_id}_{aug_idx}"
            seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            torch.manual_seed(seed_hash)
            np.random.seed(seed_hash)
            random.seed(seed_hash)

        with torch.no_grad():
            for sample in eval_loader:
                # Sample structure after wrapper:
                #   inp: (B, 4, 3, H, W) - 4-way augmented input
                #   gt_prefer: (B, 3, H, W) - preferred ground truth
                #   gt_non_prefer: (B, 3, H, W) - non-preferred ground truth
                #   coord: (B, H, W, 2) - coordinates
                #   cell: (B, 2) - cell size

                # Move to GPU
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        sample[k] = v.to(device)

                # Add raw UPE to sample (expand to match batch)
                B = sample['inp'].size(0)
                sample['upe'] = upe_raw.unsqueeze(0).expand(B, -1, -1)  # (B, 16, 1280)

                # Preprocess batch (reshape 4-way aug)
                inp, coord, cell, gt_prefer, gt_non_prefer, upe_batch, user_id = \
                    preprocess_pps_batch_with_upe(sample)

                # Now shapes are:
                #   inp: (B*4, 3, H, W)
                #   coord: (B*4, H, W, 2)
                #   cell: (B*4, 2)
                #   gt_prefer: (B*4, 3, H, W)
                #   gt_non_prefer: (B*4, 3, H, W)
                #   upe_batch: (B*4, 16, 1280)

                if eval_mode == 'full':
                    # FULL IMAGE MODE: Process with patch-based inference
                    patch_size = config.get('validation', {}).get('patch_size', 128)
                    stride = config.get('validation', {}).get('stride', 64)

                    # Process each of the 4 input versions separately
                    pred_list = []
                    for i in range(4):  # prefer_orig, prefer_aug, non_prefer_orig, non_prefer_aug
                        inp_single = inp[i:i+1]  # (1, 3, H, W)
                        coord_single = coord[i:i+1]  # (1, H, W, 2)
                        cell_single = cell[i:i+1]  # (1, 2)
                        upe_single = upe_batch[i:i+1]  # (1, 16, 1280)

                        # Extract patches from the input image
                        patches, positions, orig_shape = extract_patches(
                            inp_single[0], patch_size, stride
                        )

                        # Process patches in batches for efficiency
                        patch_batch_size = config.get('validation', {}).get('patch_batch_size', 16)
                        patches_list = list(patches)
                        num_patches = len(patches_list)

                        patch_preds = []
                        for batch_start in range(0, num_patches, patch_batch_size):
                            batch_end = min(batch_start + patch_batch_size, num_patches)
                            batch_patches = patches_list[batch_start:batch_end]
                            current_batch_size = len(batch_patches)

                            # Stack patches into batch: (N, 3, patch_size, patch_size)
                            patch_batch_tensor = torch.stack(batch_patches, dim=0)

                            # Generate coordinates for batch (N, patch_size, patch_size, 2)
                            coord_patch = utils.make_coord([patch_size, patch_size], flatten=False)
                            coord_batch = coord_patch.unsqueeze(0).repeat(current_batch_size, 1, 1, 1).to(device)

                            # Cell size for batch (N, 2)
                            cell_value = torch.tensor([2/patch_size, 2/patch_size], dtype=torch.float32)
                            cell_batch = cell_value.unsqueeze(0).repeat(current_batch_size, 1).to(device)

                            # Expand UPE for batch (N, 16, 1280)
                            upe_batch_expanded = upe_single.repeat(current_batch_size, 1, 1)

                            # Normalize batch
                            patch_batch_tensor = normalizer.normalize_input(patch_batch_tensor)

                            # Inference on batch of patches
                            pred_batch = model(patch_batch_tensor, coord_batch, cell_batch, upe_batch_expanded)

                            # Denormalize and collect predictions
                            pred_batch_denorm = normalizer.denormalize_gt(pred_batch)  # (N, 3, H, W)
                            for j in range(pred_batch_denorm.shape[0]):
                                patch_preds.append(pred_batch_denorm[j])  # Append each (3, H, W)

                        # Stitch patches back together
                        full_pred = stitch_patches(patch_preds, positions, orig_shape)
                        pred_list.append(full_pred)

                    # Stack predictions from all 4 versions
                    pred_denorm = torch.stack(pred_list, dim=0)  # (4, 3, H, W)

                    # GT is already at full resolution from PPSFullImageWrapper
                    # Denormalize GT (no need to normalize then denormalize, but for consistency)
                    gt_prefer_denorm = gt_prefer[0::4]  # Take only one copy (all 4 are identical)
                    gt_non_prefer_denorm = gt_non_prefer[0::4]

                else:
                    # PATCH MODE: Direct inference (current behavior)
                    # Normalize
                    inp = normalizer.normalize_input(inp)
                    gt_prefer = normalizer.normalize_gt(gt_prefer)
                    gt_non_prefer = normalizer.normalize_gt(gt_non_prefer)

                    # Forward pass (model processes UPE internally)
                    # Model expects coord in (B, H, W, 2) format
                    pred = model(inp, coord, cell, upe_batch)  # (B*4, 3, H, W)

                    # Denormalize
                    pred_denorm = normalizer.denormalize_gt(pred)
                    gt_prefer_denorm = normalizer.denormalize_gt(gt_prefer)
                    gt_non_prefer_denorm = normalizer.denormalize_gt(gt_non_prefer)

                # Split predictions by input type (B*4 → 4 separate predictions)
                # pred[0::4]: prefer_orig, pred[1::4]: prefer_aug
                # pred[2::4]: non_prefer_orig, pred[3::4]: non_prefer_aug
                pred_prefer_orig = pred_denorm[0::4]       # (B, 3, H, W)
                pred_prefer_aug = pred_denorm[1::4]        # (B, 3, H, W)
                pred_non_prefer_orig = pred_denorm[2::4]   # (B, 3, H, W)
                pred_non_prefer_aug = pred_denorm[3::4]    # (B, 3, H, W)

                # GT also needs to be split (though all 4 are identical)
                gt_prefer_single = gt_prefer_denorm[0::4]  # (B, 3, H, W)
                gt_non_prefer_single = gt_non_prefer_denorm[0::4]  # (B, 3, H, W)

                # Compute GT comparison (prefer_gt vs non_prefer_gt)
                gt_comp_dict = metrics_calc.compute_all(gt_prefer_single, gt_non_prefer_single)
                for name, value in gt_comp_dict.items():
                    if name in gt_comparison_metrics:
                        gt_comparison_metrics[name].append(value)

                # Compute metrics for each input type against PREFER GT
                for pred_type, pred_data in [
                    ('prefer_orig', pred_prefer_orig),
                    ('prefer_aug', pred_prefer_aug),
                    ('non_prefer_orig', pred_non_prefer_orig),
                    ('non_prefer_aug', pred_non_prefer_aug),
                ]:
                    metrics_dict = metrics_calc.compute_all(pred_data, gt_prefer_single)
                    for name, value in metrics_dict.items():
                        if name in metrics_prefer_gt[pred_type]:
                            metrics_prefer_gt[pred_type][name].append(value)

                # Compute metrics for each input type against NON_PREFER GT
                for pred_type, pred_data in [
                    ('prefer_orig', pred_prefer_orig),
                    ('prefer_aug', pred_prefer_aug),
                    ('non_prefer_orig', pred_non_prefer_orig),
                    ('non_prefer_aug', pred_non_prefer_aug),
                ]:
                    metrics_dict = metrics_calc.compute_all(pred_data, gt_non_prefer_single)
                    for name, value in metrics_dict.items():
                        if name in metrics_non_prefer_gt[pred_type]:
                            metrics_non_prefer_gt[pred_type][name].append(value)

                # Save validation images if directory is provided
                if images_dir is not None:
                    save_validation_images(sample, pred_denorm, gt_prefer_denorm, gt_non_prefer_denorm,
                                          str(images_dir), eval_mode=eval_mode, aug_mode=aug_mode, aug_idx=aug_idx)

    # Aggregate metrics
    results = {
        'num_eval_pairs': len(eval_sample_indices),
        'metrics': {
            'gt_comparison': {},
            'prefer_gt': {
                'prefer_orig': {},
                'prefer_aug': {},
                'non_prefer_orig': {},
                'non_prefer_aug': {},
                'average': {},
            },
            'non_prefer_gt': {
                'prefer_orig': {},
                'prefer_aug': {},
                'non_prefer_orig': {},
                'non_prefer_aug': {},
                'average': {},
            },
        },
        'upe_info': {
            'num_upe_pairs': num_upe,
            'upe_cached': upe_cached,
        }
    }

    # Average GT comparison metrics
    for name, values in gt_comparison_metrics.items():
        if values:
            results['metrics']['gt_comparison'][name] = float(np.mean(values))

    # Average metrics for PREFER GT (for each input type)
    for input_type in ['prefer_orig', 'prefer_aug', 'non_prefer_orig', 'non_prefer_aug']:
        for name, values in metrics_prefer_gt[input_type].items():
            if values:
                results['metrics']['prefer_gt'][input_type][name] = float(np.mean(values))

    # Average metrics for NON_PREFER GT (for each input type)
    for input_type in ['prefer_orig', 'prefer_aug', 'non_prefer_orig', 'non_prefer_aug']:
        for name, values in metrics_non_prefer_gt[input_type].items():
            if values:
                results['metrics']['non_prefer_gt'][input_type][name] = float(np.mean(values))

    # Calculate averages across all 4 input types
    for name in metric_names:
        # Average for PREFER GT
        prefer_gt_values = []
        for input_type in ['prefer_orig', 'prefer_aug', 'non_prefer_orig', 'non_prefer_aug']:
            if name in results['metrics']['prefer_gt'][input_type]:
                prefer_gt_values.append(results['metrics']['prefer_gt'][input_type][name])
        if prefer_gt_values:
            results['metrics']['prefer_gt']['average'][name] = float(np.mean(prefer_gt_values))

        # Average for NON_PREFER GT
        non_prefer_gt_values = []
        for input_type in ['prefer_orig', 'prefer_aug', 'non_prefer_orig', 'non_prefer_aug']:
            if name in results['metrics']['non_prefer_gt'][input_type]:
                non_prefer_gt_values.append(results['metrics']['non_prefer_gt'][input_type][name])
        if non_prefer_gt_values:
            results['metrics']['non_prefer_gt']['average'][name] = float(np.mean(non_prefer_gt_values))

    return results


def aggregate_results(all_results: Dict[str, Dict]) -> Dict:
    """
    Aggregate per-user results into average metrics.

    Args:
        all_results: Dict of per-user results

    Returns:
        avg_metrics: Dict with mean and std for each metric, separated by input type and GT type
    """
    if not all_results:
        return {}

    # Get all metric names from first user's results
    first_result = next(iter(all_results.values()))

    aggregated = {
        'gt_comparison': {},
        'prefer_gt': {
            'prefer_orig': {},
            'prefer_aug': {},
            'non_prefer_orig': {},
            'non_prefer_aug': {},
            'average': {},
        },
        'non_prefer_gt': {
            'prefer_orig': {},
            'prefer_aug': {},
            'non_prefer_orig': {},
            'non_prefer_aug': {},
            'average': {},
        }
    }

    # Aggregate GT comparison metrics
    if 'gt_comparison' in first_result['metrics']:
        for metric_name in first_result['metrics']['gt_comparison'].keys():
            values = [
                result['metrics']['gt_comparison'][metric_name]
                for result in all_results.values()
                if metric_name in result['metrics']['gt_comparison']
            ]
            if values:
                aggregated['gt_comparison'][metric_name] = float(np.mean(values))
                aggregated['gt_comparison'][f"{metric_name}_std"] = float(np.std(values))

    # Aggregate prefer GT metrics (for each input type)
    for input_type in ['prefer_orig', 'prefer_aug', 'non_prefer_orig', 'non_prefer_aug', 'average']:
        if input_type in first_result['metrics']['prefer_gt']:
            for metric_name in first_result['metrics']['prefer_gt'][input_type].keys():
                values = [
                    result['metrics']['prefer_gt'][input_type][metric_name]
                    for result in all_results.values()
                    if metric_name in result['metrics']['prefer_gt'].get(input_type, {})
                ]
                if values:
                    aggregated['prefer_gt'][input_type][metric_name] = float(np.mean(values))
                    aggregated['prefer_gt'][input_type][f"{metric_name}_std"] = float(np.std(values))

    # Aggregate non_prefer GT metrics (for each input type)
    for input_type in ['prefer_orig', 'prefer_aug', 'non_prefer_orig', 'non_prefer_aug', 'average']:
        if input_type in first_result['metrics']['non_prefer_gt']:
            for metric_name in first_result['metrics']['non_prefer_gt'][input_type].keys():
                values = [
                    result['metrics']['non_prefer_gt'][input_type][metric_name]
                    for result in all_results.values()
                    if metric_name in result['metrics']['non_prefer_gt'].get(input_type, {})
                ]
                if values:
                    aggregated['non_prefer_gt'][input_type][metric_name] = float(np.mean(values))
                    aggregated['non_prefer_gt'][input_type][f"{metric_name}_std"] = float(np.std(values))

    return aggregated


def save_results(all_results: Dict, avg_metrics: Dict, config: Dict, save_dir: Path, validation_time: float):
    """
    Save validation results to JSON and CSV.

    Args:
        all_results: Dict of per-user results
        avg_metrics: Dict of averaged metrics
        config: Validation config
        save_dir: Directory to save results
        validation_time: Total validation time in seconds
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare full results
    results_dict = {
        'summary': {
            'total_users': len(all_results),
            'total_eval_pairs': sum(r['num_eval_pairs'] for r in all_results.values()),
            'checkpoint': str(config.get('checkpoint', '')),
            'validation_time_seconds': validation_time,
            'validation_time_minutes': validation_time / 60.0,
        },
        'metrics_avg': avg_metrics,
        'per_user_results': all_results,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }

    # Save to JSON
    output_path = save_dir / 'validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")

    # Save summary CSV
    save_summary_csv(avg_metrics, save_dir / 'validation_summary.csv')


def save_summary_csv(avg_metrics: Dict, csv_path: Path):
    """Save summary as CSV for easy viewing."""
    import csv

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['GT Type', 'Input Type', 'Metric', 'Mean', 'Std'])

        # Write GT comparison metrics
        if 'gt_comparison' in avg_metrics:
            for metric, value in sorted(avg_metrics['gt_comparison'].items()):
                if not metric.endswith('_std'):
                    std_key = f"{metric}_std"
                    std_value = avg_metrics['gt_comparison'].get(std_key, 0.0)
                    writer.writerow(['GT Comparison', 'N/A', metric.upper(), f"{value:.4f}", f"{std_value:.4f}"])

        # Write prefer GT metrics (for each input type)
        for input_type in ['prefer_orig', 'prefer_aug', 'non_prefer_orig', 'non_prefer_aug', 'average']:
            if input_type in avg_metrics.get('prefer_gt', {}):
                for metric, value in sorted(avg_metrics['prefer_gt'][input_type].items()):
                    if not metric.endswith('_std'):
                        std_key = f"{metric}_std"
                        std_value = avg_metrics['prefer_gt'][input_type].get(std_key, 0.0)
                        writer.writerow(['Prefer GT', input_type, metric.upper(), f"{value:.4f}", f"{std_value:.4f}"])

        # Write non_prefer GT metrics (for each input type)
        for input_type in ['prefer_orig', 'prefer_aug', 'non_prefer_orig', 'non_prefer_aug', 'average']:
            if input_type in avg_metrics.get('non_prefer_gt', {}):
                for metric, value in sorted(avg_metrics['non_prefer_gt'][input_type].items()):
                    if not metric.endswith('_std'):
                        std_key = f"{metric}_std"
                        std_value = avg_metrics['non_prefer_gt'][input_type].get(std_key, 0.0)
                        writer.writerow(['Non-Prefer GT', input_type, metric.upper(), f"{value:.4f}", f"{std_value:.4f}"])

    print(f"✅ Summary saved to {csv_path}")


def check_validation_users(dataset, config):
    """
    Check how many users have cached UPEs available.

    Args:
        dataset: Base dataset
        config: Validation config

    Returns:
        valid_users: List of all user IDs (UPE requirement checked during validation)
    """
    # Simply return all user IDs - validation will handle UPE cache checking
    valid_users = list(dataset.get_user_ids())

    print(f"✅ Total users: {len(valid_users)}")
    print(f"   (UPE cache will be checked/created during validation)")

    return valid_users


def main():
    parser = argparse.ArgumentParser(description='UPE Validation with Direct Inference')
    parser.add_argument('--config', required=True, help='Path to validation config YAML')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--save_dir', default=None, help='Directory to save results (default: auto from checkpoint)')
    parser.add_argument('--gpu', default='0', help='GPU device ID')
    args = parser.parse_args()

    # Set GPU (no longer using CUDA_VISIBLE_DEVICES)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['checkpoint'] = args.checkpoint

    # Auto-determine save_dir from checkpoint if not provided
    if args.save_dir is None:
        checkpoint_dir = Path(args.checkpoint).parent
        save_dir = checkpoint_dir / 'validation'
    else:
        save_dir = Path(args.save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = save_dir / 'validation_log.txt'

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log("=" * 80)
    log("UPE Direct Inference Validation")
    log("=" * 80)
    log(f"Config: {args.config}")
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Save dir: {save_dir}")
    log("")

    # Auto-configure UPE input_dim if not specified
    log("Auto-configuring UPE processor input_dim...")
    auto_configure_upe_input_dim(config)

    # Validate config
    log("Validating configuration...")
    validate_config(config)
    log("✅ Configuration valid")

    # Load model
    log("\nLoading model...")
    model = models.make(config['model']).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    log(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'psnr' in checkpoint:
        log(f"   Best validation PSNR: {checkpoint['psnr']:.2f} dB")

    # Setup UPE extractor
    log("\nInitializing UPE extractor...")
    upe_config = config['upe_config']
    upe_extractor = UPEExtractor(
        content_model_name=upe_config.get('content_model', 'dino'),
        color_model_name=upe_config.get('color_model', 'clip'),
        num_pairs=upe_config['num_pairs'],
        normalize=upe_config.get('normalize', True),
        device=device
    )
    log("✅ UPE extractor ready")

    # Load dataset first to get dataset_name
    log("\nLoading validation dataset...")
    base_dataset = datasets.make(config['val_dataset']['dataset'])
    log(f"✅ Loaded {len(base_dataset)} samples")

    # Setup UPE cache with dataset-specific subdirectory
    cache_base_dir = upe_config.get('cache_dir', './cache/upe')

    # Check if we should use dataset-specific subdirectories
    use_dataset_subdirs = upe_config.get('use_dataset_subdirs', True)  # Default: True (backward compatible)

    if use_dataset_subdirs:
        dataset_name = base_dataset.dataset_name
        log(f"  Using dataset-specific cache subdirectory: {dataset_name}")
    else:
        dataset_name = None
        log(f"  Using shared cache directory (no dataset subdirectory)")

    upe_cache = UPECache(
        cache_dir=cache_base_dir,
        max_memory_size=100,
        preload_to_gpu=False,
        dataset_name=dataset_name
    )
    log(f"✅ UPE cache initialized at: {upe_cache.cache_dir}")

    # Check valid users
    log("\nChecking user sample counts...")
    valid_user_ids = check_validation_users(base_dataset, config)

    if not valid_user_ids:
        log("❌ No users with sufficient samples found!")
        return

    # Setup metrics calculator
    metric_names = config.get('metrics', ['psnr', 'ssim', 'lpips'])
    metrics_calc = PPSMetrics(device=str(device), enabled_metrics=metric_names)
    log(f"\n✅ Metrics calculator ready: {metric_names}")

    # Setup data normalizer
    data_norm = config.get('data_norm', {
        'inp': {'sub': [0.5], 'div': [0.5]},
        'gt': {'sub': [0.5], 'div': [0.5]}
    })
    normalizer = DataNormalizer(data_norm, device=str(device))

    # Check if random seed is configured
    random_seed = config.get('random_seed', None)
    if random_seed is not None:
        log(f"\n✅ Random seed enabled: {random_seed}")
        log("   Samples will be shuffled deterministically for reproducibility")
    else:
        log("\n⚠️  No random seed specified - using default sample order")
        log("   Add 'random_seed: <int>' to config for reproducible evaluation")

    # Get augmentation modes from config
    aug_modes = config.get('validation', {}).get('aug_modes', ['clean'])
    num_aug_imgs = config.get('validation', {}).get('num_aug_imgs', 1)

    log(f"\n✅ Augmentation modes: {aug_modes}")
    log(f"✅ Number of augmentations per image (for aug mode): {num_aug_imgs}")

    # Store results for all aug_modes
    all_aug_mode_results = {}
    start_time = time.time()

    # Loop over each aug_mode
    for aug_mode in aug_modes:
        log(f"\n{'='*80}")
        log(f"Starting validation with aug_mode='{aug_mode}' for {len(valid_user_ids)} users...")
        log(f"{'='*80}\n")

        all_results = {}

        for user_id in tqdm(valid_user_ids, desc=f'Validating users ({aug_mode})'):
            user_samples = base_dataset.get_user_samples(user_id)

            results = validate_user_with_upe(
                user_id=user_id,
                user_samples=user_samples,
                base_dataset=base_dataset,
                model=model,
                upe_extractor=upe_extractor,
                upe_cache=upe_cache,
                metrics_calc=metrics_calc,
                config=config,
                normalizer=normalizer,
                device=device,
                save_dir=save_dir,
                aug_mode=aug_mode,
                num_aug_imgs=num_aug_imgs if aug_mode == 'aug' else 1,
            )

            if results:
                all_results[user_id] = results

        # Aggregate results for this aug_mode
        log(f"\n{'='*80}")
        log(f"Aggregating results for aug_mode='{aug_mode}'...")
        avg_metrics = aggregate_results(all_results)

        # Store results for this aug_mode
        all_aug_mode_results[aug_mode] = {
            'all_results': all_results,
            'avg_metrics': avg_metrics
        }

        # Save results with mode-specific filename
        mode_save_dir = save_dir / aug_mode if len(aug_modes) > 1 else save_dir
        mode_save_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily store validation time (will update at the end with total time)
        partial_validation_time = time.time() - start_time
        save_results(all_results, avg_metrics, config, mode_save_dir, partial_validation_time)

    validation_time = time.time() - start_time

    # Print summary
    log("\n" + "="*80)
    log("✅ Validation Complete!")
    log("="*80)
    log(f"Validation time: {validation_time/60:.1f} minutes")
    log("")

    # Helper function to format metrics in a single line
    def format_metrics_line(metrics_dict):
        """Format metrics as: METRIC1: X.XX, METRIC2: Y.YY, ..."""
        items = []
        for metric in sorted(metrics_dict.keys()):
            if not metric.endswith('_std'):
                value = metrics_dict[metric]
                items.append(f"{metric.upper()}: {value:.4f}")
        return ", ".join(items)

    # Print summary for each aug_mode
    for aug_mode in aug_modes:
        all_results = all_aug_mode_results[aug_mode]['all_results']
        avg_metrics = all_aug_mode_results[aug_mode]['avg_metrics']

        log(f"\n{'='*80}")
        log(f"Results for aug_mode='{aug_mode}':")
        log(f"{'='*80}")
        log(f"Total users validated: {len(all_results)}")
        log(f"Total evaluation pairs: {sum(r['num_eval_pairs'] for r in all_results.values())}")
        log("")

        log("Average Metrics:")
        log("")

        # GT Comparison
        if avg_metrics.get('gt_comparison'):
            log("  GT Comparison (Prefer vs Non-Prefer):")
            log(f"    {format_metrics_line(avg_metrics['gt_comparison'])}")
            log("")

        # Prefer GT section
        log("  Prefer GT:")
        if 'prefer_orig' in avg_metrics['prefer_gt']:
            log(f"    prefer_orig        - {format_metrics_line(avg_metrics['prefer_gt']['prefer_orig'])}")
        if 'prefer_aug' in avg_metrics['prefer_gt']:
            log(f"    prefer_aug         - {format_metrics_line(avg_metrics['prefer_gt']['prefer_aug'])}")
        if 'non_prefer_orig' in avg_metrics['prefer_gt']:
            log(f"    non_prefer_orig    - {format_metrics_line(avg_metrics['prefer_gt']['non_prefer_orig'])}")
        if 'non_prefer_aug' in avg_metrics['prefer_gt']:
            log(f"    non_prefer_aug     - {format_metrics_line(avg_metrics['prefer_gt']['non_prefer_aug'])}")
        if 'average' in avg_metrics['prefer_gt'] and avg_metrics['prefer_gt']['average']:
            log(f"    [AVERAGE]          - {format_metrics_line(avg_metrics['prefer_gt']['average'])}")
        log("")

        # Non-Prefer GT section
        log("  Non-Prefer GT:")
        if 'prefer_orig' in avg_metrics['non_prefer_gt']:
            log(f"    prefer_orig        - {format_metrics_line(avg_metrics['non_prefer_gt']['prefer_orig'])}")
        if 'prefer_aug' in avg_metrics['non_prefer_gt']:
            log(f"    prefer_aug         - {format_metrics_line(avg_metrics['non_prefer_gt']['prefer_aug'])}")
        if 'non_prefer_orig' in avg_metrics['non_prefer_gt']:
            log(f"    non_prefer_orig    - {format_metrics_line(avg_metrics['non_prefer_gt']['non_prefer_orig'])}")
        if 'non_prefer_aug' in avg_metrics['non_prefer_gt']:
            log(f"    non_prefer_aug     - {format_metrics_line(avg_metrics['non_prefer_gt']['non_prefer_aug'])}")
        if 'average' in avg_metrics['non_prefer_gt'] and avg_metrics['non_prefer_gt']['average']:
            log(f"    [AVERAGE]          - {format_metrics_line(avg_metrics['non_prefer_gt']['average'])}")


if __name__ == '__main__':
    main()
