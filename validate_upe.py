"""
Validation Script for UPE (User Preference Embedding) System

Performs direct inference validation (NO test-time training):
1. Load trained model from checkpoint
2. For each validation user:
   - Extract UPE from first N preference pairs (default: 16)
   - Evaluate on remaining samples with direct inference
3. Compute all metrics: PSNR, SSIM, LPIPS, Delta E variants, etc.
4. Save results as JSON + optional visualizations
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

import datasets
import models
import utils
from metrics import PPSMetrics
from models.upe_extractor import UPEExtractor
from pps_utils.upe_cache import UPECache
from pps_utils.data_processing import DataNormalizer, preprocess_pps_batch_with_upe


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

    # Check minimum evaluation samples
    min_eval = config.get('min_eval_samples', 8)
    if min_eval <= 0:
        raise ValueError(f"min_eval_samples must be > 0, got {min_eval}")

    # Validate metrics config
    if 'metrics' in config:
        if not isinstance(config['metrics'], list):
            raise ValueError("'metrics' must be a list of metric names")


def create_eval_dataset(base_dataset, eval_sample_indices, config):
    """
    Create evaluation dataset from specific sample indices.

    Args:
        base_dataset: Original PPSPreferencePairDataset
        eval_sample_indices: List of sample indices to include
        config: Validation config dict

    Returns:
        eval_dataset: Dataset ready for evaluation (with transforms applied)
    """
    from torch.utils.data import Subset

    # Create subset with only eval samples
    eval_subset = Subset(base_dataset, eval_sample_indices)

    # Apply color augmentation wrapper (augment=False for eval)
    if 'wrapper' in config['val_dataset']:
        wrapper_config = config['val_dataset']['wrapper']

        if wrapper_config['name'] == 'pps-color-augmented':
            # Import wrapper class directly
            from datasets.pps_wrapper import PPSColorAugmentedWrapper

            # Create wrapper with no augmentation
            eval_dataset = PPSColorAugmentedWrapper(
                dataset=eval_subset,
                crop_size=wrapper_config['args'].get('crop_size', 128),
                augment=False,  # No augmentation during eval
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

    Returns:
        Dict with metrics and metadata, or None if user has insufficient samples
    """
    num_upe = config['upe_config']['num_pairs']
    min_eval = config.get('min_eval_samples', 8)
    min_total = num_upe + min_eval

    # Check minimum samples
    if len(user_samples) < min_total:
        warnings.warn(
            f"User {user_id} has only {len(user_samples)} samples, "
            f"need {min_total} ({num_upe} UPE + {min_eval} eval). SKIPPING."
        )
        return None

    # Split samples: first N for UPE, rest for evaluation
    upe_sample_indices = user_samples[:num_upe]
    eval_sample_indices = user_samples[num_upe:]

    if len(eval_sample_indices) < min_eval:
        warnings.warn(
            f"User {user_id} has only {len(eval_sample_indices)} eval samples "
            f"after UPE split, need {min_eval}. SKIPPING."
        )
        return None

    # Extract or load UPE (raw UPE: 16, 1280)
    upe_raw = upe_cache.load(user_id, config['upe_config'])
    upe_cached = True

    if upe_raw is None:
        # Extract UPE from preference pairs
        upe_samples = [base_dataset[i] for i in upe_sample_indices]
        upe_raw = upe_extractor(upe_samples)
        upe_cache.save(user_id, upe_raw, config['upe_config'])
        upe_cached = False

    # Move raw UPE to GPU (keep as raw for now)
    upe_raw = upe_raw.to(device)  # (16, 1280)

    # Create eval dataset
    eval_dataset = create_eval_dataset(base_dataset, eval_sample_indices, config)

    # Create eval loader (batch_size=1 for simplicity)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Evaluate with direct inference
    psnr_prefer_list = []
    psnr_non_prefer_list = []
    all_metrics = {name: [] for name in config.get('metrics', ['psnr', 'ssim', 'lpips'])}

    model.eval()
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

            # Compute PSNR
            psnr_prefer = utils.calc_psnr(pred_denorm, gt_prefer_denorm).item()
            psnr_non_prefer = utils.calc_psnr(pred_denorm, gt_non_prefer_denorm).item()

            psnr_prefer_list.append(psnr_prefer)
            psnr_non_prefer_list.append(psnr_non_prefer)

            # Compute other metrics
            metric_dict = metrics_calc.compute_all(pred_denorm, gt_prefer_denorm)
            for name, value in metric_dict.items():
                if name in all_metrics:
                    all_metrics[name].append(value)

    # Aggregate metrics
    results = {
        'num_eval_samples': len(eval_sample_indices),
        'metrics': {},
        'upe_info': {
            'num_upe_pairs': num_upe,
            'upe_cached': upe_cached,
        }
    }

    # Average metrics (convert to Python float for JSON serialization)
    results['metrics']['psnr_prefer'] = float(np.mean(psnr_prefer_list))
    results['metrics']['psnr_non_prefer'] = float(np.mean(psnr_non_prefer_list))
    results['metrics']['psnr_gap'] = results['metrics']['psnr_prefer'] - results['metrics']['psnr_non_prefer']

    for name, values in all_metrics.items():
        if values:
            results['metrics'][name] = float(np.mean(values))

    return results


def aggregate_results(all_results: Dict[str, Dict]) -> Dict:
    """
    Aggregate per-user results into average metrics.

    Args:
        all_results: Dict of per-user results

    Returns:
        avg_metrics: Dict with mean and std for each metric
    """
    if not all_results:
        return {}

    # Collect all metric names
    metric_names = list(next(iter(all_results.values()))['metrics'].keys())

    aggregated = {}
    for metric_name in metric_names:
        values = [
            result['metrics'][metric_name]
            for result in all_results.values()
            if metric_name in result['metrics']
        ]

        if values:
            aggregated[metric_name] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))

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
            'total_eval_samples': sum(r['num_eval_samples'] for r in all_results.values()),
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
        writer.writerow(['Metric', 'Mean', 'Std'])

        # Write averages
        for metric, value in sorted(avg_metrics.items()):
            if not metric.endswith('_std'):
                std_key = f"{metric}_std"
                std_value = avg_metrics.get(std_key, 0.0)
                writer.writerow([metric.upper(), f"{value:.4f}", f"{std_value:.4f}"])

    print(f"✅ Summary saved to {csv_path}")


def check_validation_users(dataset, config):
    """
    Check how many users meet the minimum sample requirement.

    Args:
        dataset: Base dataset
        config: Validation config

    Returns:
        valid_users: List of user IDs with sufficient samples
    """
    num_upe = config['upe_config']['num_pairs']
    min_eval = config.get('min_eval_samples', 8)
    min_total = num_upe + min_eval

    valid_users = []
    skipped_users = []

    for user_id in dataset.get_user_ids():
        user_samples = dataset.get_user_samples(user_id)

        if len(user_samples) >= min_total:
            valid_users.append(user_id)
        else:
            skipped_users.append((user_id, len(user_samples)))

    print(f"✅ Valid users: {len(valid_users)}/{len(valid_users) + len(skipped_users)}")

    if skipped_users:
        print(f"⚠️  Skipped users ({len(skipped_users)}):")
        for user_id, count in skipped_users[:5]:  # Show first 5
            print(f"   - {user_id}: {count} samples (need {min_total})")
        if len(skipped_users) > 5:
            print(f"   ... and {len(skipped_users) - 5} more")

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

    # Setup UPE cache
    cache_dir = upe_config.get('cache_dir', './cache/upe')
    upe_cache = UPECache(cache_dir=cache_dir, max_memory_size=100, preload_to_gpu=False)
    log(f"✅ UPE cache initialized at: {cache_dir}")

    # Load dataset
    log("\nLoading validation dataset...")
    base_dataset = datasets.make(config['val_dataset']['dataset'])
    log(f"✅ Loaded {len(base_dataset)} samples")

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

    # Validate each user
    log(f"\n{'='*80}")
    log(f"Starting validation for {len(valid_user_ids)} users...")
    log(f"{'='*80}\n")

    start_time = time.time()
    all_results = {}

    for user_id in tqdm(valid_user_ids, desc='Validating users'):
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
        )

        if results:
            all_results[user_id] = results

    validation_time = time.time() - start_time

    # Aggregate results
    log("\n" + "="*80)
    log("Aggregating results...")
    avg_metrics = aggregate_results(all_results)

    # Save results
    save_results(all_results, avg_metrics, config, save_dir, validation_time)

    # Print summary
    log("\n" + "="*80)
    log("✅ Validation Complete!")
    log("="*80)
    log(f"Total users validated: {len(all_results)}")
    log(f"Total evaluation samples: {sum(r['num_eval_samples'] for r in all_results.values())}")
    log(f"Validation time: {validation_time/60:.1f} minutes")
    log("")
    log("Average Metrics:")
    for metric, value in sorted(avg_metrics.items()):
        if not metric.endswith('_std'):
            std_key = f"{metric}_std"
            std_value = avg_metrics.get(std_key, 0.0)
            log(f"  {metric.upper()}: {value:.4f} ± {std_value:.4f}")


if __name__ == '__main__':
    main()
