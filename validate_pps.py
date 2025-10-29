"""
Validation Script for PPS (Personalized Photographic Style) Learning

Performs few-shot learning validation:
1. Load trained encoder from checkpoint
2. For each validation user:
   - Sample N training images
   - Create and train new decoder from scratch
   - Evaluate at multiple checkpoints [50, 100, 500, 1000]
   - Test with 3 augmentation types: clean, same aug, different aug
3. Compute all 6 metrics: PSNR, SSIM, LPIPS, Delta E LAB, CIEDE2000, NIQE
4. Save results as JSON + generated images
"""

import argparse
import os
import json
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from PIL import Image

import datasets
import models
import utils
from metrics import PPSMetrics

# Import validation core functions
from pps_utils.validation_core import (
    validate_user,
    save_image_tensor,
    denormalize,
    create_dataloader_with_aug,
    create_fullimage_dataloader,
    train_decoder_one_step,
    evaluate_decoder
)


def validate_config(config):
    """Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required keys are missing or values are invalid
    """
    # Check required top-level keys
    required_keys = ['model', 'val_dataset', 'decoder_training', 'augmentation']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    # Validate decoder_training parameters
    dt = config['decoder_training']
    if 'samples_per_user' not in dt:
        raise ValueError("Missing 'decoder_training.samples_per_user'")
    if dt['samples_per_user'] <= 0:
        raise ValueError(f"samples_per_user must be > 0, got {dt['samples_per_user']}")

    if 'batch_size' not in dt:
        raise ValueError("Missing 'decoder_training.batch_size'")
    if dt['batch_size'] <= 0:
        raise ValueError(f"batch_size must be > 0, got {dt['batch_size']}")

    # Validate augmentation config
    if 'train_aug' not in config['augmentation']:
        raise ValueError("Missing 'augmentation.train_aug'")

    # Validate checkpoint iterations
    if 'checkpoint_iterations' in dt:
        for ckpt_iter in dt['checkpoint_iterations']:
            if ckpt_iter <= 0:
                raise ValueError(f"checkpoint_iterations must be > 0, got {ckpt_iter}")


def main(config_, save_path_):
    """Main validation function"""
    global config, save_path
    config = config_
    save_path = save_path_

    # Setup logging
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'log.txt')

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log(f"{'='*80}")
    log(f"PPS Few-Shot Learning Validation")
    log(f"{'='*80}")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Validate config
    log("\nValidating configuration...")
    validate_config(config)
    log("Configuration validated successfully")

    # Save config
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # Load validation dataset
    log("\nLoading validation dataset...")
    base_dataset = datasets.make(config['val_dataset']['dataset'])

    user_ids = base_dataset.get_user_ids()
    log(f"Total validation users: {len(user_ids)}")
    log(f"User IDs: {user_ids}")

    # Select subset if specified
    num_val_users = config.get('num_val_users')
    if num_val_users is not None and num_val_users < len(user_ids):
        user_ids = user_ids[:num_val_users]
        log(f"Using subset: {num_val_users} users")

    # Load model with encoder from checkpoint
    log("\nLoading model...")
    model_config = config['model'].copy()
    model_config['args']['user_ids'] = user_ids  # Validation user IDs
    model = models.make(model_config).cuda()

    # Load encoder checkpoint
    encoder_ckpt_path = config['encoder_checkpoint']
    log(f"Loading encoder from: {encoder_ckpt_path}")

    if not os.path.exists(encoder_ckpt_path):
        log(f"ERROR: Encoder checkpoint not found: {encoder_ckpt_path}")
        return

    ckpt = torch.load(encoder_ckpt_path, map_location='cuda')

    # Load encoder and freq parameters
    # Checkpoint saved by CheckpointManager has keys 'encoder' and 'freq' directly
    model.encoder.load_state_dict(ckpt['encoder'])
    model.freq.load_state_dict(ckpt['freq'])

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.freq.parameters():
        param.requires_grad = False

    model.eval()  # Encoder always in eval mode
    log(f"Encoder loaded and frozen")

    # Initialize metrics calculator
    log("\nInitializing metrics calculator...")
    enabled_metrics = config.get('metrics', None)  # Get from config, or None for all
    metrics_calc = PPSMetrics(device='cuda', enabled_metrics=enabled_metrics)
    log(f"Available metrics: {metrics_calc.available_metrics}")
    if enabled_metrics:
        log(f"Enabled metrics (from config): {enabled_metrics}")
        # Warn if some enabled metrics are not available
        unavailable = [m for m in enabled_metrics if m not in metrics_calc.available_metrics]
        if unavailable:
            log(f"Warning: These metrics are enabled but not available (check dependencies): {unavailable}")
    else:
        log(f"Using all available metrics")

    # Validate each user
    all_results = {
        'config': {
            'encoder_checkpoint': encoder_ckpt_path,
            'num_users': len(user_ids),
            'samples_per_user': config['decoder_training']['samples_per_user'],
            'eval_checkpoints': config['decoder_training']['eval_checkpoints'],
        },
        'users': {}
    }

    for user_id in user_ids:
        user_samples = base_dataset.get_user_samples(user_id)

        user_results = validate_user(
            user_id=user_id,
            user_samples=user_samples,
            base_dataset=base_dataset,
            model=model,
            config=config,
            metrics_calc=metrics_calc,
            save_dir=save_path,
            log_fn=log
        )

        all_results['users'][user_id] = user_results

    # Save results
    results_path = os.path.join(save_path, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    log(f"\n{'='*80}")
    log(f"Validation Complete!")
    log(f"{'='*80}")
    log(f"Results saved to: {results_path}")
    log(f"Images saved to: {save_path}/images/")
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to validation config file')
    parser.add_argument('--name', default='_validate_pps', help='Save directory name')
    parser.add_argument('--tag', default=None, help='Additional tag for save directory')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('Validation config loaded.')

    # Create save directory
    save_name = args.name
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
