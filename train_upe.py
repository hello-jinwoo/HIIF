"""
Training Script for UPE (User Preference Embedding) System

Key differences from train_pps.py:
- Single global decoder (no user-specific decoders)
- Mixed-user batching (no PPSUserBatchSampler)
- Direct UPE injection from cache
- Single optimizer for encoder + decoder
- Simplified checkpoint management (2-3 files vs 101 files)
- No curriculum learning or decoder switching
"""

import argparse
import csv
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

import datasets
import models
import utils
from pps_utils.loss_scheduler import LossWeightScheduler, compute_dual_gt_loss
from pps_utils.data_processing import DataNormalizer, preprocess_pps_batch_with_upe
from pps_utils.upe_cache import UPECache
from pps_utils.collate import collate_with_upe
from datasets.pps_upe_wrapper import PPSUPEWrapper

torch.backends.cudnn.benchmark = True


def validate_config(config):
    """Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required keys are missing or values are invalid
    """
    # Check required top-level keys
    required_keys = ['model', 'train_dataset', 'optimizer', 'epoch_max', 'upe_config']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    # Validate UPE config
    upe_config = config['upe_config']
    if upe_config['content_model'] == upe_config['color_model']:
        raise ValueError("content_model and color_model must be different")

    # Validate train_dataset parameters
    td = config['train_dataset']
    if 'batch_size' not in td:
        raise ValueError("Missing 'train_dataset.batch_size'")
    if td['batch_size'] <= 0:
        raise ValueError(f"batch_size must be > 0, got {td['batch_size']}")

    # Validate loss schedule parameters
    if 'loss_schedule' in config:
        ws = config['loss_schedule']
        if 'w_p_start' in ws:
            if not (0 <= ws['w_p_start'] <= 2):
                raise ValueError(f"w_p_start must be in [0,2], got {ws['w_p_start']}")
        if 'w_p_end' in ws:
            if not (0 <= ws['w_p_end'] <= 2):
                raise ValueError(f"w_p_end must be in [0,2], got {ws['w_p_end']}")

    # Validate epoch parameters
    if config['epoch_max'] <= 0:
        raise ValueError(f"epoch_max must be > 0, got {config['epoch_max']}")


def make_upe_data_loader(spec, upe_cache, upe_config, tag='', base_dataset=None):
    """
    Create data loader with UPE support.

    This function creates a data pipeline that:
    1. Loads base preference pair dataset
    2. Splits samples (16 for UPE, rest for training)
    3. Applies color augmentation wrapper
    4. Adds UPE wrapper to load cached UPEs
    5. Creates DataLoader with custom collate function

    Args:
        spec: Dataset specification dict with keys:
            - dataset: Base dataset spec
            - wrapper: Augmentation wrapper spec
            - batch_size: Batch size for training
        upe_cache: UPECache instance for loading cached UPEs
        upe_config: UPE configuration dict
        tag: Tag for logging ('train' or 'val')
        base_dataset: Optional pre-created base dataset (for sharing train/val)

    Returns:
        loader: DataLoader with UPE support
        base_dataset: Base dataset instance
        split_info: Sample split information
    """
    if spec is None:
        return None, None, None

    # Create base dataset if not provided
    if base_dataset is None:
        base_dataset = datasets.make(spec['dataset'])

    # Split samples for UPE (16 pairs for UPE, rest for training)
    split_info = base_dataset.split_samples_for_upe(
        num_upe_pairs=upe_config['num_pairs'],
        seed=42  # Deterministic splitting
    )

    # Create training subset (excludes UPE samples)
    train_dataset = base_dataset.create_train_subset(split_info)

    log(f'{tag} dataset: base_size={len(base_dataset)}, train_size={len(train_dataset)}')
    log(f'  UPE samples: {upe_config["num_pairs"]} pairs per user')
    log(f'  Training samples: {len(train_dataset)} total')

    # Apply color augmentation wrapper
    if 'wrapper' in spec:
        train_dataset = datasets.make(spec['wrapper'], args={'dataset': train_dataset})
        log(f'{tag} dataset after wrapper: size={len(train_dataset)}')

    # Add UPE wrapper
    upe_dataset = PPSUPEWrapper(
        dataset=train_dataset,  # Note: parameter name is 'dataset', not 'base_dataset'
        upe_cache=upe_cache,
        upe_config=upe_config,
        device='cpu',  # Keep UPEs on CPU to avoid multiprocessing issues
        cache_policy='fail_fast'  # Ensure all UPEs are extracted
    )

    # Preload all UPEs to CPU
    log(f'Preloading UPEs...')
    upe_dataset.preload_all_upes(show_progress=True)
    stats = upe_dataset.get_cache_stats()
    log(f'  Loaded {stats["loaded_users"]} UPEs')

    # Sample output for debugging
    sample = upe_dataset[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            log(f'  {k}: shape={tuple(v.shape)}')
        else:
            log(f'  {k}: {type(v).__name__}')

    # Create DataLoader with mixed-user batching
    # UPEs are kept on CPU and moved to GPU in training loop
    # This allows using num_workers > 0 for faster data loading if needed
    loader = DataLoader(
        upe_dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),  # Shuffle for train, not for val
        num_workers=4,  # Can use multiple workers since UPEs are on CPU
        pin_memory=True,  # Enable pin_memory for faster CPU->GPU transfer
        collate_fn=collate_with_upe,
        drop_last=(tag == 'train')  # Drop last incomplete batch for training
    )

    return loader, base_dataset, split_info


def make_data_loaders(upe_cache, upe_config):
    """Create train and validation data loaders with UPE support"""
    train_loader, train_base_dataset, train_split = make_upe_data_loader(
        config.get('train_dataset'),
        upe_cache,
        upe_config,
        tag='train'
    )
    val_loader, _, val_split = make_upe_data_loader(
        config.get('val_dataset'),
        upe_cache,
        upe_config,
        tag='val',
        base_dataset=train_base_dataset  # Share base dataset
    )

    return train_loader, val_loader, train_base_dataset, train_split, val_split


def prepare_training(device):
    """
    Prepare model and optimizer.

    Args:
        device: torch.device to use for model

    Returns:
        model: HIIF_Global_UPE model
        optimizer: Single optimizer for encoder + decoder
        epoch_start: Starting epoch number
        best_val_psnr: Best validation PSNR so far
    """
    # Check for resume
    resume_path = config.get('resume', os.path.join(save_path, 'checkpoint_last.pth'))

    # Create model
    model = models.make(config['model']).to(device)

    log(f'Model: {config["model"]["name"]}')
    log(f'Model: #params={utils.compute_num_params(model, text=True)}')

    # Create single optimizer for encoder + decoder
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    log(f'Optimizer: {optimizer}')

    # Try to resume
    if os.path.exists(resume_path):
        log(f'Resuming from {resume_path}')

        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        epoch_start = checkpoint['epoch'] + 1
        best_val_psnr = checkpoint.get('best_val_psnr', 0.0)

        log(f'Resumed from epoch {checkpoint["epoch"]}')
        log(f'Best val PSNR: {best_val_psnr:.2f} dB')
    else:
        log('Starting training from scratch')
        epoch_start = 1
        best_val_psnr = 0.0

    return model, optimizer, epoch_start, best_val_psnr


def train(train_loader, model, optimizer, loss_scheduler, epoch, device):
    """
    Train for one epoch (simplified - no decoder switching).

    Args:
        train_loader: DataLoader with UPE support
        model: HIIF_Global_UPE model
        optimizer: Single optimizer for encoder + decoder
        loss_scheduler: LossWeightScheduler for w_p
        epoch: Current epoch number
        device: torch.device to use

    Returns:
        Average training loss for the epoch
    """
    model.train()
    train_loss = utils.Averager()

    # Data normalization
    data_norm = config['data_norm']
    normalizer = DataNormalizer(data_norm, device=device)

    pbar = tqdm(train_loader, leave=False, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # Move to GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Preprocess batch with UPE and normalize
        inp, coord, cell, gt_prefer, gt_non_prefer, upe, user_id = \
            preprocess_pps_batch_with_upe(batch)

        inp = normalizer.normalize_input(inp)
        gt_prefer = normalizer.normalize_gt(gt_prefer)
        gt_non_prefer = normalizer.normalize_gt(gt_non_prefer)

        # Forward pass (global decoder with UPE)
        pred = model(inp, coord, cell, upe)

        # Dual GT loss with w_p scheduling
        w_p = loss_scheduler.get_weight()
        loss = compute_dual_gt_loss(pred, gt_prefer, gt_non_prefer, w_p)

        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            log(f'WARNING: NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. Skipping batch.')
            continue

        # Backward (single optimizer)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        loss_scheduler.step()

        train_loss.add(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'w_p': f'{w_p:.3f}'})

        # Log to tensorboard
        if writer is not None:
            iteration = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), iteration)
            writer.add_scalar('train/w_p', w_p, iteration)

    return train_loss.item()


def validate(val_loader, model, epoch, device):
    """
    Validate with direct inference (no test-time training).

    Args:
        val_loader: Validation DataLoader with UPE support
        model: HIIF_Global_UPE model
        epoch: Current epoch number
        device: torch.device to use

    Returns:
        Dict with validation metrics
    """
    log(f'Validating epoch {epoch}...')

    model.eval()

    # Data normalization
    data_norm = config['data_norm']
    normalizer = DataNormalizer(data_norm, device=device)

    psnr_prefer_meter = utils.Averager()
    psnr_non_prefer_meter = utils.Averager()

    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc='Validation'):
            # Move to GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Preprocess batch with UPE
            inp, coord, cell, gt_prefer, gt_non_prefer, upe, user_id = \
                preprocess_pps_batch_with_upe(batch)

            inp = normalizer.normalize_input(inp)
            gt_prefer = normalizer.normalize_gt(gt_prefer)
            gt_non_prefer = normalizer.normalize_gt(gt_non_prefer)

            # Forward pass (direct inference)
            pred = model(inp, coord, cell, upe)

            # Denormalize for metrics
            pred_denorm = normalizer.denormalize_gt(pred)
            gt_prefer_denorm = normalizer.denormalize_gt(gt_prefer)
            gt_non_prefer_denorm = normalizer.denormalize_gt(gt_non_prefer)

            # Compute PSNR
            psnr_prefer = utils.calc_psnr(pred_denorm, gt_prefer_denorm)
            psnr_non_prefer = utils.calc_psnr(pred_denorm, gt_non_prefer_denorm)

            psnr_prefer_meter.add(psnr_prefer.item())
            psnr_non_prefer_meter.add(psnr_non_prefer.item())

    val_metrics = {
        'psnr_prefer': psnr_prefer_meter.item(),
        'psnr_non_prefer': psnr_non_prefer_meter.item(),
    }

    log(f'Validation results:')
    log(f'  PSNR (prefer): {val_metrics["psnr_prefer"]:.2f} dB')
    log(f'  PSNR (non-prefer): {val_metrics["psnr_non_prefer"]:.2f} dB')
    log(f'  PSNR gap: {val_metrics["psnr_prefer"] - val_metrics["psnr_non_prefer"]:.2f} dB')

    return val_metrics


def main(config_, save_path_, device):
    """Main training function"""
    global config, log, writer, save_path
    config = config_
    save_path = save_path_

    log, writer = utils.set_save_path(save_path, remove=False)

    # Create CSV file for training metrics logging
    csv_path = os.path.join(save_path, 'training_metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_psnr', 'w_p', 'elapsed_time'])
    csv_file.flush()
    log(f'Training metrics will be logged to: {csv_path}')

    # Validate configuration
    log('Validating configuration...')
    validate_config(config)
    log('Configuration validated successfully')

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # Initialize UPE cache
    upe_config = config['upe_config']
    upe_cache_dir = upe_config.get('cache_dir', './cache/upe')
    upe_cache = UPECache(cache_dir=upe_cache_dir, max_memory_size=100)

    log(f'UPE Cache initialized at: {upe_cache_dir}')
    upe_cache.print_stats()

    # Create data loaders with UPE support
    train_loader, val_loader, train_base_dataset, train_split, val_split = \
        make_data_loaders(upe_cache, upe_config)

    # Prepare model and optimizer
    model, optimizer, epoch_start, best_val_psnr = prepare_training(device)

    # Create loss weight scheduler
    loss_config = config.get('loss_schedule', {})
    total_iterations = config['epoch_max'] * len(train_loader)
    loss_scheduler = LossWeightScheduler(
        schedule_type=loss_config.get('schedule_type', 'exponential'),
        w_p_start=loss_config.get('w_p_start', 0.5),
        w_p_end=loss_config.get('w_p_end', 1.2),
        total_iterations=loss_config.get('total_iterations', total_iterations)
    )

    # Restore loss scheduler state if resuming
    if epoch_start > 1 and os.path.exists(os.path.join(save_path, 'checkpoint_last.pth')):
        checkpoint = torch.load(os.path.join(save_path, 'checkpoint_last.pth'))
        if 'loss_scheduler' in checkpoint:
            loss_scheduler.load_state_dict(checkpoint['loss_scheduler'])
            log(f'Restored loss scheduler: iteration={loss_scheduler.current_iteration}, w_p={loss_scheduler.get_weight():.3f}')

    # Data normalization
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0.5], 'div': [0.5]},
            'gt': {'sub': [0.5], 'div': [0.5]}
        }

    # Training loop
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val', 10)
    epoch_save = config.get('epoch_save', 10)

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = [f'Epoch {epoch}/{epoch_max}']

        # Train
        train_loss = train(train_loader, model, optimizer, loss_scheduler, epoch, device)
        log_info.append(f'train_loss={train_loss:.4f}')
        log_info.append(f'w_p={loss_scheduler.get_weight():.3f}')

        # Validation
        val_psnr = 0.0
        if val_loader is not None and (epoch % epoch_val == 0 or epoch == epoch_max):
            val_metrics = validate(val_loader, model, epoch, device)
            val_psnr = val_metrics['psnr_prefer']
            log_info.append(f'val_psnr={val_psnr:.2f}')

            # Save best model
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_psnr': best_val_psnr,
                    'loss_scheduler': loss_scheduler.state_dict(),
                    'config': config,
                }, os.path.join(save_path, 'checkpoint_best.pth'))
                log(f'✓ New best model saved! PSNR: {best_val_psnr:.2f} dB')

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_psnr': best_val_psnr,
            'loss_scheduler': loss_scheduler.state_dict(),
            'config': config,
        }, os.path.join(save_path, 'checkpoint_last.pth'))

        # Periodic checkpoint save
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth'))
            log(f'Saved checkpoint at epoch {epoch}')

        # Time estimation
        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append(f'{t_epoch} {t_elapsed}/{t_all}')

        log(', '.join(log_info))

        if writer is not None:
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/w_p', loss_scheduler.get_weight(), epoch)
            if val_psnr > 0:
                writer.add_scalar('epoch/val_psnr', val_psnr, epoch)
            writer.flush()

        # Log to CSV file
        csv_writer.writerow([epoch, train_loss, val_psnr, loss_scheduler.get_weight(), t_epoch])
        csv_file.flush()

    # Final save
    torch.save({
        'epoch': epoch_max,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_psnr': best_val_psnr,
        'loss_scheduler': loss_scheduler.state_dict(),
        'config': config,
    }, os.path.join(save_path, 'checkpoint_last.pth'))

    log('Training complete!')
    log(f'Checkpoints saved to: {save_path}/')
    log(f'  - checkpoint_last.pth (latest)')
    log(f'  - checkpoint_best.pth (best PSNR: {best_val_psnr:.2f} dB)')
    log(f'Training metrics saved to: {csv_path}')

    # Close CSV file
    csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--name', default='_train_upe', help='Save directory name')
    parser.add_argument('--tag', default=None, help='Additional tag for save directory')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    args = parser.parse_args()

    # Create device from GPU ID (no longer using CUDA_VISIBLE_DEVICES)
    device = torch.device(f'cuda:{args.gpu}')

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('Config loaded.')

    # Create save directory
    save_name = args.name
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, device)
