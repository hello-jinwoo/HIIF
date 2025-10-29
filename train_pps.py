"""
Training Script for PPS (Personalized Photographic Style) Learning

Key differences from train.py:
- Uses PPSUserBatchSampler for same-user batching
- Reshapes inputs from (B, 4, 3, H, W) to (B*4, 3, H, W)
- Computes dual GT loss with w_p scheduling
- Manages 101 checkpoints (1 encoder + 100 user decoders)
"""

import argparse
import csv
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils
from datasets.pps_sampler import PPSUserBatchSampler
from pps_utils.checkpoint_manager import CheckpointManager
from pps_utils.loss_scheduler import LossWeightScheduler, compute_dual_gt_loss
from pps_utils.data_processing import DataNormalizer, preprocess_pps_batch

torch.backends.cudnn.benchmark = True


def validate_config(config):
    """Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required keys are missing or values are invalid
    """
    # Check required top-level keys
    required_keys = ['model', 'train_dataset', 'optimizer', 'epoch_max']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    # Validate train_dataset parameters
    td = config['train_dataset']
    if 'batch_size' not in td:
        raise ValueError("Missing 'train_dataset.batch_size'")
    if td['batch_size'] <= 0:
        raise ValueError(f"batch_size must be > 0, got {td['batch_size']}")

    if 'wrapper' in td and 'args' in td['wrapper']:
        args = td['wrapper']['args']
        if 'crop_size' in args and args['crop_size'] <= 0:
            raise ValueError(f"crop_size must be > 0, got {args['crop_size']}")

    # Validate loss schedule parameters
    if 'loss' in config and 'weight_schedule' in config['loss']:
        ws = config['loss']['weight_schedule']
        if 'w_p_start' in ws:
            if not (0 <= ws['w_p_start'] <= 1):
                raise ValueError(f"w_p_start must be in [0,1], got {ws['w_p_start']}")
        if 'w_p_end' in ws:
            if not (0 <= ws['w_p_end'] <= 1):
                raise ValueError(f"w_p_end must be in [0,1], got {ws['w_p_end']}")

    # Validate epoch parameters
    if config['epoch_max'] <= 0:
        raise ValueError(f"epoch_max must be > 0, got {config['epoch_max']}")


def make_data_loader(spec, tag='', base_dataset=None):
    """Create data loader with PPSUserBatchSampler"""
    if spec is None:
        return None, None

    # Create base dataset if not provided
    if base_dataset is None:
        base_dataset = datasets.make(spec['dataset'])

    # Wrap with augmentation
    dataset = datasets.make(spec['wrapper'], args={'dataset': base_dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if isinstance(v, torch.Tensor):
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    # Create batch sampler for user-specific batching
    if tag == 'train':
        batch_sampler = PPSUserBatchSampler(
            dataset=base_dataset,
            batch_size=spec['batch_size'],
            iterations_per_user=spec.get('sampler', {}).get('iterations_per_user', 16),
            shuffle_users=True,
            shuffle_samples=True,
            drop_last=True
        )
        loader = DataLoader(dataset, batch_sampler=batch_sampler,
                            num_workers=8, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=False, num_workers=8, pin_memory=True)

    return loader, base_dataset


def make_data_loaders():
    """Create train and validation data loaders"""
    train_loader, train_base_dataset = make_data_loader(
        config.get('train_dataset'), tag='train'
    )
    val_loader, _ = make_data_loader(
        config.get('val_dataset'), tag='val', base_dataset=train_base_dataset
    )

    return train_loader, val_loader, train_base_dataset


def prepare_training(user_ids):
    """Prepare model and encoder optimizer

    Note: Decoder optimizer is created dynamically in training loop
    """

    # Check for resume
    resume_path = config.get('resume', os.path.join(save_path, 'encoder', 'epoch-last.pth'))
    ckpt_manager = CheckpointManager(save_path)

    # Create model with user IDs (decoders not created in memory)
    model_config = config['model'].copy()
    model_config['args']['user_ids'] = user_ids
    model = models.make(model_config).cuda()

    log(f'Model created with {len(user_ids)} available users')
    log(f'Model: #params={utils.compute_num_params(model, text=True)}')

    # Create encoder optimizer (persistent)
    encoder_params = list(model.encoder.parameters()) + list(model.freq.parameters())
    encoder_optimizer = utils.make_optimizer(encoder_params, config['optimizer'])
    log(f'Encoder optimizer: {encoder_optimizer}')

    # Try to resume
    if os.path.exists(resume_path):
        log(f'Resuming from {resume_path}')

        resume_info = ckpt_manager.load_encoder(model, encoder_optimizer, name='epoch-last')

        if resume_info is None:
            log('No checkpoint found, starting from scratch')
            epoch_start = 1
        else:
            epoch_start = resume_info['epoch'] + 1
            log(f'Resumed from epoch {resume_info["epoch"]}')
    else:
        log('Starting training from scratch')
        epoch_start = 1
        resume_info = None

    return model, encoder_optimizer, epoch_start, resume_info, ckpt_manager


def train(train_loader, model, encoder_optimizer, loss_scheduler, epoch, ckpt_manager):
    """Train for one epoch with dynamic decoder loading

    Args:
        train_loader: DataLoader with PPSUserBatchSampler
        model: HIIF_PPS model
        encoder_optimizer: Optimizer for encoder parameters
        loss_scheduler: LossWeightScheduler for w_p
        epoch: Current epoch number
        ckpt_manager: CheckpointManager for decoder save/load

    Returns:
        Average training loss for the epoch
    """
    model.train()
    train_loss = utils.Averager()

    # Data normalization
    data_norm = config['data_norm']
    normalizer = DataNormalizer(data_norm, device='cuda')

    pbar = tqdm(train_loader, leave=False, desc=f'Epoch {epoch}')

    # Decoder lifecycle tracking
    previous_user_id = None
    decoder_optimizer = None

    for batch_idx, batch in enumerate(pbar):
        # Move to GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        # Preprocess batch and normalize
        inp, coord, cell, gt_prefer, gt_non_prefer, user_indices = preprocess_pps_batch(batch)
        inp = normalizer.normalize_input(inp)
        gt_prefer = normalizer.normalize_gt(gt_prefer)
        gt_non_prefer = normalizer.normalize_gt(gt_non_prefer)

        # User indices: all same in batch due to PPSUserBatchSampler
        current_user_id = user_indices[0]

        # Handle user switching: offload previous, load current
        if previous_user_id != current_user_id:
            # Offload previous decoder (save to disk + free memory)
            if previous_user_id is not None:
                model.offload_user_decoder(ckpt_manager)
                decoder_optimizer = None
                log(f'[Epoch {epoch}] Offloaded decoder for {previous_user_id}')

            # Load current decoder (from disk or create new)
            model.load_user_decoder(current_user_id, ckpt_manager)

            # Create optimizer for current decoder
            decoder_optimizer = utils.make_optimizer(
                model.current_decoder.parameters(),
                config['optimizer']
            )
            log(f'[Epoch {epoch}] Loaded decoder for {current_user_id}')

            previous_user_id = current_user_id

        # Forward pass
        pred = model(inp, coord, cell, user_indices)

        # Dual GT loss with w_p scheduling
        w_p = loss_scheduler.get_weight()
        loss = compute_dual_gt_loss(pred, gt_prefer, gt_non_prefer, w_p)

        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            log(f'WARNING: NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. Skipping batch.')
            continue

        # Backward with separated optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            list(model.encoder.parameters()) + list(model.current_decoder.parameters()),
            max_norm=1.0
        )

        encoder_optimizer.step()
        decoder_optimizer.step()
        loss_scheduler.step()

        train_loss.add(loss.item())
        pbar.set_postfix({'loss': loss.item(), 'w_p': w_p, 'user': current_user_id})

        # Log to tensorboard
        if writer is not None:
            iteration = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), iteration)
            writer.add_scalar('train/w_p', w_p, iteration)

    # Offload final decoder at end of epoch
    if model.current_decoder is not None:
        model.offload_user_decoder(ckpt_manager)
        log(f'[Epoch {epoch}] Offloaded decoder for {previous_user_id} (epoch end)')

    return train_loss.item()


def main(config_, save_path_):
    """Main training function"""
    global config, log, writer, save_path
    config = config_
    save_path = save_path_

    log, writer = utils.set_save_path(save_path, remove=False)

    # Create CSV file for training metrics logging
    csv_path = os.path.join(save_path, 'training_metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'w_p', 'elapsed_time'])
    csv_file.flush()
    log(f'Training metrics will be logged to: {csv_path}')

    # Validate configuration
    log('Validating configuration...')
    validate_config(config)
    log('Configuration validated successfully')

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # Create data loaders
    train_loader, val_loader, train_base_dataset = make_data_loaders()

    # Get user IDs from dataset
    user_ids = train_base_dataset.get_user_ids()
    log(f'Training with {len(user_ids)} users')
    log(f'User IDs (first 5): {user_ids[:5]}')

    # Prepare model and encoder optimizer
    model, encoder_optimizer, epoch_start, resume_info, ckpt_manager = prepare_training(user_ids)

    # Create loss weight scheduler
    loss_config = config.get('loss_schedule', {})
    total_iterations = config['epoch_max'] * len(train_loader)
    loss_scheduler = LossWeightScheduler(
        schedule_type=loss_config.get('schedule_type', 'exponential'),
        w_p_start=loss_config.get('w_p_start', 0.5),
        w_p_end=loss_config.get('w_p_end', 0.8),
        total_iterations=loss_config.get('total_iterations', total_iterations)
    )

    # Restore loss scheduler state if resuming
    if resume_info is not None and 'loss_scheduler' in resume_info:
        loss_scheduler.load_state_dict(resume_info['loss_scheduler'])
        log(f'Restored loss scheduler: iteration={loss_scheduler.current_iteration}, w_p={loss_scheduler.get_weight():.3f}')

    # Data normalization
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0.5], 'div': [0.5]},
            'gt': {'sub': [0.5], 'div': [0.5]}
        }

    # Training loop
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = [f'Epoch {epoch}/{epoch_max}']

        # Train
        train_loss = train(train_loader, model, encoder_optimizer, loss_scheduler, epoch, ckpt_manager)
        log_info.append(f'train_loss={train_loss:.4f}')
        log_info.append(f'w_p={loss_scheduler.get_weight():.3f}')

        # Save encoder checkpoint
        ckpt_manager.save_encoder(
            model, encoder_optimizer, epoch, name='epoch-last',
            extra_info={'loss_scheduler': loss_scheduler.state_dict()}
        )

        # Periodic encoder checkpoint save
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            ckpt_manager.save_encoder(model, encoder_optimizer, epoch, name=f'epoch-{epoch}')
            log(f'Saved encoder checkpoint at epoch {epoch}')
            # Note: Decoders are already saved during training when users switch

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
            writer.flush()

        # Log to CSV file
        csv_writer.writerow([epoch, train_loss, loss_scheduler.get_weight(), t_elapsed])
        csv_file.flush()

    # Final save
    ckpt_manager.save_encoder(model, encoder_optimizer, epoch_max, name='epoch-last')
    log('Training complete!')
    log(f'Encoder checkpoint: {save_path}/encoder/epoch-last.pth')
    log(f'Decoder checkpoints: {save_path}/user_decoders/*.pth')

    # Close CSV file
    csv_file.close()
    log(f'Training metrics saved to: {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--name', default='_train_pps', help='Save directory name')
    parser.add_argument('--tag', default=None, help='Additional tag for save directory')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('Config loaded.')

    # Create save directory
    save_name = args.name
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
