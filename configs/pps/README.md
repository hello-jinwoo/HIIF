# Unified PPS Configuration Files

This folder contains **unified configuration files** that work with both `train_pps.py` and `validate_pps.py`.

## Why Unified Configs?

Previously, training and validation configs were separate:
- `configs/train-pps/train_pps_swinir.yaml` - Training only
- `configs/validate-pps/validate_pps_swinir.yaml` - Validation only

**Problems with separate configs:**
- ❌ Duplication of shared settings (model architecture, data normalization)
- ❌ Risk of inconsistency between train and validate
- ❌ Harder to maintain and version control

**Benefits of unified configs:**
- ✅ Single source of truth for model/dataset configuration
- ✅ Guaranteed consistency between training and validation
- ✅ Easier to maintain and track changes
- ✅ No code changes required to existing scripts

## Structure

Each unified config has three sections:

```yaml
# =============================================================================
# Shared Configuration (used by both train and validate)
# =============================================================================
model: ...          # Model architecture (identical for both)
val_dataset: ...    # Validation dataset (used by both)
data_norm: ...      # Data normalization (must match)

# =============================================================================
# Training Configuration (used by train_pps.py only)
# =============================================================================
train_dataset: ...  # Training dataset
optimizer: ...      # Training optimizer
epoch_max: ...      # Training epochs
loss_schedule: ...  # Loss weight scheduling

# =============================================================================
# Validation Configuration (used by validate_pps.py only)
# =============================================================================
encoder_checkpoint: ...    # Path to trained encoder
decoder_training: ...      # Few-shot decoder training settings
augmentation: ...          # Test-time augmentation configs
metrics: ...               # Evaluation metrics
```

## Available Configs

| File | Description | Encoder | Crop Size | Use Case |
|------|-------------|---------|-----------|----------|
| `pps_swinir.yaml` | Full-scale SwinIR | SwinIR | 128 | Production training/validation |
| `pps_edsr.yaml` | Full-scale EDSR | EDSR-Baseline | 64 | Production training/validation |
| `test_2users.yaml` | Quick test config | EDSR-Baseline | 64 | Testing with 2 users only |

## Usage

### Training
```bash
# Full-scale training with SwinIR
python train_pps.py --config configs/pps/pps_swinir.yaml

# Full-scale training with EDSR
python train_pps.py --config configs/pps/pps_edsr.yaml

# Quick test with 2 users
python train_pps.py --config configs/pps/test_2users.yaml --tag test_2users
```

### Validation
```bash
# Validation with SwinIR (uses same config!)
python validate_pps.py --config configs/pps/pps_swinir.yaml

# Validation with EDSR (uses same config!)
python validate_pps.py --config configs/pps/pps_edsr.yaml

# Quick validation test with 2 users
python validate_pps.py --config configs/pps/test_2users.yaml --tag test_2users
```

## Key Configuration Parameters

### Shared Parameters

#### Model Architecture
- `model.args.encoder_spec.name`: Encoder type (`swinir`, `edsr-baseline`, `rdn`)
- `model.args.hidden_dim`: Hidden dimension for decoder (default: 256)
- `model.args.blocks`: Number of decoder blocks (default: 16)

#### Data Normalization
- `data_norm.inp`: Input normalization (default: subtract 0.5, divide by 0.5)
- `data_norm.gt`: Ground truth normalization (same as input)

### Training-Only Parameters

- `train_dataset.batch_size`: Training batch size (default: 2)
- `train_dataset.sampler.iterations_per_user`: Iterations before user switch (default: 16)
- `optimizer.args.lr`: Learning rate (SwinIR: 5e-4, EDSR: 1e-4)
- `epoch_max`: Maximum training epochs (default: 200)
- `epoch_val`: Validation interval (default: 20)
- `epoch_save`: Checkpoint save interval (default: 20)
- `loss_schedule.w_p_start/end`: Prefer image weight range (default: 0.5 → 0.9)

### Validation-Only Parameters

- `encoder_checkpoint`: Path to trained encoder (e.g., `./save/train_hiif_pps_swinir/encoder/epoch-last.pth`)
- `decoder_training.samples_per_user`: Training samples per user (default: 64)
- `decoder_training.eval_checkpoints`: Evaluation checkpoints (default: [50, 100, 500, 1000])
- `decoder_training.max_iterations`: Maximum training iterations (default: 1000)
- `augmentation`: Test-time augmentation configurations
- `metrics`: List of metrics to compute (PSNR, SSIM, LPIPS, etc.)
- `num_val_users`: Number of users to validate (null = all users)

## Creating New Configs

To create a new unified config:

1. **Copy an existing config** as a template:
   ```bash
   cp configs/pps/pps_swinir.yaml configs/pps/my_new_config.yaml
   ```

2. **Modify shared parameters** (model, data_norm, val_dataset)

3. **Modify training parameters** (train_dataset, optimizer, epochs)

4. **Modify validation parameters** (decoder_training, augmentation, metrics)

5. **Test the config**:
   ```bash
   python -c "
   import yaml
   from train_pps import validate_config as train_validate
   from validate_pps import validate_config as val_validate

   with open('configs/pps/my_new_config.yaml') as f:
       config = yaml.load(f, Loader=yaml.FullLoader)

   train_validate(config)
   val_validate(config)
   print('✓ Config validated successfully!')
   "
   ```

## Migration from Old Configs

If you have existing separate configs in `configs/train-pps/` or `configs/validate-pps/`, you can still use them:

```bash
# Old configs still work (backward compatible)
python train_pps.py --config configs/train-pps/train_pps_swinir.yaml
python validate_pps.py --config configs/validate-pps/validate_pps_swinir.yaml
```

However, **we recommend migrating to unified configs** for better maintainability.

## Troubleshooting

### Config validation fails
Run the validation test to check which keys are missing:
```bash
python -c "
import yaml
from train_pps import validate_config

with open('configs/pps/your_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

validate_config(config)
"
```

### Encoder checkpoint path incorrect
Make sure `encoder_checkpoint` in the validation section points to the correct path:
- Default: `./save/train_hiif_pps_{encoder_name}/encoder/epoch-last.pth`
- Example: `./save/train_hiif_pps_swinir/encoder/epoch-last.pth`

### Model architecture mismatch
Ensure the `model` section in the unified config matches what was used during training. The model architecture must be identical for both training and validation.

## Best Practices

1. ✅ **Use unified configs** for new experiments
2. ✅ **Version control** your configs alongside code
3. ✅ **Document changes** when modifying configs
4. ✅ **Test configs** before long training runs
5. ✅ **Keep encoder checkpoint paths consistent** with training save directories
6. ✅ **Match crop sizes** between training and validation for best results

## Questions?

See the main project documentation in `.claude/CLAUDE.md` for more details about the PPS project structure and development workflow.
