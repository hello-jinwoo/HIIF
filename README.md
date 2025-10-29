# HIIF-PPS: Personalized Photographic Style Learning

Learning user-specific photographic styles from preference pairs using Hierarchical Implicit Image Functions (HIIF).

**Task**: Style Transfer (Input RGB → Style-transferred RGB, same resolution)
**Method**: Shared encoder + 100 user-specific decoders trained with dual ground truth loss

---

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify dataset structure
ls load/PPS/responses/train/      # User JSON files (e.g., user_response_example10.json)
ls load/PPS/responses/validation/ # Validation JSON files
ls load/PPS/images/               # Image folders organized by scene and style
```

**Dataset structure**:
- Each JSON file contains ~80 preference pairs (prefer vs non-prefer image paths)
- Training set: ~100 users
- Validation set: ~20-30 users (different from training)

### 2. Training

```bash
# Small test: 10 users, 50 epochs (~2 hours on single GPU)
python train_pps.py \
    --config configs/train-pps/integration_test_10users_50epochs.yaml \
    --name my_experiment \
    --gpu 0
```

**What happens**:
- Trains shared encoder + 10 user-specific decoders
- Saves checkpoints to `save/my_experiment/`
- Logs training metrics (loss, PSNR, etc.)
- Takes ~2-3 hours for 50 epochs on GTX 1080Ti/RTX 2080

### 3. Validation (Few-Shot Learning)

```bash
# Evaluate trained model with few-shot learning
python validate_pps.py \
    --config configs/validate-pps/validate_pps.yaml \
    --checkpoint save/my_experiment/encoder/epoch-last.pth \
    --gpu 0
```

**Few-shot validation process**:
1. Loads trained shared encoder (frozen)
2. For each validation user:
   - Samples N training images (default: 20)
   - Creates and trains a **new decoder from scratch**
   - Evaluates at multiple checkpoints: [50, 100, 500, 1000] iterations
   - Tests with 3 augmentation types: clean, same aug, different aug
3. Computes 6 metrics: PSNR, SSIM, LPIPS, ΔE LAB, CIEDE2000, NIQE
4. Saves results to `save/my_experiment/validation/results.json`

**Time**: ~10-20 min for 20 validation users on single GPU

---

## Key Config Parameters

Config files are in `configs/train-pps/*.yaml`. Key parameters for research:

### Model Architecture

```yaml
model:
  args:
    encoder_spec:
      name: edsr-baseline  # Options: edsr-baseline, swinir, rdn
      args:
        n_resblocks: 16    # Encoder depth: 16 (small), 32 (large)
        n_feats: 64        # Feature channels: 64 (small), 128 (large)
    hidden_dim: 256        # HIIF decoder hidden dim: 128, 256, 512
    blocks: 16             # HIIF decoder blocks: 8, 16, 32
```

**Impact**:
- Larger encoder: Better feature extraction, slower training
- Larger decoder: More user-specific capacity, more memory
- **Recommended**: Start with baseline (n_resblocks=16, hidden_dim=256)

### Training Strategy

```yaml
train_dataset:
  batch_size: 2              # Samples per batch (× 4 augmented versions = 8 images)
  sampler:
    iterations_per_user: 16  # Train each user for N iterations before switching

optimizer:
  args:
    lr: 1.e-4               # Learning rate: 1e-4 (stable), 1e-3 (faster but risky)

epoch_max: 200              # Total epochs: 50 (test), 200 (full), 500 (best quality)
epoch_val: 10               # Validate every N epochs
epoch_save: 10              # Save checkpoint every N epochs
```

**Impact**:
- `iterations_per_user`: Higher = less decoder switching overhead, but slower convergence for individual users
  - **Recommended**: 8-16 for 10-50 users, 4-8 for 100 users
- `lr`: Higher = faster training but less stable, Lower = slower but more stable
  - **Recommended**: 1e-4 (safe), 5e-5 (if loss plateaus)
- `epoch_max`: More epochs = better quality, but diminishing returns after 200-300
  - **Recommended**: 50 (quick test), 200 (good results), 500 (publication quality)

### Loss Configuration

```yaml
loss_schedule:
  schedule_type: exponential   # Options: linear, exponential, cosine
  w_p_start: 0.5              # Initial weight for prefer images (0-1)
  w_p_end: 0.8                # Final weight for prefer images (0-1)
```

**Dual Ground Truth Loss**: `L_total = w_p × L1(pred, prefer) + (1 - w_p) × L1(pred, non_prefer)`

**Impact**:
- `w_p_start`: Start balanced (0.5) to learn from both prefer and non-prefer
- `w_p_end`: End prefer-focused (0.7-0.9) to match user preferences
- `schedule_type`: Exponential reaches high w_p faster than linear
  - **Recommended**: w_p_start=0.5, w_p_end=0.8, exponential schedule

### Data Augmentation

```yaml
wrapper:
  args:
    crop_size: 128            # Random crop size: 64 (fast), 128 (balanced), 256 (slow)
    augment: true
    augment_params:
      brightness: 0.1         # HSV brightness shift: 0-0.2
      contrast: 0.1           # HSV saturation scaling: 0-0.2
      saturation: 0.1         # HSV saturation shift: 0-0.2
      hue: 0.05               # HSV hue shift: 0-0.1
```

**Impact**:
- `crop_size`: Larger crops = better context, slower training, more memory
  - **Recommended**: 64 (testing), 128 (training), 256 (final fine-tuning)
- Augmentation strength: Prevents overfitting, increases robustness
  - **Recommended**: Keep default values (brightness=0.1, hue=0.05) for natural photos

---

## Results

### Checkpoints

Training saves:
```
save/my_experiment/
├── encoder/
│   ├── epoch-last.pth     # Latest encoder (always updated)
│   ├── epoch-10.pth       # Saved every epoch_save epochs
│   └── epoch-best.pth     # Best validation PSNR (if epoch_val is set)
└── user_decoders/
    ├── user_response_example10.pth  # User-specific decoder
    ├── user_response_example11.pth
    └── ...
```

### Logs

```bash
# View training logs
cat save/my_experiment/log_train.txt

# Monitor training in real-time
tail -f save/my_experiment/log_train.txt

# Check TensorBoard (if enabled)
tensorboard --logdir save/my_experiment/tensorboard
```

**Key metrics to watch**:
- `loss_total`: Should decrease steadily (target: < 0.05 after 50 epochs)
- `psnr_prefer`: Should increase (target: > 28 dB)
- `psnr_non_prefer`: Should be lower than prefer (target: 24-26 dB)
- `w_p`: Weight for prefer images (should increase from 0.5 to 0.8)

### Validation Results

```bash
# View validation results
cat save/my_experiment/validation/results.json

# Format: JSON with per-user metrics
{
  "user_response_example10": {
    "ckpt_1000": {
      "psnr_prefer": 28.5,
      "ssim_prefer": 0.89,
      "lpips_prefer": 0.12,
      ...
    }
  },
  "overall": {
    "psnr_prefer_mean": 28.2,
    "psnr_prefer_std": 1.3,
    ...
  }
}
```

**Target metrics** (prefer images):
- PSNR: 28-32 dB (higher = better pixel accuracy)
- SSIM: 0.85-0.92 (higher = better structural similarity)
- LPIPS: 0.05-0.15 (lower = better perceptual quality)
- CIEDE2000 (ΔE00): 2-8 (lower = better color accuracy)
- NIQE: 3-6 (lower = better natural image quality)

---

## Advanced Usage

### Using SwinIR Encoder

SwinIR (Swin Transformer) provides stronger feature extraction than EDSR:

```bash
# Quick test: 10 users, 50 epochs with SwinIR
python train_pps.py \
    --config configs/train-pps/train_pps_swinir_test.yaml \
    --name swinir_test \
    --gpu 0

# Full training: ~100 users, 200 epochs with SwinIR
python train_pps.py \
    --config configs/train-pps/train_pps_swinir.yaml \
    --name swinir_full \
    --gpu 0
```

**SwinIR vs EDSR**:
- **SwinIR**: Transformer-based, better perceptual quality, slower training (~1.5x)
- **EDSR**: CNN-based, faster training, good baseline
- **Recommended**: Start with EDSR for quick experiments, use SwinIR for final results

### Custom Config

Create your own config in `configs/train-pps/my_config.yaml`:

```yaml
model:
  name: hiif-pps
  args:
    encoder_spec:
      name: swinir  # Options: edsr-baseline, swinir, rdn
      args:
        no_upsampling: true
    hidden_dim: 512  # Larger decoder for more capacity

train_dataset:
  batch_size: 4  # If you have more GPU memory
  sampler:
    iterations_per_user: 8
  wrapper:
    args:
      crop_size: 256  # Larger crops for better context

optimizer:
  args:
    lr: 5.e-5  # Lower learning rate for stability

epoch_max: 300  # Longer training for better convergence

loss_schedule:
  schedule_type: cosine  # Options: linear, exponential, cosine
  w_p_start: 0.5
  w_p_end: 0.9  # More prefer-focused
```

Then run:
```bash
python train_pps.py --config configs/train-pps/my_config.yaml --name my_custom_exp --gpu 0
```

### Multi-GPU Training

Currently single-GPU only. For multi-GPU, modify `train_pps.py` to use `DataParallel` for encoder (decoders are user-specific, one at a time).

### Resume Training

```bash
# Training auto-resumes from epoch-last.pth if it exists
python train_pps.py \
    --config configs/train-pps/my_config.yaml \
    --name my_experiment \  # Same name as before
    --gpu 0
```

---

## Troubleshooting

### Issue: Loss becomes NaN

**Cause**: Usually color space conversion issues or too high learning rate

**Solution**:
1. Lower learning rate: `lr: 5.e-5` or `lr: 1.e-5`
2. Check augmentation is not too strong: reduce `hue` to 0.02
3. Add gradient clipping in config (if available)

### Issue: Low PSNR (< 25 dB)

**Cause**: Underfitting or insufficient training

**Solution**:
1. Train longer: increase `epoch_max` to 200-500
2. Use larger model: increase `hidden_dim` to 512 or `blocks` to 32
3. Lower learning rate: try `lr: 5.e-5`
4. Check loss is decreasing in logs

### Issue: PSNR for prefer and non-prefer are similar

**Cause**: Model not learning user preferences

**Solution**:
1. Increase prefer weight: set `w_p_end: 0.9`
2. Check dataset: verify prefer/non-prefer images are different
3. Train longer: model may need more epochs to learn preferences
4. Increase decoder capacity: `hidden_dim: 512`

### Issue: Out of memory (OOM)

**Cause**: Batch size or crop size too large

**Solution**:
1. Reduce crop size: `crop_size: 64` or `crop_size: 96`
2. Reduce batch size: `batch_size: 1`
3. Use smaller encoder: `n_feats: 32` instead of 64
4. Use smaller decoder: `hidden_dim: 128` instead of 256

### Issue: Training too slow

**Cause**: Large crop size or too many iterations per user

**Solution**:
1. Reduce crop size: `crop_size: 64` (fastest) or `crop_size: 96`
2. Reduce iterations per user: `iterations_per_user: 4` or `iterations_per_user: 8`
3. Use faster encoder: `edsr-baseline` instead of `swinir`
4. Increase num_workers in dataloader (edit train_pps.py)

---

## Experimental Guidelines

### For ablation studies:

**Effect of loss weight**:
- Run with `w_p_end: 0.6, 0.7, 0.8, 0.9`
- Compare PSNR gap between prefer and non-prefer

**Effect of model capacity**:
- Run with `hidden_dim: 128, 256, 512, 1024`
- Compare final PSNR and training time

**Effect of encoder type**:
- Run with `encoder: edsr-baseline, swinir, rdn`
- Compare perceptual quality (LPIPS, SSIM)

**Effect of training strategy**:
- Run with `iterations_per_user: 4, 8, 16, 32`
- Compare training speed and convergence

### Recommended experiment pipeline:

1. **Quick test** (verify code works): 10 users, 50 epochs, crop_size=64
2. **Pilot study** (find good hyperparameters): 20 users, 100 epochs, crop_size=128
3. **Full experiment** (final results): 100 users, 200-500 epochs, crop_size=128-256

---

## Citation

If you use this code for research, please cite (update with your paper details):

```bibtex
@article{hiif-pps2025,
  title={Personalized Photographic Style Learning with Hierarchical Implicit Image Functions},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

**Note**: This is a placeholder. Update with your actual publication information.

---

## Project Structure

```
HIIF/
├── train_pps.py              # Training script
├── validate_pps.py           # Validation script
├── configs/
│   ├── train-pps/            # Training configs
│   └── validate-pps/         # Validation configs
├── datasets/
│   ├── pps_preference_pair.py   # Dataset loader
│   ├── pps_wrapper.py           # Augmentation wrapper
│   └── pps_sampler.py           # User batch sampler
├── models/
│   ├── hiif_pps.py           # Main model
│   ├── color_utils.py        # Color space conversions
│   └── edsr.py, swinir.py    # Encoders
├── pps_utils/
│   ├── checkpoint_manager.py  # Save/load checkpoints
│   └── loss_scheduler.py      # Dual GT loss
├── metrics.py                # Evaluation metrics
└── load/PPS/                 # Dataset (not included)
```

---

**For detailed implementation notes, see `.claude/CLAUDE.md`**
