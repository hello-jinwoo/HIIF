# UPE Configuration Guide

This directory contains configuration files for User Preference Embedding (UPE) extraction.

## Quick Start

### 1. Extract UPEs for Training
```bash
python scripts/extract_all_upes.py \
    --config configs/upe/upe_extraction_config.yaml \
    --split train
```

### 2. Extract UPEs for Validation
```bash
python scripts/extract_all_upes.py \
    --config configs/upe/upe_extraction_config.yaml \
    --split validation
```

### 3. Check Cache Status
```bash
python -c "
from pps_utils.upe_cache import UPECache
cache = UPECache('./cache/upe')
cache.print_stats()
"
```

## Configuration Options

### Model Selection

**Content Model** (semantic/structure information):
- `dino` (recommended): 768-dim output
- `clip`: 512-dim output

**Color Model** (style/color information):
- `clip` (recommended): 512-dim output
- `dino`: 768-dim output

**Important**: Content and color models MUST be different!

### Recommended Configurations

**Default (balanced)**:
```yaml
content_model: dino  # 768-dim
color_model: clip    # 512-dim
num_pairs: 16
```

**Alternative**:
```yaml
content_model: clip  # 512-dim
color_model: dino    # 768-dim
num_pairs: 16
```

## Output Dimensions

| Config | Content | Color | Raw UPE | Processed UPE |
|--------|---------|-------|---------|---------------|
| dino + clip | (16, 768) | (16, 512) | (16, 1280) | (512,) |
| clip + dino | (16, 512) | (16, 768) | (16, 1280) | (512,) |

## Cache Location

Default: `./cache/upe/`

Cache files are named: `{user_id}_{content_model}_{color_model}_{num_pairs}.pt`

Example: `user_example10_dino_clip_16.pt`

## Troubleshooting

### "content_model and color_model must be different"
Make sure you're using different models for content and color (e.g., dino + clip, not clip + clip).

### "CUDA out of memory"
Add `--device cpu` to use CPU instead of GPU (slower but uses less memory).

### "User has only X samples, but 16 required"
Some users may not have enough samples. These will be automatically skipped with a warning.

### Cache is stale
Use `--force_recompute` to regenerate all caches.

## Advanced Usage

### Custom Number of Pairs
```bash
python scripts/extract_all_upes.py \
    --num_pairs 32 \
    --split train
```

### Verbose Output
```bash
python scripts/extract_all_upes.py \
    --verbose \
    --split train
```

### Custom Cache Directory
```bash
python scripts/extract_all_upes.py \
    --cache_dir ./my_custom_cache \
    --split train
```

## Expected Extraction Time

- **10 users**: ~30 seconds
- **100 users**: ~5 minutes
- **1000 users**: ~50 minutes

*Times are approximate and depend on hardware (GPU vs CPU).*
