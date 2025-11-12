"""
Extract and cache UPEs for all users in the dataset.

Usage:
    python scripts/extract_all_upes.py --config configs/upe/upe_extraction_config.yaml
    python scripts/extract_all_upes.py --split train --num_pairs 16
    python scripts/extract_all_upes.py --split validation --force_recompute
"""

import argparse
import torch
import yaml
import warnings
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.upe_extractor import UPEExtractor
from models.upe_processor import UPEProcessor
from pps_utils.upe_cache import UPECache
from datasets.pps_preference_pair import PPSPreferencePairDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Extract UPEs for all users')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file (YAML)')
    parser.add_argument('--split', type=str, default=None,
                       help='Dataset split (can be any folder name under responses/)')
    parser.add_argument('--root_path', type=str, default=None,
                       help='Root path to images (default: ./load/PPS/images)')
    parser.add_argument('--response_dir', type=str, default=None,
                       help='Response directory (auto-set based on split if None)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory (default: ./cache/upe)')
    parser.add_argument('--content_model', type=str, default=None,
                       choices=['clip', 'dino', 'sam'],
                       help='Content model (default: dino)')
    parser.add_argument('--color_model', type=str, default=None,
                       choices=['clip', 'dino', 'dinov2'],
                       help='Color model (default: clip)')
    parser.add_argument('--num_pairs', type=int, default=None,
                       help='Number of preference pairs for UPE extraction (default: 16)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu) (default: cuda)')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recompute even if cache exists')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Load config if provided
    if args.config is not None:
        config = load_config(args.config)
        # Only override with explicitly provided command line args
        # (i.e., skip None values which were not provided)
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
    else:
        config = vars(args)

    # Set defaults for required fields if not specified in config or CLI
    if config.get('split') is None:
        config['split'] = 'train'
    if config.get('root_path') is None:
        config['root_path'] = './load/PPS/images'
    if config.get('cache_dir') is None:
        config['cache_dir'] = './cache/upe'
    if config.get('content_model') is None:
        config['content_model'] = 'dino'
    if config.get('color_model') is None:
        config['color_model'] = 'clip'
    if config.get('num_pairs') is None:
        config['num_pairs'] = 16
    if config.get('device') is None:
        config['device'] = 'cuda'
    if config.get('force_recompute') is None:
        config['force_recompute'] = False
    if config.get('verbose') is None:
        config['verbose'] = False

    # Auto-set response_dir based on split
    if config.get('response_dir') is None:
        config['response_dir'] = f"./load/PPS/responses/{config['split']}"

    # Check device
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        warnings.warn("CUDA not available, using CPU")
        config['device'] = 'cpu'

    print("=" * 60)
    print("UPE Extraction Configuration")
    print("=" * 60)
    print(f"Split: {config['split']}")
    print(f"Root path: {config['root_path']}")
    print(f"Response dir: {config['response_dir']}")
    print(f"Cache dir: {config['cache_dir']}")
    print(f"Content model: {config['content_model']} (using fixed variant)")
    print(f"Color model: {config['color_model']} (using fixed variant)")
    print(f"Num pairs: {config['num_pairs']}")
    print(f"Device: {config['device']}")
    print(f"Force recompute: {config['force_recompute']}")
    print("=" * 60)

    # Initialize dataset
    print("\n[1/5] Loading dataset...")
    dataset = PPSPreferencePairDataset(
        root_path=config['root_path'],
        response_dir=config['response_dir'],
    )
    user_ids = dataset.get_user_ids()
    print(f"  Found {len(user_ids)} users")
    print(f"  Dataset name: {dataset.dataset_name}")

    # Initialize UPE extractor
    print("\n[2/5] Initializing UPE extractor...")
    upe_extractor = UPEExtractor(
        content_model_name=config['content_model'],
        color_model_name=config['color_model'],
        num_pairs=config['num_pairs'],
        device=config['device']
    )

    # Determine dimensions
    content_dim = upe_extractor.content_dim
    color_dim = upe_extractor.color_dim

    # Initialize cache
    print("\n[3/5] Initializing cache...")
    upe_cache = UPECache(
        cache_dir=config['cache_dir'],
        max_memory_size=100,
        preload_to_gpu=False,
        dataset_name=dataset.dataset_name  # 🆕 Use dataset-specific subdirectory
    )
    print(f"  Cache directory: {upe_cache.cache_dir}")

    # Prepare config for cache
    cache_config = {
        'content_model': config['content_model'],
        'color_model': config['color_model'],
        'num_pairs': config['num_pairs'],
        'content_dim': content_dim,
        'color_dim': color_dim,
    }

    # Extract UPEs
    print("\n[4/5] Extracting UPEs...")
    success_count = 0
    skip_count = 0
    error_count = 0

    for user_id in tqdm(user_ids, desc="Extracting UPEs"):
        try:
            # Check if cache exists and skip if not force recompute
            if not config['force_recompute'] and upe_cache.exists(user_id, cache_config):
                skip_count += 1
                if config.get('verbose'):
                    print(f"  [SKIP] {user_id}: Cache exists")
                continue

            # Get user samples
            user_sample_indices = dataset.get_user_samples(user_id)

            # Check if user has enough samples
            if len(user_sample_indices) < config['num_pairs']:
                warnings.warn(
                    f"User {user_id} has only {len(user_sample_indices)} samples, "
                    f"but {config['num_pairs']} required. SKIPPING."
                )
                error_count += 1
                continue

            # Get first N pairs for UPE extraction
            upe_sample_indices = user_sample_indices[:config['num_pairs']]

            # Load pairs
            pairs = []
            for idx in upe_sample_indices:
                sample = dataset[idx]
                pairs.append({
                    'prefer': sample['prefer'],
                    'non_prefer': sample['non_prefer']
                })

            # Extract UPE
            upe_raw = upe_extractor(pairs)

            # Validate shape
            expected_shape = (config['num_pairs'], content_dim + color_dim)
            if upe_raw.shape != expected_shape:
                raise ValueError(
                    f"UPE shape mismatch: expected {expected_shape}, got {upe_raw.shape}"
                )

            # Save to cache
            upe_cache.save(user_id, upe_raw, cache_config, metadata={
                'split': config['split'],
                'num_total_samples': len(user_sample_indices),
            })

            success_count += 1
            if config.get('verbose'):
                print(f"  [OK] {user_id}: UPE extracted and cached")

        except Exception as e:
            error_count += 1
            warnings.warn(f"Failed to extract UPE for {user_id}: {e}")

    # Print summary
    print("\n[5/5] Summary")
    print("=" * 60)
    print(f"Total users: {len(user_ids)}")
    print(f"Successfully extracted: {success_count}")
    print(f"Skipped (cached): {skip_count}")
    print(f"Errors: {error_count}")
    print("=" * 60)

    # Print cache stats
    upe_cache.print_stats()

    print("\n✓ UPE extraction completed!")


if __name__ == '__main__':
    main()
