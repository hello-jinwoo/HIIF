"""
Test script to check if GT prefer and GT non-prefer are identical in the dataset.
"""

import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, '/workspace/HIIF')

from datasets import make
from pps_utils.upe_cache import UPECache


def test_gt_identity(config_path, num_samples=10):
    """
    Test if GT prefer and non-prefer are identical.

    Args:
        config_path: Path to config file
        num_samples: Number of samples to test
    """
    print("=" * 80)
    print("Testing GT Identity")
    print("=" * 80)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfig: {config_path}")
    print(f"Testing {num_samples} samples\n")

    # Create validation dataset
    val_spec = config['val_dataset']

    # Create base dataset
    base_dataset = make(val_spec['dataset'])

    # Wrap with color augmentation wrapper
    dataset = make(val_spec['wrapper'], args={'dataset': base_dataset})

    print(f"Dataset size: {len(dataset)}")

    # Test samples
    identical_count = 0
    near_identical_count = 0
    different_count = 0

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        gt_prefer = sample['gt_prefer']  # (3, H, W)
        gt_non_prefer = sample['gt_non_prefer']  # (3, H, W)

        # Check if identical
        is_identical = torch.equal(gt_prefer, gt_non_prefer)

        # Check MSE
        mse = F.mse_loss(gt_prefer, gt_non_prefer).item()

        # Print result
        print(f"Sample {i}:")
        print(f"  User: {sample['user_id']}, Image: {sample['image_id']}")
        print(f"  Identical: {is_identical}")
        print(f"  MSE: {mse:.8f}")

        if is_identical:
            identical_count += 1
            print(f"  ⚠️  BUG: GTs are IDENTICAL!")
        elif mse < 1e-6:
            near_identical_count += 1
            print(f"  ⚠️  WARNING: GTs are nearly identical")
        else:
            different_count += 1
            print(f"  ✓ OK: GTs are different")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Samples tested: {num_samples}")
    print(f"Identical: {identical_count} ({identical_count/num_samples*100:.1f}%)")
    print(f"Nearly identical: {near_identical_count} ({near_identical_count/num_samples*100:.1f}%)")
    print(f"Different: {different_count} ({different_count/num_samples*100:.1f}%)")
    print()

    if identical_count > 0:
        print("❌ BUG CONFIRMED: Some GT pairs are identical!")
        return False
    elif near_identical_count > 0:
        print("⚠️  WARNING: Some GT pairs are nearly identical")
        return True
    else:
        print("✓ All GT pairs are different")
        return True


if __name__ == '__main__':
    config_path = 'configs/pps/pps_upe_edsr.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    num_samples = 20
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])

    success = test_gt_identity(config_path, num_samples)
    sys.exit(0 if success else 1)
