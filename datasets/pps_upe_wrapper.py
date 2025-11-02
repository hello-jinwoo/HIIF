"""
PPS UPE Wrapper
Wrapper that adds User Preference Embeddings (UPE) to each sample.
"""

from typing import Dict, Optional
import warnings

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import register


@register('pps-upe-wrapper')
class PPSUPEWrapper(Dataset):
    """
    Wrapper that adds UPE to each sample from cache.

    This wrapper:
    1. Wraps a base dataset (e.g., PPSColorAugmentedWrapper)
    2. Loads UPE from cache for each user
    3. Adds UPE to each sample in __getitem__
    4. Supports lazy loading or preloading

    Usage:
        >>> from pps_utils.upe_cache import UPECache
        >>>
        >>> # Create base dataset
        >>> base_dataset = PPSColorAugmentedWrapper(...)
        >>>
        >>> # Create UPE cache
        >>> upe_cache = UPECache(cache_dir='./cache/upe')
        >>> upe_config = {
        >>>     'content_model': 'dino',
        >>>     'color_model': 'clip',
        >>>     'num_pairs': 16,
        >>>     'content_dim': 768,
        >>>     'color_dim': 512,
        >>> }
        >>>
        >>> # Create wrapper
        >>> dataset = PPSUPEWrapper(base_dataset, upe_cache, upe_config, device='cuda')
        >>>
        >>> # Optional: Preload all UPEs
        >>> dataset.preload_all_upes()
        >>>
        >>> # Use in DataLoader
        >>> loader = DataLoader(dataset, batch_size=8, collate_fn=collate_with_upe)
    """

    def __init__(self,
                 dataset: Dataset,
                 upe_cache,  # UPECache instance
                 upe_config: Dict,
                 device: str = 'cpu',
                 cache_policy: str = 'fail_fast'):
        """
        Args:
            dataset: Base dataset (e.g., PPSColorAugmentedWrapper)
            upe_cache: UPECache instance for loading UPEs
            upe_config: UPE configuration dict:
                {
                    'content_model': str (e.g., 'dino'),
                    'color_model': str (e.g., 'clip'),
                    'num_pairs': int (e.g., 16),
                    'content_dim': int (e.g., 768),
                    'color_dim': int (e.g., 512),
                }
            device: Device to load UPEs on ('cuda' or 'cpu')
            cache_policy: How to handle cache misses:
                - 'fail_fast': Raise error on cache miss (default, recommended)
                - 'skip': Skip samples with missing UPE (not recommended)
                - 'extract_on_demand': Extract UPE on-the-fly (slow, not implemented)
        """
        super().__init__()
        self.dataset = dataset
        self.upe_cache = upe_cache
        self.upe_config = upe_config
        self.device = device
        self.cache_policy = cache_policy

        # Dictionary to store loaded UPEs (lazy loading)
        # Maps user_id -> UPE tensor
        self.upe_dict = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get sample with UPE added.

        Returns:
            dict with same keys as base dataset + 'upe' key:
                - All keys from base dataset (user_id, image_id, inp, gt_prefer, etc.)
                - 'upe': torch.Tensor (16, 1280) - Raw UPE for this user

        Raises:
            RuntimeError: If cache_policy='fail_fast' and UPE not found
        """
        # Get base sample
        sample = self.dataset[idx]
        user_id = sample['user_id']

        # Load UPE (from cache or memory)
        if user_id not in self.upe_dict:
            upe = self.upe_cache.load(user_id, self.upe_config, device=self.device)

            if upe is None:
                # Handle cache miss based on policy
                if self.cache_policy == 'fail_fast':
                    raise RuntimeError(
                        f"UPE not found for user {user_id}. "
                        f"Please run UPE extraction first:\n"
                        f"  python scripts/extract_all_upes.py --split train/validation\n"
                        f"Cache dir: {self.upe_cache.cache_dir}"
                    )

                elif self.cache_policy == 'skip':
                    warnings.warn(f"Skipping user {user_id} (UPE not found)")
                    return None  # Collate function must handle None

                elif self.cache_policy == 'extract_on_demand':
                    raise NotImplementedError(
                        "On-demand UPE extraction not implemented. "
                        "Please run extract_all_upes.py before training."
                    )

                else:
                    raise ValueError(f"Unknown cache_policy: {self.cache_policy}")

            # Store in memory for future use
            self.upe_dict[user_id] = upe

        # Add UPE to sample
        sample['upe'] = self.upe_dict[user_id]
        return sample

    def preload_all_upes(self, show_progress: bool = True):
        """
        Preload all UPEs before training.

        This loads all unique user UPEs into memory, which is efficient if:
        - Total users < 1000 (memory usage ~160 MB)
        - Training on GPU with sufficient VRAM

        Args:
            show_progress: Show tqdm progress bar (default: True)

        Example:
            >>> dataset = PPSUPEWrapper(...)
            >>> dataset.preload_all_upes()  # Preload before training
            >>> # Now __getitem__ will be faster (no disk I/O)
        """
        # Collect all unique user IDs
        user_ids = set()
        for i in range(len(self)):
            sample = self.dataset[i]
            user_ids.add(sample['user_id'])

        # Load UPEs
        iterator = tqdm(user_ids, desc="Preloading UPEs") if show_progress else user_ids
        loaded_count = 0
        missing_users = []

        for user_id in iterator:
            upe = self.upe_cache.load(user_id, self.upe_config, device=self.device)

            if upe is not None:
                self.upe_dict[user_id] = upe
                loaded_count += 1
            else:
                missing_users.append(user_id)

        # Report results
        print(f"Preloaded {loaded_count}/{len(user_ids)} UPEs")

        if missing_users:
            if self.cache_policy == 'fail_fast':
                raise RuntimeError(
                    f"Missing UPEs for {len(missing_users)} users: {missing_users[:5]}...\n"
                    f"Please run UPE extraction first."
                )
            else:
                warnings.warn(
                    f"Missing UPEs for {len(missing_users)} users. "
                    f"These samples will be skipped during training."
                )

    def get_loaded_user_count(self) -> int:
        """
        Get number of users with loaded UPEs.

        Returns:
            Number of users currently in memory
        """
        return len(self.upe_dict)

    def clear_cache(self):
        """
        Clear in-memory UPE cache.

        Use this to free GPU/CPU memory if needed.
        """
        self.upe_dict.clear()
        print("Cleared in-memory UPE cache")

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about UPE cache usage.

        Returns:
            Dict with cache statistics:
                - loaded_users: Number of users in memory
                - total_samples: Total dataset size
                - device: Device where UPEs are stored
        """
        return {
            'loaded_users': len(self.upe_dict),
            'total_samples': len(self),
            'device': self.device,
            'cache_policy': self.cache_policy,
        }
