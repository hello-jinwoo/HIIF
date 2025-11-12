"""
PPS Preference Pair Dataset
Loads user preference data and corresponding image pairs.
"""

import os
import json
import glob
import copy
import hashlib
import warnings
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('pps-preference-pair')
class PPSPreferencePairDataset(Dataset):
    """
    Base dataset for PPS preference pairs.

    Loads JSON metadata and corresponding image pairs from disk.
    Each sample contains a prefer and non-prefer image pair for a specific user.
    """

    def __init__(self, root_path, response_dir, user_id=None, cache='none', dataset_name=None):
        """
        Args:
            root_path: str, path to images root (e.g., './load/PPS/images')
            response_dir: str, path to responses directory (e.g., './load/PPS/responses/train_toy')
            user_id: str or None, specific user ID from JSON filename
                     (e.g., 'user_response_example10'). If None, load all users.
            cache: str, caching strategy:
                   'none' - Load from disk each time
                   'in_memory' - Load all images into RAM
            dataset_name: str or None, explicit dataset name for UPE cache organization
                         (e.g., 'train_toy', 'validation_small')
                         If None, auto-extracted from response_dir
        """
        super().__init__()

        self.root_path = root_path
        self.response_dir = response_dir
        self.cache = cache
        self.user_id = user_id

        # Auto-detect dataset name from response_dir if not provided
        if dataset_name is None:
            # Extract last directory name from response_dir
            # e.g., './load/PPS/responses/train_toy' -> 'train_toy'
            dataset_name = os.path.basename(os.path.normpath(response_dir))
        self.dataset_name = dataset_name

        # Transforms
        self.to_tensor = transforms.ToTensor()

        # Load all JSON files from response_dir
        json_files = sorted(glob.glob(os.path.join(response_dir, '*.json')))

        # Build sample list
        self.samples = []
        self.user_to_samples = {}  # Map user_id to sample indices

        for json_file in json_files:
            # Extract user ID from JSON filename (not from image folder!)
            # e.g., 'user_response_example10.json' → 'user_response_example10'
            user_name = os.path.splitext(os.path.basename(json_file))[0]

            with open(json_file, 'r') as f:
                responses = json.load(f)

            for image_id, prefs in responses.items():
                # Extract image type/folder from image_id (e.g., 'D1' from 'D1-00002')
                image_type = image_id.split('-')[0]

                # Filter by user_id if specified
                if self.user_id is not None and user_name != self.user_id:
                    continue

                prefer_idx = prefs['prefer']
                non_prefer_idx = prefs['non_prefer']

                # Construct paths
                prefer_path = os.path.join(self.root_path, image_type,
                                          f"{image_id}_{prefer_idx}.jpg")
                non_prefer_path = os.path.join(self.root_path, image_type,
                                              f"{image_id}_{non_prefer_idx}.jpg")

                # Verify files exist
                if not os.path.exists(prefer_path):
                    print(f"Warning: Missing file {prefer_path}")
                    continue
                if not os.path.exists(non_prefer_path):
                    print(f"Warning: Missing file {non_prefer_path}")
                    continue

                # Create sample
                sample = {
                    'image_id': image_id,
                    'user_id': user_name,  # Actual user ID from JSON filename
                    'image_type': image_type,  # D1, E3, etc. for reference
                    'prefer_idx': prefer_idx,
                    'non_prefer_idx': non_prefer_idx,
                    'prefer_path': prefer_path,
                    'non_prefer_path': non_prefer_path,
                    'json_file': json_file,
                }

                # Cache images if requested
                if self.cache == 'in_memory':
                    sample['prefer_img'] = self._load_image(prefer_path)
                    sample['non_prefer_img'] = self._load_image(non_prefer_path)

                sample_idx = len(self.samples)
                self.samples.append(sample)

                # Build user index (by actual user_id, not image_type)
                if user_name not in self.user_to_samples:
                    self.user_to_samples[user_name] = []
                self.user_to_samples[user_name].append(sample_idx)

        # Validate non-empty dataset
        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples found in {response_dir}. "
                f"Checked {len(json_files)} JSON files. "
                f"Please verify JSON files and image paths are correct."
            )

        print(f"Loaded {len(self.samples)} preference pairs from {len(json_files)} JSON files")
        print(f"Dataset name: {self.dataset_name}")
        print(f"Users: {sorted(self.user_to_samples.keys())}")

    def _load_image(self, path):
        """Load image and convert to tensor with error handling.

        Args:
            path: str, path to image file

        Returns:
            torch.Tensor: (3, H, W), values in [0, 1]

        Raises:
            IOError: If image cannot be loaded
        """
        try:
            img = Image.open(path).convert('RGB')
            return self.to_tensor(img)
        except Exception as e:
            raise IOError(f"Failed to load image {path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'image_id': str, image identifier (e.g., 'D1-00002')
                - 'user_id': str, actual user ID from JSON filename (e.g., 'user_response_example10')
                - 'image_type': str, image type/folder (e.g., 'D1', 'E3')
                - 'prefer': torch.Tensor (3, H, W), values in [0, 1]
                - 'non_prefer': torch.Tensor (3, H, W), values in [0, 1]
                - 'prefer_idx': str, style index for prefer
                - 'non_prefer_idx': str, style index for non_prefer
        """
        sample = self.samples[idx]

        # Load images
        if self.cache == 'in_memory':
            prefer_img = sample['prefer_img']
            non_prefer_img = sample['non_prefer_img']
        else:
            prefer_img = self._load_image(sample['prefer_path'])
            non_prefer_img = self._load_image(sample['non_prefer_path'])

        # DEBUG: Check if paths are the same
        if sample['prefer_path'] == sample['non_prefer_path']:
            print(f"[BUG DETECTED] User {sample['user_id']}, Image {sample['image_id']}: prefer_path == non_prefer_path")
            print(f"  Path: {sample['prefer_path']}")
            print(f"  prefer_idx: {sample['prefer_idx']}, non_prefer_idx: {sample['non_prefer_idx']}")

        # DEBUG: Check if images are identical
        if torch.equal(prefer_img, non_prefer_img):
            print(f"[BUG DETECTED] User {sample['user_id']}, Image {sample['image_id']}: Loaded images are IDENTICAL!")
            print(f"  prefer_path: {sample['prefer_path']}")
            print(f"  non_prefer_path: {sample['non_prefer_path']}")

        return {
            'image_id': sample['image_id'],
            'user_id': sample['user_id'],
            'image_type': sample['image_type'],
            'prefer': prefer_img,
            'non_prefer': non_prefer_img,
            'prefer_idx': sample['prefer_idx'],
            'non_prefer_idx': sample['non_prefer_idx'],
        }

    def get_user_ids(self):
        """Get list of unique user IDs in dataset."""
        return list(self.user_to_samples.keys())

    def get_user_samples(self, user_id):
        """Get sample indices for a specific user."""
        return self.user_to_samples.get(user_id, [])

    def split_samples_for_upe(self, num_upe_pairs: int = 16, seed: int = 42) -> Dict[str, Dict]:
        """
        Split samples for each user: first A for UPE extraction, rest for training.

        Uses deterministic splitting based on cryptographic hash to ensure:
        - Same split across multiple runs with same seed
        - No seed collisions between different users
        - Reproducibility across train and validation datasets

        Args:
            num_upe_pairs: Number of pairs to reserve for UPE extraction (default: 16)
            seed: Global random seed for reproducibility (default: 42)

        Returns:
            split_info: Dict mapping user_id to split information:
                {
                    user_id: {
                        'upe_indices': List[int],    # Indices for UPE extraction
                        'train_indices': List[int],  # Indices for training
                        'total': int,                # Total samples for this user
                        'user_seed': int,            # User-specific seed (for debugging)
                    }
                }

        Example:
            >>> dataset = PPSPreferencePairDataset(...)
            >>> split_info = dataset.split_samples_for_upe(num_upe_pairs=16, seed=42)
            >>> print(f"User user_001 has {len(split_info['user_001']['upe_indices'])} UPE pairs")
            >>> print(f"User user_001 has {len(split_info['user_001']['train_indices'])} training pairs")
        """
        split_info = {}

        for user_id in self.get_user_ids():
            user_samples = self.get_user_samples(user_id)

            # Validate sufficient samples
            if len(user_samples) < num_upe_pairs:
                warnings.warn(
                    f"User {user_id} has only {len(user_samples)} samples, "
                    f"need {num_upe_pairs}. SKIPPING this user."
                )
                continue

            # Create deterministic user-specific seed using cryptographic hash
            # This prevents seed collisions and ensures reproducibility
            hash_obj = hashlib.md5(f"{user_id}_{seed}".encode())
            user_seed = int(hash_obj.hexdigest(), 16) % (2**32)

            # Shuffle samples with user-specific seed
            rng = np.random.RandomState(user_seed)
            shuffled_indices = rng.permutation(user_samples).tolist()

            # Split: first num_upe_pairs for UPE, rest for training
            split_info[user_id] = {
                'upe_indices': shuffled_indices[:num_upe_pairs],
                'train_indices': shuffled_indices[num_upe_pairs:],
                'total': len(user_samples),
                'user_seed': user_seed,  # For debugging
            }

        print(f"Split {len(split_info)} users: {num_upe_pairs} pairs for UPE + "
              f"{sum(len(info['train_indices']) for info in split_info.values())} pairs for training")

        return split_info

    def get_upe_samples(self, user_id: str, split_info: Dict[str, Dict]) -> List[Dict]:
        """
        Get UPE extraction samples for a specific user.

        Args:
            user_id: User identifier (e.g., 'user_response_example10')
            split_info: Split information from split_samples_for_upe()

        Returns:
            List of sample dicts (same format as __getitem__)

        Example:
            >>> split_info = dataset.split_samples_for_upe()
            >>> upe_samples = dataset.get_upe_samples('user_001', split_info)
            >>> print(f"Got {len(upe_samples)} UPE samples")
        """
        if user_id not in split_info:
            raise ValueError(f"User {user_id} not found in split_info. "
                           f"Available users: {list(split_info.keys())}")

        upe_indices = split_info[user_id]['upe_indices']
        return [self[idx] for idx in upe_indices]

    def create_train_subset(self, split_info: Dict[str, Dict]) -> 'PPSPreferencePairDataset':
        """
        Create a new dataset containing only training samples (excluding UPE samples).

        This is useful for training to ensure UPE samples are not used in training loss,
        maintaining proper separation between UPE extraction and training data.

        Args:
            split_info: Split information from split_samples_for_upe()

        Returns:
            A new PPSPreferencePairDataset containing only training samples

        Example:
            >>> split_info = dataset.split_samples_for_upe(num_upe_pairs=16)
            >>> train_dataset = dataset.create_train_subset(split_info)
            >>> print(f"Original dataset: {len(dataset)} samples")
            >>> print(f"Training subset: {len(train_dataset)} samples")
        """
        # Collect all training indices
        train_indices = []
        for user_id, info in split_info.items():
            train_indices.extend(info['train_indices'])

        # Sort for deterministic ordering
        train_indices = sorted(train_indices)

        # Create a shallow copy of the dataset
        subset = copy.copy(self)

        # Replace samples with training subset
        subset.samples = [self.samples[i] for i in train_indices]

        # Rebuild user_to_samples mapping
        subset.user_to_samples = {}
        for new_idx, sample in enumerate(subset.samples):
            user_id = sample['user_id']
            if user_id not in subset.user_to_samples:
                subset.user_to_samples[user_id] = []
            subset.user_to_samples[user_id].append(new_idx)

        print(f"Created training subset: {len(subset.samples)} samples "
              f"(original: {len(self.samples)} samples)")

        return subset
