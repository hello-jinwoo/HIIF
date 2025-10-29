"""
PPS Preference Pair Dataset
Loads user preference data and corresponding image pairs.
"""

import os
import json
import glob
from PIL import Image
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

    def __init__(self, root_path, response_dir, user_id=None, cache='none'):
        """
        Args:
            root_path: str, path to images root (e.g., './load/PPS/images')
            response_dir: str, path to responses directory (e.g., './load/PPS/responses/train')
            user_id: str or None, specific user ID from JSON filename
                     (e.g., 'user_response_example10'). If None, load all users.
            cache: str, caching strategy:
                   'none' - Load from disk each time
                   'in_memory' - Load all images into RAM
        """
        super().__init__()

        self.root_path = root_path
        self.cache = cache
        self.user_id = user_id

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
