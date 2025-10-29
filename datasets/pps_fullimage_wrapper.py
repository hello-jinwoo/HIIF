"""
PPS Full-Image Wrapper for Evaluation

Loads images at original resolution without resize/crop for patch-based evaluation.
"""

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import make_coord


@register('pps-full-image')
class PPSFullImageWrapper(Dataset):
    """
    Wrapper for PPS evaluation with full-resolution images.

    Processing pipeline:
    1. Load images at original resolution (no resize, no crop)
    2. Optionally apply color augmentation
    3. Create 4 versions: prefer_orig, prefer_aug, non_prefer_orig, non_prefer_aug
    4. Generate coordinates for full-resolution implicit function

    Used for patch-based evaluation to get full-resolution outputs.
    """

    def __init__(self, dataset: Dataset, augment: bool = False,
                 augment_params: Optional[Dict] = None):
        """
        Args:
            dataset: Base dataset (PPSPreferencePairDataset)
            augment: bool, whether to apply color augmentation
            augment_params: dict, augmentation parameters:
                - brightness: float (default: 0.1)
                - contrast: float (default: 0.1)
                - saturation: float (default: 0.1)
                - hue: float (default: 0.05)
                - noise_std: float, Gaussian noise std (default: 0.02)
        """
        super().__init__()
        self.dataset = dataset
        self.augment = augment

        # Parse augmentation parameters
        if augment_params is None:
            augment_params = {}
        brightness = augment_params.get('brightness', 0.1)
        contrast = augment_params.get('contrast', 0.1)
        saturation = augment_params.get('saturation', 0.1)
        hue = augment_params.get('hue', 0.05)
        self.noise_std = augment_params.get('noise_std', 0.02)

        # Color jitter for augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def _apply_color_augmentation(self, img):
        """
        Apply random color augmentation.

        Args:
            img: torch.Tensor (3, H, W), RGB in [0, 1]

        Returns:
            aug_img: torch.Tensor (3, H, W), augmented RGB in [0, 1]
        """
        # Convert to PIL for ColorJitter, then back to tensor
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img_pil = to_pil(img)
        img_aug = self.color_jitter(img_pil)
        img_aug = to_tensor(img_aug)

        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(img_aug) * self.noise_std
            img_aug = (img_aug + noise).clamp(0, 1)

        return img_aug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns batch with 4 input versions and 2 ground truths at full resolution.

        Input versions: prefer_orig, prefer_aug, non_prefer_orig, non_prefer_aug
        Ground truths: prefer (original), non_prefer (original)

        Returns:
            dict with keys:
                - 'user_id': str
                - 'image_id': str
                - 'inp': torch.Tensor (4, 3, H, W) - 4 input versions at full resolution
                - 'gt_prefer': torch.Tensor (3, H, W) - prefer ground truth
                - 'gt_non_prefer': torch.Tensor (3, H, W) - non-prefer ground truth
                - 'coord': torch.Tensor (H, W, 2) - normalized coordinates
                - 'cell': torch.Tensor (2,) - pixel size in normalized space
        """
        # Get base sample (full resolution)
        sample = self.dataset[idx]
        prefer_img = sample['prefer']  # (3, H, W)
        non_prefer_img = sample['non_prefer']  # (3, H, W)

        # Get image dimensions
        H, W = prefer_img.shape[-2:]

        # Create 4 input versions (no resize, no crop)
        versions = []

        # 1. Prefer original
        versions.append(prefer_img)

        # 2. Prefer augmented
        if self.augment:
            prefer_aug = self._apply_color_augmentation(prefer_img)
        else:
            prefer_aug = prefer_img
        versions.append(prefer_aug)

        # 3. Non-prefer original
        versions.append(non_prefer_img)

        # 4. Non-prefer augmented
        if self.augment:
            non_prefer_aug = self._apply_color_augmentation(non_prefer_img)
        else:
            non_prefer_aug = non_prefer_img
        versions.append(non_prefer_aug)

        # Stack input versions
        inp_batch = torch.stack(versions, dim=0)  # (4, 3, H, W)

        # Ground truths at full resolution
        gt_prefer = prefer_img
        gt_non_prefer = non_prefer_img

        # Generate coordinates for full resolution (same resolution - no upsampling)
        coord = make_coord([H, W], flatten=False)
        # coord: (H, W, 2), values in [-1, 1]

        # Cell size (pixel size in normalized space)
        cell = torch.tensor([2 / H, 2 / W], dtype=torch.float32)

        return {
            'user_id': sample['user_id'],
            'image_id': sample['image_id'],
            'inp': inp_batch,           # (4, 3, H, W)
            'gt_prefer': gt_prefer,     # (3, H, W)
            'gt_non_prefer': gt_non_prefer,  # (3, H, W)
            'coord': coord,             # (H, W, 2)
            'cell': cell,               # (2,)
        }
