"""
PPS Color Augmented Wrapper
Applies color augmentation, preprocessing, and batch construction for PPS training.
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import make_coord


@register('pps-color-augmented')
class PPSColorAugmentedWrapper(Dataset):
    """
    Wrapper for PPS training with color augmentation and preprocessing.

    Processing pipeline:
    1. Random resize (short side in [resize_range[0], resize_range[1]])
    2. Random crop to crop_size x crop_size
    3. Color augmentation (HSV perturbations) if enabled
    4. Create 4 versions: prefer_orig, prefer_aug, non_prefer_orig, non_prefer_aug
    5. Generate coordinates for implicit function
    """

    def __init__(self, dataset: Dataset, resize_range: List[int] = [256, 1024],
                 crop_size: int = 128, augment: bool = True,
                 augment_params: Optional[Dict] = None):
        """
        Args:
            dataset: Base dataset (PPSPreferencePairDataset)
            resize_range: [int, int], range for random resize on short side
            crop_size: int, size of random crop
            augment: bool, whether to apply color augmentation
            augment_params: dict, augmentation parameters:
                - hue_shift: float, max hue shift in [-h, h] (default: 0.1)
                - sat_scale: [float, float], saturation scale range (default: [0.8, 1.2])
                - val_scale: [float, float], value scale range (default: [0.9, 1.1])
                - noise_std: float, Gaussian noise std (default: 0.02)
        """
        super().__init__()
        self.dataset = dataset
        self.resize_range = resize_range
        self.crop_size = crop_size
        self.augment = augment

        # Default augmentation parameters
        if augment_params is None:
            augment_params = {}
        self.hue_shift = augment_params.get('hue_shift', 0.1)
        self.sat_scale = augment_params.get('sat_scale', [0.8, 1.2])
        self.val_scale = augment_params.get('val_scale', [0.9, 1.1])
        self.noise_std = augment_params.get('noise_std', 0.02)

        # Color jitter for augmentation (simpler alternative to manual HSV)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.2,
            hue=0.1
        )

    def _resize_image(self, img, target_size):
        """
        Resize image so that the shorter side equals target_size.

        Args:
            img: torch.Tensor (3, H, W)
            target_size: int, short side target size

        Returns:
            resized_img: torch.Tensor (3, H', W')
        """
        h, w = img.shape[-2:]
        if h < w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)

        img = img.unsqueeze(0)  # (1, 3, H, W)
        img = F.interpolate(img, size=(new_h, new_w), mode='bicubic', align_corners=False)
        return img.squeeze(0)  # (3, H', W')

    def _random_crop(self, img, crop_size):
        """
        Random crop from image.

        Args:
            img: torch.Tensor (3, H, W)
            crop_size: int

        Returns:
            cropped_img: torch.Tensor (3, crop_size, crop_size)
        """
        h, w = img.shape[-2:]

        if h < crop_size or w < crop_size:
            # Pad if necessary
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            h, w = img.shape[-2:]

        # Random crop position
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)

        return img[:, top:top+crop_size, left:left+crop_size]

    def _random_crop_pair(self, img1, img2, crop_size):
        """
        Apply same random crop to two images.

        Args:
            img1: torch.Tensor (3, H, W)
            img2: torch.Tensor (3, H, W)
            crop_size: int

        Returns:
            crop1: torch.Tensor (3, crop_size, crop_size)
            crop2: torch.Tensor (3, crop_size, crop_size)
        """
        h1, w1 = img1.shape[-2:]
        h2, w2 = img2.shape[-2:]

        # Pad if necessary
        if h1 < crop_size or w1 < crop_size:
            pad_h = max(0, crop_size - h1)
            pad_w = max(0, crop_size - w1)
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='reflect')
            h1, w1 = img1.shape[-2:]

        if h2 < crop_size or w2 < crop_size:
            pad_h = max(0, crop_size - h2)
            pad_w = max(0, crop_size - w2)
            img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode='reflect')
            h2, w2 = img2.shape[-2:]

        # Single random crop position (use minimum dimensions to ensure valid for both)
        h_min = min(h1, h2)
        w_min = min(w1, w2)
        top = random.randint(0, h_min - crop_size)
        left = random.randint(0, w_min - crop_size)

        crop1 = img1[:, top:top+crop_size, left:left+crop_size]
        crop2 = img2[:, top:top+crop_size, left:left+crop_size]

        return crop1, crop2

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
        Returns batch with 4 input versions and 2 ground truths per sample.

        Input versions: prefer_orig, prefer_aug, non_prefer_orig, non_prefer_aug
        Ground truths: prefer (original), non_prefer (original)

        Each input is evaluated against BOTH ground truths during training:
        L_total = w_p * L1(pred, gt_prefer) + (1-w_p) * L1(pred, gt_non_prefer)

        Returns:
            dict with keys:
                - 'user_id': str
                - 'image_id': str
                - 'inp': torch.Tensor (4, 3, crop_size, crop_size) - 4 input versions
                - 'gt_prefer': torch.Tensor (3, crop_size, crop_size) - prefer ground truth
                - 'gt_non_prefer': torch.Tensor (3, crop_size, crop_size) - non-prefer ground truth
                - 'coord': torch.Tensor (crop_size, crop_size, 2) - normalized coordinates
                - 'cell': torch.Tensor (2,) - pixel size in normalized space
        """
        # Get base sample
        sample = self.dataset[idx]
        prefer_img = sample['prefer']
        non_prefer_img = sample['non_prefer']

        # Random resize (same for both images)
        target_size = random.randint(self.resize_range[0], self.resize_range[1])
        prefer_img = self._resize_image(prefer_img, target_size)
        non_prefer_img = self._resize_image(non_prefer_img, target_size)

        # Random crop (same position for both images)
        prefer_crop, non_prefer_crop = self._random_crop_pair(
            prefer_img, non_prefer_img, self.crop_size
        )

        # Create 4 input versions
        versions = []

        # 1. Prefer original
        versions.append(prefer_crop)

        # 2. Prefer augmented
        if self.augment:
            prefer_aug = self._apply_color_augmentation(prefer_crop)
        else:
            prefer_aug = prefer_crop
        versions.append(prefer_aug)

        # 3. Non-prefer original
        versions.append(non_prefer_crop)

        # 4. Non-prefer augmented
        if self.augment:
            non_prefer_aug = self._apply_color_augmentation(non_prefer_crop)
        else:
            non_prefer_aug = non_prefer_crop
        versions.append(non_prefer_aug)

        # Stack input versions
        inp_batch = torch.stack(versions, dim=0)  # (4, 3, crop_size, crop_size)

        # Ground truths: Only 2 unique GTs (prefer and non-prefer)
        # Each input will compute loss against BOTH GTs
        # L_total = w_p * L1(pred, gt_prefer) + (1-w_p) * L1(pred, gt_non_prefer)
        gt_prefer = prefer_crop      # (3, H, W)
        gt_non_prefer = non_prefer_crop  # (3, H, W)

        # DEBUG: Check if GTs are the same
        if torch.equal(gt_prefer, gt_non_prefer):
            print(f"[BUG DETECTED] User {sample['user_id']}, Image {sample['image_id']}: GT prefer and non_prefer are IDENTICAL!")
            print(f"  prefer_idx: {sample.get('prefer_idx', 'N/A')}, non_prefer_idx: {sample.get('non_prefer_idx', 'N/A')}")
        mse_diff = F.mse_loss(gt_prefer, gt_non_prefer).item()
        if mse_diff < 1e-6:
            print(f"[WARNING] User {sample['user_id']}, Image {sample['image_id']}: GTs are nearly identical (MSE: {mse_diff:.8f})")

        # Generate coordinates (same resolution - no upsampling)
        coord = make_coord([self.crop_size, self.crop_size], flatten=False)
        # coord: (crop_size, crop_size, 2), values in [-1, 1]

        # Cell size (pixel size in normalized space)
        cell = torch.tensor([2 / self.crop_size, 2 / self.crop_size], dtype=torch.float32)

        return {
            'user_id': sample['user_id'],
            'image_id': sample['image_id'],
            'inp': inp_batch,           # (4, 3, H, W)
            'gt_prefer': gt_prefer,     # (3, H, W)
            'gt_non_prefer': gt_non_prefer,  # (3, H, W)
            'coord': coord,             # (H, W, 2)
            'cell': cell,               # (2,)
        }
