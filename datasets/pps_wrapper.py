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
                 augment_params: Optional[Dict] = None,
                 patches_per_image: int = 1):
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
            patches_per_image: int, number of patches to extract per image (default: 1)
                - When > 1, extracts multiple random crops from the same image
                - More efficient data loading (fewer image I/O operations)
        """
        super().__init__()
        self.dataset = dataset
        self.resize_range = resize_range
        self.crop_size = crop_size
        self.augment = augment
        self.patches_per_image = patches_per_image

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

        When patches_per_image > 1, extracts multiple random crops from the same image.

        Returns:
            dict with keys:
                - 'user_id': str or list[str]
                - 'image_id': str or list[str]
                - 'inp': torch.Tensor
                    - If patches_per_image=1: (4, 3, crop_size, crop_size)
                    - If patches_per_image>1: (N, 4, 3, crop_size, crop_size)
                - 'gt_prefer': torch.Tensor
                    - If patches_per_image=1: (3, crop_size, crop_size)
                    - If patches_per_image>1: (N, 3, crop_size, crop_size)
                - 'gt_non_prefer': torch.Tensor (same shape as gt_prefer)
                - 'coord': torch.Tensor (crop_size, crop_size, 2) or (N, crop_size, crop_size, 2)
                - 'cell': torch.Tensor (2,) or (N, 2)
        """
        # Get base sample
        sample = self.dataset[idx]
        prefer_img = sample['prefer']
        non_prefer_img = sample['non_prefer']

        # Random resize (same for both images)
        target_size = random.randint(self.resize_range[0], self.resize_range[1])
        prefer_img = self._resize_image(prefer_img, target_size)
        non_prefer_img = self._resize_image(non_prefer_img, target_size)

        # Extract multiple patches if requested
        if self.patches_per_image > 1:
            all_inp = []
            all_gt_prefer = []
            all_gt_non_prefer = []

            for _ in range(self.patches_per_image):
                # Random crop (independent position for each patch)
                prefer_crop, non_prefer_crop = self._random_crop_pair(
                    prefer_img, non_prefer_img, self.crop_size
                )

                # Create 4 input versions with independent augmentation
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

                all_inp.append(inp_batch)
                all_gt_prefer.append(prefer_crop)
                all_gt_non_prefer.append(non_prefer_crop)

            # Stack all patches
            inp_batch = torch.stack(all_inp, dim=0)  # (N, 4, 3, crop_size, crop_size)
            gt_prefer = torch.stack(all_gt_prefer, dim=0)  # (N, 3, crop_size, crop_size)
            gt_non_prefer = torch.stack(all_gt_non_prefer, dim=0)  # (N, 3, crop_size, crop_size)

            # Coordinates and cell (same for all patches)
            coord = make_coord([self.crop_size, self.crop_size], flatten=False)
            coord = coord.unsqueeze(0).expand(self.patches_per_image, -1, -1, -1)  # (N, H, W, 2)
            cell = torch.tensor([2 / self.crop_size, 2 / self.crop_size], dtype=torch.float32)
            cell = cell.unsqueeze(0).expand(self.patches_per_image, -1)  # (N, 2)

            # Metadata (repeat for each patch)
            user_ids = [sample['user_id']] * self.patches_per_image
            image_ids = [sample['image_id']] * self.patches_per_image

            return {
                'user_id': user_ids,
                'image_id': image_ids,
                'inp': inp_batch,
                'gt_prefer': gt_prefer,
                'gt_non_prefer': gt_non_prefer,
                'coord': coord,
                'cell': cell,
            }

        else:
            # Original behavior (single crop)
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

            # Ground truths
            gt_prefer = prefer_crop
            gt_non_prefer = non_prefer_crop

            # Generate coordinates (same resolution - no upsampling)
            coord = make_coord([self.crop_size, self.crop_size], flatten=False)
            cell = torch.tensor([2 / self.crop_size, 2 / self.crop_size], dtype=torch.float32)

            return {
                'user_id': sample['user_id'],
                'image_id': sample['image_id'],
                'inp': inp_batch,
                'gt_prefer': gt_prefer,
                'gt_non_prefer': gt_non_prefer,
                'coord': coord,
                'cell': cell,
            }


def pps_multi_crop_collate_fn(batch):
    """
    Custom collate function for multi-crop batching.

    Handles both single-crop (patches_per_image=1) and multi-crop (patches_per_image>1) cases.

    Args:
        batch: list of dict, each dict is a sample returned by PPSColorAugmentedWrapper

    Returns:
        dict with batched tensors

    Shape transformation for multi-crop:
        Input:  batch_image items, each with shape (N, 4, 3, H, W)
        Output: single batch with shape (batch_size, 4, 3, H, W)
                where batch_size = batch_image × N
    """
    # Check if multi-crop (first item has list of user_ids)
    is_multi_crop = isinstance(batch[0]['user_id'], list)

    if is_multi_crop:
        # Flatten multi-crop batches
        # Each item has shape (N, ...), stack and flatten to (batch_image*N, ...)
        all_user_ids = []
        all_image_ids = []
        all_inp = []
        all_gt_prefer = []
        all_gt_non_prefer = []
        all_coord = []
        all_cell = []

        for item in batch:
            # item['user_id'] is a list of N user_ids
            # item['inp'] has shape (N, 4, 3, H, W)
            patches_per_image = len(item['user_id'])

            all_user_ids.extend(item['user_id'])
            all_image_ids.extend(item['image_id'])

            # Flatten first dimension (N) into the batch
            for i in range(patches_per_image):
                all_inp.append(item['inp'][i])
                all_gt_prefer.append(item['gt_prefer'][i])
                all_gt_non_prefer.append(item['gt_non_prefer'][i])
                all_coord.append(item['coord'][i])
                all_cell.append(item['cell'][i])

        # Stack all patches
        return {
            'user_id': all_user_ids,
            'image_id': all_image_ids,
            'inp': torch.stack(all_inp, dim=0),
            'gt_prefer': torch.stack(all_gt_prefer, dim=0),
            'gt_non_prefer': torch.stack(all_gt_non_prefer, dim=0),
            'coord': torch.stack(all_coord, dim=0),
            'cell': torch.stack(all_cell, dim=0),
        }
    else:
        # Single-crop: use default PyTorch collate behavior
        # Stack each field
        return {
            'user_id': [item['user_id'] for item in batch],
            'image_id': [item['image_id'] for item in batch],
            'inp': torch.stack([item['inp'] for item in batch], dim=0),
            'gt_prefer': torch.stack([item['gt_prefer'] for item in batch], dim=0),
            'gt_non_prefer': torch.stack([item['gt_non_prefer'] for item in batch], dim=0),
            'coord': torch.stack([item['coord'] for item in batch], dim=0),
            'cell': torch.stack([item['cell'] for item in batch], dim=0),
        }
