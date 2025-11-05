# modified from: https://github.com/yinboc/liif

import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import math

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))


def downsample(img, scale_min=1, scale_max=4, inp_size=None, augment=False, epoch=None):
    if epoch < 20: s = random.randint(scale_min, scale_max)
    s = random.uniform(scale_min, scale_max)
    # print(s)

    if inp_size is None:
        h_lr = math.floor(img.shape[-2] / s + 1e-9)
        w_lr = math.floor(img.shape[-1] / s + 1e-9)
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        img = img[:, :, :h_hr, :w_hr]
        img_down = torch.stack([resize_fn(x, (h_lr, w_lr)) for x in img], dim=0)
        crop_lr, crop_hr = img_down, img
    else:
        h_lr = inp_size
        w_lr = inp_size
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, :, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = torch.stack([resize_fn(x, w_lr) for x in crop_hr], dim=0)

    if augment == True:
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip: x = x.flip(-2)
            if vflip: x = x.flip(-1)
            if dflip: x = x.transpose(-2, -1)
            return x

        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)

    coord = make_coord([h_hr, w_hr], flatten=False)
    coord = coord.unsqueeze(0).expand(img.shape[0], *coord.shape[:2], 2)

    cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32).unsqueeze(0).expand(
        img.shape[0], 2)
    return {
        'inp': crop_lr.contiguous(),
        'coord': coord.contiguous(),
        'cell': cell.contiguous(),
        'gt': crop_hr.contiguous()
    }


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.2f}M'.format(tot / 1e6)
        else:
            return '{:.2f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]

    # Convert string lr to float (handles YAML parsing issues with scientific notation)
    args = optimizer_spec['args'].copy()
    if 'lr' in args and isinstance(args['lr'], str):
        args['lr'] = float(args['lr'])

    optimizer = Optimizer(param_list, **args)
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


# Patch-based image processing utilities for full-resolution inference
from typing import List, Tuple


def extract_patches(image: torch.Tensor, patch_size: int = 128, stride: int = 64) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], Tuple[int, int, int]]:
    """
    Extract overlapping patches from an image using sliding window.

    Args:
        image: torch.Tensor (C, H, W) - Input image
        patch_size: int - Size of each square patch (default: 128)
        stride: int - Stride between patches (default: 64)

    Returns:
        patches: List of torch.Tensor (C, patch_size, patch_size)
        positions: List of (top, left) positions for each patch
        original_shape: Tuple (C, H, W) - Original image shape before padding

    Example:
        >>> image = torch.randn(3, 512, 768)
        >>> patches, positions, orig_shape = extract_patches(image, patch_size=128, stride=64)
        >>> len(patches)  # Number of patches
    """
    # Input validation
    assert image.ndim == 3, f"Expected 3D tensor (C, H, W), got {image.ndim}D"
    assert patch_size > 0, f"patch_size must be positive, got {patch_size}"
    assert stride > 0, f"stride must be positive, got {stride}"

    C, H, W = image.shape
    assert H > 0 and W > 0, f"Invalid image dimensions: ({H}, {W})"

    original_shape = (C, H, W)  # Save before padding
    patches = []
    positions = []

    # Handle small images by padding
    if H < patch_size or W < patch_size:
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        # Pad with replication (safe for any size)
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
        H, W = image.shape[1], image.shape[2]

    # Calculate number of patches needed
    # Ensure we cover the entire image
    for top in range(0, H, stride):
        for left in range(0, W, stride):
            # Adjust patch boundaries if they exceed image dimensions
            bottom = min(top + patch_size, H)
            right = min(left + patch_size, W)

            # Adjust top-left if patch is at image boundary
            actual_top = max(0, bottom - patch_size)
            actual_left = max(0, right - patch_size)

            # Extract patch
            patch = image[:, actual_top:actual_top + patch_size, actual_left:actual_left + patch_size]

            # Ensure patch is exactly patch_size x patch_size
            if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                patches.append(patch)
                positions.append((actual_top, actual_left))

    return patches, positions, original_shape


def stitch_patches(patches: List[torch.Tensor], positions: List[Tuple[int, int]],
                   image_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Stitch patches back into full image with simple averaging in overlap regions.

    Handles cases where patches were extracted from padded images by stitching
    to the padded size first, then cropping to the desired output shape.

    Args:
        patches: List of torch.Tensor (C, patch_size, patch_size)
        positions: List of (top, left) positions corresponding to each patch
        image_shape: Tuple (C, H, W) - Desired output image shape (may be smaller than padded)

    Returns:
        image: torch.Tensor (C, H, W) - Reconstructed full image

    Example:
        >>> patches = [torch.randn(3, 128, 128) for _ in range(10)]
        >>> positions = [(0, 0), (0, 64), ...]
        >>> image = stitch_patches(patches, positions, (3, 512, 768))
    """
    # Input validation
    assert len(patches) > 0, "Empty patch list"
    assert len(patches) == len(positions), f"Mismatch: {len(patches)} patches, {len(positions)} positions"

    C_out, H_out, W_out = image_shape
    device = patches[0].device
    dtype = patches[0].dtype

    # Verify channel consistency
    C_patch = patches[0].shape[0]
    assert C_patch == C_out, f"Channel mismatch: patches have {C_patch} channels, expected {C_out}"

    # Infer padded dimensions from patch positions and sizes
    # The padded size is the maximum extent covered by any patch
    patch_size_h = patches[0].shape[1]
    patch_size_w = patches[0].shape[2]

    max_h = max(top + patch_size_h for top, left in positions)
    max_w = max(left + patch_size_w for top, left in positions)

    # Padded dimensions (at least as large as desired output)
    H_padded = max(max_h, H_out)
    W_padded = max(max_w, W_out)

    # Initialize accumulator and count tensors (padded size)
    accumulated = torch.zeros((C_out, H_padded, W_padded), device=device, dtype=dtype)
    count = torch.zeros((C_out, H_padded, W_padded), device=device, dtype=dtype)

    # Accumulate patches
    for patch, (top, left) in zip(patches, positions):
        patch_h, patch_w = patch.shape[1], patch.shape[2]
        accumulated[:, top:top+patch_h, left:left+patch_w] += patch
        count[:, top:top+patch_h, left:left+patch_w] += 1

    # Average overlapping regions
    # Verify coverage (all pixels in desired output should be covered)
    if __debug__:
        uncovered = (count[:, :H_out, :W_out] == 0).sum().item()
        if uncovered > 0:
            import warnings
            warnings.warn(f"Stitching: {uncovered} pixels in output region not covered by any patch")

    count = count.clamp(min=1)
    image_padded = accumulated / count

    # Crop to desired output size
    image = image_padded[:, :H_out, :W_out]

    return image


def process_image_patchwise(image: torch.Tensor, model_fn, patch_size: int = 128,
                            stride: int = 64, **model_kwargs) -> torch.Tensor:
    """
    Process a full-resolution image using patch-based inference.

    Args:
        image: torch.Tensor (C, H, W) - Input image
        model_fn: Callable that takes a patch and returns processed patch
                 Signature: model_fn(patch, **kwargs) -> processed_patch
        patch_size: int - Size of each square patch
        stride: int - Stride between patches
        **model_kwargs: Additional keyword arguments to pass to model_fn

    Returns:
        output: torch.Tensor (C', H, W) - Processed full image
                C' may differ from C depending on model output

    Example:
        >>> def my_model(patch, coord, cell):
        ...     return patch * 2  # Simple example
        >>> image = torch.randn(3, 512, 768)
        >>> output = process_image_patchwise(image, my_model, patch_size=128, stride=64)
    """
    # Extract patches
    patches, positions, original_shape = extract_patches(image, patch_size, stride)

    # Process each patch
    processed_patches = []
    for patch in patches:
        processed_patch = model_fn(patch, **model_kwargs)
        processed_patches.append(processed_patch)

    # Infer output shape from first processed patch
    C_out = processed_patches[0].shape[0]
    _, H, W = original_shape  # Use original dimensions

    # Stitch patches back together
    output = stitch_patches(processed_patches, positions, (C_out, H, W))

    return output
