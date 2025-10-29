"""
Color space conversion utilities for PPS learning.

All conversions are differentiable for gradient-based optimization.
"""

import torch


def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space (differentiable).

    Args:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]

    Returns:
        hsv: Tensor of shape (B, 3, H, W)
             H in [0, 1] (normalized from [0°, 360°])
             S in [0, 1]
             V in [0, 1]
    """
    r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

    max_rgb, argmax_rgb = rgb.max(dim=1, keepdim=True)
    min_rgb = rgb.min(dim=1, keepdim=True)[0]
    diff = max_rgb - min_rgb
    diff_clamped = torch.clamp(diff, min=1e-8)

    # Saturation
    s = diff / torch.clamp(max_rgb, min=1e-8)
    s = torch.where(max_rgb > 0, s, torch.zeros_like(s))

    # Hue
    rc = (max_rgb - r) / diff_clamped
    gc = (max_rgb - g) / diff_clamped
    bc = (max_rgb - b) / diff_clamped

    h = torch.zeros_like(s)
    h = torch.where(argmax_rgb == 0, bc - gc, h)
    h = torch.where(argmax_rgb == 1, 2.0 + rc - bc, h)
    h = torch.where(argmax_rgb == 2, 4.0 + gc - rc, h)
    h = h / 6.0
    h = h % 1.0
    h = torch.where(diff > 0, h, torch.zeros_like(h))

    # Value
    v = max_rgb

    return torch.cat([h, s, v], dim=1)


def hsv_to_rgb(hsv):
    """Convert HSV to RGB color space (differentiable).

    Args:
        hsv: Tensor of shape (B, 3, H, W)
             H in [0, 1]
             S in [0, 1]
             V in [0, 1]

    Returns:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]
    """
    h, s, v = hsv[:, 0:1, :, :], hsv[:, 1:2, :, :], hsv[:, 2:3, :, :]
    h = h * 6.0  # Scale to [0, 6]

    c = v * s
    x = c * (1 - torch.abs(h % 2.0 - 1))
    m = v - c

    h_floor = h.floor()

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    # Compute RGB based on hue sector
    mask = (h_floor == 0)
    r = torch.where(mask, c, r)
    g = torch.where(mask, x, g)

    mask = (h_floor == 1)
    r = torch.where(mask, x, r)
    g = torch.where(mask, c, g)

    mask = (h_floor == 2)
    g = torch.where(mask, c, g)
    b = torch.where(mask, x, b)

    mask = (h_floor == 3)
    g = torch.where(mask, x, g)
    b = torch.where(mask, c, b)

    mask = (h_floor == 4)
    r = torch.where(mask, x, r)
    b = torch.where(mask, c, b)

    mask = (h_floor >= 5)
    r = torch.where(mask, c, r)
    b = torch.where(mask, x, b)

    rgb = torch.cat([r + m, g + m, b + m], dim=1)
    return rgb.clamp(0, 1)


def rgb_to_oklab(rgb):
    """Convert RGB to OKLab color space (Björn Ottosson, 2020).

    OKLab is a perceptually uniform color space designed for image processing.

    Args:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]

    Returns:
        oklab: Tensor of shape (B, 3, H, W)
               L in [0, 1] approximately
               a in [-0.5, 0.5] approximately
               b in [-0.5, 0.5] approximately
    """
    # Step 1: sRGB to Linear RGB (gamma correction)
    linear_mask = rgb <= 0.04045
    linear_rgb = torch.where(
        linear_mask,
        rgb / 12.92,
        torch.pow((rgb + 0.055) / 1.055, 2.4)
    )

    # Step 2: Linear RGB to LMS (cone response)
    M1 = torch.tensor([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ], dtype=rgb.dtype, device=rgb.device)

    B, C, H, W = rgb.shape
    linear_rgb_flat = linear_rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    lms = torch.matmul(linear_rgb_flat, M1.T)

    # Step 3: Nonlinear transform (cube root for perceptual uniformity)
    # LMS is always non-negative for valid sRGB inputs (RGB ∈ [0, 1])
    # Proof: M1 has all positive coefficients, so LMS = linear_RGB @ M1.T ≥ 0
    # Therefore sign() and abs() operations are unnecessary and removed for simplicity
    lms_ = torch.pow(lms.clamp(min=1e-10), 1.0 / 3.0)

    # Step 4: LMS to OKLab
    M2 = torch.tensor([
        [0.2104542553,  0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050,  0.4505937099],
        [0.0259040371,  0.7827717662, -0.8086757660]
    ], dtype=rgb.dtype, device=rgb.device)

    oklab_flat = torch.matmul(lms_, M2.T)
    oklab = oklab_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    return oklab


def oklab_to_rgb(oklab):
    """Convert OKLab to RGB color space.

    Args:
        oklab: Tensor of shape (B, 3, H, W)

    Returns:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]
    """
    # Step 1: OKLab to LMS
    M2_inv = torch.tensor([
        [1.0000000000,  0.3963377774,  0.2158037573],
        [1.0000000000, -0.1055613458, -0.0638541728],
        [1.0000000000, -0.0894841775, -1.2914855480]
    ], dtype=oklab.dtype, device=oklab.device)

    B, C, H, W = oklab.shape
    oklab_flat = oklab.permute(0, 2, 3, 1).reshape(-1, 3)
    lms_ = torch.matmul(oklab_flat, M2_inv.T)

    # Step 2: Inverse cube root
    lms = lms_.pow(3.0)

    # Step 3: LMS to Linear RGB
    M1_inv = torch.tensor([
        [ 4.0767416621, -3.3077115913,  0.2309699292],
        [-1.2684380046,  2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147,  1.7076147010]
    ], dtype=oklab.dtype, device=oklab.device)

    linear_rgb_flat = torch.matmul(lms, M1_inv.T)
    linear_rgb = linear_rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    # Step 4: Linear RGB to sRGB (inverse gamma correction)
    srgb_mask = linear_rgb <= 0.0031308
    rgb = torch.where(
        srgb_mask,
        12.92 * linear_rgb,
        1.055 * torch.pow(linear_rgb, 1.0 / 2.4) - 0.055
    )

    return rgb.clamp(0, 1)
