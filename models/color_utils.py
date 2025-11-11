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


def rgb_to_xyz(rgb):
    """Convert sRGB to CIE XYZ color space (D65 illuminant).

    Args:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]

    Returns:
        xyz: Tensor of shape (B, 3, H, W)
             X in [0, ~0.95]
             Y in [0, 1.0]
             Z in [0, ~1.09]
    """
    # Step 1: sRGB to Linear RGB (inverse gamma correction)
    linear_mask = rgb <= 0.04045
    linear_rgb = torch.where(
        linear_mask,
        rgb / 12.92,
        torch.pow((rgb + 0.055) / 1.055, 2.4)
    )

    # Step 2: Linear RGB to XYZ (D65 illuminant, sRGB primaries)
    # Matrix from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=rgb.dtype, device=rgb.device)

    B, C, H, W = rgb.shape
    linear_rgb_flat = linear_rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz_flat = torch.matmul(linear_rgb_flat, M.T)
    xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    return xyz


def xyz_to_lab(xyz, illuminant='D65'):
    """Convert CIE XYZ to CIE LAB color space.

    Args:
        xyz: Tensor of shape (B, 3, H, W)
        illuminant: str, 'D65' or 'D50'

    Returns:
        lab: Tensor of shape (B, 3, H, W)
             L in [0, 100]
             a in [-128, 127] approximately
             b in [-128, 127] approximately
    """
    # Reference white points (from http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html)
    if illuminant == 'D65':
        ref_white = torch.tensor([0.95047, 1.00000, 1.08883],
                                 dtype=xyz.dtype, device=xyz.device)
    elif illuminant == 'D50':
        ref_white = torch.tensor([0.96422, 1.00000, 0.82521],
                                 dtype=xyz.dtype, device=xyz.device)
    else:
        raise ValueError(f"Unknown illuminant: {illuminant}")

    # Reshape for broadcasting: (3, 1, 1)
    ref_white = ref_white.reshape(3, 1, 1)

    # Normalize by reference white
    xyz_normalized = xyz / ref_white

    # Piecewise function for Lab
    # f(t) = t^(1/3) if t > (6/29)^3
    #      = (1/3) * ((29/6)^2) * t + (4/29) otherwise
    delta = 6.0 / 29.0
    delta_cubed = delta ** 3

    # Compute f(t)
    mask = xyz_normalized > delta_cubed
    f_xyz = torch.where(
        mask,
        torch.pow(xyz_normalized, 1.0 / 3.0),
        (xyz_normalized / (3 * delta ** 2)) + (4.0 / 29.0)
    )

    # Extract f(X), f(Y), f(Z)
    fx = f_xyz[:, 0:1, :, :]
    fy = f_xyz[:, 1:2, :, :]
    fz = f_xyz[:, 2:3, :, :]

    # Compute Lab
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return torch.cat([L, a, b], dim=1)


def rgb_to_lab(rgb, illuminant='D65'):
    """Convert sRGB to CIE LAB color space (GPU-accelerated).

    This is a differentiable, batched implementation that processes
    all images on GPU, providing significant speedup over CPU-based
    scikit-image rgb2lab.

    Args:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]
        illuminant: str, 'D65' (default) or 'D50'

    Returns:
        lab: Tensor of shape (B, 3, H, W)
             L in [0, 100]
             a in [-128, 127] approximately
             b in [-128, 127] approximately
    """
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz, illuminant=illuminant)
    return lab


def lab_to_xyz(lab, illuminant='D65'):
    """Convert CIE LAB to CIE XYZ color space.

    Args:
        lab: Tensor of shape (B, 3, H, W)
        illuminant: str, 'D65' or 'D50'

    Returns:
        xyz: Tensor of shape (B, 3, H, W)
    """
    # Reference white points
    if illuminant == 'D65':
        ref_white = torch.tensor([0.95047, 1.00000, 1.08883],
                                 dtype=lab.dtype, device=lab.device)
    elif illuminant == 'D50':
        ref_white = torch.tensor([0.96422, 1.00000, 0.82521],
                                 dtype=lab.dtype, device=lab.device)
    else:
        raise ValueError(f"Unknown illuminant: {illuminant}")

    ref_white = ref_white.reshape(3, 1, 1)

    L = lab[:, 0:1, :, :]
    a = lab[:, 1:2, :, :]
    b = lab[:, 2:3, :, :]

    # Inverse Lab transform
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    # Stack for easier processing
    f_xyz = torch.cat([fx, fy, fz], dim=1)

    # Inverse piecewise function
    delta = 6.0 / 29.0
    mask = f_xyz > delta
    xyz_normalized = torch.where(
        mask,
        torch.pow(f_xyz, 3.0),
        3 * delta ** 2 * (f_xyz - 4.0 / 29.0)
    )

    # Denormalize by reference white
    xyz = xyz_normalized * ref_white

    return xyz


def lab_to_rgb(lab, illuminant='D65'):
    """Convert CIE LAB to sRGB color space (GPU-accelerated).

    Args:
        lab: Tensor of shape (B, 3, H, W)
        illuminant: str, 'D65' (default) or 'D50'

    Returns:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]
    """
    xyz = lab_to_xyz(lab, illuminant=illuminant)

    # XYZ to Linear RGB (inverse of rgb_to_xyz matrix)
    M_inv = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=xyz.dtype, device=xyz.device)

    B, C, H, W = xyz.shape
    xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    linear_rgb_flat = torch.matmul(xyz_flat, M_inv.T)
    linear_rgb = linear_rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    # Linear RGB to sRGB (gamma correction)
    srgb_mask = linear_rgb <= 0.0031308
    rgb = torch.where(
        srgb_mask,
        12.92 * linear_rgb,
        1.055 * torch.pow(linear_rgb.clamp(min=0), 1.0 / 2.4) - 0.055
    )

    return rgb.clamp(0, 1)


def rgb_to_yuv(rgb):
    """Convert RGB to YUV color space (BT.709 standard, differentiable).

    Uses BT.709 (Rec.709) coefficients for HD content.

    Args:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]

    Returns:
        yuv: Tensor of shape (B, 3, H, W)
             Y in [0, 1] (luma)
             U in [-0.5, 0.5] (blue-difference chroma)
             V in [-0.5, 0.5] (red-difference chroma)
    """
    r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

    # BT.709 coefficients (for HD content)
    # Y = 0.2126*R + 0.7152*G + 0.0722*B
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # U and V are normalized to [-0.5, 0.5] for consistency with training range
    # U = (B - Y) / (2 * (1 - 0.0722)) = (B - Y) / 1.8556
    # V = (R - Y) / (2 * (1 - 0.2126)) = (R - Y) / 1.5748
    u = (b - y) / 1.8556
    v = (r - y) / 1.5748

    return torch.cat([y, u, v], dim=1)


def yuv_to_rgb(yuv):
    """Convert YUV to RGB color space (BT.709 standard, differentiable).

    Args:
        yuv: Tensor of shape (B, 3, H, W)
             Y in [0, 1]
             U in [-0.5, 0.5]
             V in [-0.5, 0.5]

    Returns:
        rgb: Tensor of shape (B, 3, H, W) with values in [0, 1]
    """
    y, u, v = yuv[:, 0:1, :, :], yuv[:, 1:2, :, :], yuv[:, 2:3, :, :]

    # Inverse BT.709 transformation
    # R = Y + V * (2 * (1 - 0.2126)) = Y + V * 1.5748
    # G = Y - U * (2 * 0.0722 * (1 - 0.0722) / 0.7152) - V * (2 * 0.2126 * (1 - 0.2126) / 0.7152)
    #   = Y - U * 0.1873 - V * 0.4681
    # B = Y + U * (2 * (1 - 0.0722)) = Y + U * 1.8556
    r = y + v * 1.5748
    g = y - u * 0.1873 - v * 0.4681
    b = y + u * 1.8556

    rgb = torch.cat([r, g, b], dim=1)
    return rgb.clamp(0, 1)
