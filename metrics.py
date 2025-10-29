"""
Metrics module for PPS validation
Implements full-reference evaluation metrics: PSNR, SSIM, LPIPS, CIEDE2000, Delta E variants
"""

import torch
import numpy as np
from utils import calc_psnr, Averager


def calc_ssim(pred, gt, window_size=11, data_range=1.0):
    """
    Calculate SSIM using pytorch-msssim

    Args:
        pred: torch.Tensor (B, 3, H, W), predicted images in [0, 1]
        gt: torch.Tensor (B, 3, H, W), ground truth images in [0, 1]
        window_size: int, default 11 (must be odd)
        data_range: float, default 1.0

    Returns:
        float: SSIM value in [0, 1], higher is better
    """
    try:
        from pytorch_msssim import ssim
    except ImportError:
        raise ImportError("pytorch-msssim not installed. Run: pip install pytorch-msssim")

    # SSIM requires minimum dimension size
    h, w = pred.shape[-2:]
    if h < window_size or w < window_size:
        window_size = min(h, w)
        if window_size % 2 == 0:
            window_size -= 1

    ssim_val = ssim(pred, gt,
                    data_range=data_range,
                    size_average=True,
                    win_size=window_size)
    return ssim_val.item()


def calc_lpips(pred, gt, net='alex', device='cuda'):
    """
    Calculate LPIPS perceptual distance

    Args:
        pred: torch.Tensor (B, 3, H, W), predicted images in [0, 1]
        gt: torch.Tensor (B, 3, H, W), ground truth images in [0, 1]
        net: str, backbone network ('alex', 'vgg')
        device: str, 'cuda' or 'cpu'

    Returns:
        float: LPIPS distance in [0, 1], lower is better
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips not installed. Run: pip install lpips")

    # Initialize model (will be cached in future calls)
    if not hasattr(calc_lpips, 'loss_fn'):
        calc_lpips.loss_fn = lpips.LPIPS(net=net)
        if device == 'cuda' and torch.cuda.is_available():
            calc_lpips.loss_fn = calc_lpips.loss_fn.cuda()
        calc_lpips.loss_fn.eval()

    with torch.no_grad():
        # Transform from [0, 1] to [-1, 1]
        pred_norm = pred * 2.0 - 1.0
        gt_norm = gt * 2.0 - 1.0

        lpips_val = calc_lpips.loss_fn(pred_norm, gt_norm)
        return lpips_val.mean().item()


def calc_delta_e_cie76(pred, gt):
    """
    Calculate CIE76 Delta E (Euclidean distance in LAB space).
    Also known as ΔE*ab or ΔE76. This is the original 1976 color difference formula.

    This is faster than CIEDE2000 but less perceptually accurate.
    Legacy industry standard, still widely used in simple applications.

    Args:
        pred: torch.Tensor (B, 3, H, W), predicted images in [0, 1]
        gt: torch.Tensor (B, 3, H, W), ground truth images in [0, 1]

    Returns:
        float: Mean Delta E (LAB distance), lower is better

    Expected ranges for PPS:
        - Excellent: < 5.0 (very similar colors)
        - Good: 5.0-10.0 (small difference)
        - Acceptable: 10.0-15.0 (noticeable)
        - Poor: > 15.0 (large difference)
    """
    try:
        from skimage import color
    except ImportError:
        raise ImportError("scikit-image not installed. Run: pip install scikit-image")

    # Convert to numpy and transpose to (B, H, W, 3)
    pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
    gt_np = gt.cpu().numpy().transpose(0, 2, 3, 1)

    delta_e_vals = []
    for i in range(pred_np.shape[0]):
        # Convert RGB to LAB
        pred_lab = color.rgb2lab(pred_np[i])
        gt_lab = color.rgb2lab(gt_np[i])

        # Calculate Euclidean distance in LAB space
        delta_e = np.sqrt(np.sum((pred_lab - gt_lab)**2, axis=-1))
        delta_e_vals.append(np.mean(delta_e))

    return np.mean(delta_e_vals)


# Backward compatibility alias
def calc_delta_e_lab(pred, gt):
    """Deprecated: Use calc_delta_e_cie76 instead. Kept for backward compatibility."""
    return calc_delta_e_cie76(pred, gt)


def calc_ciede2000(pred, gt):
    """
    Calculate CIEDE2000 (CIE Delta E 2000) perceptual color difference
    This is the proper perceptual metric with weighted lightness, chroma, and hue.

    Args:
        pred: torch.Tensor (B, 3, H, W), predicted images in [0, 1]
        gt: torch.Tensor (B, 3, H, W), ground truth images in [0, 1]

    Returns:
        float: Mean CIEDE2000 (ΔE00), lower is better

    Expected ranges for PPS:
        - Excellent: < 2.3 (imperceptible)
        - Good: 2.3-5.0 (small difference)
        - Acceptable: 5.0-8.0 (noticeable)
        - Poor: > 8.0 (large difference)
    """
    try:
        from skimage.color import rgb2lab, deltaE_ciede2000
    except ImportError:
        raise ImportError("scikit-image not installed. Run: pip install scikit-image")

    # Convert to numpy and transpose to (B, H, W, 3)
    pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
    gt_np = gt.cpu().numpy().transpose(0, 2, 3, 1)

    delta_e_vals = []
    for i in range(pred_np.shape[0]):
        # Convert RGB to LAB
        pred_lab = rgb2lab(pred_np[i])
        gt_lab = rgb2lab(gt_np[i])

        # Calculate true CIEDE2000 perceptual distance
        delta_e = deltaE_ciede2000(pred_lab, gt_lab)
        delta_e_vals.append(np.mean(delta_e))

    return np.mean(delta_e_vals)


def calc_delta_e_oklab(pred, gt):
    """
    Calculate OKLab Delta E (Euclidean distance in OKLab space).

    OKLab (Ottosson, 2020) is a modern perceptually uniform color space
    designed for image processing. Faster than CIEDE2000 and CAM16-UCS,
    with good perceptual accuracy.

    Args:
        pred: torch.Tensor (B, 3, H, W), predicted images in [0, 1]
        gt: torch.Tensor (B, 3, H, W), ground truth images in [0, 1]

    Returns:
        float: Mean OKLab Delta E, lower is better

    Expected ranges for PPS:
        - Excellent: < 0.02 (very similar colors)
        - Good: 0.02-0.05 (small difference)
        - Acceptable: 0.05-0.10 (noticeable)
        - Poor: > 0.10 (large difference)

    Note: OKLab has different scale than CIELAB, so values are smaller.
    """
    try:
        from models.color_utils import rgb_to_oklab
    except ImportError:
        raise ImportError("color_utils not found. Check models/color_utils.py")

    # Convert to OKLab
    pred_oklab = rgb_to_oklab(pred)
    gt_oklab = rgb_to_oklab(gt)

    # Calculate Euclidean distance in OKLab space
    # pred_oklab: (B, 3, H, W) where channels are [L, a, b]
    delta_e = torch.sqrt(torch.sum((pred_oklab - gt_oklab)**2, dim=1))  # (B, H, W)

    return delta_e.mean().item()


def calc_delta_e_cam16_ucs(pred, gt):
    """
    Calculate CAM16-UCS Delta E (Euclidean distance in CAM16-UCS space).

    CAM16-UCS (Uniform Color Space) is based on the CAM16 color appearance model.
    It's the successor to CAM02-UCS and provides the most accurate perceptual
    color difference measurement. Slower than other methods but most reliable.

    Args:
        pred: torch.Tensor (B, 3, H, W), predicted images in [0, 1]
        gt: torch.Tensor (B, 3, H, W), ground truth images in [0, 1]

    Returns:
        float: Mean CAM16-UCS Delta E, lower is better

    Expected ranges for PPS:
        - Excellent: < 2.0 (imperceptible)
        - Good: 2.0-4.0 (small difference)
        - Acceptable: 4.0-8.0 (noticeable)
        - Poor: > 8.0 (large difference)

    Note: Requires colour-science library.
    """
    try:
        from colour import XYZ_to_sRGB, sRGB_to_XYZ
        from colour.models import XYZ_to_CAM16UCS, CAM16UCS_to_XYZ
    except ImportError:
        raise ImportError("colour-science not installed. Run: pip install colour-science")

    # Convert to numpy and transpose to (B, H, W, 3)
    pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
    gt_np = gt.cpu().numpy().transpose(0, 2, 3, 1)

    delta_e_vals = []
    for i in range(pred_np.shape[0]):
        # Convert sRGB to XYZ
        # colour library expects input in [0, 1] range
        pred_xyz = sRGB_to_XYZ(pred_np[i])
        gt_xyz = sRGB_to_XYZ(gt_np[i])

        # Convert XYZ to CAM16-UCS
        # Using default viewing conditions (D65 illuminant)
        pred_cam16 = XYZ_to_CAM16UCS(pred_xyz)
        gt_cam16 = XYZ_to_CAM16UCS(gt_xyz)

        # Calculate Euclidean distance in CAM16-UCS space
        delta_e = np.sqrt(np.sum((pred_cam16 - gt_cam16)**2, axis=-1))
        delta_e_vals.append(np.mean(delta_e))

    return np.mean(delta_e_vals)




class MetricsAggregator:
    """Aggregates metrics across multiple samples"""

    def __init__(self, metric_names):
        """
        Args:
            metric_names: list of str, metric names to track
        """
        self.metric_names = metric_names
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.metrics = {name: Averager() for name in self.metric_names}

    def add(self, metric_dict, n=1):
        """
        Add metric values
        Args:
            metric_dict: dict, {metric_name: value}
            n: int, number of samples
        """
        for name, value in metric_dict.items():
            if name in self.metrics:
                self.metrics[name].add(value, n)

    def get(self):
        """Get current average values"""
        return {name: avg.item() for name, avg in self.metrics.items()}


class PPSMetrics:
    """Container for all PPS validation metrics"""

    def __init__(self, device='cuda', enabled_metrics=None):
        """
        Initialize all metric calculators

        Args:
            device: str, 'cuda' or 'cpu'
            enabled_metrics: list of str, metrics to enable. If None, enable all available.
                Available options: ['psnr', 'ssim', 'lpips', 'delta_e_cie76',
                'delta_e_oklab', 'ciede2000', 'delta_e_cam16', 'niqe']
        """
        self.device = device
        self.enabled_metrics = enabled_metrics

        # Check available metrics (library dependencies)
        self.available_metrics = ['psnr']  # PSNR always available

        try:
            from pytorch_msssim import ssim
            self.available_metrics.append('ssim')
        except ImportError:
            pass

        try:
            import lpips
            self.available_metrics.append('lpips')
        except ImportError:
            pass


        try:
            from skimage import color
            self.available_metrics.append('delta_e_cie76')
            self.available_metrics.append('delta_e_lab')  # Backward compatibility
            self.available_metrics.append('ciede2000')
        except ImportError:
            pass

        try:
            from models.color_utils import rgb_to_oklab
            self.available_metrics.append('delta_e_oklab')
        except ImportError:
            pass

        try:
            from colour import sRGB_to_XYZ
            from colour.models import XYZ_to_CAM16UCS
            self.available_metrics.append('delta_e_cam16')
        except ImportError:
            pass

    def _is_enabled(self, metric_name):
        """Check if a metric is enabled and available."""
        if self.enabled_metrics is None:
            # If no specific metrics enabled, use all available
            return metric_name in self.available_metrics
        # Check if enabled by config and available (library installed)
        return metric_name in self.enabled_metrics and metric_name in self.available_metrics

    def compute_all(self, pred, gt):
        """
        Compute all enabled and available metrics

        Args:
            pred: torch.Tensor (B, 3, H, W), predicted images in [0, 1]
            gt: torch.Tensor (B, 3, H, W), ground truth images in [0, 1]

        Returns:
            dict: {metric_name: value}
        """
        metrics = {}

        # PSNR (always available)
        if self._is_enabled('psnr'):
            metrics['psnr'] = calc_psnr(pred, gt).item()

        # SSIM
        if self._is_enabled('ssim'):
            try:
                metrics['ssim'] = calc_ssim(pred, gt)
            except Exception as e:
                print(f"Warning: SSIM computation failed: {e}")

        # LPIPS
        if self._is_enabled('lpips'):
            try:
                metrics['lpips'] = calc_lpips(pred, gt, device=self.device)
            except Exception as e:
                print(f"Warning: LPIPS computation failed: {e}")

        # Delta E CIE76 (LAB Euclidean)
        if self._is_enabled('delta_e_cie76') or self._is_enabled('delta_e_lab'):
            try:
                metrics['delta_e_cie76'] = calc_delta_e_cie76(pred, gt)
                # Backward compatibility
                if self._is_enabled('delta_e_lab'):
                    metrics['delta_e_lab'] = metrics['delta_e_cie76']
            except Exception as e:
                print(f"Warning: Delta E CIE76 computation failed: {e}")

        # Delta E OKLab
        if self._is_enabled('delta_e_oklab'):
            try:
                metrics['delta_e_oklab'] = calc_delta_e_oklab(pred, gt)
            except Exception as e:
                print(f"Warning: Delta E OKLab computation failed: {e}")

        # CIEDE2000 (CIE Delta E 2000)
        if self._is_enabled('ciede2000'):
            try:
                metrics['ciede2000'] = calc_ciede2000(pred, gt)
            except Exception as e:
                print(f"Warning: CIEDE2000 computation failed: {e}")

        # Delta E CAM16-UCS
        if self._is_enabled('delta_e_cam16'):
            try:
                metrics['delta_e_cam16'] = calc_delta_e_cam16_ucs(pred, gt)
            except Exception as e:
                print(f"Warning: Delta E CAM16-UCS computation failed: {e}")

        return metrics
