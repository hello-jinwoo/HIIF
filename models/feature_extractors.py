"""
Vision Feature Extractors for User Preference Embedding (UPE)

This module provides wrappers around pretrained vision models (CLIP, DINO)
for extracting features from preference pair images.

Key Features:
- Frozen pretrained weights (no training)
- Standardized interface (extract_features API)
- Global Average Pooling (GAP) for fixed-size outputs
- Configurable model variants

Usage:
    # Create CLIP extractor
    clip_extractor = create_feature_extractor('clip', variant='ViT-B/16')
    features = clip_extractor.extract_features(images)  # (B, 512)

    # Create DINO extractor
    dino_extractor = create_feature_extractor('dino', variant='dino_vitb16')
    features = dino_extractor.extract_features(images)  # (B, 768)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class VisionFeatureExtractor(nn.Module):
    """Base class for vision feature extractors."""

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        self.output_dim = None  # To be set by subclass

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features with Global Average Pooling.

        Args:
            images: (B, 3, H, W) in [0, 1]

        Returns:
            features: (B, D) where D is model-specific
        """
        raise NotImplementedError


class CLIPFeatureExtractor(VisionFeatureExtractor):
    """CLIP vision encoder wrapper."""

    def __init__(self,
                 variant: str = 'ViT-B/16',
                 device: str = 'cuda'):
        super().__init__(freeze=True)

        # Load CLIP
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP is not installed. Install it with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )

        self.model, self.preprocess = clip.load(variant, device=device)
        self.device = device

        # Set to eval and freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Output dimension (variant-dependent)
        if 'ViT-B' in variant or 'RN50' in variant:
            self.output_dim = 512
        elif 'ViT-L' in variant:
            self.output_dim = 768
        else:
            # Default to 512 for unknown variants
            self.output_dim = 512

        print(f"[CLIPFeatureExtractor] Loaded {variant}, output_dim={self.output_dim}")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP features.

        Args:
            images: (B, 3, H, W) in [0, 1]

        Returns:
            features: (B, 512) or (B, 768) depending on variant
        """
        # Ensure correct device
        images = images.to(self.device)

        # Resize to CLIP input size (224x224)
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images,
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            )

        # Normalize to [-1, 1] if input is [0, 1]
        # CLIP preprocessing expects this range
        if images.min() >= 0 and images.max() <= 1:
            # Convert [0, 1] → [-1, 1]
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
            images = (images - mean) / std

        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(images).float()

        return features  # (B, output_dim)


class DINOFeatureExtractor(VisionFeatureExtractor):
    """DINO vision encoder wrapper."""

    def __init__(self,
                 variant: str = 'dino_vitb16',
                 device: str = 'cuda'):
        super().__init__(freeze=True)

        # Load DINO via torch.hub
        # Fix import conflict with project's utils.py
        import sys
        original_sys_path = sys.path.copy()
        original_utils_module = sys.modules.get('utils', None)

        try:
            # Temporarily remove 'utils' from sys.modules to allow DINO to import its own
            if 'utils' in sys.modules:
                del sys.modules['utils']

            # Remove project root from sys.path to avoid utils.py conflict
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            sys.path = [p for p in sys.path if os.path.abspath(p) != project_root]

            self.model = torch.hub.load('facebookresearch/dino:main', variant, trust_repo=True)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINO model '{variant}'. "
                f"Make sure you have internet connection. Error: {e}"
            )
        finally:
            # Restore original sys.path and utils module
            sys.path = original_sys_path
            if original_utils_module is not None:
                sys.modules['utils'] = original_utils_module

        self.model = self.model.to(device)
        self.device = device

        # Set to eval and freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Output dimension (variant-dependent)
        if 'vitb' in variant.lower():
            self.output_dim = 768
        elif 'vits' in variant.lower():
            self.output_dim = 384
        else:
            raise ValueError(f"Unknown DINO variant: {variant}")

        print(f"[DINOFeatureExtractor] Loaded {variant}, output_dim={self.output_dim}")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DINO features.

        Args:
            images: (B, 3, H, W) in [0, 1]

        Returns:
            features: (B, 768) for ViT-B or (B, 384) for ViT-S
        """
        # Ensure correct device
        images = images.to(self.device)

        # Resize to DINO input size (224x224)
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images,
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            )

        # DINO expects [0, 1] normalized images (no mean/std normalization needed)

        # Extract features
        with torch.no_grad():
            features = self.model(images)

            # DINO returns either:
            # - (B, D) if using get_intermediate_layers(n=1)
            # - (B, n_patches+1, D) if using forward()
            if features.dim() == 3:
                # Average pool across all tokens (including CLS)
                features = features.mean(dim=1)  # (B, D)

        return features


def create_feature_extractor(model_name: str,
                             variant: str = None,
                             device: str = 'cuda') -> VisionFeatureExtractor:
    """
    Factory function to create feature extractors.

    Args:
        model_name: 'clip' or 'dino'
        variant: Model variant (optional, uses default if None)
        device: Device to load model on

    Returns:
        VisionFeatureExtractor instance

    Examples:
        >>> clip_extractor = create_feature_extractor('clip')
        >>> dino_extractor = create_feature_extractor('dino', variant='dino_vits16')
    """
    model_name = model_name.lower()

    if model_name == 'clip':
        variant = variant or 'ViT-B/16'
        return CLIPFeatureExtractor(variant=variant, device=device)

    elif model_name == 'dino':
        variant = variant or 'dino_vitb16'
        return DINOFeatureExtractor(variant=variant, device=device)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose 'clip' or 'dino'."
        )
