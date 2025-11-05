"""
Vision Feature Extractors for User Preference Embedding (UPE)

This module provides wrappers around pretrained vision models (CLIP, DINO, DINOv3, SAM)
for extracting features from preference pair images.

Key Features:
- Frozen pretrained weights (no training)
- Standardized interface (extract_features API)
- Global Average Pooling (GAP) for fixed-size outputs
- Configurable model variants

Supported Models:
- CLIP: OpenAI's vision-language model (512/768 dims)
- DINO: Self-supervised ViT (384/768 dims)
- DINOv3: Latest DINO version (384/768/1024/1280/4096 dims)
- SAM: Segment Anything Model (768/1024/1280 dims)

Usage:
    # Create CLIP extractor
    clip_extractor = create_feature_extractor('clip', variant='ViT-B/16')
    features = clip_extractor.extract_features(images)  # (B, 512)

    # Create DINO extractor
    dino_extractor = create_feature_extractor('dino', variant='dino_vitb16')
    features = dino_extractor.extract_features(images)  # (B, 768)

    # Create SAM extractor
    sam_extractor = create_feature_extractor('sam', variant='vit_b')
    features = sam_extractor.extract_features(images)  # (B, 768)
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


class DINOv2FeatureExtractor(VisionFeatureExtractor):
    """DINOv2 vision encoder wrapper (Meta AI 2023) - publicly available."""

    def __init__(self,
                 variant: str = 'dinov2-base',
                 device: str = 'cuda'):
        super().__init__(freeze=True)

        # Load DINOv2 via HuggingFace transformers (publicly available, no gating)
        from transformers import AutoModel, AutoImageProcessor

        print(f"[DINOv2FeatureExtractor] Loading {variant} via HuggingFace transformers...")

        # Map short variant names to HuggingFace model IDs
        variant_mapping = {
            'dinov2-small': 'facebook/dinov2-small',
            'dinov2-base': 'facebook/dinov2-base',
            'dinov2-large': 'facebook/dinov2-large',
            'dinov2-giant': 'facebook/dinov2-giant',
            # 2024 models with registers (better attention maps)
            'dinov2-small-registers': 'facebook/dinov2-with-registers-small',
            'dinov2-base-registers': 'facebook/dinov2-with-registers-base',
            'dinov2-large-registers': 'facebook/dinov2-with-registers-large',
            'dinov2-giant-registers': 'facebook/dinov2-with-registers-giant',
        }

        model_id = variant_mapping.get(variant, f"facebook/{variant}")

        try:
            # Load image processor (for normalization)
            self.processor = AutoImageProcessor.from_pretrained(model_id)

            # Load model
            self.model = AutoModel.from_pretrained(
                model_id,
                device_map=device
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 model '{variant}' from HuggingFace. "
                f"Model ID: {model_id}. "
                f"Make sure you have internet connection. Error: {e}"
            )

        self.device = device

        # Set to eval and freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Output dimension (variant-dependent)
        # DINOv2 embedding dimensions:
        # small: 384, base: 768, large: 1024, giant: 1536
        variant_lower = variant.lower()
        if 'giant' in variant_lower:
            self.output_dim = 1536
        elif 'large' in variant_lower:
            self.output_dim = 1024
        elif 'base' in variant_lower:
            self.output_dim = 768
        elif 'small' in variant_lower:
            self.output_dim = 384
        else:
            raise ValueError(f"Cannot determine output dimension for variant: {variant}")

        print(f"[DINOv2FeatureExtractor] Loaded {variant}, output_dim={self.output_dim}")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 features using CLS token.

        Args:
            images: (B, 3, H, W) in [0, 1]

        Returns:
            features: (B, output_dim) - 768 for base, 1024 for large, etc.
        """
        # Ensure correct device
        images = images.to(self.device)

        # Resize to DINOv2 input size (224x224, same as DINO v1)
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images,
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            )

        # Extract features
        with torch.no_grad():
            # HuggingFace transformers returns ModelOutput object
            # For DINOv2, typical structure:
            # - last_hidden_state: (B, num_patches+1, hidden_dim)
            # - pooler_output: (B, hidden_dim) - CLS token (if available)
            outputs = self.model(pixel_values=images)

            # Try different output attributes in order of preference
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                # Preferred: pre-computed CLS token
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Fallback: extract CLS token from full sequence
                # CLS token is the first token
                features = outputs.last_hidden_state[:, 0, :]
            else:
                raise RuntimeError(
                    f"Cannot extract features from DINOv2 output. "
                    f"Available attributes: {dir(outputs)}"
                )

        return features  # (B, output_dim)


class DINOv3FeatureExtractor(VisionFeatureExtractor):
    """DINOv3 vision encoder wrapper (Meta AI 2025)."""

    def __init__(self,
                 variant: str = 'dinov3_vitb16',
                 device: str = 'cuda'):
        super().__init__(freeze=True)

        # Load DINOv3 via HuggingFace transformers (more reliable than torch.hub)
        from transformers import AutoModel, AutoImageProcessor

        print(f"[DINOv3FeatureExtractor] Loading {variant} via HuggingFace transformers...")

        # Map short variant names to HuggingFace model IDs
        variant_mapping = {
            'dinov3_vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
            'dinov3_vitsplus16': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
            'dinov3_vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'dinov3_vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'dinov3_vitlplus16': 'facebook/dinov3-vitl16plus-pretrain-lvd1689m',
            'dinov3_vithplus16': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
            'dinov3_vit7b16': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
        }

        model_id = variant_mapping.get(variant, f"facebook/{variant}-pretrain-lvd1689m")

        try:
            # Load image processor (for normalization)
            self.processor = AutoImageProcessor.from_pretrained(model_id)

            # Load model
            self.model = AutoModel.from_pretrained(
                model_id,
                device_map=device,
                trust_remote_code=True
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 model '{variant}' from HuggingFace. "
                f"Model ID: {model_id}. "
                f"Make sure you have internet connection and accepted model terms. Error: {e}"
            )

        self.device = device

        # Set to eval and freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Output dimension (variant-dependent)
        # DINOv3 embedding dimensions:
        # ViT-S: 384, ViT-S+: 384, ViT-B: 768, ViT-L: 1024, ViT-H+: 1280, ViT-7B: 4096
        variant_lower = variant.lower()
        if 'vit7b' in variant_lower or '7b' in variant_lower:
            self.output_dim = 4096
        elif 'vithplus' in variant_lower or 'hplus' in variant_lower:
            self.output_dim = 1280
        elif 'vitl' in variant_lower:
            self.output_dim = 1024
        elif 'vitb' in variant_lower:
            self.output_dim = 768
        elif 'vits' in variant_lower:
            self.output_dim = 384
        else:
            raise ValueError(f"Cannot determine output dimension for variant: {variant}")

        print(f"[DINOv3FeatureExtractor] Loaded {variant}, output_dim={self.output_dim}")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv3 features using CLS token (similar to DINO v1).

        Args:
            images: (B, 3, H, W) in [0, 1]

        Returns:
            features: (B, output_dim) - 768 for ViT-B, 1024 for ViT-L, etc.
        """
        # Ensure correct device
        images = images.to(self.device)

        # Resize to DINOv3 input size (224x224, same as DINO v1 for compatibility)
        # Note: DINOv3 can handle larger sizes (518x518) but 224 is standard
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images,
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            )

        # Apply processor normalization
        # HuggingFace models expect normalized inputs
        # The processor handles converting from [0, 1] to model-specific normalization
        # However, since we're passing tensors directly, we'll use the model's expected format

        # Extract features
        with torch.no_grad():
            # HuggingFace transformers returns ModelOutput object
            # For vision transformers, typical structure:
            # - last_hidden_state: (B, num_patches+1, hidden_dim)
            # - pooler_output: (B, hidden_dim) - CLS token (if available)
            outputs = self.model(pixel_values=images)

            # Try different output attributes in order of preference
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                # Preferred: pre-computed CLS token
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Fallback: extract CLS token from full sequence
                # CLS token is typically the first token
                features = outputs.last_hidden_state[:, 0, :]
            elif isinstance(outputs, torch.Tensor):
                # Direct tensor output (unlikely with transformers)
                if outputs.dim() == 2:
                    features = outputs
                elif outputs.dim() == 3:
                    features = outputs[:, 0, :]
                else:
                    raise RuntimeError(f"Unexpected DINOv3 output shape: {outputs.shape}")
            else:
                raise RuntimeError(
                    f"Cannot extract features from DINOv3 output. "
                    f"Available attributes: {dir(outputs)}"
                )

        return features  # (B, output_dim)


class SAMFeatureExtractor(VisionFeatureExtractor):
    """SAM (Segment Anything Model) vision encoder wrapper."""

    def __init__(self,
                 variant: str = 'vit_b',
                 device: str = 'cuda'):
        super().__init__(freeze=True)

        # Load SAM
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "Segment Anything is not installed. Install it with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        # Map variant names to checkpoint URLs and model types
        checkpoint_urls = {
            'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        }

        if variant not in checkpoint_urls:
            raise ValueError(
                f"Unknown SAM variant: {variant}. "
                f"Choose from {list(checkpoint_urls.keys())}"
            )

        # Download checkpoint if needed
        checkpoint_path = self._download_checkpoint(checkpoint_urls[variant], variant)

        # Load SAM model
        self.sam = sam_model_registry[variant](checkpoint=checkpoint_path)
        self.sam = self.sam.to(device)
        self.device = device

        # Set to eval and freeze
        self.sam.eval()
        for param in self.sam.parameters():
            param.requires_grad = False

        # Output dimension (variant-dependent)
        # SAM ViT encoder embedding dimensions
        if variant == 'vit_h':
            self.output_dim = 1280
        elif variant == 'vit_l':
            self.output_dim = 1024
        elif variant == 'vit_b':
            self.output_dim = 768
        else:
            raise ValueError(f"Unknown SAM variant: {variant}")

        print(f"[SAMFeatureExtractor] Loaded {variant}, output_dim={self.output_dim}")

    def _download_checkpoint(self, url: str, variant: str) -> str:
        """Download SAM checkpoint if not already cached."""
        import urllib.request
        from pathlib import Path

        # Cache directory
        cache_dir = Path.home() / '.cache' / 'sam'
        cache_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = cache_dir / f'sam_{variant}.pth'

        if not checkpoint_path.exists():
            print(f"[SAMFeatureExtractor] Downloading {variant} checkpoint...")
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"[SAMFeatureExtractor] Downloaded to {checkpoint_path}")

        return str(checkpoint_path)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract SAM image encoder features.

        Args:
            images: (B, 3, H, W) in [0, 1]

        Returns:
            features: (B, output_dim) - 768 for ViT-B, 1024 for ViT-L, 1280 for ViT-H
        """
        # Ensure correct device
        images = images.to(self.device)

        # SAM expects 1024x1024 input (native resolution)
        # For efficiency, we use 224x224 (same as other models) and adjust
        if images.shape[-2:] != (1024, 1024):
            images = F.interpolate(
                images,
                size=(1024, 1024),
                mode='bicubic',
                align_corners=False
            )

        # SAM expects RGB images in [0, 255] range
        # Convert from [0, 1] to [0, 255]
        images = images * 255.0

        # Normalize with SAM's normalization (ImageNet stats)
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std

        # Extract features using image encoder
        with torch.no_grad():
            # SAM image encoder outputs (B, C, H, W) where C is embed_dim
            # For ViT-B: (B, 768, 64, 64)
            # For ViT-L: (B, 1024, 64, 64)
            # For ViT-H: (B, 1280, 64, 64)
            features = self.sam.image_encoder(images)  # (B, C, H, W)

            # Global Average Pooling to get fixed-size vectors
            features = features.mean(dim=[2, 3])  # (B, C)

        return features  # (B, output_dim)


def create_feature_extractor(model_name: str,
                             variant: str = None,
                             device: str = 'cuda') -> VisionFeatureExtractor:
    """
    Factory function to create feature extractors.

    Args:
        model_name: 'clip', 'dino', 'dinov3', or 'sam'
        variant: Model variant (optional, uses default if None)
        device: Device to load model on

    Returns:
        VisionFeatureExtractor instance

    Examples:
        >>> clip_extractor = create_feature_extractor('clip')
        >>> dino_extractor = create_feature_extractor('dino', variant='dino_vits16')
        >>> dinov3_extractor = create_feature_extractor('dinov3', variant='dinov3_vitb16')
        >>> sam_extractor = create_feature_extractor('sam', variant='vit_b')
    """
    model_name = model_name.lower()

    if model_name == 'clip':
        variant = variant or 'ViT-B/16'
        return CLIPFeatureExtractor(variant=variant, device=device)

    elif model_name == 'dino':
        variant = variant or 'dino_vitb16'
        return DINOFeatureExtractor(variant=variant, device=device)

    elif model_name == 'dinov2':
        variant = variant or 'dinov2-base'
        return DINOv2FeatureExtractor(variant=variant, device=device)

    elif model_name == 'dinov3':
        variant = variant or 'dinov3_vitb16'
        return DINOv3FeatureExtractor(variant=variant, device=device)

    elif model_name == 'sam':
        variant = variant or 'vit_b'
        return SAMFeatureExtractor(variant=variant, device=device)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose 'clip', 'dino', 'dinov3', or 'sam'."
        )
