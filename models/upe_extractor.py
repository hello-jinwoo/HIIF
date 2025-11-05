"""
User Preference Embedding (UPE) Extractor

This module extracts User Preference Embeddings from preference pairs using
pretrained vision models (CLIP, DINO, DINOv3, SAM).

**UPDATED SPECIFICATION** (2025-11-05):
- Input: A preference pairs (default A=16)
- Output: (A, content_dim + color_dim)
  - Without projection: (16, 1280) for DINO(768) + CLIP(512)
  - With projection: (16, 1536) for 768 + 768
- Content: AVERAGED per pair → (A, content_dim)
- Color: DIFFERENCE per pair → (A, color_dim)
- Same model can now be used for both content and color (dual extraction)

Key Process:
1. Extract content embeddings from both prefer/non-prefer images using content model
2. AVERAGE the two embeddings per pair → (A, content_dim_native)
3. Extract color embeddings from both images using color model
4. SUBTRACT to get difference (prefer - non_prefer) → (A, color_dim_native)
5. (Optional) Apply adaptive projection to standardize dimensions
6. Normalize both content and color separately (L2 norm)
7. Concatenate: [content_avg, color_diff] → (A, content_dim + color_dim)

Supported Models:
- CLIP: ViT-B/16 (512), ViT-L/14 (768)
- DINO: dino_vitb16 (768), dino_vits16 (384)
- DINOv3: dinov3_vitb16 (768), dinov3_vitl16 (1024), etc.
- SAM: vit_b (768), vit_l (1024), vit_h (1280)

Examples:
    >>> # Default: Different models (DINO + CLIP)
    >>> upe_extractor = UPEExtractor(
    ...     content_model_name='dino',
    ...     color_model_name='clip',
    ...     num_pairs=16
    ... )
    >>> upe_raw = upe_extractor(pairs)  # (16, 1280)

    >>> # Same model with adaptive projection (CLIP + CLIP)
    >>> upe_extractor = UPEExtractor(
    ...     content_model_name='clip',
    ...     color_model_name='clip',
    ...     use_adaptive_projection=True,
    ...     projection_dim=768,
    ...     num_pairs=16
    ... )
    >>> upe_raw = upe_extractor(pairs)  # (16, 1536)

    >>> # SAM for both content and color
    >>> upe_extractor = UPEExtractor(
    ...     content_model_name='sam',
    ...     color_model_name='sam',
    ...     num_pairs=16
    ... )
    >>> upe_raw = upe_extractor(pairs)  # (16, 1536) for SAM vit_b
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from .feature_extractors import create_feature_extractor


class UPEExtractor(nn.Module):
    """
    Extract User Preference Embeddings from preference pairs.

    Process:
    1. Extract content embeddings using model X (per image)
    2. AVERAGE content embeddings per pair
    3. Extract color embeddings using model Y (per image)
    4. SUBTRACT to get color difference per pair (prefer - non_prefer)
    5. (Optional) Apply adaptive projection to standardize dimensions
    6. Normalize both separately (L2)
    7. Concatenate → (num_pairs, content_dim + color_dim)

    Note: content_model and color_model can now be the same model.
    The difference is in the aggregation method (average vs subtraction).
    """

    def __init__(self,
                 content_model_name: str = 'dino',
                 content_model_variant: str = None,
                 color_model_name: str = 'clip',
                 color_model_variant: str = None,
                 num_pairs: int = 16,
                 normalize: bool = True,
                 use_adaptive_projection: bool = False,
                 projection_dim: int = 768,
                 device: str = 'cuda'):
        super().__init__()

        # Create extractors
        self.content_extractor = create_feature_extractor(
            content_model_name, content_model_variant, device
        )
        self.color_extractor = create_feature_extractor(
            color_model_name, color_model_variant, device
        )

        self.num_pairs = num_pairs
        self.normalize = normalize
        self.use_adaptive_projection = use_adaptive_projection
        self.projection_dim = projection_dim
        self.device = device

        # Native dimensions from extractors
        self.content_dim_native = self.content_extractor.output_dim
        self.color_dim_native = self.color_extractor.output_dim

        # Adaptive projection layers (optional)
        if use_adaptive_projection:
            self.content_projection = nn.Linear(self.content_dim_native, projection_dim).to(device)
            self.color_projection = nn.Linear(self.color_dim_native, projection_dim).to(device)

            # Initialize projections
            nn.init.xavier_uniform_(self.content_projection.weight)
            nn.init.zeros_(self.content_projection.bias)
            nn.init.xavier_uniform_(self.color_projection.weight)
            nn.init.zeros_(self.color_projection.bias)

            # Output dimensions after projection
            self.content_dim = projection_dim
            self.color_dim = projection_dim
        else:
            self.content_projection = None
            self.color_projection = None
            self.content_dim = self.content_dim_native
            self.color_dim = self.color_dim_native

        self.output_dim = self.content_dim + self.color_dim

        print(f"[UPEExtractor] Initialized:")
        print(f"  Content model: {content_model_name} (native_dim={self.content_dim_native})")
        print(f"  Color model: {color_model_name} (native_dim={self.color_dim_native})")
        if use_adaptive_projection:
            print(f"  Adaptive projection: {self.content_dim_native} → {self.content_dim}, {self.color_dim_native} → {self.color_dim}")
        print(f"  Output shape: ({self.num_pairs}, {self.output_dim})")
        print(f"  Normalize: {self.normalize}")

    def extract_content_embedding(self,
                                 prefer_imgs: torch.Tensor,
                                 non_prefer_imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract content embedding by AVERAGING features from both images.

        Args:
            prefer_imgs: (A, 3, H, W)
            non_prefer_imgs: (A, 3, H, W)

        Returns:
            content_emb: (A, content_dim), L2 normalized if self.normalize=True
        """
        # Extract features for both
        prefer_feat = self.content_extractor.extract_features(prefer_imgs)
        non_prefer_feat = self.content_extractor.extract_features(non_prefer_imgs)

        # CRITICAL: Average per pair (not keep separate!)
        content_emb = (prefer_feat + non_prefer_feat) / 2.0  # (A, content_dim_native)

        # Apply adaptive projection if enabled
        if self.use_adaptive_projection:
            content_emb = self.content_projection(content_emb)  # (A, projection_dim)

        # Normalize
        if self.normalize:
            content_emb = F.normalize(content_emb, p=2, dim=-1)

        return content_emb

    def extract_color_difference(self,
                                prefer_imgs: torch.Tensor,
                                non_prefer_imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract color difference embedding (prefer - non_prefer).

        Args:
            prefer_imgs: (A, 3, H, W)
            non_prefer_imgs: (A, 3, H, W)

        Returns:
            color_diff: (A, color_dim), L2 normalized if self.normalize=True
        """
        prefer_feat = self.color_extractor.extract_features(prefer_imgs)
        non_prefer_feat = self.color_extractor.extract_features(non_prefer_imgs)

        # Difference (prefer - non_prefer)
        color_diff = prefer_feat - non_prefer_feat  # (A, color_dim_native)

        # Apply adaptive projection if enabled
        if self.use_adaptive_projection:
            color_diff = self.color_projection(color_diff)  # (A, projection_dim)

        # Normalize
        if self.normalize:
            color_diff = F.normalize(color_diff, p=2, dim=-1)

        return color_diff

    def forward(self, preference_pairs: List[Dict]) -> torch.Tensor:
        """
        Extract UPE from A preference pairs.

        Args:
            preference_pairs: List of A dicts, each with:
                - 'prefer': (3, H, W) tensor in [0, 1]
                - 'non_prefer': (3, H, W) tensor in [0, 1]

        Returns:
            upe_raw: (A, content_dim + color_dim)
                    Default: (16, 1280) for DINO(768) + CLIP(512)

        Raises:
            ValueError: If number of pairs doesn't match self.num_pairs
        """
        if len(preference_pairs) != self.num_pairs:
            raise ValueError(
                f"Expected {self.num_pairs} pairs, got {len(preference_pairs)}"
            )

        # Extract features for each pair individually (images may have different sizes)
        content_embs_list = []
        color_diffs_list = []

        for p in preference_pairs:
            prefer_img = p['prefer'].unsqueeze(0).to(self.device)  # (1, 3, H, W)
            non_prefer_img = p['non_prefer'].unsqueeze(0).to(self.device)  # (1, 3, H, W)

            # Extract content embedding (averaged)
            content_emb = self.extract_content_embedding(prefer_img, non_prefer_img)  # (1, content_dim)
            content_embs_list.append(content_emb)

            # Extract color difference
            color_diff = self.extract_color_difference(prefer_img, non_prefer_img)  # (1, color_dim)
            color_diffs_list.append(color_diff)

        # Stack results
        content_embs = torch.cat(content_embs_list, dim=0)  # (A, content_dim)
        color_diffs = torch.cat(color_diffs_list, dim=0)  # (A, color_dim)

        # Concatenate
        upe_raw = torch.cat([content_embs, color_diffs], dim=-1)  # (A, total_dim)

        return upe_raw


def test_upe_extractor():
    """Quick sanity check of UPE extractor dimensions."""
    import warnings
    warnings.filterwarnings('ignore')

    print("Testing UPE Extractor...")

    # Create mock preference pairs
    num_pairs = 16
    preference_pairs = []
    for i in range(num_pairs):
        prefer = torch.rand(3, 256, 256)
        non_prefer = torch.rand(3, 256, 256)
        preference_pairs.append({'prefer': prefer, 'non_prefer': non_prefer})

    # Initialize extractor (will fail if CLIP/DINO not available)
    try:
        upe_extractor = UPEExtractor(
            content_model_name='dino',
            color_model_name='clip',
            num_pairs=16,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Extract UPE
        upe_raw = upe_extractor(preference_pairs)

        print(f"✓ UPE extracted successfully!")
        print(f"  Shape: {upe_raw.shape}")
        print(f"  Expected: (16, 1280)")
        print(f"  Match: {upe_raw.shape == (16, 1280)}")

        # Check normalization
        content_norms = torch.norm(upe_raw[:, :768], p=2, dim=1)
        color_norms = torch.norm(upe_raw[:, 768:], p=2, dim=1)
        print(f"  Content norms: {content_norms.mean():.3f} ± {content_norms.std():.3f}")
        print(f"  Color norms: {color_norms.mean():.3f} ± {color_norms.std():.3f}")

    except ImportError as e:
        print(f"✗ Dependencies not installed: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == '__main__':
    test_upe_extractor()
