"""
User Preference Embedding (UPE) Processor

This module processes raw UPE embeddings using a shallow transformer to produce
a single processed UPE vector per user.

**Dimension Flow**:
1. Raw UPE: (num_pairs, input_dim) = (16, 1280)
2. Input projection: (16, 1280) → (16, hidden_dim=512)
3. Transformer: (16, 512) → (16, 512)
4. Average pooling: (16, 512) → (512,)
5. Output projection: (512,) → (output_dim=512)

Key Features:
- Strictly permutation-invariant (via average pooling, no positional encoding)
- Shallow architecture (3 layers, 8 heads)
- Pre-LN transformer for stability
- Order-independent preference representation

Usage:
    >>> upe_processor = UPEProcessor(
    ...     input_dim=1280,
    ...     hidden_dim=512,
    ...     output_dim=512,
    ... )
    >>> raw_upe = torch.randn(16, 1280)  # From UPEExtractor
    >>> processed_upe = upe_processor(raw_upe)  # (512,)
"""

import torch
import torch.nn as nn
import math


class UPEProcessor(nn.Module):
    """
    Process raw UPE with shallow transformer.

    Features:
    - Permutation-invariant aggregation
    - Project to target dimension
    - Average pooling to single vector
    """

    def __init__(self,
                 input_dim: int,      # content_dim + color_dim (e.g., 1280)
                 hidden_dim: int = 512,
                 output_dim: int = 512,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Validation
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Shallow transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Layer norm for final output
        self.layer_norm = nn.LayerNorm(output_dim)

        print(f"[UPEProcessor] Initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Transformer: {num_layers} layers, {num_heads} heads")
        print(f"  FFN dim: {hidden_dim * 4}")

    def forward(self, upe_raw: torch.Tensor) -> torch.Tensor:
        """
        Process raw UPE.

        Args:
            upe_raw: (N, input_dim) or (B, N, input_dim)
                    where N = num_pairs (e.g., 16)

        Returns:
            upe: (output_dim,) or (B, output_dim)

        Examples:
            >>> processor = UPEProcessor(1280, 512, 512)
            >>> raw = torch.randn(16, 1280)
            >>> processed = processor(raw)  # (512,)
            >>>
            >>> # Batched processing
            >>> raw_batch = torch.randn(4, 16, 1280)
            >>> processed_batch = processor(raw_batch)  # (4, 512)
        """
        # Handle both batched and unbatched inputs
        if upe_raw.dim() == 2:
            upe_raw = upe_raw.unsqueeze(0)  # (1, N, input_dim)
            squeeze_output = True
        else:
            squeeze_output = False

        B, N, _ = upe_raw.shape

        # Validate input dimension
        if upe_raw.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {upe_raw.shape[-1]}"
            )

        # Input projection
        x = self.input_proj(upe_raw)  # (B, N, hidden_dim)

        # Transformer
        x = self.transformer(x)  # (B, N, hidden_dim)

        # Average pooling (permutation-invariant)
        x = x.mean(dim=1)  # (B, hidden_dim)

        # Output projection
        x = self.output_proj(x)  # (B, output_dim)
        x = self.layer_norm(x)

        if squeeze_output:
            return x.squeeze(0)  # (output_dim,)
        else:
            return x  # (B, output_dim)

    def get_param_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_permutation_invariance():
    """Test that UPE processor is permutation-invariant."""
    print("Testing permutation invariance...")

    processor = UPEProcessor(
        input_dim=1280,
        hidden_dim=512,
        output_dim=512,
    )
    processor.eval()

    # Create random UPE
    upe_raw = torch.randn(16, 1280)

    # Process original
    with torch.no_grad():
        output1 = processor(upe_raw)

    # Permute input
    perm_indices = torch.randperm(16)
    upe_permuted = upe_raw[perm_indices]

    # Process permuted
    with torch.no_grad():
        output2 = processor(upe_permuted)

    # Check if outputs are close
    diff = (output1 - output2).abs().max().item()
    is_invariant = diff < 1e-5  # Strict tolerance for true permutation invariance

    print(f"  Max difference: {diff:.8f}")
    print(f"  Permutation invariant (tolerance=1e-5): {is_invariant}")

    if is_invariant:
        print("  ✓ Processor is STRICTLY permutation invariant!")
        print("  Order of preference pairs does not affect output.")
    else:
        print("  WARNING: Processor is NOT permutation invariant!")
        print("  This should not happen without positional encoding.")


def test_batch_processing():
    """Test batched processing."""
    print("\nTesting batch processing...")

    processor = UPEProcessor(
        input_dim=1280,
        hidden_dim=512,
        output_dim=512,
    )
    processor.eval()

    # Single sample
    upe_single = torch.randn(16, 1280)
    with torch.no_grad():
        output_single = processor(upe_single)
    print(f"  Single input shape: {upe_single.shape} → {output_single.shape}")

    # Batched
    upe_batch = torch.randn(4, 16, 1280)
    with torch.no_grad():
        output_batch = processor(upe_batch)
    print(f"  Batch input shape: {upe_batch.shape} → {output_batch.shape}")

    # Verify shapes
    assert output_single.shape == (512,), f"Expected (512,), got {output_single.shape}"
    assert output_batch.shape == (4, 512), f"Expected (4, 512), got {output_batch.shape}"
    print("  ✓ Batch processing works correctly")


def test_parameter_count():
    """Test parameter count."""
    print("\nTesting parameter count...")

    processor = UPEProcessor(
        input_dim=1280,
        hidden_dim=512,
        output_dim=512,
        num_layers=3,
        num_heads=8,
    )

    param_count = processor.get_param_count()
    print(f"  Total parameters: {param_count:,}")
    print(f"  Size: ~{param_count / 1e6:.2f}M")

    # Expected: ~10M parameters (no positional encoding reduces from ~10.4M)
    assert 5e6 < param_count < 15e6, f"Unexpected param count: {param_count}"
    print("  ✓ Parameter count is reasonable")


if __name__ == '__main__':
    print("=" * 60)
    print("UPE Processor Tests")
    print("=" * 60)

    test_batch_processing()
    test_permutation_invariance()
    test_parameter_count()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
