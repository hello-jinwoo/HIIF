"""
Multi-level UPE Cache System

This module provides a multi-level caching system for User Preference Embeddings:
1. Disk cache: Persistent .pt files
2. Memory cache: LRU cache in RAM
3. GPU cache: Pre-loaded on GPU (optional)

Key Features:
- Automatic cache invalidation on config changes
- LRU eviction for memory management
- Batch preloading for efficient training
- Corruption detection and recovery

Usage:
    >>> # Initialize cache
    >>> cache = UPECache(cache_dir='./cache/upe', max_memory_size=100)
    >>>
    >>> # Save UPE
    >>> cache.save(user_id='user_001', upe=upe_tensor, config=config)
    >>>
    >>> # Load UPE (with multi-level caching)
    >>> upe = cache.load(user_id='user_001', config=config, device='cuda')
    >>>
    >>> # Preload all users to GPU
    >>> cache.preload_all(user_ids, config, device='cuda')
"""

import torch
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
import hashlib


class UPECache:
    """
    Multi-level caching for User Preference Embeddings.

    Levels:
    1. Disk: Persistent .pt files
    2. Memory: LRU cache in RAM
    3. GPU: Pre-loaded on GPU (optional)
    """

    def __init__(self,
                 cache_dir: str = './cache/upe',
                 max_memory_size: int = 100,
                 preload_to_gpu: bool = True,
                 dataset_name: Optional[str] = None):
        """
        Initialize UPE cache.

        Args:
            cache_dir: Base directory to store cache files
            max_memory_size: Maximum number of UPEs in memory cache
            preload_to_gpu: Whether to keep loaded UPEs on GPU
            dataset_name: Optional dataset name to create subdirectory
                         (e.g., 'train_toy', 'validation_small')
                         If provided, cache will be stored in cache_dir/dataset_name/
        """
        self.base_cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name

        # Create dataset-specific subdirectory if dataset_name is provided
        if dataset_name:
            self.cache_dir = self.base_cache_dir / dataset_name
        else:
            self.cache_dir = self.base_cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_size = max_memory_size
        self.preload_to_gpu = preload_to_gpu

        # Memory cache (LRU)
        self.memory_cache: Dict[str, torch.Tensor] = {}
        self.access_order: List[str] = []

        # GPU cache
        self.gpu_cache: Dict[str, torch.Tensor] = {}

        # Metadata file
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()

        print(f"[UPECache] Initialized at {self.cache_dir}")
        if dataset_name:
            print(f"  Dataset name: {dataset_name}")
        print(f"  Max memory size: {max_memory_size}")
        print(f"  Preload to GPU: {preload_to_gpu}")

    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'cache_entries': {},
                'total_users': 0
            }

    def _save_metadata(self):
        """Save cache metadata."""
        self.metadata['total_users'] = len(self.metadata['cache_entries'])
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_config_hash(self, config: dict) -> str:
        """Generate hash from config for cache filename."""
        # Use only relevant keys for cache invalidation
        relevant_keys = ['content_model', 'color_model', 'num_pairs']
        config_str = '_'.join([
            f"{key}={config.get(key, 'unknown')}"
            for key in relevant_keys
        ])
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_cache_path(self, user_id: str, config: dict) -> Path:
        """
        Generate cache path with model-pair subdirectory.

        Path format: {cache_dir}/{model1}_{model2}/{user_id}_{model1}_{model2}.pt
        Example: ./cache/upe/sam_dinov2/ZWvAVhb_desktop_sam_dinov2.pt
        """
        content_model = config.get('content_model', 'dino')
        color_model = config.get('color_model', 'clip')

        # Create model-pair subdirectory
        model_pair_dir = self.cache_dir / f"{content_model}_{color_model}"
        model_pair_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename (without num_pairs)
        filename = f"{user_id}_{content_model}_{color_model}.pt"

        return model_pair_dir / filename

    def exists(self, user_id: str, config: dict) -> bool:
        """Check if cache exists for user."""
        return self._get_cache_path(user_id, config).exists()

    def save(self, user_id: str, upe: torch.Tensor, config: dict,
             metadata: dict = None):
        """
        Save UPE to disk.

        Args:
            user_id: User identifier
            upe: UPE tensor to save
            config: Configuration dict (for cache key)
            metadata: Optional metadata to store
        """
        cache_path = self._get_cache_path(user_id, config)

        # Prepare data
        data = {
            'user_id': user_id,
            'upe': upe.cpu(),  # Always save on CPU
            'config': config,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'shape': tuple(upe.shape),
        }

        # Save to disk
        torch.save(data, cache_path)

        # Update metadata
        config_hash = self._get_config_hash(config)
        self.metadata['cache_entries'][user_id] = {
            'filename': cache_path.name,
            'shape': list(upe.shape),
            'created': data['timestamp'],
            'config_hash': config_hash,
        }
        self._save_metadata()

    def load(self, user_id: str, config: dict,
             device: str = 'cpu') -> Optional[torch.Tensor]:
        """
        Load UPE with multi-level caching.

        Lookup order:
        1. GPU cache (if preload_to_gpu=True)
        2. Memory cache
        3. Disk cache

        Args:
            user_id: User identifier
            config: Configuration dict
            device: Device to load tensor on

        Returns:
            UPE tensor or None if not found
        """
        # Level 1: GPU cache
        if self.preload_to_gpu and user_id in self.gpu_cache:
            return self.gpu_cache[user_id]

        # Level 2: Memory cache
        if user_id in self.memory_cache:
            upe = self.memory_cache[user_id]
            self._update_lru(user_id)

            if device == 'cuda' and self.preload_to_gpu:
                upe = upe.cuda()
                self.gpu_cache[user_id] = upe

            return upe

        # Level 3: Disk cache
        cache_path = self._get_cache_path(user_id, config)
        if not cache_path.exists():
            return None

        try:
            data = torch.load(cache_path, map_location='cpu')
            upe = data['upe']

            # Validate shape
            # Calculate expected dimensions from model names (if specified)
            content_dim = config.get('content_dim')
            color_dim = config.get('color_dim')

            if content_dim is None or color_dim is None:
                # Try to infer from model names
                try:
                    from models.feature_extractors import get_feature_extractor_dim
                    content_model = config.get('content_model')
                    color_model = config.get('color_model')

                    if content_model and color_model:
                        content_dim = get_feature_extractor_dim(content_model)
                        color_dim = get_feature_extractor_dim(color_model)
                    else:
                        # Fallback to old defaults if model names not specified
                        content_dim = content_dim or 768
                        color_dim = color_dim or 512
                except Exception:
                    # Fallback to old defaults on error
                    content_dim = content_dim or 768
                    color_dim = color_dim or 512

            expected_shape = (
                config.get('num_pairs', 16),
                content_dim + color_dim
            )
            if tuple(upe.shape) != expected_shape:
                warnings.warn(
                    f"Shape mismatch for {user_id}: {upe.shape} != {expected_shape}. "
                    f"Cache may be stale. Consider regenerating."
                )
                return None

            # Add to memory cache
            self._add_to_memory(user_id, upe)

            # Move to device
            upe = upe.to(device)

            if device == 'cuda' and self.preload_to_gpu:
                self.gpu_cache[user_id] = upe

            return upe

        except Exception as e:
            warnings.warn(f"Failed to load cache for {user_id}: {e}")
            # Try to remove corrupted cache
            try:
                cache_path.unlink()
            except:
                pass
            return None

    def _add_to_memory(self, user_id: str, upe: torch.Tensor):
        """Add to memory cache with LRU eviction."""
        if len(self.memory_cache) >= self.max_memory_size:
            # Evict oldest
            if len(self.access_order) > 0:
                oldest = self.access_order.pop(0)
                if oldest in self.memory_cache:
                    del self.memory_cache[oldest]

        self.memory_cache[user_id] = upe
        self.access_order.append(user_id)

    def _update_lru(self, user_id: str):
        """Update LRU order."""
        if user_id in self.access_order:
            self.access_order.remove(user_id)
        self.access_order.append(user_id)

    def preload_all(self, user_ids: List[str], config: dict, device: str = 'cuda'):
        """
        Preload UPEs for all users.

        Args:
            user_ids: List of user IDs to preload
            config: Configuration dict
            device: Device to preload on
        """
        print(f"[UPECache] Preloading {len(user_ids)} users to {device}...")

        missing = []
        for user_id in tqdm(user_ids, desc="Loading UPEs"):
            upe = self.load(user_id, config, device=device)
            if upe is None:
                missing.append(user_id)

        if missing:
            warnings.warn(
                f"UPEs not found for {len(missing)} users: {missing[:5]}..."
            )

        loaded = len(user_ids) - len(missing)
        print(f"[UPECache] Loaded {loaded}/{len(user_ids)} UPEs to {device}")
        if device == 'cuda':
            print(f"  GPU cache size: {len(self.gpu_cache)}")

    def get_batch(self, user_ids: List[str], config: dict,
                 device: str = 'cuda') -> torch.Tensor:
        """
        Get batched UPEs for multiple users.

        Args:
            user_ids: List of user IDs
            config: Configuration dict
            device: Device to load on

        Returns:
            Batched UPE tensor (B, num_pairs, upe_dim)

        Raises:
            RuntimeError: If any UPE is missing
        """
        upes = []
        for user_id in user_ids:
            upe = self.load(user_id, config, device=device)
            if upe is None:
                raise RuntimeError(f"UPE not found for user {user_id}")
            upes.append(upe)

        return torch.stack(upes, dim=0)

    def clear_memory(self):
        """Clear memory and GPU caches."""
        self.memory_cache.clear()
        self.gpu_cache.clear()
        self.access_order.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_disk(self, user_id: str = None, config: dict = None):
        """
        Clear disk cache.

        Args:
            user_id: If provided, only clear this user's cache
            config: If provided with user_id, only clear specific config
        """
        if user_id is not None:
            # Clear specific user
            if config is not None:
                cache_path = self._get_cache_path(user_id, config)
                if cache_path.exists():
                    cache_path.unlink()
                    if user_id in self.metadata['cache_entries']:
                        del self.metadata['cache_entries'][user_id]
                        self._save_metadata()
            else:
                # Clear all configs for this user (search all model-pair subdirs)
                pattern = f"{user_id}_*.pt"
                for model_pair_dir in self.cache_dir.iterdir():
                    if model_pair_dir.is_dir():
                        for cache_file in model_pair_dir.glob(pattern):
                            cache_file.unlink()
                if user_id in self.metadata['cache_entries']:
                    del self.metadata['cache_entries'][user_id]
                    self._save_metadata()
        else:
            # Clear all caches (in all model-pair subdirectories)
            for model_pair_dir in self.cache_dir.iterdir():
                if model_pair_dir.is_dir():
                    for cache_file in model_pair_dir.glob("*.pt"):
                        cache_file.unlink()
            self.metadata['cache_entries'] = {}
            self._save_metadata()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        # Search for .pt files in all model-pair subdirectories
        cache_files = []
        for model_pair_dir in self.cache_dir.iterdir():
            if model_pair_dir.is_dir():
                cache_files.extend(model_pair_dir.glob("*.pt"))

        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        return {
            'disk_cache_count': len(cache_files),
            'memory_cache_count': len(self.memory_cache),
            'gpu_cache_count': len(self.gpu_cache),
            'disk_cache_size_mb': total_size_bytes / (1024 ** 2),
            'cache_dir': str(self.cache_dir),
        }

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        print(f"\n[UPECache] Statistics:")
        print(f"  Disk cache: {stats['disk_cache_count']} files ({stats['disk_cache_size_mb']:.2f} MB)")
        print(f"  Memory cache: {stats['memory_cache_count']} entries")
        print(f"  GPU cache: {stats['gpu_cache_count']} entries")
        print(f"  Cache dir: {stats['cache_dir']}")


def test_cache():
    """Test cache functionality."""
    print("Testing UPE Cache...")

    # Create test cache
    cache = UPECache(cache_dir='./cache/upe_test', max_memory_size=5)

    # Test config
    config = {
        'content_model': 'dino',
        'color_model': 'clip',
        'num_pairs': 16,
        'content_dim': 768,
        'color_dim': 512,
    }

    # Create test UPEs
    test_users = ['user_001', 'user_002', 'user_003']
    for user_id in test_users:
        upe = torch.randn(16, 1280)
        cache.save(user_id, upe, config)
        print(f"  Saved {user_id}")

    # Test loading
    for user_id in test_users:
        upe = cache.load(user_id, config, device='cpu')
        assert upe is not None, f"Failed to load {user_id}"
        assert upe.shape == (16, 1280), f"Wrong shape for {user_id}"
        print(f"  Loaded {user_id}: shape={upe.shape}")

    # Test stats
    cache.print_stats()

    print("\n✓ Cache tests passed!")


if __name__ == '__main__':
    test_cache()
