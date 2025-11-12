"""
Config validator for UPE system.

Validates configuration files and applies default values.
"""

import warnings
from typing import Dict, Any


class UPEConfigValidator:
    """Validate UPE configuration."""

    @staticmethod
    def validate(config: Dict[str, Any]) -> None:
        """
        Run all validation checks.

        Args:
            config: Configuration dictionary loaded from YAML

        Raises:
            ValueError: If required fields are missing or invalid
        """
        UPEConfigValidator._check_upe_config(config)
        UPEConfigValidator._check_model_selection(config)
        UPEConfigValidator._check_dimensions(config)
        UPEConfigValidator._check_data_config(config)
        UPEConfigValidator._check_value_ranges(config)
        UPEConfigValidator._check_defaults(config)
        print("✅ Config validation passed")

    @staticmethod
    def _check_upe_config(config: Dict[str, Any]) -> None:
        """Check UPE config exists."""
        if 'upe_config' not in config:
            raise ValueError("Missing top-level 'upe_config' in config")

        upe_cfg = config['upe_config']
        required = ['num_pairs', 'content_model', 'color_model']
        for field in required:
            if field not in upe_cfg:
                raise ValueError(f"Missing upe_config.{field}")

    @staticmethod
    def _check_model_selection(config: Dict[str, Any]) -> None:
        """Validate model selection (content_model and color_model can be the same)."""
        upe_cfg = config['upe_config']
        content = upe_cfg['content_model']
        color = upe_cfg['color_model']

        # Note: As of 2025-11-05 update, content_model and color_model CAN be the same
        # They use different aggregation methods (average vs subtraction)
        # So we only validate that the model names are valid

        valid = ['clip', 'dino', 'dinov2', 'dinov3', 'sam']
        if content not in valid or color not in valid:
            raise ValueError(f"Models must be in {valid}, got content={content}, color={color}")

    @staticmethod
    def _check_dimensions(config: Dict[str, Any]) -> None:
        """Check dimension consistency (uses fixed variants per model)."""
        from models.feature_extractors import get_feature_extractor_dim

        upe_cfg = config['upe_config']
        content_name = upe_cfg['content_model']
        color_name = upe_cfg['color_model']

        # Get expected dimensions from feature extractors (uses fixed variants)
        try:
            expected_content_dim = get_feature_extractor_dim(content_name)
            expected_color_dim = get_feature_extractor_dim(color_name)
        except ValueError as e:
            raise ValueError(f"Invalid model configuration: {e}")

        # Check if dimensions are specified (optional)
        if 'content_dim' in upe_cfg:
            content_dim = upe_cfg['content_dim']
            if content_dim != expected_content_dim:
                warnings.warn(f"Content dim mismatch: {content_dim} != {expected_content_dim}")

        if 'color_dim' in upe_cfg:
            color_dim = upe_cfg['color_dim']
            if color_dim != expected_color_dim:
                warnings.warn(f"Color dim mismatch: {color_dim} != {expected_color_dim}")

    @staticmethod
    def _check_data_config(config: Dict[str, Any]) -> None:
        """Check data configuration consistency (uses fixed variants per model)."""
        from models.feature_extractors import get_feature_extractor_dim

        if 'train_dataset' not in config:
            return  # Validation config might not have train_dataset

        upe_cfg = config['upe_config']
        model_pairs = upe_cfg['num_pairs']

        # Check if model has upe_processor_config (optional)
        if 'model' in config and 'args' in config['model']:
            model_args = config['model']['args']
            if 'upe_processor_config' in model_args:
                processor_cfg = model_args['upe_processor_config']

                # Check input_dim matches expected UPE dimension (if specified)
                content_name = upe_cfg['content_model']
                color_name = upe_cfg['color_model']

                try:
                    content_dim = get_feature_extractor_dim(content_name)
                    color_dim = get_feature_extractor_dim(color_name)
                    expected_input_dim = content_dim + color_dim
                except ValueError as e:
                    raise ValueError(f"Invalid model configuration: {e}")

                if 'input_dim' in processor_cfg:
                    actual_input_dim = processor_cfg['input_dim']
                    if actual_input_dim != expected_input_dim:
                        warnings.warn(
                            f"UPE processor input_dim mismatch: "
                            f"expected {expected_input_dim} (content={content_dim} + "
                            f"color={color_dim}), got {actual_input_dim}. "
                            f"This will be auto-corrected during training/validation."
                        )

    @staticmethod
    def _check_value_ranges(config: Dict[str, Any]) -> None:
        """Validate parameter value ranges."""
        # Model parameters
        if 'model' in config and 'args' in config['model']:
            model_args = config['model']['args']

            if 'hidden_dim' in model_args:
                hidden_dim = model_args['hidden_dim']
                if not (128 <= hidden_dim <= 1024):
                    raise ValueError(f"hidden_dim must be 128-1024, got {hidden_dim}")

            if 'blocks' in model_args:
                blocks = model_args['blocks']
                if not (8 <= blocks <= 32):
                    raise ValueError(f"blocks must be 8-32, got {blocks}")

        # UPE parameters
        upe_cfg = config['upe_config']
        num_pairs = upe_cfg['num_pairs']
        if not (8 <= num_pairs <= 64):
            raise ValueError(f"num_pairs must be 8-64, got {num_pairs}")

        # UPE processor parameters (if present)
        if 'model' in config and 'args' in config['model']:
            model_args = config['model']['args']
            if 'upe_processor_config' in model_args:
                processor_cfg = model_args['upe_processor_config']

                if 'output_dim' in processor_cfg:
                    upe_dim = processor_cfg['output_dim']
                    if upe_dim not in [256, 512, 1024]:
                        warnings.warn(f"Unusual upe output_dim: {upe_dim}. Recommended: 256, 512, or 1024")

        # Training parameters (if present)
        if 'train_dataset' in config:
            if 'batch_size' in config['train_dataset']:
                batch_size = config['train_dataset']['batch_size']
                if not (1 <= batch_size <= 32):
                    warnings.warn(f"Unusual batch_size: {batch_size}")

            if 'wrapper' in config['train_dataset'] and 'args' in config['train_dataset']['wrapper']:
                wrapper_args = config['train_dataset']['wrapper']['args']
                if 'crop_size' in wrapper_args:
                    crop_size = wrapper_args['crop_size']
                    if crop_size not in [64, 128, 256]:
                        warnings.warn(f"Unusual crop_size: {crop_size}")

    @staticmethod
    def _check_defaults(config: Dict[str, Any]) -> None:
        """Apply default values for missing optional fields."""
        # Model defaults
        if 'model' in config and 'args' in config['model']:
            model_args = config['model']['args']

            if 'hidden_dim' not in model_args:
                model_args['hidden_dim'] = 384
                print("ℹ️  Applied default: hidden_dim=384")

            if 'blocks' not in model_args:
                model_args['blocks'] = 20
                print("ℹ️  Applied default: blocks=20")

            if 'n_hi_layers' not in model_args:
                model_args['n_hi_layers'] = 6
                print("ℹ️  Applied default: n_hi_layers=6")

            # UPE processor defaults
            if 'upe_processor_config' in model_args:
                processor_cfg = model_args['upe_processor_config']

                if 'num_layers' not in processor_cfg:
                    processor_cfg['num_layers'] = 3
                    print("ℹ️  Applied default: upe_processor_config.num_layers=3")

                if 'num_heads' not in processor_cfg:
                    processor_cfg['num_heads'] = 8
                    print("ℹ️  Applied default: upe_processor_config.num_heads=8")

                if 'dropout' not in processor_cfg:
                    processor_cfg['dropout'] = 0.1
                    print("ℹ️  Applied default: upe_processor_config.dropout=0.1")

        # UPE defaults
        upe_cfg = config['upe_config']
        if 'normalize' not in upe_cfg:
            upe_cfg['normalize'] = True
            print("ℹ️  Applied default: upe_config.normalize=True")

        if 'cache_dir' not in upe_cfg:
            upe_cfg['cache_dir'] = './cache/upe'
            print("ℹ️  Applied default: upe_config.cache_dir='./cache/upe'")

        # Training defaults
        if 'train_dataset' in config:
            train_cfg = config['train_dataset']

            if 'batch_size' not in train_cfg:
                train_cfg['batch_size'] = 8
                print("ℹ️  Applied default: train_dataset.batch_size=8")


def validate_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated config dictionary

    Raises:
        ValueError: If config is invalid
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    UPEConfigValidator.validate(config)

    return config


if __name__ == '__main__':
    """Test config validator on all UPE configs."""
    import sys
    import yaml
    from pathlib import Path

    # Find all UPE config files
    train_configs = list(Path('configs/train-upe').glob('*.yaml'))
    val_configs = list(Path('configs/validate-upe').glob('*.yaml'))

    print("=" * 60)
    print("Testing UPE Config Validator")
    print("=" * 60)

    all_passed = True

    # Test training configs
    print("\n📋 Training Configs:")
    for config_path in train_configs:
        print(f"\n  Testing: {config_path}")
        try:
            config = yaml.safe_load(open(config_path))
            UPEConfigValidator.validate(config)
            print(f"  ✅ {config_path.name} passed")
        except Exception as e:
            print(f"  ❌ {config_path.name} failed: {e}")
            all_passed = False

    # Test validation configs
    print("\n📋 Validation Configs:")
    for config_path in val_configs:
        print(f"\n  Testing: {config_path}")
        try:
            config = yaml.safe_load(open(config_path))
            UPEConfigValidator.validate(config)
            print(f"  ✅ {config_path.name} passed")
        except Exception as e:
            print(f"  ❌ {config_path.name} failed: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All configs passed validation!")
        sys.exit(0)
    else:
        print("❌ Some configs failed validation")
        sys.exit(1)
