"""
Checkpoint Manager for PPS Training

Manages encoder and user decoder checkpoints:
- Encoder: save/_train_pps/encoder/ (epoch-last.pth, epoch-{N}.pth, epoch-best.pth)
- User decoders: save/_train_pps/user_decoders/ (D1.pth, E3.pth, ...)
"""

import os
import torch


class CheckpointManager:
    """Manage encoder and user decoder checkpoints for PPS training"""

    def __init__(self, save_dir):
        """
        Args:
            save_dir: Root save directory (e.g., './save/_train_pps')
        """
        self.save_dir = save_dir
        self.encoder_dir = os.path.join(save_dir, 'encoder')
        self.decoder_dir = os.path.join(save_dir, 'user_decoders')

        os.makedirs(self.encoder_dir, exist_ok=True)
        os.makedirs(self.decoder_dir, exist_ok=True)

    def save_encoder(self, model, optimizer, epoch, name='epoch-last', extra_info=None):
        """
        Save encoder checkpoint with error handling and atomic write

        Args:
            model: Full model (HIIF_PPS)
            optimizer: Optimizer
            epoch: Current epoch
            name: Checkpoint name (default: 'epoch-last')
            extra_info: Optional dict with additional info to save

        Returns:
            str: Path to saved checkpoint, or None if save failed
        """
        save_path = os.path.join(self.encoder_dir, f'{name}.pth')
        temp_path = save_path + '.tmp'

        try:
            # Get encoder and freq layer state dicts
            checkpoint = {
                'encoder': model.encoder.state_dict(),
                'freq': model.freq.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            if extra_info is not None:
                checkpoint.update(extra_info)

            # Atomic save: write to temp file first, then rename
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, save_path)  # Atomic on POSIX systems

            return save_path

        except Exception as e:
            print(f"ERROR: Failed to save encoder checkpoint to {save_path}: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return None

    def load_encoder(self, model, optimizer=None, name='epoch-last'):
        """
        Load encoder checkpoint

        Args:
            model: Full model (HIIF_PPS)
            optimizer: Optional optimizer to load state
            name: Checkpoint name (default: 'epoch-last')

        Returns:
            Dict with epoch and any extra info, or None if not found
        """
        load_path = os.path.join(self.encoder_dir, f'{name}.pth')

        if not os.path.exists(load_path):
            return None

        checkpoint = torch.load(load_path, map_location='cpu')

        model.encoder.load_state_dict(checkpoint['encoder'])
        model.freq.load_state_dict(checkpoint['freq'])

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        return {k: v for k, v in checkpoint.items()
                if k not in ['encoder', 'freq', 'optimizer']}

    def save_user_decoder(self, model, user_id):
        """
        Save current user decoder to disk with error handling and atomic write

        Args:
            model: Full model (HIIF_PPS)
            user_id: User ID (e.g., 'user_response_example10')

        Returns:
            str: Path to saved checkpoint, or None if save failed

        Note: This saves the currently loaded decoder (model.current_decoder)
        """
        if model.current_user_id != user_id:
            raise ValueError(
                f"Cannot save decoder for {user_id}: "
                f"current decoder is {model.current_user_id}"
            )

        if model.current_decoder is None:
            raise ValueError(f"No decoder loaded to save for {user_id}")

        save_path = os.path.join(self.decoder_dir, f'{user_id}.pth')
        temp_path = save_path + '.tmp'

        try:
            decoder_state = model.current_decoder.state_dict()

            # Atomic save: write to temp file first, then rename
            torch.save(decoder_state, temp_path)
            os.replace(temp_path, save_path)  # Atomic on POSIX systems

            return save_path

        except Exception as e:
            print(f"ERROR: Failed to save decoder for {user_id} to {save_path}: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return None

    def load_user_decoder_weights(self, decoder, user_id):
        """
        Load decoder weights from disk into a decoder module

        Args:
            decoder: Decoder module (nn.ModuleDict) to load weights into
            user_id: User ID (e.g., 'user_response_example10')

        Returns:
            True if loaded successfully, False if checkpoint not found

        Note: Decoder must already be created; this only loads weights
        """
        load_path = os.path.join(self.decoder_dir, f'{user_id}.pth')

        if not os.path.exists(load_path):
            return False

        decoder_state = torch.load(load_path, map_location='cpu')
        decoder.load_state_dict(decoder_state)

        return True


    def checkpoint_exists(self, name='epoch-last', user_id=None):
        """
        Check if checkpoint exists

        Args:
            name: Encoder checkpoint name (used if user_id is None)
            user_id: Optional user ID to check decoder checkpoint

        Returns:
            True if exists, False otherwise
        """
        if user_id is not None:
            path = os.path.join(self.decoder_dir, f'{user_id}.pth')
        else:
            path = os.path.join(self.encoder_dir, f'{name}.pth')

        return os.path.exists(path)

    def get_all_decoder_files(self):
        """
        Get list of all decoder checkpoint files

        Returns:
            List of decoder filenames (without path)
        """
        import glob
        decoder_files = glob.glob(os.path.join(self.decoder_dir, '*.pth'))
        return [os.path.basename(f) for f in decoder_files]
