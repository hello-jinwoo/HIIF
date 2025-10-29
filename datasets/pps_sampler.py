"""
PPS User Batch Sampler
Ensures each batch contains samples from only one user (for user-specific decoder training).
"""

import random
from torch.utils.data import Sampler


class PPSUserBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch contains samples from only one user.

    This is critical for PPS training where we train user-specific decoders.
    Each batch will contain samples from the same user, and users are cycled
    through in a specified order.

    Usage:
        dataset = PPSPreferencePairDataset(...)
        sampler = PPSUserBatchSampler(
            dataset=dataset,
            batch_size=8,
            iterations_per_user=16,
            shuffle_users=True
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(self, dataset, batch_size, iterations_per_user=16,
                 shuffle_users=True, shuffle_samples=True, drop_last=False):
        """
        Args:
            dataset: PPSPreferencePairDataset instance with get_user_ids() and get_user_samples()
            batch_size: int, number of samples per batch
            iterations_per_user: int, number of batches per user before switching (default: 16)
            shuffle_users: bool, whether to shuffle user order each epoch (default: True)
            shuffle_samples: bool, whether to shuffle samples within each user (default: True)
            drop_last: bool, whether to drop last incomplete batch for each user (default: False)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.iterations_per_user = iterations_per_user
        self.shuffle_users = shuffle_users
        self.shuffle_samples = shuffle_samples
        self.drop_last = drop_last

        # Get user information
        self.user_ids = dataset.get_user_ids()
        self.user_to_samples = {
            user_id: dataset.get_user_samples(user_id)
            for user_id in self.user_ids
        }

        # Calculate total batches
        self._calculate_num_batches()

    def _calculate_num_batches(self):
        """Calculate total number of batches across all users."""
        self.user_num_batches = {}
        total_batches = 0

        for user_id in self.user_ids:
            num_samples = len(self.user_to_samples[user_id])
            if self.drop_last:
                num_batches = num_samples // self.batch_size
            else:
                num_batches = (num_samples + self.batch_size - 1) // self.batch_size

            # Limit to iterations_per_user batches per user
            num_batches = min(num_batches, self.iterations_per_user)

            self.user_num_batches[user_id] = num_batches
            total_batches += num_batches

        self.total_batches = total_batches

    def __iter__(self):
        """
        Generate batches.

        For each user:
        1. Get all samples for that user
        2. Optionally shuffle samples
        3. Create iterations_per_user batches (or fewer if not enough samples)
        4. Each batch contains only samples from this user
        """
        # Determine user order
        user_order = list(self.user_ids)
        if self.shuffle_users:
            random.shuffle(user_order)

        # Iterate through users
        for user_id in user_order:
            # Get samples for this user
            user_samples = self.user_to_samples[user_id].copy()

            # Optionally shuffle samples
            if self.shuffle_samples:
                random.shuffle(user_samples)

            # Create batches for this user
            num_batches = self.user_num_batches[user_id]

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(user_samples))

                batch = user_samples[start_idx:end_idx]

                # Skip if drop_last and batch is incomplete
                if self.drop_last and len(batch) < self.batch_size:
                    continue

                yield batch

    def __len__(self):
        """Return total number of batches."""
        return self.total_batches


class PPSUserSequentialSampler:
    """
    Simple sequential sampler that groups samples by user.
    Useful for validation where we don't need to limit iterations per user.

    Usage:
        dataset = PPSPreferencePairDataset(...)
        sampler = PPSUserSequentialSampler(dataset)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    """

    def __init__(self, dataset, shuffle_users=False):
        """
        Args:
            dataset: PPSPreferencePairDataset instance
            shuffle_users: bool, whether to shuffle user order (default: False for validation)
        """
        self.dataset = dataset
        self.shuffle_users = shuffle_users

        # Get user information
        self.user_ids = dataset.get_user_ids()
        self.user_to_samples = {
            user_id: dataset.get_user_samples(user_id)
            for user_id in self.user_ids
        }

    def __iter__(self):
        """Iterate through all samples, grouped by user."""
        user_order = list(self.user_ids)
        if self.shuffle_users:
            random.shuffle(user_order)

        for user_id in user_order:
            for sample_idx in self.user_to_samples[user_id]:
                yield sample_idx

    def __len__(self):
        """Return total number of samples."""
        return len(self.dataset)
