"""
Loss Weight Scheduler for Dual GT Loss

Schedules w_p (prefer weight) during training:
L_total = w_p × L1(pred, gt_prefer) + (1-w_p) × L1(pred, gt_non_prefer)

Supports: uniform, linear, exponential, logarithmic schedules
"""

import math


class LossWeightScheduler:
    """Schedule loss weight w_p during training"""

    def __init__(self, schedule_type='exponential', w_p_start=0.5, w_p_end=0.8,
                 total_iterations=100000):
        """
        Args:
            schedule_type: 'uniform', 'linear', 'exponential', or 'logarithmic'
            w_p_start: Initial prefer weight
            w_p_end: Final prefer weight
            total_iterations: Total training iterations
        """
        self.schedule_type = schedule_type
        self.w_p_start = w_p_start
        self.w_p_end = w_p_end
        self.total_iterations = total_iterations
        self.current_iteration = 0

    def get_weight(self):
        """
        Get current w_p value

        Returns:
            Current prefer weight in [w_p_start, w_p_end]
        """
        if self.schedule_type == 'uniform':
            return self.w_p_start

        # Progress in [0, 1]
        progress = min(self.current_iteration / self.total_iterations, 1.0)

        if self.schedule_type == 'linear':
            w_p = self.w_p_start + progress * (self.w_p_end - self.w_p_start)

        elif self.schedule_type == 'exponential':
            # Exponential growth (slow start, fast end)
            w_p = self.w_p_start + (self.w_p_end - self.w_p_start) * (progress ** 2)

        elif self.schedule_type == 'logarithmic':
            # Logarithmic growth (fast start, slow end)
            if progress == 0:
                w_p = self.w_p_start
            else:
                # Use log(1 + progress) to ensure smooth growth from 0
                # Normalize by log(2) so progress=1 gives maximum growth
                w_p = self.w_p_start + (self.w_p_end - self.w_p_start) * math.log(1 + progress) / math.log(2)

        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        return w_p

    def step(self):
        """Advance iteration counter"""
        self.current_iteration += 1

    def state_dict(self):
        """Save scheduler state"""
        return {
            'current_iteration': self.current_iteration,
            'schedule_type': self.schedule_type,
            'w_p_start': self.w_p_start,
            'w_p_end': self.w_p_end,
            'total_iterations': self.total_iterations,
        }

    def load_state_dict(self, state):
        """Load scheduler state"""
        self.current_iteration = state['current_iteration']
        self.schedule_type = state.get('schedule_type', self.schedule_type)
        self.w_p_start = state.get('w_p_start', self.w_p_start)
        self.w_p_end = state.get('w_p_end', self.w_p_end)
        self.total_iterations = state.get('total_iterations', self.total_iterations)


def compute_dual_gt_loss(pred, gt_prefer, gt_non_prefer, w_p):
    """
    Compute dual ground truth loss

    Args:
        pred: Predictions (B, 3, H, W)
        gt_prefer: Prefer ground truths (B, 3, H, W)
        gt_non_prefer: Non-prefer ground truths (B, 3, H, W)
        w_p: Prefer weight

    Returns:
        Weighted L1 loss
    """
    import torch.nn.functional as F

    loss_prefer = F.l1_loss(pred, gt_prefer)
    loss_non_prefer = F.l1_loss(pred, gt_non_prefer)

    return w_p * loss_prefer + (1 - w_p) * loss_non_prefer
