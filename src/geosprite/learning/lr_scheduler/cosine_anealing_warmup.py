import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithWarmup(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps(int): Linear warmup step size. Default: 0.
        annealing_steps (int): annealing steps
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 annealing_steps: int,
                 max_lr: float,
                 min_lr: float,
                 ):

        self.warmup_steps = warmup_steps  # warmup step size
        self.annealing_steps = annealing_steps
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.current_epoch = -1

        # will call self.step() in __init__
        super().__init__(optimizer)

    def get_lr(self):
        assert self.current_epoch >= 0

        if self.current_epoch <= self.warmup_steps:
            lr = self.max_lr * self.current_epoch / self.warmup_steps
        elif self.current_epoch < self.warmup_steps + self.annealing_steps:
            # 0 -> 1
            annealing_phase = (self.current_epoch - self.warmup_steps) / self.annealing_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * annealing_phase)) / 2
        else:
            lr = self.min_lr

        return [lr] * len(self.optimizer.param_groups)

    def step(self):
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
