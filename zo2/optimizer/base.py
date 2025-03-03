import torch
from torch.optim.optimizer import Optimizer

class BaseOptimizer(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        self.lr = defaults["lr"]
        if len(self.param_groups) > 1:
            raise NotImplementedError("Currently ZO2 does not support multi-group optimizing.")
    
    def _update_lr(self):
        self.lr = self.param_groups[0]["lr"]
    
    def _set_lr(self):
        self.param_groups[0]["lr"] = self.lr