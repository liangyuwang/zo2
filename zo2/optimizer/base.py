# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
from torch.optim.optimizer import Optimizer

class BaseOptimizer(Optimizer):    
    """
    Base class for Zeroth-Order Optimization handling basic setup, including learning rate management.
    This class is not intended for direct use but provides core functionalities for derived classes.
    """
    def __init__(self, params, defaults):
        """
        Initializes the BaseOptimizer.

        Args:
            params (iterable): Parameters to optimize or dicts defining parameter groups.
            defaults (dict): Default optimization options.
        """
        super().__init__(params, defaults)
        self.lr = defaults["lr"]
        if len(self.param_groups) > 1:
            raise NotImplementedError("Currently ZO2 does not support multi-group optimizing.")
    
    def _update_lr(self):
        self.lr = self.param_groups[0]["lr"]
    
    def _set_lr(self):
        self.param_groups[0]["lr"] = self.lr