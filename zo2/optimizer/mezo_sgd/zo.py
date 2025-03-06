# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import sys
sys.path.append('./zo2')

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseOptimizer
import numpy as np

from ...config.mezo_sgd import MeZOSGDConfig


class MeZOSGD(BaseOptimizer):
    """
    Implements the [MeZO-SGD](https://arxiv.org/abs/2305.17333) optimization method, 
    particularly suited for scenarios with limited compute resources.
    """
    def __init__(self, model: nn.Module, config: MeZOSGDConfig):
        """
        Initializes the MeZOSGD optimizer which applies zeroth-order optimization techniques to the model parameters.

        Args:
            model (nn.Module): The model whose parameters will be optimized.
            config (MeZOSGDConfig): Configuration object containing optimizer settings.
        """
        self.config = config
        self.model = model
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.zo_eps = config.eps
        self.max_zo_random_seed = config.max_zo_random_seed
        self.debug_mode = config.debug_mode
        defaults = dict(
            lr=self.lr,
            weight_decay=self.weight_decay,
            maximize=False,
            foreach=None,
            differentiable=False,
            fused=None,
        )
        super().__init__(model.parameters(), defaults)
        
    @torch.inference_mode
    def zo_perturb_parameters(self, module: nn.Module, scaling_factor: float=1):
        """
        Applies Gaussian noise to parameters of a module, facilitating zeroth-order optimization.

        Args:
            module (nn.Module): Module whose parameters will be perturbed.
            scaling_factor (float): Scaling factor for the noise applied to the parameters.
        """
        for _, param in module.named_parameters():
            if param.requires_grad:
                # Resample z
                if self.debug_mode:
                    z = torch.ones_like(param.data) # for debug
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data.add_(scaling_factor * z * self.zo_eps)

    @torch.inference_mode
    def zo_update(self, module, weight_decay=None):
        """
        Updates the parameters of a module based on zeroth-order perturbations and optional weight decay.

        Args:
            module (nn.Module): Module whose parameters will be updated.
            weight_decay (float, optional): Weight decay coefficient. If None, it defaults to the configuration.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                # Resample z
                if self.debug_mode:
                    z = torch.ones_like(param.data) # for debug
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if weight_decay != None:
                    param.data.sub_(
                        self.lr * (self.projected_grad * z + weight_decay * param.data))
                else:
                    if all(x not in name for x in ["bias", "layer_norm", "layernorm", "ln"]):
                        param.data.sub_(
                            self.lr * (self.projected_grad * z + self.weight_decay * param.data))
                    else:
                        param.data.sub_(self.lr * self.projected_grad * z)
    
    def zo_perturb_shifts(self, first_perturb_shift=1, stride=2):
        """
        Generates shifts for perturbing parameters in a pattern conducive to zeroth-order optimization.

        Returns:
            list: A list of perturb shifts used during the forward and update passes.
        """
        return [first_perturb_shift, -stride, stride-first_perturb_shift]

    @torch.inference_mode
    def zo_forward(self, *args, zo_random_seed: int=None, **kwargs):
        """
        Forward pass that applies zeroth-order perturbations to compute the loss, used for gradient estimation.
        Notice that the application of Gaussian perturbations for the parameters during both the perturbation and update phases should be the same.

        Args:
            zo_random_seed (int, optional): Random seed for reproducibility of perturbations.
        """
        self._update_lr()
        self.zo_random_seed = zo_random_seed if zo_random_seed else np.random.randint(self.max_zo_random_seed)
        torch.manual_seed(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=self.zo_perturb_shifts()[0])
        loss1 = self.inner_zo_forward(*args, **kwargs)
        torch.manual_seed(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=self.zo_perturb_shifts()[1])
        loss2 = self.inner_zo_forward(*args, **kwargs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
        torch.manual_seed(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=self.zo_perturb_shifts()[2])
        torch.manual_seed(self.zo_random_seed)
        self.zo_update(self.model)
        return loss1

    #*********************** evaluate ***********************#

    @torch.inference_mode()
    def zo_eval_forward(self, *args, **kwargs):
        """
        Forward pass in evaluation mode.
        """
        output = self.inner_zo_eval_forward(*args, **kwargs)
        return output
    
    #*********************** api ***********************#

    @torch.inference_mode
    def inner_zo_forward(self, idx, pos, targets):
        """
        Example of ZO inner_zo_forward:
            Match the same args as the original model forward,
            and replace all 'self' to 'self.model'.
        """
        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        x = self.model.lm_head(x)
        loss = F.cross_entropy(
            x[:, :-1, :].reshape(-1, x.size(-1)), 
            targets[:, 1:].reshape(-1)
        )
        return loss.detach()

    @torch.inference_mode()   
    def inner_zo_eval_forward(self, eval_fn, idx, pos, targets):
        output = eval_fn(idx, pos, targets)
        return output
    
