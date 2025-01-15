import sys
sys.path.append('./zo2')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...config.mezo_sgd import MeZOSGDConfig


class MeZOSGD:
    """
        MeZO-SGD
    """
    def __init__(self, model, config: MeZOSGDConfig):
        self.model = model
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.zo_eps = config.eps
        self.max_zo_random_seed = config.max_zo_random_seed
    
    @torch.inference_mode
    def zo_perturb_parameters(self, module: nn.Module, scaling_factor: float=1):       
        for _, param in module.named_parameters():
            if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data.add_(scaling_factor * z * self.zo_eps)

    @torch.inference_mode
    def zo_update(self, module):

        for name, param in module.named_parameters():
            if param.requires_grad:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if all(x not in name for x in ["bias", "layer_norm", "layernorm", "ln"]):
                    param.data.sub_(
                        self.lr * (self.projected_grad * z + self.weight_decay * param.data))
                else:
                    param.data.sub_(self.lr * self.projected_grad * z)
    
    @torch.inference_mode
    def zo_forward(self, *args, zo_random_seed: int=None, **kwargs):
        self.zo_random_seed = zo_random_seed if zo_random_seed else np.random.randint(self.max_zo_random_seed)
        torch.manual_seed(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=1)
        loss1 = self.inner_zo_forward(*args, **kwargs)
        torch.manual_seed(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=-2)
        loss2 = self.inner_zo_forward(*args, **kwargs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
        torch.manual_seed(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=1)
        torch.manual_seed(self.zo_random_seed)
        self.zo_update(self.model)
        return loss1

    #*********************** example ***********************#

    @torch.inference_mode
    def inner_zo_forward(self, idx, pos, targets):
        # example of model forward
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

