import torch
import torch.nn.functional as F

from .. import model
from ...base import BaseZOModel
from ....optimizer.mezo_sgd.zo import MeZOSGD
from ....config.mezo_sgd import MeZOSGDConfig


class GPT(model.GPT, BaseZOModel):
    def __init__(self, config: model.GPTConfig, zo_config: MeZOSGDConfig):
        super().__init__(config)
        self.opt = Optimizer(model=self, config=zo_config)

    def forward(self, idx, pos, targets=None):
        if self.zo_training:
            return self.opt.zo_forward(idx, pos, targets)
        else:
            # for evaluate and inference purpose
            return self.opt.zo_eval_forward(super().forward, idx, pos, targets)


class Optimizer(MeZOSGD):

    @torch.inference_mode
    def inner_zo_forward(self, idx, pos, targets):
        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        x = self.model.lm_head(x)
        loss = F.cross_entropy(
            x.reshape(-1, x.size(-1)), 
            targets.reshape(-1)
        )
        return loss.detach()

    @torch.inference_mode()   
    def inner_zo_eval_forward(self, eval_fn, idx, pos, targets):
        output = eval_fn(idx, pos, targets)
        return output
    