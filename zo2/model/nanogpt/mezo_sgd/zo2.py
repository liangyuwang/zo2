import torch
import torch.nn.functional as F
import numpy as np

from .. import model
from ...base import BaseZOModel
from ....optimizer.mezo_sgd.zo2 import MeZO2SGD
from ....config.mezo_sgd import MeZOSGDConfig


class GPT(model.GPT, BaseZOModel):
    def __init__(self, config: model.GPTConfig, zo_config: MeZOSGDConfig):
        super().__init__(config)
        self.opt = Optimizer(model=self, config=zo_config)

    def forward(self, idx, pos, targets=None):
        if self.zo_training:
            return self.opt.zo_forward(idx, pos, targets)
        else:
            # for inference purpose
            return super().forward(idx, pos, targets)


class Optimizer(MeZO2SGD):
    
    def init_zo2_upload(self):
        print("Upload head and tail to cuda.")
        self.model.transformer.wte = self.model.transformer.wte.to(self.device)
        self.model.transformer.wpe = self.model.transformer.wpe.to(self.device)
        self.model.transformer.ln_f = self.model.transformer.ln_f.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)
        
        self.num_blocks = len(self.model.transformer.h)
        if self.offloading_blocks is not None:
            self.offloading_blocks = self.offloading_blocks
        else:
            self.offloading_blocks = list(range(self.num_blocks))
        print(f"Transformer blocks {self.offloading_blocks} will be offloaded to {self.offloading_device}")
        for i in range(self.num_blocks):
            if i in self.offloading_blocks:
                continue
            else:
                self.model.transformer.h[i] = self.model.transformer.h[i].to(self.device)
                print(f"Upload block {i} to cuda.")

    @torch.inference_mode()   
    def inner_zo_forward(self, idx, pos, targets):
        we1, we2 = self.task_compute_module(self.model.transformer.wte,
                                inputs1={"input": idx},
                                inputs2={"input": idx},
                                grad=self.projected_grad)
        pe1, pe2 = self.task_compute_module(self.model.transformer.wpe, 
                                 {"input": pos}, 
                                 {"input": pos}, 
                                 self.projected_grad,
                                 compute_sync=False)
        hidden_states1, hidden_states2 = self.task_compute_function(torch.add,
                                                                    {"input": we1, "other": pe1},
                                                                    {"input": we2, "other": pe2},
                                                                    compute_sync=False)
        if 0 in self.offloading_blocks:
            self.model.transformer.h[0] = self.task_upload(
                module=self.model.transformer.h[0], 
                device=self.device)
        N = len(self.model.transformer.h)
        for i in range(1, N):
            if i != 1:
                if i-2 in self.offloading_blocks:
                    self.model.transformer.h[i-2] = self.task_offload(
                        module=self.model.transformer.h[i-2], 
                        device=self.offloading_device)
            hidden_states1, hidden_states2 = self.task_compute_module(
                self.model.transformer.h[i-1], 
                inputs1={"x": hidden_states1}, 
                inputs2={"x": hidden_states2}, 
                grad=self.projected_grad)
            if i in self.offloading_blocks:
                self.model.transformer.h[i] = self.task_upload(
                    module=self.model.transformer.h[i], 
                    device=self.device)
        if N-2 in self.offloading_blocks:
            self.model.transformer.h[N-2] = self.task_offload(
                self.model.transformer.h[N-2], device=self.offloading_device)
        hidden_states1, hidden_states2 = self.task_compute_module(
                    self.model.transformer.h[N-1], 
                    inputs1={"x": hidden_states1}, 
                    inputs2={"x": hidden_states2}, 
                    grad=self.projected_grad
                )
        if N-1 in self.offloading_blocks:
            self.model.transformer.h[N-1] = self.task_offload(
                self.model.transformer.h[N-1], device=self.offloading_device)
        logits1, logits2 = self.task_compute_module(self.model.transformer.ln_f,
                                             inputs1={"input": hidden_states1}, 
                                             inputs2={"input": hidden_states2}, 
                                             grad=self.projected_grad,
                                             weight_decay=0.)   
                    # 'task_compute_module' will remove the first name 'ln_f', so we need to disable weight_decay manually.
        logits1, logits2 = self.task_compute_module(self.model.lm_head,
                                             inputs1={"input": logits1}, 
                                             inputs2={"input": logits2}, 
                                             grad=self.projected_grad)
        loss1, loss2 = self.task_compute_function(F.cross_entropy,
                                                  {"input": logits1[:, :-1, :].reshape(-1, logits1.size(-1)), 
                                                   "target": targets[:, 1:].reshape(-1)},
                                                  {"input": logits2[:, :-1, :].reshape(-1, logits2.size(-1)), 
                                                   "target": targets[:, 1:].reshape(-1)})
        return loss1, loss2
    