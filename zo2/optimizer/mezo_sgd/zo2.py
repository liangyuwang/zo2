import sys
sys.path.append('./zo2')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from .zo import MeZOSGD
from ...config.mezo_sgd import MeZOSGDConfig


class MeZO2SGD(MeZOSGD):
    """
        MeZO-SGD with Offloading
    """
    def __init__(self, model, config: MeZOSGDConfig):
        assert config.zo2, "MeZO2SGD can only work with offloading."
        self.model = model
        self.zo_eps = config.eps
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.device = config.working_device
        self.offloading_device = config.offloading_device
        self.max_zo_random_seed = config.max_zo_random_seed
        self.overlap = config.overlap
        self.offloading_blocks = config.offloading_blocks
        self.init_zo2()
    
    def init_zo2(self):
        self.upload_stream = torch.cuda.Stream()
        self.offload_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.rstate = None
        self.rstate_queue = deque(maxlen=2)
        self.last_rstate = None
        self.projected_grad = 0
        self.init_zo2_upload()
    
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
    
    @torch.inference_mode
    def zo_update(self, module):
        torch.cuda.set_rng_state(self.last_rstate)
        super().zo_update(module)
        self.last_rstate = torch.cuda.get_rng_state()
        return module
    
    @torch.inference_mode()
    def module_dual_forward(self, module, inputs1, inputs2, projected_grad=0.):
        if projected_grad != 0:
            module = self.zo_update(module, projected_grad=projected_grad)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=1)
        output1 = module(**inputs1)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=-2)
        output2 = module(**inputs2)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=1)
        self.rstate = torch.cuda.get_rng_state()
        return output1, output2
    
    #*********************** tasks ***********************#

    def task_upload_to(self, module, device='cuda'):
        if self.overlap:
            self.upload_stream.synchronize()
            with torch.cuda.stream(self.upload_stream):
                module = module.to(device, non_blocking=True)
        else:
            module = module.to(device)
        return module

    def task_offload_to(self, module, device='cpu'):
        if self.overlap:
            self.offload_stream.synchronize()
            with torch.cuda.stream(self.offload_stream):
                module = module.to(device, non_blocking=True)
        else:
            module = module.to(device)
        return module
    
    def task_compute(self, module, inputs1, inputs2, grad):
        if self.overlap:
            self.compute_stream.synchronize()
            with torch.cuda.stream(self.compute_stream):
                o1, o2 = self.module_dual_forward(
                    module=module, 
                    inputs1=inputs1, 
                    inputs2=inputs2,
                    projected_grad=grad)
        else:
            o1, o2 = self.module_dual_forward(
                    module=module, 
                    inputs1=inputs1, 
                    inputs2=inputs2,
                    projected_grad=grad)
        return o1, o2
    
    #*********************** example ***********************#

    @torch.inference_mode()   
    def zo_forward(self, input_ids, pos):
        self.rstate_queue.append(self.rstate.clone())
        if len(self.rstate_queue) == 2:
            self.last_rstate = self.rstate_queue.popleft()
        if 0 in self.offloading_blocks:
            self.model.transformer.h[0] = self.task_upload_to(
                module=self.model.transformer.h[0], 
                device=self.device)
        we1, we2 = self.task_compute(self.model.transformer.wte,
                                inputs1={"input": input_ids},
                                inputs2={"input": input_ids},
                                grad=self.projected_grad)
        pe1, pe2 = self.task_compute(self.model.transformer.wpe, 
                                 {"input": pos}, 
                                 {"input": pos}, 
                                 self.projected_grad)
        # torch.cuda.synchronize()
        hidden_states1 = we1 + pe1
        hidden_states2 = we2 + pe2
        N = len(self.model.transformer.h)
        for i in range(1, N):
            if i == 1:
                if i in self.offloading_blocks:
                    self.model.transformer.h[i] = self.task_upload_to(
                        module=self.model.transformer.h[i], 
                        device=self.device)
                hidden_states1, hidden_states2 = self.task_compute(
                    self.model.transformer.h[i-1], 
                    inputs1={"x": hidden_states1}, 
                    inputs2={"x": hidden_states2}, 
                    grad=self.projected_grad)
            else:
                if i in self.offloading_blocks:
                    self.model.transformer.h[i] = self.task_upload_to(
                        module=self.model.transformer.h[i], 
                        device=self.device)
                if i-2 in self.offloading_blocks:
                    self.model.transformer.h[i-2] = self.task_offload_to(
                        module=self.model.transformer.h[i-2], 
                        device=self.offloading_device)
                hidden_states1, hidden_states2 = self.task_compute(
                    self.model.transformer.h[i-1], 
                    inputs1={"x": hidden_states1}, 
                    inputs2={"x": hidden_states2}, 
                    grad=self.projected_grad)
            torch.cuda.synchronize()    #TODO: remove global sync
        if N-2 in self.offloading_blocks:
            self.model.transformer.h[N-2] = self.task_offload_to(
                self.model.transformer.h[N-2], device=self.offloading_device)
        hidden_states1, hidden_states2 = self.task_compute(
                    self.model.transformer.h[N-1], 
                    inputs1={"x": hidden_states1}, 
                    inputs2={"x": hidden_states2}, 
                    grad=self.projected_grad
                )
        torch.cuda.synchronize()    #TODO: remove global sync
        if N-1 in self.offloading_blocks:
            self.model.transformer.h[N-1] = self.task_offload_to(
                self.model.transformer.h[N-1], device=self.offloading_device)
        logits1, logits2 = self.task_compute(self.model.transformer.ln_f,
                                             inputs1={"input": hidden_states1}, 
                                             inputs2={"input": hidden_states2}, 
                                             grad=self.projected_grad)
        logits1, logits2 = self.task_compute(self.model.lm_head,
                                             inputs1={"input": logits1}, 
                                             inputs2={"input": logits2}, 
                                             grad=self.projected_grad)
        torch.cuda.synchronize()    #TODO: remove global sync
        return logits1, logits2
    
    @torch.inference_mode()
    def zo_step(self, inputs, seed: int=None):
        self.zo_random_seed = seed if seed else np.random.randint(self.max_zo_random_seed)
        torch.manual_seed(self.zo_random_seed)
        torch.cuda.manual_seed(self.zo_random_seed)
        self.rstate = torch.cuda.get_rng_state()
        logits1, logits2 = self.zo_forward(
            input_ids=inputs["idx"], 
            pos=inputs['pos']
        )
        loss1 = F.cross_entropy(
            logits1[:, :-1, :].reshape(-1, logits1.size(-1)), 
            inputs['targets'][:, 1:].reshape(-1)
        )
        loss2 = F.cross_entropy(
            logits2[:, :-1, :].reshape(-1, logits2.size(-1)), 
            inputs['targets'][:, 1:].reshape(-1)
        )
        self.projected_grad = ((loss1 - loss2) / (self.zo_eps * 2)).item()
        return loss1.item()
    
    
class MeZO2SGDAMP(MeZO2SGD):
    ...
