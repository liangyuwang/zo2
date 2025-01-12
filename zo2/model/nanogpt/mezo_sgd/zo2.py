import torch
import torch.nn.functional as F
import numpy as np

from .. import model
from ....optimizer.mezo_sgd.zo2 import MeZO2SGD
from ....config.mezo_sgd import MeZOSGDConfig


class GPT(model.GPT):
    def __init__(self, config, zo_config: MeZOSGDConfig, zo_training=True):
        super().__init__(config)
        self.zo_training = zo_training
        self.opt = Optimizer(model=self, config=zo_config)

    def forward(self, idx, pos, targets=None):
        if self.zo_training:
            return self.opt.zo_step({"idx": idx, "pos": pos, "targets": targets})
        else:
            # for inference purpose
            return super().forward(idx, pos, targets)


class Optimizer(MeZO2SGD):
    
    def init_zo2(self):
        self.init_zo2_upload()
        print("Upload embedding and lm head to cuda.")
        for i in range(self.num_blocks):
            if i in self.offloading_blocks:
                continue
            else:
                self.model.transformer.h[i] = self.model.transformer.h[i].to(self.device)
                print(f"Upload block {i} to cuda.")
    
    def init_zo2_upload(self):
        self.model.transformer.wte = self.model.transformer.wte.to(self.device)
        self.model.transformer.wpe = self.model.transformer.wpe.to(self.device)
        self.model.transformer.ln_f = self.model.transformer.ln_f.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)

    @torch.inference_mode()   
    def zo_forward(self, input_ids, pos, projected_grad):
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
                                grad=projected_grad)
        pe1, pe2 = self.task_compute(self.model.transformer.wpe, 
                                 {"input": pos}, 
                                 {"input": pos}, 
                                 projected_grad)
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
                    grad=projected_grad)
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
                    grad=projected_grad)
            torch.cuda.synchronize()    #TODO: remove global sync
        if N-2 in self.offloading_blocks:
            self.model.transformer.h[N-2] = self.task_offload_to(
                self.model.transformer.h[N-2], device=self.offloading_device)
        hidden_states1, hidden_states2 = self.task_compute(
                    self.model.transformer.h[N-1], 
                    inputs1={"x": hidden_states1}, 
                    inputs2={"x": hidden_states2}, 
                    grad=projected_grad
                )
        torch.cuda.synchronize()    #TODO: remove global sync
        if N-1 in self.offloading_blocks:
            self.model.transformer.h[N-1] = self.task_offload_to(
                self.model.transformer.h[N-1], device=self.offloading_device)
        logits1, logits2 = self.task_compute(self.model.transformer.ln_f,
                                             inputs1={"input": hidden_states1}, 
                                             inputs2={"input": hidden_states2}, 
                                             grad=projected_grad)
        logits1, logits2 = self.task_compute(self.model.lm_head,
                                             inputs1={"input": logits1}, 
                                             inputs2={"input": logits2}, 
                                             grad=projected_grad)
        torch.cuda.synchronize()
        return logits1, logits2
    
    @torch.inference_mode()
    def zo_step(self, inputs, projected_grad=0., seed: int=None):
        self.zo_random_seed = seed if seed else np.random.randint(self.max_zo_random_seed)
        torch.manual_seed(self.zo_random_seed)
        torch.cuda.manual_seed(self.zo_random_seed)
        self.rstate = torch.cuda.get_rng_state()
        logits1, logits2 = self.zo_forward(
            input_ids=inputs["idx"], 
            pos=inputs['pos'],
            projected_grad=projected_grad
        )
        loss1 = F.cross_entropy(
            logits1[:, :-1, :].reshape(-1, logits1.size(-1)), 
            inputs['targets'][:, 1:].reshape(-1)
        )
        loss2 = F.cross_entropy(
            logits2[:, :-1, :].reshape(-1, logits2.size(-1)), 
            inputs['targets'][:, 1:].reshape(-1)
        )
        projected_grad = ((loss1 - loss2) / (self.zo_eps * 2)).item()
        return projected_grad, loss1.item()
    