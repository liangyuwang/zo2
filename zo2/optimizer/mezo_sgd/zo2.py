import sys
sys.path.append('./zo2')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from .zo import MeZOSGD
from ...config.mezo_sgd import MeZOSGDConfig
from .utils import *


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
        self.debug_mode = config.debug_mode
        self.compute_module_optimize_method = config.compute_module_optimize_method
        self.compute_function_optimize_method = config.compute_function_optimize_method
        self.communicate_optimize_method = config.communicate_optimize_method
        self.init_zo2()
    
    def init_zo2(self):
        self.upload_stream = torch.cuda.Stream()
        self.offload_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.zo_random_seed = None
        self.rstate = None
        self.rstate_queue = deque(maxlen=2)
        self.last_rstate = None
        self.projected_grad = 0
        self.init_zo2_upload()
    
    def assign_zo2_attributes(self, source, target):
        """
            For nested model code.
            Assign source's zo2 attributes to target.
        """
        attrs_to_assign = ['upload_stream', 'offload_stream', 'compute_stream', 
                           'zo_random_seed', 'rstate', 'rstate_queue', 'last_rstate', 
                           'projected_grad']
        for attr in attrs_to_assign:
            setattr(target, attr, getattr(source, attr))
    
    def checkpoint_zo_attributes(self):
        return {
            'zo_random_seed': self.zo_random_seed, 
            'rstate': self.rstate, 
            'rstate_queue': self.rstate_queue, 
            'last_rstate': self.last_rstate, 
            'projected_grad': self.projected_grad
        }
    
    @torch.inference_mode
    def zo_update(self, module, weight_decay=None):
        torch.cuda.set_rng_state(self.last_rstate)
        super().zo_update(module, weight_decay=weight_decay)
        self.last_rstate = torch.cuda.get_rng_state()
        return module
    
    @torch.inference_mode()
    def module_dual_forward(self, module, inputs1, inputs2, projected_grad=0., weight_decay=None):
        if projected_grad != 0:
            module = self.zo_update(module, weight_decay)
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
    
    @torch.inference_mode()
    def function_dual_forward(self, fn, inputs1, inputs2):
        output1 = fn(**inputs1)
        output2 = fn(**inputs2)
        return output1, output2
    
    @torch.inference_mode()
    def zo_forward(self, *args, seed: int=None, **kwargs):
        self.zo_random_seed = seed if seed else np.random.randint(self.max_zo_random_seed)
        torch.manual_seed(self.zo_random_seed)
        torch.cuda.manual_seed(self.zo_random_seed)
        self.rstate = torch.cuda.get_rng_state()
        self.rstate_queue.append(self.rstate.clone())
        if len(self.rstate_queue) == 2:
            self.last_rstate = self.rstate_queue.popleft()
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        loss1, loss2 = self.inner_zo_forward(*args, **kwargs)
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        self.projected_grad = ((loss1 - loss2) / (self.zo_eps * 2)).item()
        return loss1.item()
    
    #*********************** tasks ***********************#

    def task_upload(self, module, device='cuda', upload_sync=True, *args, **kwargs):
        if self.overlap:
            if upload_sync:
                self.upload_stream.synchronize()
            with torch.cuda.stream(self.upload_stream):
                module = self.upload_impl(
                    module, 
                    device, 
                    self.offloading_device,
                    self.communicate_optimize_method, 
                    non_blocking=True, 
                    *args, **kwargs
                )
        else:
            module = self.upload_impl(
                module, 
                device, 
                self.offloading_device,
                self.communicate_optimize_method, 
                *args, **kwargs
            )
        return module

    def task_offload(self, module, device='cpu', offload_sync=True, *args, **kwargs):
        if self.overlap:
            if offload_sync:
                self.offload_stream.synchronize()
            self.compute_stream.synchronize()   # offload depends on compute task
            with torch.cuda.stream(self.offload_stream):
                module = self.offload_impl(
                    module, 
                    device, 
                    self.offloading_device,
                    self.communicate_optimize_method, 
                    non_blocking=True, 
                    *args, **kwargs
                )
        else:
            module = self.offload_impl(
                module, 
                device, 
                self.offloading_device,
                self.communicate_optimize_method, 
                *args, **kwargs
            )
        return module
    
    def task_compute_module(self, module, inputs1, inputs2, grad, compute_sync=True, weight_decay=None, *args, **kwargs):
        if self.overlap:
            if compute_sync:
                self.compute_stream.synchronize()
            self.upload_stream.synchronize()   # module compute depends on upload task
            with torch.cuda.stream(self.compute_stream):
                o1, o2 = self.compute_module_impl(
                    self.module_dual_forward,
                    module,
                    self.compute_module_optimize_method,
                    inputs1=inputs1, 
                    inputs2=inputs2,
                    projected_grad=grad,
                    weight_decay=weight_decay,
                    *args, **kwargs
                )
        else:
            o1, o2 = self.compute_module_impl(
                self.module_dual_forward,
                module,
                self.compute_module_optimize_method,
                inputs1=inputs1, 
                inputs2=inputs2,
                projected_grad=grad,
                weight_decay=weight_decay,
                *args, **kwargs
            )
        return o1, o2
    
    def task_compute_function(self, fn, inputs1, inputs2, compute_sync=True, *args, **kwargs):
        if self.overlap:
            if compute_sync:
                self.compute_stream.synchronize()
            with torch.cuda.stream(self.compute_stream):
                o1, o2 = self.compute_function_impl(
                    self.function_dual_forward,
                    fn,
                    self.compute_function_optimize_method,
                    inputs1=inputs1, 
                    inputs2=inputs2,
                    *args, **kwargs
                )
        else:
            o1, o2 = self.compute_function_impl(
                self.function_dual_forward,
                fn,
                self.compute_function_optimize_method,
                inputs1=inputs1, 
                inputs2=inputs2,
                *args, **kwargs
            )
        return o1, o2
    
    #*********************** evaluate ***********************#

    @torch.inference_mode()
    def zo_eval_forward(self, *args, **kwargs):
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        output = self.inner_zo_eval_forward(*args, **kwargs)
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        return output
    
    def add_zo2_eval_comm_hooks(self, blocks):
        handles = []
        for block in blocks:
            if isinstance(block, nn.Module):
                pre_handle = block.register_forward_pre_hook(self.eval_upload_hook)
                post_handle = block.register_forward_hook(self.eval_offload_hook)
                handles.append(pre_handle)
                handles.append(post_handle)
        return handles
    
    def clear_zo2_eval_comm_hooks(self, handles):
        for handle in handles:
            handle.remove()
    
    def eval_upload_hook(self, module, input):
        self.upload_impl(
            module, 
            self.device, 
            self.offloading_device
        )
        return input

    def eval_offload_hook(self, module, input, output):
        self.offload_impl(
            module, 
            self.offloading_device, 
            self.offloading_device
        )
        return output
    
    #*********************** backend ***********************#

    def upload_impl(
            self,
            module: nn.Module, 
            device: str, 
            offloading_device: str,
            optimize_method: str = "", 
            module_id: str = None,
            *args, **kwargs
        ):
        def _upload_impl(module, device, offloading_device, *args, **kwargs):
            if offloading_device == "cpu":
                return module.to(device, *args, **kwargs)
            else:
                if module_id == None:
                    raise ValueError("For disk offloading mode, 'module_id' cannot be None.")
                offloading_disk_path = get_disk_offload_path(offloading_device, module_id)
                match type(module):
                    case torch.Tensor:
                        module = torch.load(offloading_disk_path, map_location=device)
                    case nn.Module:
                        module.load_state_dict(torch.load(offloading_disk_path, map_location=device))
                    case _:
                        raise ValueError
                clear_disk_offload_path(offloading_device, module_id)
        match optimize_method:
            case "":
                return _upload_impl(module, device, offloading_device, *args, **kwargs)
            case "bucket":  # works on large-scale models
                bucket, shapes = module_to_bucket_inplace(module)
                bucket = _upload_impl(bucket, device, offloading_device, *args, **kwargs)
                return bucket_to_module_inplace(bucket, module, shapes)
            case _:
                raise NotImplementedError

    def offload_impl(
            self,
            module: nn.Module, 
            device: str, 
            offloading_device: str,
            optimize_method: str = "", 
            module_id: str = None,
            *args, **kwargs
        ):
        def _offload_impl(module, device, offloading_device, *args, **kwargs):
            if offloading_device == "cpu":
                return module.to(device, *args, **kwargs)
            else:
                if module_id == None:
                    raise ValueError("For disk offloading mode, 'module_id' cannot be None.")
                offloading_disk_path = create_disk_offload_path(offloading_device, module_id)
                match type(module):
                    case torch.Tensor:
                        torch.save(module, offloading_disk_path)
                    case nn.Module:
                        torch.save(module.state_dict(), offloading_disk_path)
                    case _:
                        raise ValueError
                return module
        match optimize_method:
            case "":
                return _offload_impl(module, device, offloading_device, *args, **kwargs)
            case "bucket":  # works on large-scale models
                bucket, shapes = module_to_bucket_inplace(module)
                bucket = _offload_impl(bucket, device, offloading_device, *args, **kwargs)
                return bucket_to_module_inplace(bucket, module, shapes)
            case _:
                raise NotImplementedError
        
    def compute_module_impl(
            self,
            module_dual_forward,
            module: torch.nn.Module,
            optimize_method: str,
            *args, 
            optimize_kwargs = None,
            **kwargs
        ):
        match optimize_method:
            case "":
                pass
            case "torch.compile":   # may introduce some precision mismatch
                module = torch.compile(module, **optimize_kwargs)
            case _:
                raise NotImplementedError
        return module_dual_forward(module=module, *args, **kwargs)

    def compute_function_impl(
            self,
            function_dual_forward,
            fn,
            optimize_method: str,
            *args, 
            optimize_kwargs = None,
            **kwargs
        ):
        match optimize_method:
            case "":
                pass
            case "torch.jit.script":   # may introduce some precision mismatch
                fn = torch.jit.script(fn, **optimize_kwargs)
            case _:
                raise NotImplementedError
        return function_dual_forward(fn, *args, **kwargs)

    #*********************** api ***********************#

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
        """
        Example of ZO2 inner_zo_forward:
            Match the same args as the original model forward,
            and replace all 'self' to 'self.model'.
        """
        we1, we2 = self.task_compute_module(self.model.transformer.wte,
                                inputs1={"input": idx},
                                inputs2={"input": idx},
                                grad=self.projected_grad)
        pe1, pe2 = self.task_compute_module(self.model.transformer.wpe, 
                                 {"input": pos}, 
                                 {"input": pos}, 
                                 self.projected_grad)
        hidden_states1, hidden_states2 = self.task_compute_function(torch.add,
                                                                    {"input": we1, "other": pe1},
                                                                    {"input": we2, "other": pe2})
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
    
    @torch.inference_mode()   
    def inner_zo_eval_forward(self, eval_fn, idx, pos, targets):
        handles = self.add_zo2_eval_comm_hooks(self.model.transformer.h)
        output = eval_fn(idx, pos, targets)
        self.clear_zo2_eval_comm_hooks(handles)
        return output
    
class MeZO2SGDAMP(MeZO2SGD):
    ...
