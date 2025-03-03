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
    Extends MeZOSGD to support advanced offloading techniques that enhance the capability
    to train large models on systems with limited GPU memory. It manages the intricate
    balance between CPU and GPU, leveraging zeroth-order optimization with dynamic memory
    management through offloading.
    """
    def __init__(self, model, config: MeZOSGDConfig):
        """
        Initializes the MeZO2SGD optimizer, setting up the necessary configuration for
        offloading and optimization techniques.

        Args:
            model (nn.Module): The model whose parameters will be optimized.
            config (MeZOSGDConfig): Configuration object specifying optimizer settings including
                                    offloading and overlapping options.
        """
        assert config.zo2, "MeZO2SGD can only work with offloading."
        super().__init__(model, config)
        self.device = config.working_device
        self.offloading_device = config.offloading_device
        self.overlap = config.overlap
        self.offloading_blocks = config.offloading_blocks
        self.compute_module_optimize_method = config.compute_module_optimize_method
        self.compute_function_optimize_method = config.compute_function_optimize_method
        self.communicate_optimize_method = config.communicate_optimize_method
        self.amp = config.amp
        self.amp_precision = config.amp_precision
        self.precision_on_offloading_device = config.precision_on_offloading_device
        self.precision_on_working_device = config.precision_on_working_device
        self.amp_compress_method = config.amp_compress_method
        self.init_zo2()
    
    def init_zo2(self):
        """
        Sets up CUDA streams and initializes the offloading and uploading mechanisms
        required for efficient computation management across devices.
        """
        self.upload_stream = torch.cuda.Stream()
        self.offload_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.zo_random_seed = None
        self.rstate = None
        self.rstate_queue = deque(maxlen=2)
        self.last_rstate = None
        self.projected_grad = 0
        self.init_zo2_upload()
        if self.amp: self.init_zo2_amp()
    
    def init_zo2_amp(self):
        working_device = torch.device(self.device)
        offloading_device = torch.device(self.offloading_device)
        for p in self.model.parameters():
            if p.device == working_device:
                p.data = p.data.to(dtype=self.precision_on_working_device)
            elif p.device == offloading_device:
                p.data = p.data.to(dtype=self.precision_on_offloading_device)
            else:
                raise ValueError(f"Unsupported device found for parameter: {p.device}")

    def assign_zo2_attributes(self, source, target):
        """
        Utility function to transfer ZO2 specific attributes from one module to another,
        aiding in maintaining consistency across nested model architectures.

        Args:
            source: The source module from which attributes are copied.
            target: The target module to which attributes are assigned.
        """
        attrs_to_assign = ['upload_stream', 'offload_stream', 'compute_stream', 
                           'zo_random_seed', 'rstate', 'rstate_queue', 'last_rstate', 
                           'projected_grad']
        for attr in attrs_to_assign:
            setattr(target, attr, getattr(source, attr))
    
    @torch.inference_mode
    def zo_update(self, module, weight_decay=None):
        """
        Applies the computed gradients to update parameters of the module, potentially
        including a weight decay term. This method is enhanced by managing CUDA state
        to ensure consistent random number generation across calls.

        Args:
            module (nn.Module): The module whose parameters are to be updated.
            weight_decay (float, optional): Optional weight decay for regularization.
        """
        torch.cuda.set_rng_state(self.last_rstate)
        super().zo_update(module, weight_decay=weight_decay)
        self.last_rstate = torch.cuda.get_rng_state()
        return module
    
    @torch.inference_mode()
    def module_dual_forward(self, module, inputs1, inputs2, projected_grad=0., weight_decay=None):
        """
        Performs two parallel forward computations with perturbed parameters to estimate
        gradients. This function is key for zeroth-order gradient estimation with support
        for optional weight decay during parameter update. 
        
        Notice that the application of Gaussian perturbations for the parameters 
        during both the perturbation and update phases should be the same.

        Args:
            module (nn.Module): The module on which forward passes are conducted.
            inputs1 (dict): Inputs for the first forward pass.
            inputs2 (dict): Inputs for the second forward pass.
            projected_grad (float): Projected gradient value used for updating parameters.
            weight_decay (float, optional): Optional weight decay for regularization.
        """
        if projected_grad != 0:
            module = self.zo_update(module, weight_decay)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=self.zo_perturb_shifts()[0])
        output1 = module(**inputs1)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=self.zo_perturb_shifts()[1])
        output2 = module(**inputs2)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=self.zo_perturb_shifts()[2])
        self.rstate = torch.cuda.get_rng_state()
        return output1, output2
    
    @torch.inference_mode()
    def function_dual_forward(self, fn, inputs1, inputs2):
        """
        Executes a provided function twice with dual inputs, supporting the zeroth-order optimization process
        by enabling the estimation of gradients through function outputs.

        Args:
            fn (callable): The function to be executed.
            inputs1 (dict): Arguments for the first execution of the function.
            inputs2 (dict): Arguments for the second execution of the function.

        Returns:
            tuple: Outputs from the two executions of the function.
        """
        output1 = fn(**inputs1)
        output2 = fn(**inputs2)
        return output1, output2
    
    @torch.inference_mode()
    def zo_forward(self, *args, seed: int=None, **kwargs):
        """
        The overarching forward function that integrates perturbation, gradient estimation,
        and parameter update within a single coherent process, controlled by the seed for reproducibility.

        Args:
            seed (int, optional): Seed for random number generation to ensure reproducibility.
        """
        self._update_lr()
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
        return loss1.detach()
    
    #*********************** tasks ***********************#

    def task_upload(self, module, device='cuda', upload_sync=True, *args, **kwargs):
        """
        Handles the uploading of modules to the GPU, utilizing CUDA streams to potentially overlap
        computation and communication for efficiency.

        Args:
            module (nn.Module): Module to be uploaded.
            device (str): Target device for the upload.
            upload_sync (bool): Whether to synchronize the upload stream before proceeding.
        """
        if self.overlap:
            if upload_sync:
                self.upload_stream.synchronize()
        with torch.cuda.stream(self.upload_stream if self.overlap else torch.cuda.current_stream()):
            module = self.upload_impl(
                module, 
                device, 
                self.offloading_device,
                self.communicate_optimize_method, 
                non_blocking=self.overlap, 
                *args, **kwargs
            )
        return module

    def task_offload(self, module, device='cpu', offload_sync=True, *args, **kwargs):
        """
        Manages the offloading of modules to an alternative storage (e.g., CPU or disk), using CUDA streams
        to manage dependencies and potentially overlap tasks.

        Args:
            module (nn.Module): Module to be offloaded.
            device (str): Target device for the offload.
            offload_sync (bool): Whether to synchronize the offload stream before proceeding.
        """
        if self.overlap:
            if offload_sync:
                self.offload_stream.synchronize()
            self.compute_stream.synchronize()   # offload depends on compute task
        with torch.cuda.stream(self.offload_stream if self.overlap else torch.cuda.current_stream()):
            module = self.offload_impl(
                module, 
                device, 
                self.offloading_device,
                self.communicate_optimize_method, 
                non_blocking=self.overlap, 
                *args, **kwargs
            )
        return module
    
    def task_compute_module(self, module, inputs1, inputs2, grad, compute_sync=True, weight_decay=None, *args, **kwargs):
        """
        Conducts computations on a module with optional dual inputs for gradient estimation,
        applying synchronization and CUDA streams for efficiency.

        Args:
            module (nn.Module): The module on which computations are to be performed.
            inputs1 (dict): Inputs for the first computation.
            inputs2 (dict, optional): Inputs for the second computation, if performing dual forward.
            grad (float): Gradient value to be applied.
            compute_sync (bool): Whether to synchronize the compute stream before proceeding.
            weight_decay (float, optional): Optional weight decay during the update.
        """
        if self.overlap:
            if compute_sync:
                self.compute_stream.synchronize()
            self.upload_stream.synchronize()   # module compute depends on upload task
        with torch.cuda.stream(self.compute_stream if self.overlap else torch.cuda.current_stream()):
            if inputs2 is not None:
                return self.compute_module_impl(
                    self.module_dual_forward,
                    module,
                    self.compute_module_optimize_method,
                    inputs1=inputs1, 
                    inputs2=inputs2,
                    projected_grad=grad,
                    weight_decay=weight_decay,
                    *args, **kwargs
                )
            elif isinstance(inputs1, list):
                return self.compute_module_impl(
                    None,
                    module,
                    self.compute_module_optimize_method,
                    *inputs1,
                    *args,
                    **kwargs
                )
            elif isinstance(inputs1, dict):
                return self.compute_module_impl(
                    None,
                    module,
                    self.compute_module_optimize_method,
                    *args,
                    **inputs1,
                    **kwargs
                )
            elif isinstance(inputs1, tuple):
                return self.compute_module_impl(
                    None,
                    module,
                    self.compute_module_optimize_method,
                    *inputs1[0],
                    *args,
                    **inputs1[1],
                    **kwargs
                )
            else:
                raise ValueError("Invalid inputs type.")
    
    def task_compute_function(self, fn, inputs1, inputs2, compute_sync=True, *args, **kwargs):
        """
        Executes a provided function with dual input sets to facilitate parallel operations
        and gradient estimation. This method integrates CUDA streams for efficient task execution.

        Args:
            fn (callable): The function to execute, typically a PyTorch operation or custom function.
            inputs1 (dict): Arguments for the first execution of the function.
            inputs2 (dict): Arguments for the second execution of the function.
            compute_sync (bool): Whether to synchronize the compute stream before execution to ensure data readiness.
        """
        if self.overlap:
            if compute_sync:
                self.compute_stream.synchronize()
        with torch.cuda.stream(self.compute_stream if self.overlap else torch.cuda.current_stream()):
            if inputs2 is not None:
                return self.compute_function_impl(
                    self.function_dual_forward,
                    fn,
                    self.compute_function_optimize_method,
                    inputs1=inputs1, 
                    inputs2=inputs2,
                    *args, **kwargs
                )
            elif isinstance(inputs1, list):
                return self.compute_function_impl(
                    None,
                    fn, 
                    self.compute_function_optimize_method,
                    *inputs1,
                    *args,
                    **kwargs
                )
            elif isinstance(inputs1, dict):
                return self.compute_function_impl(
                    None,
                    fn, 
                    self.compute_function_optimize_method,
                    *args,
                    **inputs1,
                    **kwargs
                )
            elif isinstance(inputs1, tuple):
                return self.compute_function_impl(
                    None,
                    fn, 
                    self.compute_function_optimize_method,
                    *inputs1[0],
                    *args,
                    **inputs1[1],
                    **kwargs
                )
            else:
                raise ValueError("Invalid inputs type.")

    #*********************** evaluate ***********************#

    @torch.inference_mode()
    def zo_eval_forward(self, *args, **kwargs):
        """
        Conducts a model evaluation using the internal forward method without applying any perturbations.
        This method ensures all tasks finish before and after the evaluation to maintain synchronization.

        Args:
            *args, **kwargs: Arguments and keyword arguments for the model's forward method.
        """
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        output = self.inner_zo_eval_forward(*args, **kwargs)
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        return output
    
    def add_zo2_eval_comm_hooks(self, blocks):
        """
        Attaches communication hooks to model blocks to manage data uploading and offloading during evaluation.
        This helps in managing memory more efficiently during the eval phase.

        Args:
            blocks (list): List of model blocks to attach hooks to.

        Returns:
            list: A list of hook handles for managing lifecycle.
        """
        handles = []
        for block in blocks:
            if isinstance(block, nn.Module):
                pre_handle = block.register_forward_pre_hook(self.eval_upload_hook)
                post_handle = block.register_forward_hook(self.eval_offload_hook)
                handles.append(pre_handle)
                handles.append(post_handle)
        return handles
    
    def clear_zo2_eval_comm_hooks(self, handles):
        """
        Removes communication hooks from model blocks after evaluation to clean up and prevent memory leaks.

        Args:
            handles (list): List of hook handles to be removed.
        """
        for handle in handles:
            handle.remove()
    
    def eval_upload_hook(self, module, input):
        """
        A forward pre-hook to upload a module to the GPU before its evaluation.

        Args:
            module (nn.Module): Module to be uploaded.
            input: Input data for the module.
        """
        self.upload_impl(
            module, 
            self.device, 
            self.offloading_device
        )
        return input

    def eval_offload_hook(self, module, input, output):
        """
        A forward hook to offload a module from the GPU after its evaluation to free up memory.

        Args:
            module (nn.Module): Module to be offloaded.
            input: Input data for the module.
            output: Output from the module evaluation.
        """
        if self.overlap:
            with torch.cuda.stream(self.offload_stream):
                self.offload_impl(
                    module, 
                    self.offloading_device, 
                    self.offloading_device
                )
        else:
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
        """
        Implements the logic for uploading model components to a specified device.
        Supports various optimization methods to tailor the upload process for different computing environments.
        """
        def _upload_impl(module, device, offloading_device, *args, **kwargs):
            if offloading_device == "cpu":
                module = module.to(device, *args, **kwargs)
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
            return module
        match optimize_method:
            case "":
                module = _upload_impl(module, device, offloading_device, *args, **kwargs)
            case "bucket":  # works on large-scale models
                bucket, shapes = module_to_bucket_inplace(module)
                bucket = _upload_impl(bucket, device, offloading_device, *args, **kwargs)
                module = bucket_to_module_inplace(bucket, module, shapes)
            case _:
                raise NotImplementedError
        if self.amp:    # after uploading, decompress the module to higher precision
            module = self.amp_decompress_impl(module)
        return module

    def offload_impl(
            self,
            module: nn.Module, 
            device: str, 
            offloading_device: str,
            optimize_method: str = "", 
            module_id: str = None,
            *args, **kwargs
        ):
        """
        Implements the logic for offloading model components from the GPU to another storage,
        such as CPU or disk, to manage GPU memory more efficiently.
        """
        def _offload_impl(module, device, offloading_device, *args, **kwargs):
            if offloading_device == "cpu":
                module = module.to(device, *args, **kwargs)
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
        if self.amp:    # before offloading, compress the module to lower precision
            module = self.amp_compress_impl(module)
        match optimize_method:
            case "":
                module = _offload_impl(module, device, offloading_device, *args, **kwargs)
            case "bucket":  # works on large-scale models
                bucket, shapes = module_to_bucket_inplace(module)
                bucket = _offload_impl(bucket, device, offloading_device, *args, **kwargs)
                module = bucket_to_module_inplace(bucket, module, shapes)
            case _:
                raise NotImplementedError
        return module
        
    def compute_module_impl(
            self,
            forward_fn,
            module: torch.nn.Module,
            optimize_method: str,
            *args, 
            optimize_kwargs = None,
            **kwargs
        ):
        """
        Manages the computation tasks on a module, applying various optimization methods
        to enhance execution speed and efficiency.
        """
        match optimize_method:
            case "":
                pass
            case "torch.compile":   # may introduce some precision mismatch
                module = torch.compile(module, **optimize_kwargs)
            case _:
                raise NotImplementedError
        with torch.autocast(device_type=self.device, dtype=self.amp_precision, enabled=self.amp):
            if forward_fn is None:
                return module(*args, **kwargs)
            else:
                return forward_fn(module=module, *args, **kwargs)

    def compute_function_impl(
            self,
            function_fn,
            fn,
            optimize_method: str,
            *args, 
            optimize_kwargs = None,
            **kwargs
        ):
        """
        Manages the computation tasks on a function, applying various optimization methods
        to enhance function execution speed and efficiency.
        """
        match optimize_method:
            case "":
                pass
            case "torch.jit.script":   # may introduce some precision mismatch
                fn = torch.jit.script(fn, **optimize_kwargs)
            case _:
                raise NotImplementedError
        with torch.autocast(device_type=self.device, dtype=self.amp_precision, enabled=self.amp):
            if function_fn is None:
                return fn(*args, **kwargs)
            else:
                return function_fn(fn, *args, **kwargs)

    def amp_decompress_impl(self, module: nn.Module) -> nn.Module:
        for p in module.parameters():
            match self.amp_compress_method:
                case "naive":
                    p.data = p.data.to(dtype=self.precision_on_working_device)
                case _:
                    raise NotImplementedError
        return module

    def amp_compress_impl(self, module: nn.Module) -> nn.Module:
        for p in module.parameters():
            match self.amp_compress_method:
                case "naive":
                    p.data = p.data.to(dtype=self.precision_on_offloading_device)
                case _:
                    raise NotImplementedError
        return module

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
    