import torch
from dataclasses import dataclass

@dataclass
class MeZOSGDConfig:
    # zo method
    zo_method: str = "mezo-sgd" # zo method name, every zo config must include this attribute

    # zo config
    lr: float = 1e-3
    weight_decay: float = 1e-1
    eps: float = 1e-3
    max_zo_random_seed = 1000000000

    # zo2 config
    zo2: bool = True    # use offloading or not
    offloading_blocks: list = None  # specify offloading blocks or not
    offloading_device: str = 'cpu'  # offload device, can be CPU or a file path (disk offloading, but currently unavailable)
    working_device: str = 'cuda'    # compute device, can be any CUDA device
    overlap: bool = True    # use scheduler to overlap or not
    compute_module_optimize_method: str = ''   # '', 'torch.compile'
    compute_function_optimize_method: str = ''   # '', 'torch.jit.script'
    communicate_optimize_method: str = ''   # '', 'bucket'
    amp: bool = False   # use amp or not
    precision_on_offloading_device: torch.dtype = torch.float16 # precision on offloading device, valid when using amp
    precision_on_working_device: torch.dtype = torch.float32    # precision on working device, valid when using amp
    debug_mode: bool = False    # set 'True' to disable random noise