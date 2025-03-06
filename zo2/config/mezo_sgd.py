# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

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
    offloading_device: str = 'cpu'  # offload device, can be CPU or a path (for disk offloading, but currently unavailable)
    working_device: str = 'cuda'    # compute device, can be any CUDA device
    overlap: bool = True    # use scheduler to overlap or not
    compute_module_optimize_method: str = ''   # possible values are: ['', 'torch.compile']
    compute_function_optimize_method: str = ''   # possible values are: ['', 'torch.jit.script']
    communicate_optimize_method: str = ''   # possible values are: ['', 'bucket']
    amp: bool = False   # use amp or not
    amp_precision: torch.dtype = torch.bfloat16 # amp autocast precision, possible values are: [torch.bfloat16, torch.float32], valid when using amp
    precision_on_offloading_device: torch.dtype = torch.float16 # precision on offloading device, valid when using amp
    precision_on_working_device: torch.dtype = torch.float32    # precision on working device, valid when using amp
    amp_compress_method: str = 'naive'  # currently only support naive amp compress, valid when using amp

    # debug
    debug_mode: bool = False    # set 'True' to disable random noise