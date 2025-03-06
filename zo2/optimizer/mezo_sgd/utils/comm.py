# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import os
import torch
import torch.nn as nn


def module_to_bucket_inplace(module: nn.Module):
    params = list(module.parameters())
    bucket = torch.empty(sum(p.numel() for p in params), device=params[0].device)
    offset = 0
    for p in params:
        num_elements = p.numel()
        bucket[offset:offset + num_elements].copy_(p.view(-1))
        offset += num_elements
    shapes = [p.size() for p in params]
    return bucket, shapes

def bucket_to_module_inplace(bucket: torch.Tensor, module: nn.Module, shapes: list):
    offset = 0
    for param, shape in zip(module.parameters(), shapes):
        num_elements = param.numel()
        param.data.copy_(bucket[offset:offset + num_elements].view(shape))
        offset += num_elements
        print(param.device)
    return module


def create_disk_offload_path(path, module_id):
    if os.path.isfile(path):
        raise ValueError("'path' must be a dir.")
    elif os.path.isdir(path):
        file_path = os.path.join(path, module_id, 'tmp.pt')
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        os.makedirs(path)
        file_path = os.path.join(path, module_id, 'tmp.pt')
    return file_path

def get_disk_offload_path(path, module_id):
    return os.path.join(path, module_id, 'tmp.pt')

def clear_disk_offload_path(path, module_id):
    disk_offload_path = os.path.join(path, module_id)
    if os.path.isdir(disk_offload_path):
        if not os.listdir(disk_offload_path):
            os.rmdir(disk_offload_path)