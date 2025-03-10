# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import os
import torch
import torch.nn as nn


def module_to_bucket_inplace(module: nn.Module):
    bucket = torch.cat([p.view(-1) for p in module.parameters()])
    return bucket

def bucket_to_module_inplace(bucket: torch.Tensor, module: nn.Module):
    offset = 0
    for name, param in module.named_parameters():
        num_elements = param.numel()
        new_param = bucket[offset: offset+num_elements].view_as(param)
        set_nested_attr(module, name, nn.Parameter(new_param, requires_grad=param.requires_grad))
        offset += num_elements
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



def set_nested_attr(obj, attr, value):
    attrs = attr.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)
