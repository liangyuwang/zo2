import os
import torch
import torch.nn as nn


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