import os
import torch
import torch.nn as nn


def upload_impl(
        module: nn.Module, 
        device: str, 
        offloading_device: str,
        optimize_method: str = "", 
        id: str = None,
        *args, **kwargs
    ):
    def _upload_impl(module, device, offloading_device, *args, **kwargs):
        if offloading_device == "cpu":
            return module.to(device, *args, **kwargs)
        else:
            offloading_path = offloading_device.disk_path.pop(-1)
            match type(module):
                case torch.Tensor:
                    module = torch.load(offloading_path)
                case nn.Module:
                    module.load_state_dict(torch.load(offloading_path))
                case _:
                    raise ValueError
            return module
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
        module: nn.Module, 
        device: str, 
        offloading_device: str,
        optimize_method: str = "", 
        id: str = None,
        *args, **kwargs
    ):
    def _offload_impl(module, device, offloading_device, *args, **kwargs):
        if offloading_device == "cpu":
            return module.to(device, *args, **kwargs)
        else:
            if not hasattr(offloading_device, "disk_path"):
                offloading_device.disk_path = []
            offloading_device.disk_path += [ensure_disk_offload_path_valid(offloading_device+id)]
            offloading_path = offloading_device.disk_path[-1]
            match type(module):
                case torch.Tensor:
                    torch.save(module, offloading_path)
                case nn.Module:
                    torch.save(module.state_dict(), offloading_path)
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


def ensure_disk_offload_path_valid(path):
    if os.path.isfile(path):
        raise ValueError("'path' must be a dir.")
    elif os.path.isdir(path):
        file_path = os.path.join(path, 'tmp.pt')
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        os.makedirs(path)
        file_path = os.path.join(path, 'tmp.pt')
    return file_path