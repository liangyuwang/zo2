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

def bucket_to_module_inplace(bucket: torch.Tensor, module: nn.Module, shapes):
    offset = 0
    for param, shape in zip(module.parameters(), shapes):
        num_elements = torch.prod(torch.tensor(shape)).item()
        param.data.copy_(bucket[offset:offset + num_elements].view(shape))
        offset += num_elements

