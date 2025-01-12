from torch import nn
import torch
import os
import random
import numpy as np

def print_all(module: nn.Module, inputs, outputs):
    print("Param: ")
    for p in module.parameters():
        print(p.min().item(), p.max().item(), p.mean().item())
    print("Inputs: ")
    if isinstance(inputs, torch.Tensor):
        print(inputs.min().item(), inputs.max().item())
    else:
        for _, input in inputs.items():
            if isinstance(input, torch.Tensor):
                print(input.min().item(), input.max().item())
    print("Output: ")
    if isinstance(outputs, torch.Tensor):
        print(outputs.min().item(), outputs.max().item(), outputs.mean().item())
    else:
        print("Unrecongized outputs.")
    print("*" * 20)
        

def print_hook(module, input, output):
    print(module, f"{module.weight.min().item():.4f}, {module.weight.max().item():.4f}")
    print(f"{output.min().item():.8f} {output.max().item():.8f} {output.mean().item():.8f}")

def print_para_and_device(model):
    for p, v in model.named_parameters():
        print(f"{p}: {v.device}")

def cal_self_reg_loss(logits, labels):
    loss = nn.CrossEntropyLoss()(
        logits[:, :-1, :].reshape(-1, logits.size(-1)), 
        labels[:, 1:].reshape(-1)
    )
    return loss

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
