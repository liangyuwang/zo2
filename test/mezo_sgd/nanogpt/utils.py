import torch
import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_id", type=str, default="gpt2")
    args.add_argument("--verbose", action="store_true")
    args.add_argument("--max_steps", type=int, default=3)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--zo_eps", type=float, default=1e-3)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--device", type=str, default="cuda")
    args = args.parse_args()
    return args


def model_size(model: torch.nn.Module):
    total_size = sum(p.numel() for p in model.parameters())
    trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_size, "trainable": trainable_size}


def prepare_data(input_ids, labels=None, device='cuda'):
    input_ids = input_ids.to(device)
    pos = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    if labels is not None:
        labels = labels.to(device)
    else:
        labels = input_ids.clone().to(device)
    # return {
    #     'idx': input_ids,
    #     "pos": pos, 
    #     'targets': labels
    # }
    return input_ids, pos, labels
