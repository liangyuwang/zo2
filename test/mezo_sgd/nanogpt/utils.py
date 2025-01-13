import torch
import time
import argparse
from tqdm import tqdm


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--zo_method", type=str, default="zo2")
    args.add_argument("--model_id", type=str, default="gpt2")
    args.add_argument("--verbose", action="store_true")
    args.add_argument("--max_steps", type=int, default=3)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--weight_decay", type=float, default=1e-1)
    args.add_argument("--zo_eps", type=float, default=1e-3)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--offloading_device", type=str, default="cpu")
    args.add_argument("--working_device", type=str, default="cuda:0")
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


def check_peak_memory_usage(iter, device="cuda:0", use_tqdm=False):
    # Check the peak memory usage
    peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
    if use_tqdm:
        tqdm.write("Peak GPU Memory after iteration {}: {:.2f} MB".format(iter+1, peak_memory))
    else:
        print(f"Peak GPU Memory after iteration {iter+1}: {peak_memory:.2f} MB")
    torch.cuda.reset_peak_memory_stats(device=device)


def check_throughput(iter, total_token_batch_size_per_iter, fn, *args, use_tqdm=False, **kwargs):
    t1 = time.time()
    out = fn(*args, **kwargs)
    t2 = time.time()
    time_cost = t2-t1
    throughtput = total_token_batch_size_per_iter / time_cost
    if use_tqdm:
        tqdm.write("Time cost after iteration {}: {:.2f} ms, {:.2f} tok/s".format(iter+1, time_cost*1e3, throughtput))
    else:
        print("Time cost after iteration {}: {:.2f} ms, {:.2f} tok/s".format(iter+1, time_cost*1e3, throughtput))
