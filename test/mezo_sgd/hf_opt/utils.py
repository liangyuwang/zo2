# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import time
import argparse
from tqdm import tqdm
import psutil
import os
from transformers import OPTConfig
import pynvml

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--zo_method", type=str, default="zo2")
    args.add_argument("--eval", action="store_true")
    args.add_argument("--task", type=str, default="causalLM")
    args.add_argument("--model_name", type=str, default="opt_125m")
    args.add_argument("--model_dtype", type=str, default="fp16")
    args.add_argument("--verbose", action="store_true")
    args.add_argument("--max_steps", type=int, default=3)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--weight_decay", type=float, default=1e-1)
    args.add_argument("--zo_eps", type=float, default=1e-3)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--sequence_length", type=int, default=2048)
    args.add_argument("--overlap", type=str, default="all")
    args.add_argument("--offloading_device", type=str, default="cpu")
    args.add_argument("--working_device", type=str, default="cuda:0")
    args = args.parse_args()
    args.model_dtype = dtype_lookup[args.model_dtype]
    return args


class OPTConfigs:
    opt_125m: OPTConfig = OPTConfig(num_hidden_layers=12, num_attention_heads=12, hidden_size=768, ffn_dim=3072, max_position_embeddings=2048)
    opt_350m: OPTConfig = OPTConfig(num_hidden_layers=24, num_attention_heads=16, hidden_size=1024, ffn_dim=4096, max_position_embeddings=2048)
    opt_1_3b: OPTConfig = OPTConfig(num_hidden_layers=24, num_attention_heads=32, hidden_size=2048, ffn_dim=8192, max_position_embeddings=2048)
    opt_2_7b: OPTConfig = OPTConfig(num_hidden_layers=32, num_attention_heads=32, hidden_size=2560, ffn_dim=10240, max_position_embeddings=2048)
    opt_6_7b: OPTConfig = OPTConfig(num_hidden_layers=32, num_attention_heads=32, hidden_size=4096, ffn_dim=16384, max_position_embeddings=2048)
    opt_13b: OPTConfig = OPTConfig(num_hidden_layers=40, num_attention_heads=40, hidden_size=5120, ffn_dim=20480, max_position_embeddings=2048)
    opt_30b: OPTConfig = OPTConfig(num_hidden_layers=48, num_attention_heads=56, hidden_size=7168, ffn_dim=28672, max_position_embeddings=2048)
    opt_66b: OPTConfig = OPTConfig(num_hidden_layers=64, num_attention_heads=72, hidden_size=9216, ffn_dim=36864, max_position_embeddings=2048)
    opt_175b: OPTConfig = OPTConfig(num_hidden_layers=96, num_attention_heads=96, hidden_size=12288, ffn_dim=49152, max_position_embeddings=2048)


dtype_lookup = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}


def model_size(model: torch.nn.Module):
    total_size = sum(p.numel() for p in model.parameters())
    trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_size, "trainable": trainable_size}


def prepare_data_for_causalLM(V, B, T, device='cuda'):
    data_batch = torch.randint(0, V, (B, T)).to(device)
    input_ids = data_batch
    labels = data_batch
    return input_ids, labels

def prepare_data_for_sequence_classification(V, B, T, device='cuda'):
    input_ids = torch.randint(0, V, (B, T)).to(device)
    labels = torch.randint(0, 1, (B, )).to(device)
    return input_ids, labels

def prepare_data_for_question_answering(V, B, T, device='cuda'):
    input_ids = torch.randint(0, V, (B, T)).to(device)
    start_positions = torch.randint(0, 1, (B, )).to(device)
    end_positions = torch.randint(1, 2, (B, )).to(device)
    return input_ids, start_positions, end_positions


# GPU Memory Monitoring
pynvml.nvmlInit()
def check_peak_gpu_memory_usage(iter, device=0, use_tqdm=False):
    # Check the peak memory usage
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)  # Adjust index if necessary
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    peak_memory = info.used / 1024**2
    if use_tqdm:
        tqdm.write("Peak GPU Memory after iteration {}: {:.2f} MB".format(iter+1, peak_memory))
    else:
        print(f"Peak GPU Memory after iteration {iter+1}: {peak_memory:.2f} MB")

# CPU Memory Monitoring
peak_memory_cpu = 0
def check_and_update_peak_cpu_memory_usage(iter, use_tqdm=False):
    global peak_memory_cpu
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    if current_memory > peak_memory_cpu:
        peak_memory_cpu = current_memory
    if use_tqdm:
        tqdm.write(f"Peak CPU Memory after iteration {iter+1}: {peak_memory_cpu:.2f} MB")
    else:
        print(f"Peak CPU Memory after iteration {iter+1}: {peak_memory_cpu:.2f} MB")

def reset_peak_cpu_memory_usage():
    global peak_memory_cpu
    peak_memory_cpu = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


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
