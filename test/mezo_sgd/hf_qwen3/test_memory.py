# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import sys
sys.path.append("../zo2")

import torch
from tqdm import tqdm

from zo2.config.mezo_sgd import MeZOSGDConfig
from zo2.model.huggingface.qwen3.mezo_sgd import zo, zo2
from zo2.utils.utils import seed_everything
from utils import (
    Qwen3Configs,
    prepare_data_for_causalLM, 
    model_size, 
    get_args,
    reset_peak_cpu_memory_usage,
    check_peak_gpu_memory_usage,
    check_and_update_peak_cpu_memory_usage,
)

def train_mezo_sgd_causalLM(model_config, zo_config, device='cuda:0'):
    input_ids, labels = prepare_data_for_causalLM(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo.Qwen3ForCausalLM(model_config).to(device)
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    torch.cuda.reset_peak_memory_stats()
    reset_peak_cpu_memory_usage()
    for i in tqdm(range(args.max_steps)):
        model.zo_train()
        model(input_ids=input_ids, labels=labels)
        check_peak_gpu_memory_usage(i, int(device[-1]), True)
        check_and_update_peak_cpu_memory_usage(i, True)

def train_mezo2_sgd_causalLM(model_config, zo_config, device='cuda:0'):
    input_ids, labels = prepare_data_for_causalLM(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo2.Qwen3ForCausalLM(model_config)
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    torch.cuda.reset_peak_memory_stats()
    reset_peak_cpu_memory_usage()
    for i in tqdm(range(args.max_steps)):
        model.zo_train()
        model(input_ids=input_ids, labels=labels)
        check_peak_gpu_memory_usage(i, int(device[-1]), True)
        check_and_update_peak_cpu_memory_usage(i, True)

def eval_mezo_sgd_causalLM(model_config, zo_config, device='cuda:0'):
    input_ids, labels = prepare_data_for_causalLM(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo.Qwen3ForCausalLM(model_config).to(device)
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    torch.cuda.reset_peak_memory_stats()
    reset_peak_cpu_memory_usage()
    for i in tqdm(range(args.max_steps)):
        model.zo_eval()
        model(input_ids=input_ids, labels=labels)
        check_peak_gpu_memory_usage(i, int(device[-1]), True)
        check_and_update_peak_cpu_memory_usage(i, True)

def eval_mezo2_sgd_causalLM(model_config, zo_config, device='cuda:0'):
    input_ids, labels = prepare_data_for_causalLM(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo2.Qwen3ForCausalLM(model_config)
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    torch.cuda.reset_peak_memory_stats()
    reset_peak_cpu_memory_usage()
    for i in tqdm(range(args.max_steps)):
        model.zo_eval()
        model(input_ids=input_ids, labels=labels)
        check_peak_gpu_memory_usage(i, int(device[-1]), True)
        check_and_update_peak_cpu_memory_usage(i, True)



def test_mezo_sgd_causalLM_training():
    seed_everything(args.seed)
    model_configs = Qwen3Configs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    model_config.max_position_embeddings = args.sequence_length
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    train_mezo_sgd_causalLM(model_config, zo_cfg, device=args.working_device)

def test_mezo2_sgd_causalLM_training():
    seed_everything(args.seed)
    model_configs = Qwen3Configs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    model_config.max_position_embeddings = args.sequence_length
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        offloading_device=args.offloading_device, working_device=args.working_device)
    zo_cfg.zo2 = True
    train_mezo2_sgd_causalLM(model_config, zo_cfg, device=args.working_device)

def test_mezo_sgd_causalLM_eval():
    seed_everything(args.seed)
    model_configs = Qwen3Configs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    model_config.max_position_embeddings = args.sequence_length
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    eval_mezo_sgd_causalLM(model_config, zo_cfg, device=args.working_device)

def test_mezo2_sgd_causalLM_eval():
    seed_everything(args.seed)
    model_configs = Qwen3Configs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    model_config.max_position_embeddings = args.sequence_length
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        offloading_device=args.offloading_device, working_device=args.working_device)
    zo_cfg.zo2 = True
    eval_mezo2_sgd_causalLM(model_config, zo_cfg, device=args.working_device)



if __name__=="__main__":
    args = get_args()
    original_dtype = torch.get_default_dtype()
    if args.zo_method == "zo":
        if args.eval:
            test_mezo_sgd_causalLM_eval()
        else:
            test_mezo_sgd_causalLM_training()
    elif args.zo_method == "zo2":
        if args.eval:
            test_mezo2_sgd_causalLM_eval()
        else:
            test_mezo2_sgd_causalLM_training()
    else:
        raise NotImplementedError