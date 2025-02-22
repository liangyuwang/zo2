import sys
sys.path.append("../zo2")

import torch
from tqdm import tqdm

from zo2.config.mezo_sgd import MeZOSGDConfig
from zo2.model.nanogpt.mezo_sgd import get_nanogpt_mezo_sgd
from zo2.model.nanogpt.model import GPTConfig, GPTConfigs
from zo2.utils.utils import seed_everything
from utils import model_size, prepare_data, get_args, check_throughput

def train_mezo_sgd(model, args, modelConfig, device='cuda'):
    seed_everything(args.seed)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    print("Init dataset")
    input_ids, pos, labels = prepare_data(modelConfig.vocab_size, args.batch_size, modelConfig.block_size, device=device)
    for i in tqdm(range(args.max_steps)):
        check_throughput(i, args.batch_size*modelConfig.block_size, model, input_ids, pos, labels, use_tqdm=True)

def train_mezo2_sgd(model, args, modelConfig, device='cuda'):
    seed_everything(args.seed)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    print("Init dataset")
    input_ids, pos, labels = prepare_data(modelConfig.vocab_size, args.batch_size, modelConfig.block_size, device=device)
    for i in tqdm(range(args.max_steps)):
        check_throughput(i, args.batch_size*modelConfig.block_size, model, input_ids, pos, labels, use_tqdm=True)

def test_mezo_sgd_training():
    seed_everything(args.seed)
    cfgs = GPTConfigs()
    cfg = getattr(cfgs, args.model_id)
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    torch.set_default_dtype(args.model_dtype)
    model_mezo = get_nanogpt_mezo_sgd(zo_cfg)(cfg, zo_cfg).to(args.working_device)
    torch.set_default_dtype(original_dtype)
    train_mezo_sgd(model=model_mezo, 
               args=args, 
               modelConfig=cfg, 
               device=args.working_device)

def test_mezo2_sgd_training():
    seed_everything(args.seed)
    cfgs = GPTConfigs()
    cfg = getattr(cfgs, args.model_id)
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        offloading_device=args.offloading_device, working_device=args.working_device)
    zo_cfg.zo2 = True
    torch.set_default_dtype(args.model_dtype)
    model = get_nanogpt_mezo_sgd(zo_cfg)(cfg, zo_cfg)
    torch.set_default_dtype(original_dtype)
    train_mezo2_sgd(model=model, 
                          args=args, 
                          modelConfig=cfg, 
                          device=args.working_device)


if __name__ == "__main__":
    args = get_args()
    original_dtype = torch.get_default_dtype()
    if args.zo_method == "zo":
        test_mezo_sgd_training()
    elif args.zo_method == "zo2":
        test_mezo2_sgd_training()
    else:
        raise NotImplementedError