import sys
sys.path.append("../zo2")

import torch
from tqdm import tqdm

from zo2.config.mezo_sgd import MeZOSGDConfig
from zo2.model.nanogpt.mezo_sgd import get_nanogpt_mezo_sgd
from zo2.model.nanogpt.model import GPTConfig, GPTConfigs
from zo2.utils.utils import seed_everything
from utils import model_size, prepare_data, get_args

def train_mezo_sgd(model, args, modelConfig, device='cuda'):
    seed_everything(args.seed)
    total_parameters = model_size(model)["total"]
    model.eval()
    print(f"model size: {total_parameters/1024**3:.2f} B")
    print("Init dataset")
    B, T = args.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T)).to(device)
    input_ids, pos, labels = prepare_data(data, labels=None, device=device)
    for i in tqdm(range(args.max_steps)):
        loss = model(input_ids, pos, labels)
        res = "Iteration {}, loss: {}, projected grad: {}"
        tqdm.write(res.format(i, loss, model.opt.projected_grad))

def train_mezo2_sgd(model, args, modelConfig, device='cuda'):
    seed_everything(args.seed)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    print("Init dataset")
    B, T = args.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T)).to(device)
    input_ids, pos, labels = prepare_data(data, labels=None, device=device)
    model.eval()
    for i in tqdm(range(args.max_steps)):
        p_grad, loss = model(input_ids, pos, labels)
        res = "Iteration {}, loss: {}, projected grad: {}"
        tqdm.write(res.format(i, loss, p_grad))

def test_mezo_sgd_training():
    seed_everything(args.seed)
    cfgs = GPTConfigs()
    cfg = getattr(cfgs, args.model_id)
    zo_cfg = MeZOSGDConfig()
    zo_cfg.zo2 = False
    model_mezo = get_nanogpt_mezo_sgd(zo_cfg)(cfg, zo_cfg).to(args.device)
    train_mezo_sgd(model=model_mezo, 
               args=args, 
               modelConfig=cfg, 
               device=args.device)

def test_mezo2_sgd_training():
    seed_everything(args.seed)
    cfgs = GPTConfigs()
    cfg = getattr(cfgs, args.model_id)
    zo_cfg = MeZOSGDConfig()
    zo_cfg.zo2 = True
    model = get_nanogpt_mezo_sgd(zo_cfg)(cfg, zo_cfg)
    train_mezo2_sgd(model=model, 
                          args=args, 
                          modelConfig=cfg, 
                          device=args.device)


if __name__ == "__main__":
    args = get_args()
    if args.zo_method == "zo":
        test_mezo_sgd_training()
    elif args.zo_method == "zo2":
        test_mezo2_sgd_training()
    else:
        raise ValueError