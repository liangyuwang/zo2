import sys
sys.path.append("../zo2")

import argparse
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from transformers import TrainingArguments
# from trl import SFTTrainer
from zo2.trainer.hf_trl.sft_trainer import ZOSFTTrainer as SFTTrainer
from zo2 import zo_hf_init, ZOConfig
from zo2.utils.utils import seed_everything

# Hyper
args = argparse.ArgumentParser()
args.add_argument("--zo_method", type=str, default="zo2")
args.add_argument("--eval", action="store_true")
args.add_argument("--model_name", type=str, default="facebook/opt-125m")
args.add_argument("--verbose", action="store_true")
args.add_argument("--max_steps", type=int, default=10)
args.add_argument("--lr", type=float, default=1e-6)
args.add_argument("--weight_decay", type=float, default=1e-1)
args.add_argument("--zo_eps", type=float, default=1e-3)
args.add_argument("--seed", type=int, default=42)
args.add_argument("--offloading_device", type=str, default="cpu")
args.add_argument("--working_device", type=str, default="cuda:0")
args.add_argument("--max_train_data", type=int, default=None)
# For evaluate
args.add_argument("--max_eval_data", type=int, default=100)
args.add_argument("--log_every_step", type=int, default=20)
# For inference
args.add_argument("--use_cache", action="store_true")
args.add_argument("--max_new_tokens", type=int, default=50)
args = args.parse_args()
# notice that ZO2 does not optimize the evaluate and the inference, so it may be very slow.

seed_everything(args.seed)

# Prepare dataset
train_dataset = load_dataset("stanfordnlp/imdb", split="train")
if args.max_train_data: train_dataset = train_dataset.select(range(args.max_train_data))
eval_dataset = load_dataset("stanfordnlp/imdb", split="test")
if args.max_eval_data: eval_dataset = eval_dataset.select(range(args.max_eval_data))

# ZO steps
zo_config = ZOConfig(
    method="mezo-sgd", 
    zo2=args.zo_method=="zo2", 
    lr=args.lr,
    weight_decay=args.weight_decay,
    eps=args.zo_eps,
    offloading_device=args.offloading_device,
    working_device=args.working_device,
)

# Load ZO model
with zo_hf_init(zo_config):
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(args.model_name)
    model.zo_init(zo_config)
if args.zo_method != "zo2": model = model.to(args.working_device)
print(f"Check if zo2 init correctly: {hasattr(model, 'zo_training')}")

training_args = TrainingArguments(
    output_dir = "./tmp",
    evaluation_strategy = "steps",
    logging_steps = args.log_every_step,
)

trainer = SFTTrainer(
    model,
    train_dataset = train_dataset,
    eval_dataset=eval_dataset,
    args = training_args,
    dataset_text_field = "text",
    max_seq_length = 512,
)

trainer.train()