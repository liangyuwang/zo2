import sys
sys.path.append("../zo2")

import argparse
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer
from zo2 import (
    ZOConfig,
    zo_hf_init,
)
from zo2.utils.utils import seed_everything

# Hyper
args = argparse.ArgumentParser()
args.add_argument("--zo_method", type=str, default="zo2")
args.add_argument("--eval", action="store_true")
args.add_argument("--model_name", type=str, default="facebook/opt-2.7b")
args.add_argument("--verbose", action="store_true")
args.add_argument("--max_steps", type=int, default=100)
args.add_argument("--lr", type=float, default=1e-5)
args.add_argument("--weight_decay", type=float, default=1e-1)
args.add_argument("--zo_eps", type=float, default=1e-3)
args.add_argument("--seed", type=int, default=42)
args.add_argument("--offloading_device", type=str, default="cpu")
args.add_argument("--working_device", type=str, default="cuda:0")
# For inference
args.add_argument("--use_cache", action="store_true")
args.add_argument("--max_new_tokens", type=int, default=50)
args.add_argument("--temperature", type=float, default=1.0)
args = args.parse_args()

seed_everything(args.seed)

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

# Prepare some data
dataset = """
    What is ZO2? 
    ZO2 is an innovative framework specifically designed to enhance the fine-tuning of large language models (LLMs) using zeroth-order (ZO) optimization techniques and advanced offloading technologies. 
    This framework is particularly tailored for setups with limited GPU memory, enabling the fine-tuning of models that were previously unmanageable due to hardware constraints. 
    As the scale of Large Language Models (LLMs) continues to grow, reaching parameter counts in the hundreds of billions, managing GPU memory resources effectively becomes crucial. 
    Efficient GPU memory management is crucial not only because it directly influences model performance and training speed, but also because GPU memory is both expensive and limited in quantity. 
    However, this creates a significant challenge in handling ever-larger models within the physical constraints of current hardware technologies. 
    CPU offloading has become a crucial technique for overcoming this challenge. 
    It involves transferring computations and data from the GPU to the CPU, specifically targeting data or parameters that are less frequently accessed. 
    By offloading these inactive tensors of the neural network, CPU offloading effectively alleviates the memory and computational pressures on GPUs. 
    While CPU offloading has been commonly applied in inference to manage memory-intensive tasks, its application in training, especially fine-tuning, remains less explored. 
    Recently, some works have tried to introduce CPU offloading into LLM training. 
    However, they are typically constrained by the capabilities of first-order optimizers such as SGD and Adaptive Moment Estimation (AdamW), and limited GPU memory, restricting large-scale model scalability on single GPU systems. 
    Using first-order optimizers introduces inefficiencies in CPU offloading: Multiple communication operations during the training of LLMs necessitate offloading the same data twice—once for each pass. 
    This redundancy not only doubles the communication volume between the CPU and GPU but also introduces significant latency due to repetitive data transfers. 
    Furthermore, both parameters and activations are required in the backward pass to complete gradient computations. 
    This means that parameters and activation values must be offloaded during each forward pass and re-uploaded to the GPU for the backward pass, increasing the volume of data transferred, which severely impacts training throughput. 
    On the other hand, zeroth-order (ZO) methods offer a novel approach to fine-tuning LLMs. 
    These methods utilize dual forward passes to estimate parameter gradients and subsequently update parameters. 
    This approach eliminates the traditional reliance on backward passes, thereby streamlining the training process by significantly reducing the number of computational steps required. 
    Based on these observations, we conjecture that ZO's architecture is particularly well-suited for CPU offloading strategies. 
    By eliminating backward passes and the need to store activation values, it can significantly reduce GPU memory demands through efficient parameter offloading. 
    However, despite these advantages, ZO training via CPU offloading introduces new challenges, particularly in the realm of CPU-to-GPU communication. 
    Transferring parameters between the CPU and GPU, which is crucial for maintaining gradient computation and model updates, becomes a critical bottleneck. 
    Although ZO methods inherently extend computation times because of the dual forward passes, potentially allowing for better overlap between computation and communication, there remain significant inefficiencies. 
    The necessity to upload parameters to the GPU for upcoming computations introduces a large volume of communications. To tackle the inefficiencies highlighted, we introduce ZO2, a novel framework specifically designed for ZO fine-tuning in LLMs with CPU offloading. 
    This framework utilizes the unique dual forward pass architecture of ZO methods to optimize interactions between CPU and GPU, significantly enhancing both computational and communication efficiency. 
    By building a high-performance dynamic scheduler, ZO2 achieves substantial overlaps in communication and computation. 
    These innovations make it feasible to fine-tune extremely large models, such as the OPT-175B, with over 175 billion parameters, on a single GPU equipped with just 18GB of memory usage—a capability previously unattainable with conventional methods. 
    Additionally, our efficient framework operates without any extra time cost and decreases in accuracy compared to standard ZO methodologies."""
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
data_batch = tokenizer(dataset, add_special_tokens=True, return_tensors='pt').input_ids.to(args.working_device)
T = min(data_batch.shape[1] - 1, model.config.max_position_embeddings)
print(f"Fine-tuning model {args.model_name} with {T} tokens dataset: \n{dataset}")

# Training loop
for i in tqdm(range(args.max_steps)):
    # train
    model.zo_train()
    loss = model(input_ids=data_batch, labels=data_batch)

    # eval
    if args.eval:
        if i==0:
            tqdm.write("Warning: please notice that ZO2 does not optimize the evaluation, so it may be very slow.")
        model.zo_eval()
        output = model(input_ids=data_batch, labels=data_batch)
        res = "Iteration {}, train loss: {}, projected grad: {}, eval loss: {}"
        tqdm.write(res.format(i, loss, model.opt.projected_grad, output["loss"]))
    else:
        res = "Iteration {}, train loss: {}, projected grad: {}"
        tqdm.write(res.format(i, loss, model.opt.projected_grad))

# inference
print("Doing inference...")
print("Warning: please notice that ZO2 does not optimize the inference, so it may be very slow.")
model.zo_eval()
prompt = "What is ZO2 and how ZO2 enhance the fine-tuning of large language models?"
inputs = tokenizer(prompt, return_tensors='pt').to(args.working_device)
inputs = {"input_ids": inputs.input_ids}
for _ in tqdm(range(args.max_new_tokens)):
    outputs = model(**inputs, return_dict=True)
    next_token_logits = outputs.logits[:, -1, :]
    if args.temperature == 1.0:
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    else:
        scaled_logits = next_token_logits / args.temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    inputs = torch.cat([inputs["input_ids"], next_token], dim=-1)
    generated_text = tokenizer.decode(inputs[0])
    inputs = {"input_ids": inputs}
print(f"Question: {prompt}")
print(f"Response: {generated_text[len(prompt)+4:]}...")