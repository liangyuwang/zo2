# ZO2 (Zeroth-Order Offloading): Full Parameter Fine-Tuning 175B LLMs with 18GB GPU Memory

👋 Welcome! **ZO2** is an innovative framework specifically designed to enhance the fine-tuning of large language models (LLMs) using **zeroth-order (ZO)** optimization techniques and advanced **offloading** technologies. This framework is particularly tailored for setups with limited GPU memory (e.g. fine-tune **[OPT-175B](https://arxiv.org/abs/2205.01068)** with just **18GB GPU memory**), enabling the fine-tuning of models that were previously unmanageable due to hardware constraints.

- The table below displays the GPU memory usage for various OPT model sizes when fine-tuned using the ZO2 framework:

|        OPT Models        |   1.3B   |   2.7B   |   6.7B   |   13B   |   30B   |    66B    |        175B        |
| :-----------------------: | :------: | :------: | :------: | :------: | :------: | :-------: | :-----------------: |
| **GPU memory (GB)** | `3.75` | `4.14` | `4.99` | `6.18` | `8.86` | `12.07` | **`18.04`** |

- [Install](#️installation) the package and execute the following test to see the memory usage:

```shell
bash test/mezo_sgd/hf_opt/record_zo2_memory.sh
```

## 📰 News

## 💡 Key Features

- **Optimized ZO CPU Offloading**: ZO2 leverages `zeroth-order (ZO)` methods to efficiently use `CPU offloading`, avoiding redundant data transfers and significantly reducing GPU memory demands. This allows for handling large-scale models on hardware with limited GPU resources.
- **Dynamic Scheduling**: Incorporates a high-performance scheduler to optimize the `computation-communication overlap`, enhancing GPU utilization and preventing training delays.
- **Capability for Very Large Models**: Enables the fine-tuning of extraordinarily large models, such as those with over `175 billion parameters`, on single GPUs with as little as `18GB` of memory, previously impossible with traditional methods.
- **Empirical Validation**: ZO2 has demonstrated through rigorous testing that it can efficiently fine-tune massive models `without extra time costs or accuracy losses`, confirming its effectiveness for large-scale model training.

## ⚙️ Installation

```shell
git clone https://github.com/liangyuwang/zo2.git
cd zo2/
conda env create -f env.yaml
conda activate zo2
```

## 🛠️ Usage

We utilize the [OPT](https://arxiv.org/abs/2205.01068) models and [MeZO-SGD](https://arxiv.org/abs/2305.17333) as examples. For additional information, please refer to the section on [Supported Models and ZO methods](#-supported-models-zo-methods-and-tasks-support).

### 1. Using [MeZO-Runner](example/mezo_runner/) to evaluate fine-tuning tasks

```shell
cd example/mezo_runner/
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 STEPS=20000 EVAL_STEPS=4000 bash mezo.sh
```

### 2. Supervised Fine-Tuning HF Models with ZOTrainer / ZOSFTTrainer [[Trainer](./tutorial/huggingface.ipynb)]

```python
from zo2 import ZOConfig, zo_hf_init
from zo2.hf_trl import ZOTrainer, ZOSFTTrainer
from transformers import TrainingArguments

# Model and optimizer init
zo_config = ZOConfig(method="mezo-sgd", zo2=True, offloading_device='cpu', working_device='cuda', lr=1e-5)
with zo_hf_init(zo_config):
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
    model.zo_init(zo_config)

training_args = TrainingArguments("test-trainer")

trainer = ZOSFTTrainer(  # or ZOTrainer
    model,
    args = training_args,
    train_dataset=...,   # get training dataset
    eval_dataset=...,    # get eval dataset
    data_collator=...,   # get data_collator
    tokenizer=...,       # use suitable tokenizer
    compute_metrics=..., # define compute_metrics func,
    ...
)

trainer.train()
```

### 3. Train HF Models with Custom Training Loop [[demo](./tutorial/demo.ipynb)]

```python
from zo2 import ZOConfig, zo_hf_init

# Model and optimizer init
zo_config = ZOConfig(method="mezo-sgd", zo2=True, offloading_device='cpu', working_device='cuda', lr=1e-5)
with zo_hf_init(zo_config):
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
    model.zo_init(zo_config)

# Training loop
for i in range(max_training_step):
    # Train
    training_input_ids, training_labels = ...   # get training data batch
    model.zo_train()
    loss = model(input_ids=training_input_ids, labels=training_labels)
    # Evaluate
    eval_input_ids, eval_labels = ...   # get eval data batch
    model.zo_eval()     
    with torch.no_grad():
        output = model(input_ids=eval_input_ids, labels=eval_labels)

# Final training update
model.opt.zo_update(model)
```

## ✨ Tutorial

Please refer to [tutorial](./tutorial/).

## 🤖 Supported Models, ZO methods, and Tasks

- **Models**:

  * [NanoGPT](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py)   (mainly for idea evaluation)
  * [Transformers](https://github.com/huggingface/transformers):
    * [OPT](https://arxiv.org/abs/2205.01068)
- **ZO methods**:

  * [MeZO-SGD](https://arxiv.org/abs/2305.17333)
- **Tasks**: Please refer to [MeZO-Runner](example/mezo_runner/)

## 🧪 Test

Please refer to [test](./test/).

## 🧭 Future Directions

- [ ] Support more models

  - [ ] HF LLaMA
  - [ ] HF GPT
- [ ] Support more ZO methods
- [ ] Support more offloading strategies

  - [ ] Disk offloading

## 🚶 Contributing

## 📲 Contact

* Liangyu Wang: liangyu.wang@kaust.edu.sa

## 👥 Authors
