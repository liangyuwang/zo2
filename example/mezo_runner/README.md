# Example: Apply MeZO on LLMs

Modified from [MeZO](https://github.com/princeton-nlp/MeZO/blob/main/large_models/README.md)

## Usage

Use `run.py` for all functions (zero-shot/MeZO):

```bash
python run.py {ARGUMENTS}
```

Please read [run.py](./run.py) for a complete list of arguments. We introduce some of the most important ones below.

* `--num_train`: Number of training examples.
* `--num_dev`: Number of validation examples.
* `--num_test`: Number of testing examples.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot) or `zo` (MeZO).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
* `--zo_eps`: MeZO hyperparameter epsilon.
* `--zo_method`: choose zeroth-order methods.
* `--zo_mode`: can be `zo` (on device) or `zo2` (offloading).
* `--offloading_device`: offloading device.
* `--working_device`: main working device.

Example:

1. MeZO (full-parameter fine-tuning)
```bash
# You can adjust the following model size and other hyperparameters.
# OPT-2.7B
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 STEPS=20000 EVAL_STEPS=4000 bash mezo.sh
# Qwen3-1.7B
MODEL=Qwen/Qwen3-1.7B TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 STEPS=20000 EVAL_STEPS=4000 bash mezo.sh
```

## Supported Tasks (See [tasks.py](./tasks.py))

- **SST2**
- **Copa**
- **BoolQ**
- **MultiRC**
- **CB**
- **WIC**
- **WSC**
- **ReCoRD**
- **RTE**
- **SQuAD**
- **DROP**
