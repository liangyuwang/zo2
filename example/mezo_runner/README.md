# Example: Apply MeZO on LLMs

Modified from [MeZO](https://github.com/princeton-nlp/MeZO/blob/main/large_models/README.md)

## Usage

Use `run.py` for all functions (zero-shot/MeZO):

```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments. We introduce some of the most important ones below.

* `--num_train`: Number of training examples. For ICL, this is the number of demonstrations.
* `--num_dev`: Number of validation examples.
* `--num_test`: Number of testing examples.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot), `regular` (fine-tuning), or `zo` (MeZO).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
* `--zo_eps`: MeZO hyperparameter epsilon.

```bash
# MeZO (full-parameter fine-tuning)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 STEPS=4000 EVAL_STEPS=4000 bash mezo.sh
```
