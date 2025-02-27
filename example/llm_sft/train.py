"""
train.py: Main module for training and evaluation.
Encapsulates the training process into the TrainingRunner class.
"""
import sys
sys.path.append("../zo2")
import logging
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

# Importing dependencies from transformers and datasets library.
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, DataCollatorForTokenClassification)
from datasets import load_dataset

# Import custom modules (assumed to be in current PYTHONPATH)
from tasks import get_task
from metrics import calculate_metric
from utils import *
from zo2.trainer.hf_trl.sft_trainer import ZOSFTTrainer as SFTTrainer
from zo2 import zo_hf_init, ZOConfig
from zo2.utils.utils import seed_everything

# Import configuration
from config import get_args

# Set up logging formatting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define namedtuple for prediction results (assuming it is not defined elsewhere)
from collections import namedtuple
Prediction = namedtuple("Prediction", ["correct_candidate", "predicted_candidate"])

class HFDataset(Dataset):
    """
    HuggingFace-compatible Dataset to hold tokenized data.
    """
    def __init__(self, data):
        """
        Args:
            data (list): List of tokenized samples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TrainingRunner:
    """
    TrainingRunner encapsulates the whole training and evaluation process.
    """
    def __init__(self, args):
        """
        Initialize TrainingRunner with given configuration.

        Args:
            args (argparse.Namespace): Command-line arguments and configurations.
        """
        self.args = args
        # Seed initialization for reproducibility
        seed_everything(args.seed)

        # Set up ZO configuration
        self.zo_config = ZOConfig(
            method="mezo-sgd",
            zo2=(args.zo_method == "zo2"),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.zo_eps,
            offloading_device=args.offloading_device,
            working_device=args.working_device,
        )

        # Load tokenizer and task if applicable
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        if args.task_name not in ["SST2", "RTE", "CB", "BoolQ", "WSC", "WIC", "MultiRC", "Copa", "ReCoRD", "SQuAD", "DROP"]:
            self.task = None
        else:
            self.task = get_task(args.task_name)

        # Initialize model within zo_hf_init context
        with zo_hf_init(self.zo_config):
            from transformers import OPTForCausalLM
            self.model = OPTForCausalLM.from_pretrained(args.model_name)
            self.model.zo_init(self.zo_config)
        # If using a method other than zo2, move model to working device
        if args.zo_method != "zo2":
            self.model = self.model.to(args.working_device)
        logger.info(f"Check if zo2 init correctly: {hasattr(self.model, 'zo_training')}")

        # Set up training arguments for the HuggingFace Trainer
        self.training_args = TrainingArguments(
            output_dir="./tmp",
            evaluation_strategy="steps",
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            logging_steps=args.log_every_step,
            max_steps=args.max_steps,
        )

        # Prepare datasets and data collator
        self._prepare_datasets()

    def _prepare_datasets(self):
        """
        Prepare training and evaluation datasets using either HuggingFace datasets or a custom task.
        """
        if self.task is None:
            # For non-specified tasks, use load_dataset
            self.train_dataset = load_dataset(self.args.task_name, split="train")
            if self.args.max_train_data:
                self.train_dataset = self.train_dataset.select(range(self.args.max_train_data))
            self.eval_dataset = load_dataset(self.args.task_name, split="test")
            if self.args.max_eval_data:
                self.eval_dataset = self.eval_dataset.select(range(self.args.max_eval_data))
            # Dummy collator since text field may vary; adjust if needed.
            self.collator = DataCollatorForTokenClassification(AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False),
                                                                 max_length=self.args.max_length)
        else:
            # Use custom task and examples
            self.eval_samples = self.task.valid_samples

            def _convert(samples):
                """
                Convert raw samples to a format compatible with HuggingFace dataset.

                Args:
                    samples (list): List of raw samples from the task.

                Returns:
                    list: List of dictionaries containing tokenized input_ids and labels.
                """
                data = []
                for sample in samples:
                    encoded_candidates, option_lens = encode_prompt(
                        self.task,
                        self.task.get_template(),
                        [],
                        sample,
                        self.tokenizer,
                        max_length=self.args.max_length,
                        generation=self.task.generation,
                        generation_with_gold=True,
                        max_new_tokens=self.args.max_new_tokens,
                    )
                    if self.task.generation:
                        correct_candidate_id = 0
                    elif isinstance(sample.correct_candidate, list):
                        correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                    else:
                        correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                    data.append({
                        "input_ids": encoded_candidates[correct_candidate_id],
                        "labels": encoded_candidates[correct_candidate_id],
                        "option_len": option_lens[correct_candidate_id]
                    })
                return data

            with count_time("Tokenizing training samples"):
                self.train_dataset = HFDataset(_convert(self.task.samples["train"][:self.args.max_train_data]))
                self.eval_dataset = HFDataset(_convert(self.task.samples["valid"][:self.args.max_eval_data]))
            self.collator = DataCollatorForTokenClassification(self.tokenizer, max_length=self.args.max_length)

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Forward pass for inference.

        Args:
            input_ids (list or torch.Tensor): Tokenized input ids.
            option_len (int, optional): Length of the option part. Defaults to None.
            generation (bool): Whether to perform autoregressive generation.

        Returns:
            torch.Tensor or str: Log-likelihood for classification or generated text for generation task.
        """
        input_tensor = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            # Autoregressive text generation
            eos_token_ids = self.tokenizer.encode(self.args.eos_token, add_special_tokens=False, padding=True,
                                                    truncation=True, max_length=self.args.max_length)
            outputs = self.model.generate(
                input_tensor,
                max_new_tokens=min(self.args.max_new_tokens, self.args.max_length - input_tensor.size(1)),
                num_return_sequences=1,
                eos_token_id=[eos_token_ids[-1], self.tokenizer.eos_token_id],
            )
            # Decode generated tokens
            output_text = self.tokenizer.decode(outputs[0][input_tensor.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            # Compute log probabilities
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_tensor).logits
            # Shift logits and labels for computing log-softmax
            labels = input_tensor[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Return only the option part
            return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Perform one-step prediction for a given evaluation sample possibly using in-context learning.

        Args:
            train_samples (list): List of training samples (demonstrations).
            eval_sample (object): Evaluation sample.
            verbose (bool): Whether to log detailed information.

        Returns:
            Prediction: Named tuple containing correct candidate and predicted candidate.
        """
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens
        )
        outputs = []
        if self.task.generation:
            # For generation tasks, return the generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}")
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice tasks, calculate probabilities for all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info(f"=== Candidate {candidate_id} ===")
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info(f"=== Candidate {candidate_id} (without context) ===")
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")
                outputs.append({"log_probs": selected_log_probs})
            scores = [x['log_probs'].mean().item() for x in outputs]
            if verbose:
                logger.info(f"Prediction scores: {scores}")
            if isinstance(eval_sample.correct_candidate, list):
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)
            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate the model on evaluation samples.

        Args:
            train_samples (list): List of training samples for in-context learning.
            eval_samples (list): List of evaluation samples.
            one_train_set_per_eval_sample (bool): Whether each evaluation sample has its own training set.

        Returns:
            dict: Dictionary of calculated metrics.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            demonstrations = train_samples[eval_id] if one_train_set_per_eval_sample else train_samples
            predictions.append(self.one_step_pred(demonstrations, eval_sample, verbose=(eval_id < 3)))
        # Compute metric (default: accuracy)
        metric_name = "accuracy"
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def train(self):
        """
        Execute training using SFTTrainer.
        """
        trainer = SFTTrainer(
            self.model,
            train_dataset=self.train_dataset,
            data_collator=self.collator,
            eval_dataset=self.eval_dataset,
            args=self.training_args,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.args.max_length,
        )
        trainer.train()

    def run(self):
        """
        Run the full training process and evaluate if required.
        """
        logger.info("Starting training...")
        self.train()
        if self.args.eval:
            # For evaluation without in-context learning, passing an empty train set
            # Adjust this part if in-context evaluation is required
            metrics = self.evaluate([], self.eval_samples[:self.args.max_dev_data])
            logger.info(f"Evaluation metrics: {metrics}")

def result_file_tag(args):
    """
    Generate a tag for result file based on task and model name.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        str: Result file tag string.
    """
    save_model_name = args.model_name.split("/")[-1]
    return f"{args.task_name}-{save_model_name}"

def main():
    """
    Main entry point: Parse arguments, initialize TrainingRunner and run training (and evaluation if specified).
    """
    args = get_args()
    runner = TrainingRunner(args)
    runner.run()

if __name__ == "__main__":
    main()

