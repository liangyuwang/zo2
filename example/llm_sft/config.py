"""
config.py: Module for argument parsing and configuration initialization.
"""

import argparse

def get_args():
    """
    Parse command-line arguments for training and evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Training and Evaluation Configuration")
    # ZO method and evaluation
    parser.add_argument("--zo_method", type=str, default="zo2", help="ZO optimizer method, default is 'zo2'")
    parser.add_argument("--eval", action="store_true", help="Whether to perform evaluation after training")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m", help="Model name or path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logging information during evaluation")
    
    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay factor")
    parser.add_argument("--zo_eps", type=float, default=1e-3, help="ZO optimization epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--offloading_device", type=str, default="cpu", help="Device used for offloading")
    parser.add_argument("--working_device", type=str, default="cuda:0", help="Device used for training")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")

    # Data configuration
    parser.add_argument("--max_train_data", type=int, default=None, help="Maximum number of training samples to use")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eos_token", type=str, default="\n", help="End-of-sentence token")
    parser.add_argument("--max_dev_data", type=int, default=None, help="Maximum number of development samples for evaluation")
    
    # Task configuration
    parser.add_argument("--task_name", type=str, default="WIC",
                        help="Dataset/task name. Choose from available datasets such as SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP")
    parser.add_argument("--max_eval_data", type=int, default=None, help="Maximum number of evaluation samples")
    parser.add_argument("--log_every_step", type=int, default=20, help="Logging interval (in steps)")
    
    # Inference parameters
    parser.add_argument("--use_cache", action="store_true", help="Use cache when generating text")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--sampling", action="store_true", help="Use sampling during generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for text generation")

    args = parser.parse_args()
    return args

