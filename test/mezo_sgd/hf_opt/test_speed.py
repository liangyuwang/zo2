import sys
sys.path.append("../zo2")

import torch
from tqdm import tqdm

from zo2.config.mezo_sgd import MeZOSGDConfig
from zo2.model.huggingface.opt.mezo_sgd import zo, zo2
from zo2.utils.utils import seed_everything
from utils import (
    OPTConfigs,
    prepare_data_for_causalLM, 
    prepare_data_for_sequence_classification,
    prepare_data_for_question_answering,
    model_size, 
    get_args,
    check_throughput
)

def train_mezo_sgd_causalLM(model_config, zo_config, device='cuda'):
    input_ids, labels = prepare_data_for_causalLM(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo.OPTForCausalLM(model_config).to(device)
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    for i in tqdm(range(args.max_steps)):
        model.zo_train()
        check_throughput(i, args.batch_size*model_config.max_position_embeddings, model, input_ids=input_ids, labels=labels, use_tqdm=True)

def train_mezo2_sgd_causalLM(model_config, zo_config, device='cuda'):
    input_ids, labels = prepare_data_for_causalLM(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo2.OPTForCausalLM(model_config)
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    for i in tqdm(range(args.max_steps)):
        model.zo_train()
        check_throughput(i, args.batch_size*model_config.max_position_embeddings, model, input_ids=input_ids, labels=labels, use_tqdm=True)


def train_mezo_sgd_sequence_classification(model_config, zo_config, device='cuda'):
    input_ids, labels = prepare_data_for_sequence_classification(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo.OPTForSequenceClassification(model_config).to(device)
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    for i in tqdm(range(args.max_steps)):
        model.zo_train()
        check_throughput(i, args.batch_size*model_config.max_position_embeddings, model, input_ids=input_ids, labels=labels, use_tqdm=True)

def train_mezo2_sgd_sequence_classification(model_config, zo_config, device='cuda'):
    input_ids, labels = prepare_data_for_sequence_classification(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo2.OPTForSequenceClassification(model_config).to(device)
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    for i in tqdm(range(args.max_steps)):
        model.zo_train()
        check_throughput(i, args.batch_size*model_config.max_position_embeddings, model, input_ids=input_ids, labels=labels, use_tqdm=True)


def train_mezo_sgd_question_answering(model_config, zo_config, device='cuda'):
    input_ids, start_positions, end_positions = prepare_data_for_question_answering(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo.OPTForQuestionAnswering(model_config).to("cuda")
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    for i in tqdm(range(args.max_steps)):
        model.zo_train()
        check_throughput(i, args.batch_size*model_config.max_position_embeddings, model, input_ids=input_ids, start_positions=start_positions, end_positions=end_positions, use_tqdm=True)

def train_mezo2_sgd_question_answering(model_config, zo_config, device='cuda'):
    input_ids, start_positions, end_positions = prepare_data_for_question_answering(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    torch.set_default_dtype(args.model_dtype)
    model = zo2.OPTForQuestionAnswering(model_config).to("cuda")
    model.zo_init(zo_config)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    torch.set_default_dtype(original_dtype)
    for i in tqdm(range(args.max_steps)):
        model.zo_train()
        check_throughput(i, args.batch_size*model_config.max_position_embeddings, model, input_ids=input_ids, start_positions=start_positions, end_positions=end_positions, use_tqdm=True)


def test_mezo_sgd_causalLM_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    train_mezo_sgd_causalLM(model_config, zo_cfg, device=args.working_device)

def test_mezo2_sgd_causalLM_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        offloading_device=args.offloading_device, working_device=args.working_device)
    zo_cfg.zo2 = True
    train_mezo2_sgd_causalLM(model_config, zo_cfg, device=args.working_device)


def test_mezo_sgd_sequence_classification_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    train_mezo_sgd_sequence_classification(model_config, zo_cfg, device=args.working_device)

def test_mezo2_sgd_sequence_classification_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        offloading_device=args.offloading_device, working_device=args.working_device)
    zo_cfg.zo2 = True
    train_mezo2_sgd_sequence_classification(model_config, zo_cfg, device=args.working_device)


def test_mezo_sgd_question_answering_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    train_mezo_sgd_question_answering(model_config, zo_cfg, device=args.working_device)

def test_mezo2_sgd_question_answering_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    model_config.tie_word_embeddings=False
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        offloading_device=args.offloading_device, working_device=args.working_device)
    zo_cfg.zo2 = True
    train_mezo2_sgd_question_answering(model_config, zo_cfg, device=args.working_device)


if __name__=="__main__":
    args = get_args()
    original_dtype = torch.get_default_dtype()
    if args.zo_method == "zo":
        if args.task == "causalLM":
            test_mezo_sgd_causalLM_training()
        elif args.task == "sequence_classification":
            test_mezo_sgd_sequence_classification_training()
        elif args.task == "question_answering":
            test_mezo_sgd_question_answering_training()
        else:
            raise NotImplementedError(f"Task {args.task} is unsupported.")
    elif args.zo_method == "zo2":
        if args.task == "causalLM":
            test_mezo2_sgd_causalLM_training()
        elif args.task == "sequence_classification":
            test_mezo2_sgd_sequence_classification_training()
        elif args.task == "question_answering":
            test_mezo2_sgd_question_answering_training()
        else:
            raise NotImplementedError(f"Task {args.task} is unsupported.")
    else:
        raise NotImplementedError