import sys
sys.path.append("../zo2")

import torch
from tqdm import tqdm

from zo2.config.mezo_sgd import MeZOSGDConfig
from zo2.model.huggingface.opt.mezo_sgd.zo import (
    OPTConfig,
    OPTForCausalLM,
    OPTForSequenceClassification,
    OPTForQuestionAnswering,
)
from zo2.utils.utils import seed_everything
from utils import (
    OPTConfigs,
    prepare_data_for_causalLM, 
    prepare_data_for_sequence_classification,
    prepare_data_for_question_answering,
    model_size, 
    get_args
)

def train_mezo_sgd_causalLM(model_config, zo_config, device='cuda'):
    zo_config = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
            working_device=args.working_device)
    model = OPTForCausalLM(model_config, zo_config).to(device)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    input_ids, labels = prepare_data_for_causalLM(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    model.eval()
    for i in tqdm(range(args.max_steps)):
        loss = model(input_ids=input_ids, labels=labels)
        res = "Iteration {}, loss: {}, projected grad: {}"
        tqdm.write(res.format(i, loss, model.opt.projected_grad))


def train_mezo_sgd_sequence_classification(model_config, zo_config, device='cuda'):
    model_config = OPTConfig()
    zo_config = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
            working_device=args.working_device)
    model = OPTForSequenceClassification(model_config, zo_config).to(device)
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    input_ids, labels = prepare_data_for_sequence_classification(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    model.eval()
    for i in tqdm(range(args.max_steps)):
        loss = model(input_ids=input_ids, labels=labels)
        res = "Iteration {}, loss: {}, projected grad: {}"
        tqdm.write(res.format(i, loss, model.opt.projected_grad))


def train_mezo_sgd_question_answering(args, model_config, zo_config, device='cuda'):
    model_config = OPTConfig()
    zo_config = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
            working_device=args.working_device)
    model = OPTForQuestionAnswering(model_config, zo_config).to("cuda")
    total_parameters = model_size(model)["total"]
    print(f"model size: {total_parameters/1024**3:.2f} B")
    input_ids, start_positions, end_positions = prepare_data_for_question_answering(
        model_config.vocab_size, args.batch_size, model_config.max_position_embeddings, device)
    model.eval()
    for i in tqdm(range(args.max_steps)):
        loss = model(input_ids=input_ids, start_positions=start_positions, end_positions=end_positions)
        res = "Iteration {}, loss: {}, projected grad: {}"
        tqdm.write(res.format(i, loss, model.opt.projected_grad))


def test_mezo_sgd_causalLM_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    train_mezo_sgd_causalLM(model_config, zo_cfg, device=args.working_device)


def test_mezo_sgd_sequence_classification_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    train_mezo_sgd_sequence_classification(model_config, zo_cfg, device=args.working_device)


def test_mezo_sgd_question_answering_training():
    seed_everything(args.seed)
    model_configs = OPTConfigs()
    model_config = getattr(model_configs, args.model_name)
    zo_cfg = MeZOSGDConfig(lr=args.lr, weight_decay=args.weight_decay, eps=args.zo_eps,
        working_device=args.working_device)
    zo_cfg.zo2 = False
    train_mezo_sgd_question_answering(model_config, zo_cfg, device=args.working_device)


if __name__=="__main__":
    args = get_args()
    if args.zo_method == "zo":
        if args.task == "causalLM":
            test_mezo_sgd_causalLM_training()
        elif args.task == "sequence_classification":
            test_mezo_sgd_sequence_classification_training()
        elif args.task == "question_answering":
            test_mezo_sgd_question_answering_training()
        else:
            raise NotImplementedError(f"Task {args.task} is unsupported.")
    else:
        raise NotImplementedError