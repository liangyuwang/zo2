from contextlib import contextmanager
import torch
import transformers

from . import (
    opt,
    # llama,
)

_zo2_supported_models = {
    transformers.OPTForCausalLM: opt.get_opt_for_causalLM,
    transformers.OPTForSequenceClassification: opt.get_opt_for_sequence_classification,
    transformers.OPTForQuestionAnswering: opt.get_opt_for_question_answering,

    # transformers.LlamaForCausalLM: llama.get_llama_for_causalLM,
    # transformers.LlamaForSequenceClassification: llama.get_llama_for_sequence_classification,
}

@contextmanager
def zo2_hf_init(zo_config):
    original_models = {}
    try:
        for orig_class, get_zo2_class in _zo2_supported_models.items():
            if hasattr(transformers, orig_class.__name__):
                original_models[orig_class] = getattr(transformers, orig_class.__name__)
                setattr(transformers, orig_class.__name__, get_zo2_class(zo_config))
            else:
                raise NotImplementedError(f"Model '{orig_class.__name__}' is not supported in ZO2. Currently, ZO2 only supports {[model_class.__name__ for model_class in _zo2_supported_models.keys()]}")
        yield
    finally:
        for orig_class, get_zo2_class in _zo2_supported_models.items():
            setattr(transformers, orig_class.__name__, original_models[orig_class])

def main():
    # user api:
    from transformers import OPTForCausalLM
    with zo2_hf_init(zo_config):
        model = OPTForCausalLM.from_pretrained(...)
    print(type(model))  # should be zo2.OPTForCausalLM