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
def zo_hf_init(zo_config):
    try:
        for orig_class, get_zo2_class in _zo2_supported_models.items():
            if hasattr(transformers, orig_class.__name__):
                zo2_class = get_zo2_class(zo_config)
                setattr(transformers, orig_class.__name__, zo2_class)
            else:
                raise NotImplementedError(f"Model '{orig_class.__name__}' is not supported in transformers.")
        yield
    finally:
        pass

def main():
    # user api:
    with zo_hf_init(zo_config):
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(...)
        model.zo_init(zo_config)
    print(type(model))  # should be zo2.OPTForCausalLM