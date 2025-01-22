from contextlib import contextmanager
import torch
import transformers

import zo2

_zo2_supported_models = {
    transformers.OPTForCausalLM: zo2.get_opt_for_causalLM,
    transformers.OPTForSequenceClassification: zo2.get_opt_for_sequence_classification,
    transformers.OPTForQuestionAnswering: zo2.get_opt_for_question_answering,

    # transformers.LlamaForCausalLM: zo2.get_llama_for_causalLM,
    # transformers.LlamaForSequenceClassification: zo2.get_llama_for_sequence_classification,
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
                raise NotImplementedError(f"Model '{orig_class.__name__}' is not spuuorted in ZO2.")
        yield
    finally:
        for orig_class, get_zo2_class in _zo2_supported_models.items():
            setattr(transformers, orig_class.__name__, original_models[orig_class])

def main():
    # user api:
    from transformers import OPTForCausalLM
    with zo2_hf_init():
        model = OPTForCausalLM.from_pretrained(...)
    print(type(model))  # should be zo2.OPTForCausalLM