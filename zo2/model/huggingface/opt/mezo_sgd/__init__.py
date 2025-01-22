from . import zo, zo2
from .....config.mezo_sgd import MeZOSGDConfig

def get_opt_for_causalLM_mezo_sgd(config: MeZOSGDConfig):
    return zo2.OPTForCausalLM if config.zo2 else zo.OPTForCausalLM

def get_opt_for_sequence_classification_mezo_sgd(config: MeZOSGDConfig):
    return zo2.OPTForSequenceClassification if config.zo2 else zo.OPTForSequenceClassification

def get_opt_for_question_answering_mezo_sgd(config: MeZOSGDConfig):
    return zo2.OPTForQuestionAnswering if config.zo2 else zo.OPTForQuestionAnswering
