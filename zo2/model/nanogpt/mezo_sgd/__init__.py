from ..model import GPTConfig, GPTConfigs, GPT
from .zo import GPT as GPT_MeZOSGD
from .zo2 import GPT as GPT_MeZO2SGD
from ....config.mezo_sgd import MeZOSGDConfig

def get_nanogpt_mezo_sgd(config: MeZOSGDConfig):
    return GPT_MeZO2SGD if config.zo2 else GPT_MeZOSGD