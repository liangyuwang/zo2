# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from . import zo, zo2
from .....config.mezo_sgd import MeZOSGDConfig

def get_qwen3_for_causalLM_mezo_sgd(config: MeZOSGDConfig):
    return zo2.Qwen3ForCausalLM if config.zo2 else zo.Qwen3ForCausalLM
