# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from . import (
    mezo_sgd,
)

def get_qwen3_for_causalLM(zo_config):
    zo2_supported_configs = {
        "mezo-sgd": mezo_sgd.get_qwen3_for_causalLM_mezo_sgd,
    }
    return zo2_supported_configs[zo_config.zo_method](zo_config)
