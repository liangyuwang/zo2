from . import (
    mezo_sgd,
)

def get_nanogpt(zo_config):
    zo2_supported_configs = {
        "mezo-sgd": mezo_sgd.get_nanogpt_mezo_sgd,
    }
    return zo2_supported_configs[zo_config.zo_method](zo_config)
