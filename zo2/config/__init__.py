# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from .mezo_sgd import MeZOSGDConfig


def ZOConfig(method: str = "mezo-sgd", **kwargs):
    match method:
        case "mezo-sgd":
            return MeZOSGDConfig(**kwargs)
        # case "another-method":
        #     return AnotherConfig(**kwargs)
        case _:
            raise ValueError(f"Unsupported method {method}")
