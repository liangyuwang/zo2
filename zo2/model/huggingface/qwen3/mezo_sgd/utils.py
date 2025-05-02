# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch

def fn_get_qwen3_decoder_hidden_states_from_layer_outputs(input):
    return input[0]

def fn_get_qwen3_sliced_logits_from_hidden_states(hidden_states, slice_indices):
    return hidden_states[:, slice_indices, :]