# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama import modeling_llama