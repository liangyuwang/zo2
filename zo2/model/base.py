# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch

class BaseZOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zo_training = True
        self.zo_train_loss_fn_pre_hooks = []
        self.zo_train_loss_fn_post_hooks = []
        self.zo_eval_loss_fn_pre_hooks = []
        self.zo_eval_loss_fn_post_hooks = []
        self.zo_custom_train_loss_fn = None
        self.zo_custom_eval_loss_fn = None

    def zo_train(self):
        """
            Zeroth-order training
        """
        self.zo_training = True
        self.eval()

    def zo_eval(self):
        """
            Zeroth-order evaluation
        """
        self.zo_training = False
        self.eval()

    def register_zo_train_loss_fn_pre_hook(self, hook_fn):
        self.zo_train_loss_fn_pre_hooks.append(hook_fn)

    def register_zo_train_loss_fn_post_hook(self, hook_fn):
        self.zo_train_loss_fn_post_hooks.append(hook_fn)

    def register_zo_eval_loss_fn_pre_hook(self, hook_fn):
        self.zo_eval_loss_fn_pre_hooks.append(hook_fn)

    def register_zo_eval_loss_fn_post_hook(self, hook_fn):
        self.zo_eval_loss_fn_post_hooks.append(hook_fn)

    def register_custom_opt(self, custom_opt_obj):
        if hasattr(self, "opt"):
            self.opt = custom_opt_obj
        for module in self.children():
            if hasattr(module, "opt"):
                module.opt = custom_opt_obj