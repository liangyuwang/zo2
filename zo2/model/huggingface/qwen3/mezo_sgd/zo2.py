# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import random
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3DecoderLayer,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    KwargsForCausalLM,
    FlashAttentionKwargs,
    partial,
    can_return_tuple,
    deprecate_kwarg,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    QWEN3_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
)
from transformers.utils import logging

from typing import List, Optional, Tuple, Union, Unpack

from ....base import BaseZOModel
from .....optimizer.mezo_sgd.zo2 import MeZO2SGD
from .....config.mezo_sgd import MeZOSGDConfig
from .utils import *

logger = logging.get_logger(__name__)


class Qwen3Model(modeling_qwen3.Qwen3Model, Qwen3PreTrainedModel, BaseZOModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """
    def __init__(self, config: Qwen3Config):
        """
        !!! Module register must follow the execution order.
        """
        config.use_cache = False
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def zo_init(self, zo_config):
        # Initialize ZO2
        self.opt = OptimizerQwen3Model(model=self, config=zo_config)
    
    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if self.zo_training:
            return self.opt.inner_zo_forward(input_ids, attention_mask, position_ids, 
                past_key_values, inputs_embeds, use_cache, 
                output_attentions, output_hidden_states, cache_position,
                **flash_attn_kwargs)
        else:
            return self.opt.zo_eval_forward(input_ids, attention_mask, position_ids, 
                past_key_values, inputs_embeds, use_cache, 
                output_attentions, output_hidden_states, cache_position,
                **flash_attn_kwargs)


class Qwen3ForCausalLM(modeling_qwen3.Qwen3ForCausalLM, Qwen3PreTrainedModel, BaseZOModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        BaseZOModel.__init__(self)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def zo_init(self, zo_config):
        self.model.zo_init(zo_config)
        # Initialize ZO2
        self.opt = OptimizerQwen3ForCausalLM(model=self, config=zo_config)

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if self.zo_training:
            return self.opt.zo_forward(
                input_ids, attention_mask, position_ids,
                past_key_values, inputs_embeds, labels, use_cache,
                output_attentions, output_hidden_states, cache_position,
                logits_to_keep, **kwargs)
        else:
            return self.opt.zo_eval_forward(
                input_ids, attention_mask, position_ids,
                past_key_values, inputs_embeds, labels, use_cache,
                output_attentions, output_hidden_states, cache_position,
                logits_to_keep, **kwargs)


class OptimizerQwen3Model(MeZO2SGD):

    def init_zo2(self):
        self.upload_stream = None
        self.offload_stream = None
        self.compute_stream = None
        self.zo_random_seed = None
        self.rstate = None
        self.rstate_queue = None
        self.last_rstate = None
        self.projected_grad = None
        self.init_zo2_upload()
    
    def init_zo2_upload(self):
        self.model.embed_tokens = self.model.embed_tokens.to(self.device)
        self.model.rotary_emb = self.model.rotary_emb.to(self.device)
        self.model.norm = self.model.norm.to(self.device)
        self.num_blocks = len(self.model.layers)
        if self.offloading_blocks is not None:
            self.offloading_blocks = self.offloading_blocks
        else:
            self.offloading_blocks = list(range(self.num_blocks))
        print(f"Transformer blocks {self.offloading_blocks} will be offloaded to {self.offloading_device}")
        for i in range(self.num_blocks):
            if i in self.offloading_blocks:
                continue
            else:
                self.model.layers[i] = self.model.layers[i].to(self.device)
                print(f"Upload block {i} to {self.device}.")
        
    @torch.inference_mode
    def inner_zo_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else self.model.config.use_cache
        use_cache = False
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            # inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds1, inputs_embeds2 = self.task_compute_module(
                self.model.embed_tokens,
                inputs1={"input": input_ids},
                inputs2={"input": input_ids},
                grad=self.projected_grad
            )
        else:
            inputs_embeds1 = inputs_embeds2 = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds1.shape[1], device=inputs_embeds1.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask1, causal_mask2 = self.task_compute_function(
            self.model._update_causal_mask,
            inputs1={"attention_mask": attention_mask, "input_tensor": inputs_embeds1, "cache_position": cache_position, "past_key_values": past_key_values, "output_attentions": output_attentions},
            inputs2={"attention_mask": attention_mask, "input_tensor": inputs_embeds2, "cache_position": cache_position, "past_key_values": past_key_values, "output_attentions": output_attentions},
            compute_sync=False,
        )

        hidden_states1, hidden_states2 = inputs_embeds1, inputs_embeds2

        # create position embeddings to be shared across the decoder layers
        position_embeddings1, position_embeddings2 = self.task_compute_module(
            self.model.rotary_emb,
            inputs1={"x": hidden_states1, "position_ids": position_ids},
            inputs2={"x": hidden_states2, "position_ids": position_ids},
            grad=self.projected_grad,
            compute_sync=False
        )

        if 0 in self.offloading_blocks:
            self.model.layers[0] = self.task_upload(
                module=self.model.layers[0],
                device=self.device
            )

        N = self.model.config.num_hidden_layers
        for i in range(1, N):

            if i != 1:
                if i-2 in self.offloading_blocks:
                    self.model.layers[i-2] = self.task_offload(
                        module=self.model.layers[i-2],
                        device=self.offloading_device)
            
            layer_outputs1, layer_outputs2 = self.task_compute_module(
                self.model.layers[i-1],
                inputs1={"hidden_states": hidden_states1, "attention_mask": causal_mask1, "position_ids": position_ids, 
                         "past_key_value": past_key_values, "output_attentions": output_attentions, "use_cache": use_cache, 
                         "cache_position": cache_position, "position_embeddings": position_embeddings1, **flash_attn_kwargs},
                inputs2={"hidden_states": hidden_states2, "attention_mask": causal_mask2, "position_ids": position_ids, 
                         "past_key_value": past_key_values, "output_attentions": output_attentions, "use_cache": use_cache, 
                         "cache_position": cache_position, "position_embeddings": position_embeddings2, **flash_attn_kwargs},
                grad=self.projected_grad
            )

            hidden_states1, hidden_states2 = self.task_compute_function(
                fn=fn_get_qwen3_decoder_hidden_states_from_layer_outputs,
                inputs1={"input": layer_outputs1},
                inputs2={"input": layer_outputs2},
                compute_sync=False
            )

            if i in self.offloading_blocks:
                self.model.layers[i] = self.task_upload(
                    module=self.model.layers[i],
                    device=self.device)

        if N-2 in self.offloading_blocks:
            self.model.layers[N-2] = self.task_offload(
                module=self.model.layers[N-2],
                device=self.offloading_device)
        
        layer_outputs1, layer_outputs2 = self.task_compute_module(
            self.model.layers[N-1],
            inputs1={"hidden_states": hidden_states1, "attention_mask": causal_mask1, "position_ids": position_ids, 
                        "past_key_value": past_key_values, "output_attentions": output_attentions, "use_cache": use_cache, 
                        "cache_position": cache_position, "position_embeddings": position_embeddings1, **flash_attn_kwargs},
            inputs2={"hidden_states": hidden_states2, "attention_mask": causal_mask2, "position_ids": position_ids, 
                        "past_key_value": past_key_values, "output_attentions": output_attentions, "use_cache": use_cache, 
                        "cache_position": cache_position, "position_embeddings": position_embeddings2, **flash_attn_kwargs},
            grad=self.projected_grad
        )

        hidden_states1, hidden_states2 = self.task_compute_function(
            fn=fn_get_qwen3_decoder_hidden_states_from_layer_outputs,
            inputs1={"input": layer_outputs1},
            inputs2={"input": layer_outputs2},
            compute_sync=False
        )

        if N-1 in self.offloading_blocks:
            self.model.layers[N-1] = self.task_offload(
                module=self.model.layers[N-1],
                device=self.offloading_device)
            
        hidden_states1, hidden_states2 = self.task_compute_module(
            module=self.model.norm,
            inputs1={"hidden_states": hidden_states1},
            inputs2={"hidden_states": hidden_states2},
            grad=self.projected_grad,
            # weight_decay=0.
        )

        return hidden_states1, hidden_states2


class OptimizerQwen3ForCausalLM(MeZO2SGD):

    def init_zo2_upload(self):
        self.model.lm_head = self.model.lm_head.to(self.device)
    
    @torch.inference_mode
    def inner_zo_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        self.model.model.zo_training = True
        self.assign_zo2_attributes(self, self.model.model.opt)
        hidden_states1, hidden_states2 = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        self.assign_zo2_attributes(self.model.model.opt, self)
        
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states1, hidden_states2 = self.task_compute_function(
            fn_get_qwen3_sliced_logits_from_hidden_states,
            inputs1={"hidden_states": hidden_states1, "slice_indices": slice_indices},
            inputs2={"hidden_states": hidden_states2, "slice_indices": slice_indices},
        )
        logits1, logits2 = self.task_compute_module(self.model.lm_head,
                                                    inputs1={"input": hidden_states1},
                                                    inputs2={"input": hidden_states2},
                                                    grad=self.projected_grad)

        if self.model.zo_train_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_train_loss_fn_pre_hooks:
                (input_ids, logits1, labels), (input_ids, logits2, labels) = \
                    self.task_compute_function(pre_hook_fn,
                        inputs1={"self": self.model, "input_ids": input_ids, "logits": logits1, "labels": labels},
                        inputs2={"self": self.model, "input_ids": input_ids, "logits": logits2, "labels": labels})
        
        if labels is not None:
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            loss1, loss2 = self.task_compute_function(
                self.model.loss_function,
                inputs1={"logits": logits1, "labels": labels, "vocab_size": self.model.config.vocab_size, **kwargs},
                inputs2={"logits": logits2, "labels": labels, "vocab_size": self.model.config.vocab_size, **kwargs},
            )

        if self.model.zo_train_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_train_loss_fn_post_hooks:
                (loss1, input_ids, logits1, labels), (loss2, input_ids, logits2, labels) = \
                    self.task_compute_function(post_hook_fn,
                        inputs1={"self": self.model, "loss": loss1, "input_ids": input_ids, "logits": logits1, "labels": labels},
                        inputs2={"self": self.model, "loss": loss2, "input_ids": input_ids, "logits": logits2, "labels": labels})

        return loss1, loss2
