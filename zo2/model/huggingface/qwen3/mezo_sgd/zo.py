# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

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
    can_return_tuple,
    deprecate_kwarg,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    QWEN3_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
)
from transformers.utils import logging

import random
from typing import List, Optional, Tuple, Union, Unpack

from ....base import BaseZOModel
from .....optimizer.mezo_sgd.zo import MeZOSGD
from .....config.mezo_sgd import MeZOSGDConfig

logger = logging.get_logger(__name__)


class Qwen3Model(modeling_qwen3.Qwen3Model, Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
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


class Qwen3ForCausalLM(modeling_qwen3.Qwen3ForCausalLM, Qwen3PreTrainedModel, BaseZOModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3Config):
        Qwen3PreTrainedModel.__init__(self, config)
        BaseZOModel.__init__(self)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def zo_init(self, zo_config):
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
            use_cache = False
            return self.opt.zo_forward(
                input_ids, attention_mask, position_ids, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, 
                cache_position, logits_to_keep, **kwargs)
        else:
            return self.opt.zo_eval_forward(super().forward, 
                input_ids, attention_mask, position_ids, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, 
                cache_position, logits_to_keep, **kwargs)


class OptimizerQwen3ForCausalLM(MeZOSGD):
    
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
        """
            copy the original forward code and replace all 'self' to 'self.model'.
        """

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model.model(
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

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.model.lm_head(hidden_states[:, slice_indices, :])

        if self.model.zo_train_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_train_loss_fn_pre_hooks:
                input_ids, logits, labels = pre_hook_fn(self.model, input_ids, logits, labels)

        loss = None
        if labels is not None:
            if self.model.zo_custom_train_loss_fn:
                loss = self.model.zo_custom_train_loss_fn(self.model, input_ids, logits, labels, **kwargs)
            else:
                loss = self.model.loss_function(logits=logits, labels=labels, vocab_size=self.model.config.vocab_size, **kwargs)

        if self.model.zo_train_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_train_loss_fn_post_hooks:
                loss, input_ids, logits, labels = post_hook_fn(self.model, loss, input_ids, logits, labels)

        # add --> only return loss
        return loss.detach()

    @torch.inference_mode
    def inner_zo_eval_forward(
        self,
        eval_fn,
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
        if self.model.zo_eval_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_eval_loss_fn_pre_hooks:
                input_ids, logits, labels = pre_hook_fn(self.model, input_ids, logits, labels)

        if self.model.zo_custom_eval_loss_fn:
            output = eval_fn(input_ids, attention_mask, position_ids, 
                past_key_values, inputs_embeds, None, use_cache, 
                output_attentions, output_hidden_states, 
                cache_position, logits_to_keep, **kwargs)
            logits = output["logits"]
            loss = None
            if labels is not None:
                loss = self.model.zo_custom_eval_loss_fn(self.model, input_ids, logits, labels, **kwargs)
            output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=output.past_key_values,
                hidden_states=output.hidden_states,
                attentions=output.attentions,
            )
        else:
            output = eval_fn(input_ids, attention_mask, position_ids, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, 
                cache_position, logits_to_keep, **kwargs)
            
        if self.model.zo_eval_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_eval_loss_fn_post_hooks:
                output, input_ids, logits, labels = post_hook_fn(self.model, output, input_ids, logits, labels)
        return output