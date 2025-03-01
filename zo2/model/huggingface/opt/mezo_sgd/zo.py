import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.opt import modeling_opt
from transformers.models.opt.modeling_opt import (
    OPTConfig,
    OPTPreTrainedModel,
    OPTLearnedPositionalEmbedding,
    OPTDecoderLayer,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    QuestionAnsweringModelOutput,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
    replace_return_docstrings,
    OPT_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _EXPECTED_OUTPUT_SHAPE,
    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    _SEQ_CLASS_EXPECTED_OUTPUT,
    _SEQ_CLASS_EXPECTED_LOSS,
)
from transformers.utils import logging

import random
from typing import List, Optional, Tuple, Union

from ....base import BaseZOModel
from .....optimizer.mezo_sgd.zo import MeZOSGD
from .....config.mezo_sgd import MeZOSGDConfig

logger = logging.get_logger(__name__)



class OPTDecoder(modeling_opt.OPTDecoder, OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]
    
    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        """
        !!! Module register must follow the execution order.
        """
        OPTPreTrainedModel.__init__(self, config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class OPTModel(modeling_opt.OPTModel, OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        OPTPreTrainedModel.__init__(self, config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()


class OPTForCausalLM(modeling_opt.OPTForCausalLM, OPTPreTrainedModel, BaseZOModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: OPTConfig):
        OPTPreTrainedModel.__init__(self, config)
        BaseZOModel.__init__(self)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def zo_init(self, zo_config):
        self.opt = OptimizerOPTForCausalLM(model=self, config=zo_config)

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        if self.zo_training:
            return self.opt.zo_forward(
                input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, return_dict, **kwargs)
        else:
            return self.opt.zo_eval_forward(super().forward, 
                input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, return_dict, **kwargs)


class OPTForSequenceClassification(modeling_opt.OPTForSequenceClassification, OPTPreTrainedModel, BaseZOModel):
    def __init__(self, config: OPTConfig):
        OPTPreTrainedModel.__init__(self, config)
        BaseZOModel.__init__(self)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def zo_init(self, zo_config):
        self.opt = OptimizerOPTForSequenceClassification(model=self, config=zo_config)

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if self.zo_training:
            return self.opt.zo_forward(
                input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, return_dict)
        else:
            return self.opt.zo_eval_forward(super().forward, 
                input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, return_dict)


class OPTForQuestionAnswering(modeling_opt.OPTForQuestionAnswering, OPTPreTrainedModel, BaseZOModel):
    def __init__(self, config: OPTConfig):
        OPTPreTrainedModel.__init__(self, config)
        BaseZOModel.__init__(self)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()
    
    def zo_init(self, zo_config):
        self.opt = OptimizerOPTForQuestionAnswering(model=self, config=zo_config)

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        if self.zo_training:
            return self.opt.zo_forward(
                input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, start_positions, end_positions, use_cache, 
                output_attentions, output_hidden_states, return_dict)
        else:
            return self.opt.zo_eval_forward(super().forward, 
                input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, start_positions, end_positions, use_cache, 
                output_attentions, output_hidden_states, return_dict)


class OptimizerOPTForCausalLM(MeZOSGD):
    
    @torch.inference_mode
    def inner_zo_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
            copy the original forward code and replace all 'self' to 'self.model'.
        """

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.model.lm_head(outputs[0]).contiguous()

        if self.model.zo_train_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_train_loss_fn_pre_hooks:
                input_ids, logits, labels = pre_hook_fn(self.model, input_ids, logits, labels)

        loss = None
        if self.model.zo_custom_train_loss_fn:
            loss = self.model.zo_custom_train_loss_fn(self.model, input_ids, logits, labels, **kwargs)
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        if self.model.zo_train_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_train_loss_fn_post_hooks:
                loss, input_ids, logits, labels = post_hook_fn(self.model, loss, input_ids, logits, labels)

        # add --> only return loss
        return loss.detach()

    @torch.inference_mode()   
    def inner_zo_eval_forward(
        self,
        eval_fn,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.model.zo_eval_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_eval_loss_fn_pre_hooks:
                input_ids, logits, labels = pre_hook_fn(self.model, input_ids, logits, labels)

        if self.model.zo_custom_eval_loss_fn:
            output = eval_fn(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, None, use_cache, 
                output_attentions, output_hidden_states, return_dict)
            if not return_dict:
                logits = output[0]
                loss = self.model.zo_custom_eval_loss_fn(self.model, input_ids, logits, labels, **kwargs)
                output = (logits,) + output[1]
                return (loss,) + output if loss is not None else output
            logits = output["logits"]
            loss = self.model.zo_custom_eval_loss_fn(self.model, input_ids, logits, labels, **kwargs)
            output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=output["past_key_values"],
                hidden_states=output["hidden_states"],
                attentions=output["attentions"],
            )
        else:
            output = eval_fn(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, return_dict)
        
        if self.model.zo_eval_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_eval_loss_fn_post_hooks:
                output, input_ids, logits, labels = post_hook_fn(self.model, output, input_ids, logits, labels)
        return output
    

class OptimizerOPTForSequenceClassification(MeZOSGD):

    @torch.inference_mode
    def inner_zo_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
            copy the original forward code and replace all 'self' to 'self.model'.
        """

        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        transformer_outputs = self.model.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.model.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.model.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.model.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.model.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        if self.model.zo_train_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_train_loss_fn_pre_hooks:
                input_ids, logits, labels = pre_hook_fn(self.model, input_ids, logits, labels)

        loss = None
        if self.model.zo_custom_train_loss_fn:
            loss = self.model.zo_custom_train_loss_fn(self.model, input_ids, logits, labels, **kwargs)
        elif labels is not None:
            if self.model.config.problem_type is None:
                if self.model.num_labels == 1:
                    self.model.config.problem_type = "regression"
                elif self.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.model.config.problem_type = "single_label_classification"
                else:
                    self.model.config.problem_type = "multi_label_classification"

            if self.model.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.model.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.model.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.model.num_labels), labels.view(-1))
            elif self.model.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        
        if self.model.zo_train_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_train_loss_fn_post_hooks:
                loss, input_ids, logits, labels = post_hook_fn(self.model, loss, input_ids, logits, labels)

        # add --> only return loss
        if self.model.zo_training:
            return loss.detach()
        
    @torch.inference_mode
    def inner_zo_eval_forward(
        self,
        eval_fn,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        if self.model.zo_eval_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_eval_loss_fn_pre_hooks:
                input_ids, logits, labels = pre_hook_fn(self.model, input_ids, logits, labels)

        if self.model.zo_custom_eval_loss_fn:
            output = eval_fn(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, None, use_cache, 
                output_attentions, output_hidden_states, return_dict)
            if not return_dict:
                logits = output[0]
                loss = self.model.zo_custom_eval_loss_fn(self.model, input_ids, logits, labels, **kwargs)
                output = (logits,) + output[1]
                return (loss,) + output if loss is not None else output
            logits = output["logits"]
            loss = self.model.zo_custom_eval_loss_fn(self.model, input_ids, logits, labels, **kwargs)
            output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=output["past_key_values"],
                hidden_states=output["hidden_states"],
                attentions=output["attentions"],
            )
        else:
            output = eval_fn(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, return_dict)
        
        if self.model.zo_eval_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_eval_loss_fn_post_hooks:
                output, input_ids, logits, labels = post_hook_fn(self.model, output, input_ids, logits, labels)
        return output


class OptimizerOPTForQuestionAnswering(MeZOSGD):
    
    @torch.inference_mode
    def inner_zo_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        """
            copy the original forward code and replace all 'self' to 'self.model'.
        """
        
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        transformer_outputs = self.model.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.model.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        if self.model.zo_train_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_train_loss_fn_pre_hooks:
                input_ids, start_logits, start_positions, end_logits, end_positions = \
                    pre_hook_fn(self.model, input_ids, start_logits, start_positions, end_logits, end_positions)

        total_loss = None
        if self.model.zo_custom_train_loss_fn:
            loss = self.model.zo_custom_train_loss_fn(self.model, input_ids, start_logits, start_positions, end_logits, end_positions, **kwargs)
        elif start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if self.model.zo_train_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_train_loss_fn_post_hooks:
                loss, input_ids, start_logits, start_positions, end_logits, end_positions = \
                    post_hook_fn(self.model, loss, input_ids, start_logits, start_positions, end_logits, end_positions)

        # add --> only return loss
        if self.model.zo_training:
            return total_loss.detach()
        
    @torch.inference_mode
    def inner_zo_eval_forward(
        self,
        eval_fn,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        if self.model.zo_eval_loss_fn_pre_hooks != []:
            for pre_hook_fn in self.model.zo_eval_loss_fn_pre_hooks:
                input_ids, start_logits, start_positions, end_logits, end_positions = pre_hook_fn(self.model, input_ids, start_logits, start_positions, end_logits, end_positions)

        if self.model.zo_custom_eval_loss_fn:
            output = eval_fn(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, None, None, use_cache, 
                output_attentions, output_hidden_states, return_dict)
            if not return_dict:
                start_logits, end_logits = output[0], output[1]
                loss = self.model.zo_custom_eval_loss_fn(self.model, input_ids, start_logits, start_positions, end_logits, end_positions, **kwargs)
                output = (start_logits, end_logits) + output[2:]
                return (loss,) + output if loss is not None else output
            start_logits = output["start_logits"]
            end_logits = output["end_logits"]
            loss = self.model.zo_custom_eval_loss_fn(self.model, input_ids, start_logits, start_positions, end_logits, end_positions, **kwargs)
            output = QuestionAnsweringModelOutput(
                loss=loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=output["hidden_states"],
                attentions=output["attentions"],
            )
        else:
            output = eval_fn(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, start_positions, end_positions, use_cache, 
                output_attentions, output_hidden_states, return_dict)
        
        if self.model.zo_eval_loss_fn_post_hooks != []:
            for post_hook_fn in self.model.zo_eval_loss_fn_post_hooks:
                output, input_ids, start_logits, start_positions, end_logits, end_positions = post_hook_fn(self.model, output, input_ids, start_logits, start_positions, end_logits, end_positions)
        return output