import random
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

from typing import List, Optional, Tuple, Union

from ....base import BaseZOModel
from .....optimizer.mezo_sgd.zo2 import MeZO2SGD
from .....config.mezo_sgd import MeZOSGDConfig
from .utils import (
    fn_get_opt_decoder_hidden_states_from_layer_outputs,
    get_shift_logits,
    get_shift_labels,
    get_pooled_logits,
    get_start_logits_and_end_logits,
    get_qa_loss
)

logger = logging.get_logger(__name__)


class OPTDecoder(modeling_opt.OPTDecoder, OPTPreTrainedModel, BaseZOModel):
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
    
    def zo_init(self, zo_config):
        # Initialize ZO2
        self.opt = OptimizerOPTDecoder(model=self, config=zo_config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if self.zo_training:
            return self.opt.inner_zo_forward(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, use_cache, 
                output_attentions, output_hidden_states, return_dict)
        else:
            return super().forward(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, use_cache, 
                output_attentions, output_hidden_states, return_dict)


class OPTModel(modeling_opt.OPTModel, OPTPreTrainedModel, BaseZOModel):
    def __init__(self, config: OPTConfig):
        OPTPreTrainedModel.__init__(self, config)
        BaseZOModel.__init__(self)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def zo_init(self, zo_config):
        self.decoder.zo_init(zo_config)
        # Initialize ZO2
        self.opt = OptimizerOPTModel(model=self, config=zo_config)

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if self.zo_training:
            return self.opt.inner_zo_forward(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, use_cache, 
                output_attentions, output_hidden_states, return_dict)
        else:
            return super().forward(input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, use_cache, 
                output_attentions, output_hidden_states, return_dict)


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
        self.model.zo_init(zo_config)
        # Initialize ZO2
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
                output_attentions, output_hidden_states, return_dict)
        else:
            return super().forward(
                input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, labels, use_cache, 
                output_attentions, output_hidden_states, return_dict)


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
        self.model.zo_init(zo_config)
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
            return super().forward(
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
        self.model.zo_init(zo_config)
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
            return super().forward(
                input_ids, attention_mask, head_mask, 
                past_key_values, inputs_embeds, start_positions, end_positions, use_cache, 
                output_attentions, output_hidden_states, return_dict)


class OptimizerOPTDecoder(MeZO2SGD):

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
        self.model.embed_positions = self.model.embed_positions.to(self.device)
        if self.model.project_out:
            self.model.project_out = self.model.project_out.to(self.device)
        if self.model.project_in:
            self.model.project_in = self.model.project_in.to(self.device)
        if self.model.final_layer_norm:
            self.model.final_layer_norm = self.model.final_layer_norm.to(self.device)
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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache

        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            # inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds1, inputs_embeds2 = self.task_compute_module(self.model.embed_tokens,
                                                                      inputs1={"input": input_ids},
                                                                      inputs2={"input": input_ids},
                                                                      grad=self.projected_grad)
        else:
            inputs_embeds1 = inputs_embeds2 = inputs_embeds

        batch_size, seq_length = input_shape
        # past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        past_key_values_length = 0
        # required mask seq length can be calculated via length of past
        # mask_seq_length = past_key_values_length + seq_length
        mask_seq_length = seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds1.device)
        # causal_attention_mask = self.model._prepare_decoder_attention_mask(
        #     attention_mask, input_shape, inputs_embeds, past_key_values_length
        # )
        causal_attention_mask1, causal_attention_mask2 = self.task_compute_function(
            self.model._prepare_decoder_attention_mask,
            inputs1={"attention_mask": attention_mask, "input_shape": input_shape, 
                     "inputs_embeds": inputs_embeds1, "past_key_values_length": past_key_values_length},
            inputs2={"attention_mask": attention_mask, "input_shape": input_shape, 
                     "inputs_embeds": inputs_embeds2, "past_key_values_length": past_key_values_length}
        )
        # pos_embeds = self.model.embed_positions(attention_mask, past_key_values_length)
        pos_embeds1, pos_embeds2 = self.task_compute_module(self.model.embed_positions,
                                                            inputs1={"attention_mask": attention_mask, "past_key_values_length": past_key_values_length},
                                                            inputs2={"attention_mask": attention_mask, "past_key_values_length": past_key_values_length},
                                                            grad=self.projected_grad)

        if self.model.project_in is not None:
            # inputs_embeds = self.model.project_in(inputs_embeds)
            inputs_embeds1, inputs_embeds2 = self.task_compute_module(self.model.project_in,
                                                                      inputs1={"input": inputs_embeds1},
                                                                      inputs2={"input": inputs_embeds2},
                                                                      grad=self.projected_grad)

        # hidden_states = inputs_embeds + pos_embeds
        hidden_states1, hidden_states2 = self.task_compute_function(torch.add,
                                                                    inputs1={"input": inputs_embeds1, "other": pos_embeds1},
                                                                    inputs2={"input": inputs_embeds2, "other": pos_embeds2})

        if 0 in self.offloading_blocks:
            self.model.layers[0] = self.task_upload(
                module=self.model.layers[0],
                device=self.device
            )

        # if self.model.gradient_checkpointing and self.model.training:
        #     if use_cache:
        #         logger.warning_once(
        #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        #         )
        #         use_cache = False

        # # decoder layers
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attns = () if output_attentions else None
        # next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.model.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.model.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        # for idx, decoder_layer in enumerate(self.model.layers):
        #     # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        #     if output_hidden_states:
        #         all_hidden_states += (hidden_states,)

        #     dropout_probability = random.uniform(0, 1)
        #     if self.model.training and (dropout_probability < self.model.layerdrop):
        #         continue

        #     past_key_value = past_key_values[idx] if past_key_values is not None else None

        #     if self.model.gradient_checkpointing and self.model.training:

        #         def create_custom_forward(module):
        #             def custom_forward(*inputs):
        #                 # None for past_key_value
        #                 return module(*inputs, output_attentions, None)

        #             return custom_forward

        #         layer_outputs = torch.utils.checkpoint.checkpoint(
        #             create_custom_forward(decoder_layer),
        #             hidden_states,
        #             causal_attention_mask,
        #             head_mask[idx] if head_mask is not None else None,
        #             None,
        #         )
        #     else:
        #         layer_outputs = decoder_layer(
        #             hidden_states,
        #             attention_mask=causal_attention_mask,
        #             layer_head_mask=(head_mask[idx] if head_mask is not None else None),
        #             past_key_value=past_key_value,
        #             output_attentions=output_attentions,
        #             use_cache=use_cache,
        #         )

        #     hidden_states = layer_outputs[0]

        #     if use_cache:
        #         next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        #     if output_attentions:
        #         all_self_attns += (layer_outputs[1],)
        
        N = len(self.model.layers)
        for i in range(1, N):

            if i != 1:
                if i-2 in self.offloading_blocks:
                    self.model.layers[i-2] = self.task_offload(
                        module=self.model.layers[i-2],
                        device=self.offloading_device)
                    
            # layer_outputs = decoder_layer(
            #     hidden_states,
            #     attention_mask=causal_attention_mask,
            #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            #     output_attentions=output_attentions,
            #     use_cache=use_cache,
            # )
            layer_outputs1, layer_outputs2 = self.task_compute_module(
                self.model.layers[i-1],
                inputs1={"hidden_states": hidden_states1, "attention_mask": causal_attention_mask1, 
                            "layer_head_mask": (head_mask[i-1] if head_mask is not None else None),
                            "output_attentions": output_attentions},
                inputs2={"hidden_states": hidden_states2, "attention_mask": causal_attention_mask2, 
                            "layer_head_mask": (head_mask[i-1] if head_mask is not None else None),
                            "output_attentions": output_attentions},
                grad=self.projected_grad)

            # hidden_states = layer_outputs[0]
            hidden_states1, hidden_states2 = self.task_compute_function(
                fn=fn_get_opt_decoder_hidden_states_from_layer_outputs,
                inputs1={"input": layer_outputs1},
                inputs2={"input": layer_outputs2}
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
            inputs1={"hidden_states": hidden_states1, "attention_mask": causal_attention_mask1, 
                        "layer_head_mask": (head_mask[i-1] if head_mask is not None else None),
                        "output_attentions": output_attentions},
            inputs2={"hidden_states": hidden_states2, "attention_mask": causal_attention_mask2, 
                        "layer_head_mask": (head_mask[i-1] if head_mask is not None else None),
                        "output_attentions": output_attentions},
            grad=self.projected_grad)

        hidden_states1, hidden_states2 = self.task_compute_function(
            fn=fn_get_opt_decoder_hidden_states_from_layer_outputs,
            inputs1={"input": layer_outputs1},
            inputs2={"input": layer_outputs2}
        )

        if N-1 in self.offloading_blocks:
            self.model.layers[N-1] = self.task_offload(
                module=self.model.layers[N-1],
                device=self.offloading_device)
            
        if self.model.final_layer_norm is not None:
            # hidden_states = self.model.final_layer_norm(hidden_states)
            hidden_states1, hidden_states2 = self.task_compute_module(
                module=self.model.final_layer_norm,
                inputs1={"input": hidden_states1},
                inputs2={"input": hidden_states2},
                grad=self.projected_grad,
                weight_decay=0.)

        if self.model.project_out is not None:
            # hidden_states = self.model.project_out(hidden_states)
            hidden_states1, hidden_states2 = self.task_compute_module(
                module=self.model.project_out,
                inputs1={"input": hidden_states1},
                inputs2={"input": hidden_states2},
                grad=self.projected_grad)

        # # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)

        # # next_cache = next_decoder_cache if use_cache else None
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        return hidden_states1, hidden_states2


class OptimizerOPTModel(MeZO2SGD):

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
        ...
    
    @torch.inference_mode
    def inner_zo_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        self.model.decoder.zo_training = True
        self.assign_zo2_attributes(self, self.model.decoder.opt)
        output = self.model.decoder(
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
        self.assign_zo2_attributes(self.model.decoder.opt, self)
        
        return output

        # if not return_dict:
        #     return decoder_outputs

        # return BaseModelOutputWithPast(
        #     last_hidden_state=decoder_outputs.last_hidden_state,
        #     past_key_values=decoder_outputs.past_key_values,
        #     hidden_states=decoder_outputs.hidden_states,
        #     attentions=decoder_outputs.attentions,
        # )


class OptimizerOPTForCausalLM(MeZO2SGD):

    def init_zo2_upload(self):
        self.model.lm_head = self.model.lm_head.to(self.device)
    
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
        self.model.model.decoder.zo_training = True
        self.assign_zo2_attributes(self, self.model.model.decoder.opt)
        hidden_states1, hidden_states2 = self.model.model.decoder(
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
        self.assign_zo2_attributes(self.model.model.decoder.opt, self)

        # logits = self.model.lm_head(outputs[0]).contiguous()
        logits1, logits2 = self.task_compute_module(self.model.lm_head,
                                                    inputs1={"input": hidden_states1},
                                                    inputs2={"input": hidden_states2},
                                                    grad=self.projected_grad)

        # # loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            shift_logits1, shift_logits2 = self.task_compute_function(
                fn=get_shift_logits,
                inputs1={"logits": logits1},
                inputs2={"logits": logits2})
            # shift_labels = labels[..., 1:].contiguous()
            shift_labels1, shift_labels2 = self.task_compute_function(
                fn=get_shift_labels,
                inputs1={"labels": labels},
                inputs2={"labels": labels})
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
            loss1, loss2 = self.task_compute_function(
                fn=loss_fct,
                inputs1={"input": shift_logits1.view(-1, self.model.config.vocab_size), "target": shift_labels1.view(-1)},
                inputs2={"input": shift_logits2.view(-1, self.model.config.vocab_size), "target": shift_labels2.view(-1)})

        return loss1, loss2

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

class OptimizerOPTForSequenceClassification(MeZO2SGD):

    def init_zo2_upload(self):
        self.model.score = self.model.score.to(self.device)
    
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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
            copy the original forward code and replace all 'self' to 'self.model'.
        """

        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        self.model.model.decoder.zo_training = True
        self.assign_zo2_attributes(self, self.model.model.opt)
        hidden_states1, hidden_states2 = self.model.model(
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
        self.assign_zo2_attributes(self.model.model.opt, self)
        
        # hidden_states = transformer_outputs[0]
        # logits = self.model.score(hidden_states)
        logits1, logits2 = self.task_compute_module(self.model.score,
                                                    inputs1={"input": hidden_states1},
                                                    inputs2={"input": hidden_states2},
                                                    grad=self.projected_grad)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.model.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.model.config.pad_token_id).sum(-1) - 1).to(logits1.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.model.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        pooled_logits1, pooled_logits2 = self.task_compute_function(
            fn=get_pooled_logits,
            inputs1={"logits": logits1, "batch_size": batch_size, "sequence_lengths": sequence_lengths},
            inputs2={"logits": logits2, "batch_size": batch_size, "sequence_lengths": sequence_lengths},)

        # loss = None
        if labels is not None:
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
                    # loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    loss1, loss2 = self.task_compute_function(
                        fn=loss_fct,
                        inputs1={"input": pooled_logits1.squeeze(), "target": labels.squeeze()},
                        inputs2={"input": pooled_logits2.squeeze(), "target": labels.squeeze()},)
                else:
                    # loss = loss_fct(pooled_logits, labels)
                    loss1, loss2 = self.task_compute_function(
                        fn=loss_fct,
                        inputs1={"input": pooled_logits1, "target": labels},
                        inputs2={"input": pooled_logits2, "target": labels},)
            elif self.model.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # loss = loss_fct(pooled_logits.view(-1, self.model.num_labels), labels.view(-1))
                loss1, loss2 = self.task_compute_function(
                    fn=loss_fct,
                    inputs1={"input": pooled_logits1.view(-1, self.model.num_labels), "target": labels.view(-1)},
                    inputs2={"input": pooled_logits2.view(-1, self.model.num_labels), "target": labels.view(-1)},)
            elif self.model.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                # loss = loss_fct(pooled_logits, labels)
                loss1, loss2 = self.task_compute_function(
                    fn=loss_fct,
                    inputs1={"input": pooled_logits1, "target": labels},
                    inputs2={"input": pooled_logits2, "target": labels},)
        
        return loss1, loss2
        
        # if not return_dict:
        #     output = (pooled_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     past_key_values=transformer_outputs.past_key_values,
        #     hidden_states=transformer_outputs.hidden_states,
        #     attentions=transformer_outputs.attentions,
        # )


class OptimizerOPTForQuestionAnswering(MeZO2SGD):
    
    def init_zo2_upload(self):
        self.model.qa_outputs = self.model.qa_outputs.to(self.device)
    
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
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        """
            copy the original forward code and replace all 'self' to 'self.model'.
        """
        
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        self.model.model.decoder.zo_training = True
        self.assign_zo2_attributes(self, self.model.model.opt)
        hidden_states1, hidden_states2 = self.model.model(
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
        self.assign_zo2_attributes(self.model.model.opt, self)
        
        # hidden_states = transformer_outputs[0]

        # logits = self.model.qa_outputs(hidden_states)
        logits1, logits2 = self.task_compute_module(self.model.qa_outputs,
                                                    inputs1={"input": hidden_states1},
                                                    inputs2={"input": hidden_states2},
                                                    grad=self.projected_grad)
        # start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1).contiguous()
        # end_logits = end_logits.squeeze(-1).contiguous()
        (start_logits1, end_logits1), (start_logits2, end_logits2) = self.task_compute_function(
            fn=get_start_logits_and_end_logits,
            inputs1={"logits": logits1},
            inputs2={"logits": logits2},)

        # total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits1.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # start_loss = loss_fct(start_logits, start_positions)
            # end_loss = loss_fct(end_logits, end_positions)
            # total_loss = (start_loss + end_loss) / 2
            loss1, loss2 = self.task_compute_function(
                fn=get_qa_loss,
                inputs1={"loss_fct": loss_fct, "start_logits": start_logits1, "start_positions": start_positions, "end_logits": end_logits1, "end_positions": end_positions},
                inputs2={"loss_fct": loss_fct, "start_logits": start_logits2, "start_positions": start_positions, "end_logits": end_logits2, "end_positions": end_positions})

        return loss1, loss2
        
        # if not return_dict:
        #     output = (start_logits, end_logits) + transformer_outputs[2:]
        #     return ((total_loss,) + output) if total_loss is not None else output

        # return QuestionAnsweringModelOutput(
        #     loss=total_loss,
        #     start_logits=start_logits,
        #     end_logits=end_logits,
        #     hidden_states=transformer_outputs.hidden_states,
        #     attentions=transformer_outputs.attentions,
        # )