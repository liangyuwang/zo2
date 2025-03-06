# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch

def fn_get_opt_decoder_hidden_states_from_layer_outputs(input):
    return input[0]

def get_shift_logits(logits):
    return logits[..., :-1, :].contiguous()

def get_shift_labels(labels):
    return labels[..., 1:].contiguous()

def get_pooled_logits(logits, batch_size, sequence_lengths):
    return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

def get_start_logits_and_end_logits(logits):
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()
    return start_logits, end_logits

def get_qa_loss(loss_fct, start_logits, start_positions, end_logits, end_positions):
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss

def init_all_hidden_states(output_hidden_states):
    return () if output_hidden_states else None

def init_all_self_attns(output_attentions):
    return () if output_attentions else None

def init_next_decoder_cache(use_cache):
    return () if use_cache else None

def update_next_decoder_cache(use_cache, next_decoder_cache, layer_outputs, output_attentions):
    if use_cache:
        next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
    return next_decoder_cache

def update_all_self_attns(output_attentions, all_self_attns, layer_outputs):
    if output_attentions:
        all_self_attns += (layer_outputs[1],)
    return all_self_attns

def update_all_hidden_states(output_hidden_states, all_hidden_states, hidden_states):
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    return all_hidden_states

def get_past_key_value(past_key_values, idx):
    return past_key_values[idx] if past_key_values is not None else None

def get_opt_sequence_classification_pooled_logits(self, logits, input_ids, inputs_embeds):
    if input_ids is not None:
        batch_size, sequence_length = input_ids.shape[:2]
    else:
        batch_size, sequence_length = inputs_embeds.shape[:2]
    if self.config.pad_token_id is None:
        sequence_lengths = -1
    else:
        if input_ids is not None:
            sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
        else:
            sequence_lengths = -1
    return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

def get_opt_sequence_classification_loss(self, loss, pooled_logits, labels):
    if self.config.problem_type is None:
        if self.num_labels == 1:
            self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            self.config.problem_type = "single_label_classification"
        else:
            self.config.problem_type = "multi_label_classification"
    if self.config.problem_type == "regression":
        loss_fct = torch.nn.MSELoss()
        if self.num_labels == 1:
            loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
            loss = loss_fct(pooled_logits, labels)
    elif self.config.problem_type == "single_label_classification":
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
    elif self.config.problem_type == "multi_label_classification":
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(pooled_logits, labels)
    return loss

def get_opt_question_answering_start_end_logits(logits):
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()
    return start_logits, end_logits

def get_opt_question_answering_loss(total_loss, start_logits, start_positions, end_logits, end_positions):
    # If we are on multi-GPU, split add a dimension
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions = start_positions.clamp(0, ignored_index)
    end_positions = end_positions.clamp(0, ignored_index)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss