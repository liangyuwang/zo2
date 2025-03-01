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