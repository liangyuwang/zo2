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