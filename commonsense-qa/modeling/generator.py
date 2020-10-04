import torch
import torch.nn as nn
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

class Generator(nn.Module):
    """docstring for GPT2LM"""
    def __init__(self, gpt, config, max_len):
        super(Generator, self).__init__()
        self.gpt = gpt
        self.config = config
        self.max_len = max_len
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = nn.Parameter(self.gpt.wte.weight)

    def forward(self, inputs, train=False, return_path=False):
        # input: [batch, seq]
        context_len = inputs.size(1)
        print(context_len)
        generated = inputs
        next_token = inputs
        past = None
        hidden_list = []
        # accu_scores = torch.zeros(inputs.size(0)).to(inputs.device)
        with torch.no_grad():
            for step in range(self.max_len - 1):
                outputs = self.gpt(next_token, past=past)
                hidden = outputs[0][:, -1]
                hidden_list.append(hidden)
                past = outputs[1]
                next_token_logits = self.lm_head(hidden)
                next_logits, next_token = next_token_logits.topk(k=1, dim=1)
                generated = torch.cat((generated, next_token), dim=1)

        outputs = self.gpt(next_token, past=past)
        hidden = outputs[0][:, -1]
        hidden_list.append(hidden)
        hidden_list = torch.stack(hidden_list, dim=1)
        next_token_logits = self.lm_head(hidden)
        next_logits, next_token = next_token_logits.topk(k=1, dim=1)
        generated = torch.cat((generated, next_token), dim=1)
        if return_path:
            return hidden_list.mean(1), generated 
        else:
            return hidden_list.mean(1)

