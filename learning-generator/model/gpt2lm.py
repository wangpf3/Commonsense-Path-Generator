import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2LM(nn.Module):
    """docstring for GPT2LM"""
    def __init__(self, gpt, config):
        super(GPT2LM, self).__init__()
        self.gpt = gpt
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = nn.Parameter(self.gpt.wte.weight)

    def forward(self, inputs, logsoftmax=True):

        outputs = self.gpt(inputs)[0]
        lm_logits = self.lm_head(outputs)

        if logsoftmax:
            lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits
