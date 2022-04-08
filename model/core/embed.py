import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class InputEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super.__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pos_embed = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_embed[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pos_embed[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pos_embed = Variable(self.pos_embed[:, :seq_len], requires_grad = False)

        if x.is_cuda:
            pos_embed = pos_embed.cuda()
        
        x += pos_embed

        return self.dropout(x)