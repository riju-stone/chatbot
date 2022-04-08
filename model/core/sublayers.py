import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# this function simply calculates the scaled dot product attention scores


def attention(query, key, value, d_k, mask=None, dropout=None):
    score_mat = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        score_mat = score_mat.masked_fill(mask == 0, -1e9)

    score_mat = F.softmax(score_mat, dim=-1)

    if dropout is not None:
        score_mat = dropout(score_mat)

    output = torch.matmul(score_mat, value)

    return output


class Normalize(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model

        # to parameters to calibrate normalization
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        bs = query.size()

        key = self.key_linear(key).view(bs, -1, self.h, self.d_k)
        value = self.value_linear(value).view(bs, -1, self.h, self.d_k)
        query = self.query_linear(query).view(bs, -1, self.h, self.d_k)

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = query.transpose(1, 2)

        score_mat = attention(query, key, value, self.d_k, mask, self.dropout)

        concat = score_mat.transpose(
            1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
