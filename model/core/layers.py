import torch
import torch.nn as nn
from sublayers import FeedForward, MultiHeadAttention, Normalize


class EncoderLayer(nn.Module):
    def _init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Normalize(d_model)
        self.norm2 = Normalize(d_model)

        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super.__init__()
        self.norm1 = Normalize(d_model)
        self.norm2 = Normalize(d_model)
        self.norm3 = Normalize(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attention1 = MultiHeadAttention(heads, d_model, dropout)
        self.attention2 = MultiHeadAttention(heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, dropout)

    def forward(self, x, outputs, src_mask, trg_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attention1(x2, x2, x2, trg_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attention2(x2, outputs, outputs, src_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))

        return x
