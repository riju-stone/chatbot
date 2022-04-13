import torch
import torch.nn as nn
from layers import EncoderLayer, DecoderLayer
from embed import InputEmbedder, PositionalEncoder
from sublayers import Normalize
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = InputEmbedder(vocab_size, d_model)
        self.pos_encoder = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Normalize(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pos_encoder(x)

        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = InputEmbedder(vocab_size, d_model)
        self.pos_encoder = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Normalize(d_model)

    def forward(self, trg, enc_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pos_encoder(x)

        for i in range(self.N):
            x = self.layers[i](x, enc_outputs, src_mask, trg_mask)

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.output = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_outputs = self.encoder(src, src_mask)
        dec_outputs = self.decoder(trg, enc_outputs, src_mask, trg_mask)

        final_output = self.output(dec_outputs)

        return final_output


def get_model(opt, src_vocab, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model,
                        opt.layers, opt.heads, opt.dropout)

    if opt.load_weights is not None:
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if opt.device == 0:
        model = model.cuda()

    return model
