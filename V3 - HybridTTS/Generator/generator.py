import os
import torch as T
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import numpy as np


def generate(conv_hist, model, tokenizer, past=None, samples=20):
    EOS_token = tokenizer.encode("<|end of test|>")[0]

    def _make_feature(sents, eos):
        msg_index = []
        for msg in sents:
            msg_index.append(tokenizer.encode(msg))

        input_ids = [i for s in msg_index for i in s + [eos]][:-1]
        input_ids.append(eos)

        if len(input_ids) > 200:
            input_ids = input_ids[-200:]

        return input_ids

    input_ids = _make_feature(conv_hist, EOS_token)
    input_ids = T.tensor(input_ids).long().to('cuda').unsqueeze(0)

    def generate_candidates(hypothesis):
        EOS_token = tokenizer.encode("<|endoftext|>")[0]
        hypothesis_list = hypothesis.cpu().numpy().tolist()

        candidates = []

        for beam in hypothesis_list:
            if beam[-1] == EOS_token and EOS_token not in beam[:-1]:
                candidate = tokenizer.decode(beam[:-1])
                candidates.append(candidate)

        return candidates
