import os
import torch as T
import torch.nn.functional as F
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

    def p_sample(logits, top_p=0.9):
        sorted_logits, sorted_indices = T.sort(logits, dim=-1, descending=True)
        cumulative_probs = T.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        min_probs, _ = T.min(cumulative_probs, dim=-1)
        min_probs = min_probs.view(-1, 1)

        sorted_logits_ = sorted_logits.clone()

        sorted_logits = T.where(cumulative_probs > top_p, T.empty_like(
            sorted_indices).to('cuda').fill_(T.tensor(-2.0**32)), sorted_logits_)

        sorted_logits = T.where(
            min_probs > top_p, sorted_logits_, sorted_logits)

        probabilities = F.softmax(sorted_logits, dim=-1)

        next_word_sorted_index = T.multinomial(
            probabilities, num_samples=1, replacement=False)

        next_word_sorted_index = next_word_sorted_index.view(-1, 1)

        next_word_index = T.gather(
            sorted_indices, dim=-1, index=next_word_sorted_index)
        log_probs = T.gather(probabilities, dim=-1,
                             index=next_word_sorted_index)
        log_probs = T.log(log_probs+1e-8)

        log_probs = log_probs.view(-1)

        return next_word_index, log_probs

    def greedy_decoding(input_ids, samples, past=None, top_p=0.9, temperature=1):
        EOS_token = tokenizer.encode("<|endoftext|>")[0]

        i = 0

        input_ids = T.repeat_interleave(input_ids, samples, dim=0)
        _, input_size = input_ids.size()

        candidates = []

        response_ids = input_ids.clone()

        total_log_probs = T.zeros(samples).to('cuda')
        mask = T.ones(samples).to('cuda')

        while len(candidates) < samples and i < 300:
            with T.no_grad():
                outputs, past = model(input_ids, past)
                predictions = outputs
                logits = predictions[:, -1, :]
                logits = logits/temperature

                next_word_index, log_probs = p_sample(logits, top_p=top_p)
                total_log_probs = total_log_probs + log_probs * mask

                mask = T.where(next_word_index.view(-1) ==
                               EOS_token, T.zeroes(samples).to('cuda'), mask)

                input_ids = next_word_index
                response_ids = T.cat([response_ids, next_word_index], dim=-1)
                candidates += generate_candidates(response_ids[:, input_size:])

                i += 1

        return candidates, total_log_probs
    candidates, scores = greedy_decoding(input_ids, samples)

    return candidates, scores
