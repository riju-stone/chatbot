from Sentence_Encoder.response_encoder import encode as response_encoder
import Utils.functions as utils
import numpy as np
import torch as T
import copy
import random


def random_response(candidates, conversation_history, p=None):
    loop = 5

    if p is None:
        response = random.choice(candidates)
    else:
        response = np.random.choice(candidates, p=p)

    i = 0

    while response in conversation_history:
        if p is None:
            response = random.choice(candidates)
        else:
            response = np.random.choice(candidates, p=p)

        i += 1
        if i > loop:
            break

    return response


def top_candidates(candidates, scores, top=1):
    sorted_score_index = np.flip(np.argsort(scores), axis=-1)
    candidates = [candidates[i] for i in sorted_score_index.tolist()]
    scores = [scores[i] for i in sorted_score_index.tolist()]
    return candidates[0:top], scores[0:top], sorted_score_index.tolist()


def rank_and_choose(tokenizer, model_reverse, utterance, query_encoding, candidates, response_context, conversation_history, bias=None, alpha=0.4, beta=0.6):
    if bias is None:
        bias = 0.0

    EOS_token = tokenizer.encode("<|endoftext|>")[0]
    original_candidates = copy.deepcopy(candidates)

    response_encoding = response_encoder(
        candidates, response_context * len(candidates))
    rank_scores = utils.cosine_similarity_nd(query_encoding, response_encoding)

    normed_rank_scores = utils.normalize(rank_scores + bias)

    last_utterance = utterance

    def _make_feature(sents, eos):
        msg_index = []
        for msg in sents:
            msg_index.append(tokenizer.encode(msg))
        input_ids = [i for s in msg_index for i in s+[eos]][:-1]
        input_ids.append(eos)

        if len(input_ids) > 300:
            input_ids = input_ids[-300:]

        return input_ids

    output_ids = _make_feature([last_utterance], EOS_token)
