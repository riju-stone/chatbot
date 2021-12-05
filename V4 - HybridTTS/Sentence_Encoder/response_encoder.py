import sys
import numpy as np
import math
import pickle
import os

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer

sys.path.append("../")  # nopep8

import Utils.functions as utils


def encode(texts, contexts):
    texts = [utils.simple_preprocess(text) for text in texts]
    contexts = [utils.simple_preprocess(context) for context in contexts]

    encoder_model = SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2')
    response_embeddings = encoder_model.encode(texts)
    context_embeddings = encoder_model.encode(contexts)

    return np.concatenate([np.asarray(response_embeddings), np.asarray(context_embeddings)], axis=-1)
