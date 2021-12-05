import tensorflow_text
import tensorflow_hub as hub
import tensorflow as tf
import sys
sys.path.append("../")  # nopep8
import numpy as np
import math
import pickle
import os
from Classifier.model.dialogue_acts import Encoder
import Utils.functions as utils
from sentence_transformers import SentenceTransformer


def encode(texts, contexts):
    texts = [utils.simple_preprocess(text) for text in texts]
    contexts = [utils.simple_preprocess(context) for context in contexts]

    encoder_model = SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2')
    question_embeddings = encoder_model.encode(texts)
    context_embeddings = encoder_model.encode(contexts)
    return np.concatenate([np.asarray(question_embeddings), np.asarray(context_embeddings)], axis=-1)
