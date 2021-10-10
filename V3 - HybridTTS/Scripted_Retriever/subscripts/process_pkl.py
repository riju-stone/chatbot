import sys
sys.path.append("../")  # nopep8
from Sentence_Encoder.query_encoder import encode
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import numpy as np
import pickle
import collections


def process():
    with open("processed_scripts/bot_profile.pkl", "rb") as fp:
        bot_profile = pickle.load(fp)

    with open("processed_scripts/chatterbot.pkl", "rb") as fp:
        chatterbot = pickle.load(fp)

    bot_queries = [k for k, v in bot_profile.items()]
    bot_contexts = ["" for k, v in bot_profile.items()]

    chatterbot_queries = [k for k, v in chatterbot.items()]
    chatterbot_contexts = ["" for k, v in chatterbot.items()]

    embedded_bot_queries = encode(bot_queries, bot_contexts)
    embedded_chatterbot_queries = encode(
        chatterbot_queries, chatterbot_contexts)

    with open("processed_scripts/embedded_bot_queries.pkl", "wb") as fp:
        pickle.dump(embedded_bot_queries, fp)

    with open("processed_scripts/embedded_chatterbot_queries.pkl", "wb") as fp:
        pickle.dump(embedded_chatterbot_queries, fp)
