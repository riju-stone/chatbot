from multiprocessing import Process
from types import CodeType
import faiss
import time
import sqlite3
import csv
import random
import copy
import tensorflow_hub as hub
import tensorflow_text
import math
import numpy as np
import pickle

from transformers.utils.dummy_sentencepiece_objects import T5Tokenizer
from Large_Scale_Retriever.retrieve import retrive
import Utils.functions as utils
from Ranker.rank import rank_and_choose
from Generator.generator import generate as Generate
from Classifier.model.dialogue_acts import Encoder as Classifier
from Sentence_Encoder.query_encoder import encode as query_encoder
from Sentence_Encoder.response_encoder import encode as response_encoder
import tensorflow as tf
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse

# Command line arguments for activating TTS module
parser = argparse.ArgumentParser(description="Stone Bot")
parser.add_argument('--voice', dest='voice', action='store_true')
parser.add_argument('--no-voice', dest='voice', action='store_false')
parser.set_defaults(voice=True)
flags = parser.parse_args()

device = 'cuda'

# Load Faiss Data
with open('Large_Scale_Retriever/faiss_index/thread_idx.pkl', 'rb') as fp:
    idx = pickle.load(fp)

index = faiss.read_index('Large_Scale_Retriever/large.index')

# Load Database
conn = sqlite3.connect('Large_Scale_Retriever/database/reddit.db')
c = conn.cursor()

# Load Scripts
with open('Scripted_Retriever/processed_scripts/bot_profile.pkl', 'rb') as fp:
    bot_profile = pickle.load(fp)

bot_queries = [k for k, v in bot_profile.items()]

with open('Scripted_Retriever/processed_scripts/chatterbpt.pkl', 'rb') as fp:
    chatterbot = pickle.load(fp)

chatterbot_queries = pickle.load(fp)

# Load Script Embeddings
with open('Scripted_Retriever/processed_scripts/embedded_bot_queries.pkl', 'rb') as fp:
    bot_queries_embd = pickle.load(fp)

with open('Scripted_Retriever/processed_scripts/embedded_chatterbot_queries.pkl', 'rb') as fp:
    chatterbot_queries_embd = pickle.load(fp)

# Load Dialog Act Classifier
with open('Classifier/processed_data/processed_data.pkl', 'rb') as fp:
    data = pickle.load(fp)

labels2index = data["labels2idx"]
index2labels = {v: k for k, v in labels2index.items()}

with T.no_grad():
    dialog_classifier = Classifier(
        D=bot_queries_embd.shape[-1], classes_num=len(labels2index)).cuda()
    checkpoint = T.load('Classifier/model_backup/model.pt')
    dialog_classifier.load_state_dict(checkpoint['model_state_dict'])
    dialog_classifier = dialog_classifier.eval()

#  TODO: Load TTS model


# Load DialoGPT and Blender Models
with T.no_grad():
    tokenizer = GPT2TokenizerFast.from_pretrained('Generator/blender/configs/')
    weights = T.load('Generator/blender/parameters/pytorch_model.bin')
    cfg = GPT2Config.from_json_file('Generator/blender/config.json')
    model = GPT2LMHeadModel(cfg)

    model.load_state_dict(weights)
    model.to('cuda')
    model.eval()

with tf.device('/cpu:0'):
    command_codes = ["<PASS>", "<JOKE>", "<GENERATE>",
                     "<INITIATE>", "<TIL>", "<STORY>", "<SHOWER>", "<STOP>"]

    code_map = {
        "<INITIATE>": ["Scripted_Retriever/reddit_data/nostupidq.csv",
                       "Scripted_Retriever/reddit_data/jokesq.csv",
                       "Scripted_Retriever/reddit_data/showerthoughtsq.csv",
                       "Scripted_Retriever/reddit_data/tilq.csv"
                       ],
        "<TIL>": ["Scripted_Retriever/reddit_data/tilq.csv"],
        "<SHOWER>": ["Scripted_Retriever/reddit_data/showerthoughtsq.csv"],
        "<STORY>": ["Scripted_Retriever/reddit_data/writingpromptsa.csv"],
        "<JOKE>": ["Scripted_Retriever/reddit_data/jokesq.csv"]
    }

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

    def load_reddit_data(directory, conversation_history):
        candidates = []

        with open(directory, conversation_history) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            
