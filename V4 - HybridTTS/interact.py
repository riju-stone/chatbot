from multiprocessing import Process
from sys import flags
import faiss
import sqlite3
import csv
import random
import copy
import numpy as np
import pickle

from Large_Scale_Retriever.retrieve import retrive
import Utils.functions as utils
from Ranker.rank import rank_and_choose
from Generator.generator import generate as Generate
from Classifier.model.dialogue_acts import Encoder as Classifier
from Sentence_Encoder.query_encoder import encode as query_encoder
from Sentence_Encoder.response_encoder import encode as response_encoder
from Large_Scale_Retriever import retrieve

import tensorflow as tf
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import os

device = 'cuda'

# Load Faiss Data
with open('Large_Scale_Retriever/faiss_index/thread_idx.pkl', 'rb') as fp:
    idx = pickle.load(fp)

index = faiss.read_index('Large_Scale_Retriever/faiss_index/large.index')

# Load Database
conn = sqlite3.connect('Large_Scale_Retriever/database/reddit.db')
c = conn.cursor()

# Load Scripts
with open('Scripted_Retriever/processed_scripts/bot_profile.pkl', 'rb') as fp:
    bot_profile = pickle.load(fp)

bot_queries = [k for k, v in bot_profile.items()]

with open('Scripted_Retriever/processed_scripts/chatterbot.pkl', 'rb') as fp:
    chatterbot = pickle.load(fp)

chatterbot_queries = [k for k, v in chatterbot.items()]

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

# Load TTS model
# with T.no_grad():
#     text2speech = tts_class()

# Load Pretrained DialoGPT Model
# find required weights from https://github.com/microsoft/DialoGPT
with T.no_grad():
    tokenizer = GPT2Tokenizer.from_pretrained("Generator/dialogpt/")
    weights = T.load('Generator/dialogpt/large_ft.pkl')
    weights_reverse = T.load(
        'Generator/dialogpt/small_reverse.pkl')
    cfg = GPT2Config.from_json_file('Generator/dialogpt/config.json')
    model = GPT2LMHeadModel(cfg)
    model_reverse = GPT2LMHeadModel(cfg)

    # fix misused key value
    # weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    # weights.pop("lm_head.decoder.weight", None)
    # weights_reverse["lm_head.weight"] = weights_reverse["lm_head.decoder.weight"]
    # weights_reverse.pop("lm_head.decoder.weight", None)

    model.load_state_dict(weights, strict=False)
    model.to('cuda')
    model.eval()

    # model_reverse.load_state_dict(weights_reverse, strict=False)
    # model_reverse.to('cuda')
    # model_reverse.eval()

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

        with open(directory, conversation_history, encoding='cp850') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for i, row in enumerate(csv_reader):
                if 'writing' in directory:
                    parent_id = str(row['parent_id'])[3:]
                    thread_id = str(row['link_id'])[3:]
                    if parent_id == thread_id:
                        candidate = str(row['body'])
                else:
                    candidate = str(row['title'])
                    if 'joke' in directory:
                        candidate += '.... '+str(row['selftext'])
                candidates.append(candidate)

        return random_response(candidates, conversation_history)

    def top_candidates(candidates, scores, top=1):
        sorted_score_index = np.flip(np.argsort(scores), axis=-1)
        candidates = [candidates[i] for i in sorted_score_index.tolist()]
        scores = [scores[i] for i in sorted_score_index.tolist()]

        return candidates[0:top], scores[0:top], sorted_score_index.tolist()

    def generate(texts, past):
        candidates, _ = Generate(texts, model, tokenizer)
        return candidates, past

    # Main Logic

    conversation_history = []
    past = None
    stop_flag = 0
    print('\n')

    while True:
        utterance = input("Say Something: ")
        utils.delay_print("\nThinking....")

        candidates = []
        temp_candidates = []
        temp_scores = []

        if not conversation_history:
            query_context = []
            response_context = ['']
        else:
            if len(conversation_history) > 5:
                truncated_history = copy.deepcopy(conversation_history[-5:])
            else:
                truncated_history = copy.deepcopy(conversation_history)

            response_context = [conversation_history[-1]]
            query_context = [stuff for stuff in truncated_history]

        query_encoding = query_encoder([utterance], [query_context])

        if conversation_history:
            if len(conversation_history) > 5:
                truncated_history = conversation_history[-5:]
            else:
                truncated_history = conversation_history
            generated_responses, past = generate(
                truncated_history+[utterance], past)
        else:
            generated_responses, past = generate([utterance], past)

        bot_cosine_scores = utils.cosine_similarity_nd(
            query_encoding, bot_queries_embd)
        bot_queries_, bot_cosine_scores_, _ = top_candidates(
            bot_queries, bot_cosine_scores, top=-1)

        active_codes = []

        bot_candidates = bot_profile[bot_queries_[0]]

        filtered_bot_candidates = []
        for candidate in bot_candidates:
            flag = 0
            for code in command_codes:
                if code in candidate:
                    active_codes.append(code)
                    candidate = candidate.replace(code, "")
                    filtered_bot_candidates.append(candidate)
                    flag = 1
                    break

            if flag == 0:
                candidates.append(candidate)
                filtered_bot_candidates.append(candidates)
                active_codes.append("")

        with T.no_grad():
            logits = dialog_classifier(T.tensor(query_encoding).to(device))
            _, sorted_index = T.sort(logits, dim=-1, descending=True)
            sorted_index = sorted_index.squeeze(0)
            sorted_index = sorted_index[0:2].cpu().tolist()

        labels = [index2labels[i] for i in sorted_index]

        """
        Possible Dialog Acts:
        ['nonsense', 'dev_command', 'open_question_factual', 'appreciation', 'other_answers', 'statement',
        'respond_to_apology', 'pos_answer', 'closing', 'comment', 'neg_answer', 'yes_no_question', 'command',
        'hold', 'NULL', 'back-channeling', 'abandon', 'opening', 'other', 'complaint', 'opinion', 'apology',
        'thanking', 'open_question_opinion']
        """

        if bot_cosine_scores_[0] >= 0.75:
            response, id = rank_and_choose(
                tokenizer, model, utterance, query_encoding, filtered_bot_candidates, response_context, conversation_history)
            code = active_codes[id]

            if code in code_map:
                directories = code_map[code]
                directory = random.choice(directories)
                response += " " + \
                    load_reddit_data(directory, conversation_history)

            elif code == "<GENERATE>":
                response, _ = rank_and_choose(
                    tokenizer, model, utterance, query_encoding, generated_responses, response_context, conversation_history)

            elif code == "<STOP>":
                stop_flag = 1

        elif stop_flag != 1:
            mode = "DEFAULT"
            bias = None

            if 'open_question_factuals' in labels or ('yes_no_question' in labels and 'NULL' not in labels) or 'open_question_opinion' in labels or 'command' in labels:
                bias = 0.07

            elif "apology" in labels:
                mode = "BREAK"
                candidates = [
                    "Apology accepted.", "No need to apologize.", "No worries.", "You are forgiven"]
                response, _ = rank_and_choose(
                    tokenizer, model, utterance, query_encoding, candidates, response_context, conversation_history)

            elif "abandon" in labels or "nonsense" in labels:
                mode = np.random.choice(["BREAK", "INITIATE"], p=[0.6, 0.4])

                if mode == "BREAK":
                    candidates = [
                        "what?", "Can you rephrase what you mean?", "What do you mean exactly?"]
                    response, _ = rank_and_choose(
                        tokenizer, model, utterance, query_encoding, candidates, response_context, conversation_history)

                else:
                    directories = code_map['<INITIATED>']
                    directory = random.choice(directories)
                    response = load_reddit_data(
                        directory, conversation_history)

            elif 'hold' in labels:
                mode = 'BREAK'
                candidates = ["Do you want to add something more?",
                              "I feel like you want to say something more."]
                response, _ = rank_and_choose(tokenizer, model, utterance, query_encoding,
                                              generated_responses + candidates, response_context, conversation_history)

            elif 'closing' in labels:
                mode = "BREAK"
                candidates = ["Nice talking to you.",
                              "Goodbye.", "See you later.", "Peace Out !"]
                response, _ = rank_and_choose(
                    tokenizer, model, utterance, query_encoding, candidates, response_context, conversation_history)

                stop_flag = 1

            elif 'opening' in labels:
                mode = "BREAK"
                response, _ = rank_and_choose(
                    tokenizer, model, utterance, query_encoding, generated_responses, response_context, conversation_history)

                stop_flag = 1

            elif 'thanking' in labels:
                mode = np.random.choice(["BREAK", "INITIATE"], p=[0.6, 0.4])

                if mode == "BREAK":
                    candidates = ["No need to mention", "You are welcome."]
                    response, _ = rank_and_choose(tokenizer, model, utterance, query_encoding,
                                                  generated_responses + candidates, response_context, conversation_history)

                else:
                    directories = code_map["<INITIAIE>"]
                    directory = random.choice(directories)
                    response = load_reddit_data(
                        directory, conversation_history)

            elif 'apology' in labels:
                mode = "BREAK"
                candidates = [
                    "Apology accepted.", "No need to apologize.", "No worries.", "You are forgiven"]
                response, _ = rank_and_choose(
                    tokenizer, model, utterance, query_encoding, generated_responses + candidates, response_context, conversation_history)

            elif 'response_to_apology' in labels or 'pos_answer' in labels or 'neg_answer' in labels or 'appreciation' in labels or 'back_channeling' in labels:
                mode = np.random.choice(["BREAK", "INITIATE"], P=[0.6, 0.4])

                if mode == "BREAK":
                    response, _ = rank_and_choose(
                        tokenizer, model, utterance, query_encoding, generated_responses, response_context, conversation_history)

                else:
                    directories = code_map['<INITIATE>']
                    directory = random.choice(directories)
                    response = load_reddit_data(
                        directory, conversation_history)

            if mode != 'BREAK':
                chatterbot_cosine_scores = utils.cosine_similarity_nd(
                    query_context, chatterbot_queries_embd)
                chatterbot_queries_, chatterbot_cosine_scores_, _ = top_candidates(
                    chatterbot_queries, chatterbot_cosine_scores, top=1)
                candidates += chatterbot[chatterbot_queries_[0]]

                retrieved_candidates = retrieve(
                    conn, c, idx, index, query_encoding, query_context)

                if bias is not None:
                    biases = [0.0 for _ in candidates]
                    for _ in generated_responses:
                        biases.append(0.0)

                    for _ in retrieved_candidates:
                        biases.append(bias)

                    biases = np.asarray(biases, np.float32)
                else:
                    biases = None

                candidates += generated_responses + retrieved_candidates
                response, _ = rank_and_choose(
                    tokenizer, model, utterance, query_encoding, candidates, response_context, conversation_history, bias=biases)

        print("\n")

        if len(str(response).split(" ") <= 100):
            if flags.voice:
                entry = utils.simple_preprocess(
                    str(response).lower(), for_speech=True, return_tokenized=True)
                entry = " ".join(entry)

                # wavefiles = text2speech.process(entry)
                filename = 'speech.mp3'

                def f1():
                    utils.delay_print("BOT --> " + response)

                p1 = Process(target=f1)
                p1.start()
            else:
                utils.delay_print("BOT --> " + response)
        else:
            utils.delay_print("BOT --> " + response, t=0.01)

        print("\n")

        conversation_history.append(utterance)
        conversation_history.append(response)

        if stop_flag == 1:
            break
