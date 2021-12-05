import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf
import numpy as np
import pickle
import sys
sys.path.append("../../")  # nopep8
from Sentence_Encoder.query_encoder import encode


train_dir = "train.txt"
test_dir = "test.txt"

filename = train_dir
all_targets = []


def extract_data(filename):
    contexts = []
    queries = []
    acts = []

    with open(filename) as file:
        global all_targets
        lines = file.readlines()
        for line in lines:
            line = line.strip()

            split_line = line.split(" : ")
            line = split_line[1]
            context1 = split_line[0]

            if "what" in context1.lower() or "why" in context1.lower() or "where" in context1.lower() or "how" in context1.lower() or "who" in context1.lower():
                punc = "?"
            else:
                punc = "."

            split_line = line.split(" > ")
            context2 = split_line[0].strip()

            if context2 == "EMPTY":
                context = context1 + punc
            else:
                context = context1 + punc + " " + context2

            line = split_line[1]
            split_line = line.split(" ## ")
            current_uttr = split_line[0]
            targets = split_line[1]
            targets = targets.split(";")
            targets = [target for target in targets if target != '']

            if len(targets) < 2:
                targets.append("NULL")

            all_targets += targets

            contexts.append(context)
            queries.append(current_uttr)
            acts.append(targets)

    return contexts, queries, acts


train_contexts, train_queries, train_acts = extract_data(train_dir)
test_contexts, test_queries, test_acts = extract_data(test_dir)

all_targets = list(set(all_targets))

labels2index = {v: i for i, v in enumerate(all_targets)}

train_queries_vector = []
i = 0
batch_size = 2000

while i < len(train_queries):
    print(i)
    if i + batch_size > len(train_queries):
        batch_size = len(train_queries) - i

    train_query_vector = encode(
        train_queries[i: i + batch_size], train_contexts[i: i + batch_size])
    train_queries_vector.append(train_query_vector)

    i += batch_size

train_queries_vector = np.concatenate(train_queries_vector, axis=0)

test_queries_vector = []
i = 0
while i < len(test_queries):
    if i + batch_size > len(test_queries):
        batch_size = len(test_queries) - i

    test_query_vector = encode(
        test_queries[i: i + batch_size], test_contexts[i: i + batch_size])
    test_queries_vector.append(test_query_vector)

    i += batch_size

test_queries_vector = np.concatenate(test_queries_vector, axis=0)

train_acts_vec = []
for acts in train_acts:
    train_acts_vec.append([labels2index[act] for act in acts])

test_acts_vec = []
for acts in test_acts:
    test_acts_vec.append([labels2index[act] for act in acts])

train_acts_vec = np.asarray(train_acts_vec, np.int)
test_acts_vec = np.asarray(test_acts_vec, np.int)

data = {}

data["labels2idx"] = labels2index


data["train_contexts"] = train_contexts
data["test_contexts"] = test_contexts

data["train_queries"] = train_queries
data["train_acts"] = train_acts

data["test_queries"] = test_queries
data["test_acts"] = test_acts

data["test_queries_vec"] = test_queries_vector
data["test_acts_vec"] = test_acts_vec

data["train_queries_vec"] = train_queries_vector
data["train_acts_vec"] = train_acts_vec

with open("processed_data.pkl", 'wb') as fp:
    pickle.dump(data, fp)
