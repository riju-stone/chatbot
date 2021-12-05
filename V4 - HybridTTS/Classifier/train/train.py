import sys
sys.path.append("../")  # nopep8
import pickle
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.dialogue_acts import Encoder
from data_loader.bucket_and_batch import bucket_and_batch
import numpy as np
import string
import random

device = T.device('cude' if T.cuda.is_available() else 'cpu')
max_grad_norm = 1

with open("../data/processed_data.pkl", "rb") as fp:
    data = pickle.load(fp)

labels2index = data["labels2idx"]
index2labels = {i: v for v, i in labels2index.items()}

train_queries_vector = data["train_queries_vec"]
train_acts_vec = data["train_acts_vec"]

test_queries_vector = data["test_queries_vec"]
test_acts_vec = data["test_acts_vec"]

model = Encoder(D=test_queries_vector.shape[-1], classes_num=len(labels2index))
model = model.cuda()

parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Parameter Count: ", parameter_count)

optimizer = T.optim.Adam(model.parameters(), lr=1e-3)


def loss_func(logits, labels, l2=1e-6):
    regularization = T.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if 'bias' not in name and 'embedding' not in name:
            regularization += T.norm(param).pow(2)

    loss = nn.MSELoss()
    output = loss(logits, labels) + l2 * regularization

    return output


batches_train_queries, batches_train_classes = bucket_and_batch(
    train_queries_vector, train_acts_vec, 64, len(labels2index))

batches_test_queries, batches_test_classes = bucket_and_batch(
    test_queries_vector, test_acts_vec, 64, len(labels2index))


def predict(queries, classes, train=True):
    global model

    if train:
        model = model.train()
    else:
        model = model.test()

    logits = model(T.tensor(queries).to(device))

    loss = loss_func(logits, T.tensor(classes).float().to(device))

    _, sorted_index = T.sort(logits, dim=-1, descending=True)
    sorted_index = sorted_index[:, 0:2]
    sorted_index = sorted_index.cpu().numpy().tolist()

    _, gold_sorted_index = T.sort(
        T.tensor(classes).to(device), dim=-1, descending=True)
    gold_sorted_index = gold_sorted_index[:, 0:2]
    gold_sorted_index = gold_sorted_index.cpu().numpy().tolist()

    score = 0
    total = 0

    for sorted_id, gold_sorted_id in zip(sorted_index, gold_sorted_index):
        for id in sorted_id:
            if id in gold_sorted_id:
                score += 1
            total += 1

    return loss, (score/total)


best_val_accuracy = 0

for epoch in range(100):
    i = 0
    for batch_X, batch_Y in zip(batches_train_queries, batches_train_classes):
        loss, accuracy = predict(batch_X, batch_Y, train=True)

        loss.backward()
        T.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print("Step {}, Loss: {}, Accuracy: {}".format(i, loss, accuracy))
        i += 1

    print("\n\n Starting Validation \n\n")
    total_val_accuracy = 0
    i = 0

    for batch_X, batch_Y in zip(batches_test_queries, batches_test_classes):
        with T.no_grad():
            loss, accuracy = predict(batch_X, batch_Y, train=False)
            total_val_accuracy += accuracy

            if i % 100 == 0:
                print("Step {}, Loss: {}, Accuracy: {}".format(i, loss, accuracy))
            i += 1

    mean_accuracy = total_val_accuracy/len(batches_test_queries)

    print("\n\nEpoch {}, Validation Result: Accuracy: {}\n".format(
        epoch, mean_accuracy))

    if mean_accuracy > best_val_accuracy:
        best_val_accuracy = mean_accuracy

        T.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "../model_backup/model_new.pt")

        print("\n Checkpoint Saved \n")
