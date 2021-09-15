import pickle
import numpy as np
from random import sample
from parser import process_data

# load processed data from disk


def load_data():
    try:
        with open('./data/processed_data/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = None

    idx_q = np.load('./data/processed_data/idx_q.npy')
    idx_a = np.load('./data/processed_data/idx_a..npy')

    return metadata, idx_q, idx_a


'''
split the dataset into training set (70%),
testing set (15%) and validity set (15%)
'''


def split_dataset(x, y, split_ratio=[0.7, 0.15, 0.15]):
    data_len = len(x)
    lens = [len(data_len * item) for item in split_ratio]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]: lens[0] + lens[1]], y[lens[0]: lens[0] + lens[1]]
    validX, validY = [-lens[-1]:], y[-lens[-1]:]

    return (trainX, trainY), (testX, testY), (validX, validY)


# generate batches from dataset
def batch_gen(x, y, batch_size):
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[1: (i+1)*batch_size].T, y[i: (i+1)*batch_size].T


# generate bathes by random sampling
def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T


def decode(sequence, lookup, separator=''):  # 0 used for padding, is ignored
    return separator.join([lookup[element] for element in sequence if element])


if __name__ == '__main__':
    process_data()
