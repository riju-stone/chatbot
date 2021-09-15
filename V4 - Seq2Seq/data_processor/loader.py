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
    


if __name__ == '__main__':
    process_data()
