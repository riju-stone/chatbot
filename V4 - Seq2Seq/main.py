import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
from data_processor import loader
import os


def initial_setup():
    metadata, idx_q, idx_a = loader.load_data('data/processed_data/')
    (trainX, trainY), (testX, testY), (validX,
                                       validY) = loader.split_dataset(idx_q, idx_a)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validX = tl.prepro.remove_pad_sequences(validY.tolist())

    return metadata, trainX, trainY, testX, testY, validX, validY


if __name__ == "__main__":
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup()

    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    batch_size = 32
    n_step = src_len // batch_size
    src_vocab_size = len(metadata['idx2w'])
    emb_dim = 1024
