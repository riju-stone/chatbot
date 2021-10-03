import numpy as np
import sys
import pickle
from collections import defaultdict
import itertools
import nltk
import random
# space is included in whitelist
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''


DATA_PATH = './data/chat.txt'

# these values determine the length of questions and answers while training
# increase the length to get a better trained model at the cost of more time and resources...
limit = {
    'maxq': 20,
    'minq': 0,
    'maxa': 20,
    'mina': 0
}

# increase vocab size for a better trained mmodel at the cost of more time and resource
VOCAB_SIZE = 6000
UNK = 'unk'


def default():
    return 1

# read lines from the file
def read_lines(filename):
    return open(filename).read().split('\n')[:-1]


# separate sentences in a line
def split_sentences(line):
    return line.split('.')


# remove anything that isn't in the vocabulary
def filter_lines(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])


# read words and create index to word and word to index dictionaries
def index(tokenized_sentences, vocab_size):
    # get frequency distribution of the tokenized words which are most used
    freq_dist = nltk.FreqDist(itertools.chain(tokenized_sentences))
    # get vocabulary of size VOCAB_SIZE
    vocab = freq_dist.most_common(vocab_size)
    # generate index to word dictionary
    index2word = ['_'] + [UNK] + [x[0] for x in vocab]
    # generate word to index dictionary
    word2index = dict([(w, i) for i, w in enumerate(index2word)])

    return index2word, word2index, freq_dist


# filter sequences based on set min length and max length
def filter_data(sequences):
    filter_q, filter_a = [], []
    raw_data_len = len(sequences) // 2

    for i in range(0, len(sequences), 2):
        qlen = len(sequences[i].split(' '))
        alen = len(sequences[i+1].split(' '))

        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filter_q.append(sequences[i])
                filter_a.append(sequences[i+1])

    filter_data_len = len(filter_q)
    filter_percent = int((raw_data_len - filter_data_len) / raw_data_len * 100)
    print('{} filtered from original data'.format(filter_percent))

    return filter_q, filter_a

'''
Replacing words with indices in a sequcnce
Replace with unknown if word not present in vocabulary
'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    
    return indices + [0] * (maxlen - len(seq))

'''
generating the final dataset by creating and array of indices 
and adding zero paddig. Zero Padding is simply a process of 
adding layers of zeros to our inputs
'''
def zero_pad(tokenized_q, tokenized_a, word2index):
    data_len = len(tokenized_q)

    index_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    index_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(tokenized_q[i], word2index, limit['maxq'])
        a_indices = pad_seq(tokenized_a[i], word2index, limit['maxa'])

        index_q[i] = np.array(q_indices)
        index_a[i] = np.array(a_indices)

    return index_q, index_a


def process_data():

    print('\n[ READING LINES FROM FILE ]')
    lines = read_lines(filename=DATA_PATH)

    # connverting all characters to lowercase
    lines = [ line.lower() for line in lines ]

    print('\n[ SAMPLE FROM THE DATASET ]')
    print(lines[25:35])

    # filter out unnecessary characters
    print('\n[ 1ST LAYER OF FILTERING ]')
    lines = [ filter_lines(line, EN_WHITELIST) for line in lines ]
    print(lines[25:35])

    # filter and distributing sequences into questions and answers
    print('\n[ 2ND LAYER OF FILTERING ]')
    qlines, alines = filter_data(lines)
    print('\n [ SAMPLE QUESTION ANSWER PAIR ]')
    print('\n q: {0} ; a: {1}'.format(qlines[15], alines[15]))
    print('\n q: {0} ; a: {1}'.format(qlines[20], alines[20]))

    # convert list of [ lines of text ] into list of [ list of words ]
    print('\n[ SEGMMENTING LINES OF TEXTS INTO LISTS OF WORDS ]')
    qtokenized = [ wordslist.split(' ') for wordslist in qlines ]
    atokenized = [ wordslist.split(' ') for wordslist in alines ]
    print('\n[ SAMPLE FROM SEGMENTED WORDS LIST ]')
    print('\nq : {0} ; a : {1}'.format(qtokenized[15], atokenized[15]))
    print('\nq : {0} ; a : {1}'.format(qtokenized[20], atokenized[20]))

    # indexing --> idx2w, w2idx
    idx2w, w2idx, freq_dist = index(qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    # adding zero padding
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n[ STORING NUMPY PADDED ARRAYS TO DISK ]')
    np.save('./data/processed_data/idx_q.npy', idx_q)
    np.save('./data/processed_data/idx_a.npy', idx_a)

    # saving the necessary dictionaries
    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    with open('./data/processed_data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
