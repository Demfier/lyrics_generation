import os
import re
import sys
import h5py
import math
import json
import torch
import gensim
import pickle
import random
import itertools
import unicodedata
import numpy as np
import pandas as pd
import DALI as dali_code
from itertools import zip_longest
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def process_raw(config):
    """
    process dali dataset and construct the train, val and test dataset
    NOTE: DALI info is not being used as we are not dealing with audio for now
    """
    # This step takes a bit of time
    print('Loading DALI')
    dali_data = dali_code.get_the_DALI_dataset(config['dali_path'],
                                               skip=[], keep=[])
    dataset = []
    file_ids = os.listdir(config['dali_path'])
    for f_id in file_ids:
        if f_id == 'info':
            continue
        # f_id is of the format id.gz. remove the gz part
        f_id = f_id.split('.')[0]
        # Get different level of annotations
        annotations = dali_data[f_id].annotations
        note_level = annotations['annot']['notes']
        word_level = annotations['annot']['words']
        line_level = annotations['annot']['lines']

        sample = {}
        line_id = 0
        notes_seq = []
        words_seq = []
        for n in note_level:
            if 'index' not in n:
                continue
            note = freq2note(n['freq'][0])
            word_id = n['index']
            curr_line_id = word_level[word_id]['index']
            if not notes_seq:
                line_id = curr_line_id

            if curr_line_id == line_id:
                notes_seq.append(note)
                words_seq.append(n['text'])
            else:
                sample['notes'] = notes_seq
                sample['line_sanity'] = ' '.join(words_seq)
                sample['line'] = line_level[line_id]['text']
                dataset.append(sample)
                line_id = curr_line_id
                notes_seq = []
                words_seq = []
                sample = {}
    with open('data/processed/combined_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print('Created processed dataset')


def freq2note(freq):
    return np.round(12 * np.log2(freq/440.) + 69).astype(int)


def create_train_val_split(dataset):
    train, testval = train_test_split(dataset, train_size=0.8)
    test, val = train_test_split(testval, test_size=0.5)
    with open('data/processed/train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open('data/processed/test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open('data/processed/val.pkl', 'wb') as f:
        pickle.dump(val, f)
    print('Split dataset')


def load_data(config):
    with open(config['data_dir'] + 'train.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open(config['data_dir'] + 'test.pkl', 'rb') as f:
        test_set = pickle.load(f)
    with open(config['data_dir'] + 'val.pkl', 'rb') as f:
        val_set = pickle.load(f)
    return train_set, val_set, test_set


# Vocab and file-reading part
class Vocabulary(object):
    """Vocabulary class"""
    def __init__(self):
        super(Vocabulary, self).__init__()
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4  # count the special tokens above

    def add_sentence(self, sentence):
        for word in sentence.strip().split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.word2count[word] = 1
            self.index2word[self.size] = word
            self.size += 1
        else:
            self.word2count[word] += 1

    def sentence2index(self, sentence):
        indexes = []
        for w in sentence.split():
            try:
                indexes.append(self.word2index[w])
            except KeyError as e:  # handle OOV
                indexes.append(self.word2index['<UNK>'])
        return indexes

    def index2sentence(self, indexes):
        return [self.index2word[i] for i in indexes]


def build_vocab(config):
    all_pairs = read_pairs()
    all_pairs = filter_pairs(all_pairs, config['MAX_LENGTH'])
    vocab = Vocabulary()
    for pair in all_pairs:
        vocab.add_sentence(pair[0])
    print('Vocab size: {}'.format(vocab.size))
    np.save(config['vocab_path'], vocab, allow_pickle=True)
    return vocab


def read_pairs(mode='all'):
    """
    Reads src-target sentence pairs given a mode
    """
    if mode == 'all':
        with open('data/processed/combined_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:  # if mode == 'train' / 'val' / 'test'
        with open('data/processed/{}.pkl'.format(mode), 'rb') as f:
            dataset = pickle.load(f)

    pairs = []
    for o in dataset:
        pairs.append((normalize_string('{}'.format(o['line'])), o['notes']))
    return pairs


def normalize_string(x):
    """Lower-case, trip and remove non-letter characters
    ==============
    Params:
    ==============
    x (Str): the string to normalize
    """
    x = unicode_to_ascii(x.lower().strip())
    x = re.sub(r'([.!?])', r'\1', x)
    x = re.sub(r'[^a-zA-Z.!?]+', r' ', x)
    return x


def unicode_to_ascii(x):
    return ''.join(
        c for c in unicodedata.normalize('NFD', x)
        if unicodedata.category(c) != 'Mn')


def filter_pairs(pairs, max_len):
    """
    Filter pairs with either of the sentence > max_len tokens
    ==============
    Params:
    ==============
    pairs (list of tuples): each tuple is a src-target sentence pair
    max_len (Int): Max allowable sentence length
    """
    return [pair for pair in pairs if (len(pair[0].split()) <= max_len)]


# Embeddings part
def generate_word_embeddings(vocab, config):
    # Load original (raw) embeddings
    ftype = 'bin'

    src_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
        'data/raw/english_w2v.bin', binary=True)

    # Create filtered embeddings
    # Initialize filtered embedding matrix
    combined_embeddings = np.zeros((vocab.size, config['embedding_dim']))
    for index, word in vocab.index2word.items():
        try:  # random normal for special and OOV tokens
            if index <= 4:
                combined_embeddings[index] = \
                    np.random.normal(size=(config['embedding_dim'], ))
                continue  # use continue to avoid extra `else` block
            combined_embeddings[index] = src_embeddings[word]
        except KeyError as e:
            combined_embeddings[index] = \
                np.random.normal(size=(config['embedding_dim'], ))

    with h5py.File(config['filtered_emb_path'], 'w') as f:
        f.create_dataset('data', data=combined_embeddings, dtype='f')
    return torch.from_numpy(combined_embeddings).float()


def train_w2v_model(config):
    all_pairs = read_pairs()
    all_pairs = filter_pairs(all_pairs, config['MAX_LENGTH'])
    random.shuffle(all_pairs)
    src_sentences = []
    for pair in all_pairs:
        src_sentences.append(pair[0].split())

    src_w2v = gensim.models.Word2Vec(src_sentences, size=300,
                                     min_count=1, iter=50)
    src_w2v.wv.save_word2vec_format('data/raw/english_w2v.bin', binary=True)


def load_word_embeddings(config):
    with h5py.File(config['filtered_emb_path'], 'r') as f:
        return torch.from_numpy(np.array(f['data'])).float()


def batch_to_model_compatible_data(vocab, sentences):
    """
    Returns padded source and target index sequences
    ==================
    Parameters:
    ==================
    vocab (Vocabulary object): Vocabulary built from the dataset
    pairs (list of tuples): The source and target sentence pairs
    """
    pad_token = vocab.word2index['<PAD>']
    eos_token = vocab.word2index['<EOS>']
    src_indexes, src_lens = [], []
    for sent in sentences:
        src_indexes.append(vocab.sentence2index(sent) + [eos_token])
        src_lens.append(len(sent.split()))

    # pad the batches
    src_indexes = pad_indexes(src_indexes, value=pad_token)
    src_lens = torch.tensor(src_lens)
    return src_indexes, src_lens


def _btmcd(vocab, sentences):
    """alias for batch_to_model_compatible_data"""
    return batch_to_model_compatible_data(vocab, sentences)


def pad_indexes(indexes_batch, value):
    """
    Returns a padded tensor of shape (max_seq_len, batch_size) where
    max_seq_len is the sequence with max length in indexes_batch and
    batch_size is the number of elements in indexes_batch
    ==================
    Parameters:
    ==================
    indexes_batch (list of list): the batch of indexes to pad
    value (int): the value with which to pad the indexes batch
    """
    return torch.tensor(list(zip_longest(*indexes_batch, fillvalue=value)))
