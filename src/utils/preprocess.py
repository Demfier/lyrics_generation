import os
import re
import sys
import h5py
import torch
import gensim
import pickle
import random
import skimage
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
import DALI as dali_code
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def process_raw(config):
    """
    process dali dataset and construct the train, val and test dataset
    NOTE: DALI info is not being used as we are not dealing with audio for now
    """
    if config['model_code'] == 'bilstm_scorer':
        dataset = process_bilstm(config)
    elif config['model_code'] == 'bimodal_scorer':
        dataset = process_bimodal(config)
    elif config['model_code'] in {'dae', 'vae'}:
        dataset = process_ae(config)
    elif config['model_code'] == 'clf':
        dataset = process_clf(config)

    with open('data/processed/{}/combined_dataset.pkl'.format(
            config['model_code']), 'wb') as f:
        pickle.dump(dataset, f)
    print('Created processed dataset')
    return dataset


def process_bimodal(config):
    print('Loading split dataset')
    with open(config['dali_lyrics']) as f:
        line_specs = f.readlines()

    dataset = []
    for entry in tqdm(line_specs):
        line, spec_path = entry.strip().split('\t')
        f_id = spec_path.split('/')[-1].split('.')[0]
        # list of dictionaries
        try:
            for l in get_subsequences(line):
                sample = {}
                sample['lyrics'] = l.strip()
                sample['mel_spec'] = read_spectrogram('{}{}.png'.format(config['split_spec'], f_id))
                dataset.append(sample)
        except FileNotFoundError as e:  # skip file as no spectrogram exists
            continue
    return dataset


def read_spectrogram(spectrogram_path):
    spec_arr = skimage.io.imread(spectrogram_path)
    # remove alpha dimension and resize to 224x224
    return skimage.transform.resize(spec_arr[:, :, :3], (224, 224, 3))


def process_bilstm(config):
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
            note = freq2note(n['freq'][0])
            if 'index' not in n or note < 0:
                continue
            word_id = n['index']
            curr_line_id = word_level[word_id]['index']
            if not notes_seq:
                line_id = curr_line_id

            if curr_line_id == line_id:
                notes_seq.append(note)
                words_seq.append(n['text'])
            else:
                sample['notes'] = notes_seq
                for subseq in get_subsequences(line_level[line_id]['text']):
                    sample['line'] = subseq.strip()
                    dataset.append(sample)
                line_id = curr_line_id
                notes_seq = []
                words_seq = []
                sample = {}
    return uniq_dictlist(dataset)


def process_ae(config):
    dataset = []
    print('Loading lyrics dataset')
    with open(config['dali_lyrics'], 'r') as f:
        lyrics_info = f.readlines()
    for l in lyrics_info:
        line, spec_path = l.split('\t')
        # get_subsequences() includes the complete sentence as well
        # for subseq in get_subsequences(line):
        #     dataset.append(subseq.strip())
        dataset.append(line)
    return list(set(dataset))


def process_clf(config):
    dataset = []
    print('Loading DALI')
    dali_data = dali_code.get_the_DALI_dataset(config['dali_path'],
                                               skip=[], keep=[])
    print('Loading lyrics dataset')
    with open(config['dali_lyrics'], 'r') as f:
        lyrics_info = f.readlines()
    genre_list = sorted(list(config['filter_genre']))
    for l in lyrics_info:
        line, spec_path = l.split('\t')
        # spec_path of the form:
        # {path_to_split_spectrograms}/8d2bea941a11497a984e50de3119b4d4_0.ogg
        # below command splits by '/', keep the last element, then splits by
        # '_' and keep the first element (to remove the line id and ext)
        spec_id = spec_path.split('/')[-1].split('_')[0]
        info = dali_data[spec_id].info
        genre = info['metadata']['genres'][0]  # consider 1st as the major one
        # Note: this line whould throw an error if genre is not present in
        # genre_list but this shouldn't happen since the lyrics were filtered
        # using the same config. In other words, it's like a sanity check.
        genre_id = genre_list.index(genre)
        sample = {'lyrics': normalize_string(line),
                  'genre_id': genre_id,
                  'genre': genre}
        dataset.append(sample)
    return dataset


def get_subsequences(line):
    line = line.split()
    return [' '.join(line[:i]) for i in range(1, len(line))]


def uniq_dictlist(list_of_dict):
    return list({v['line']: v for v in list_of_dict}.values())


def freq2note(freq):
    return np.round(12 * np.log2(freq/440.) + 69).astype(int)


def create_train_val_split(dataset, config):
    train, testval = train_test_split(dataset, train_size=0.8)
    test, val = train_test_split(testval, test_size=0.5)

    with open('data/processed/{}/train.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(train, f)
    with open('data/processed/{}/test.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(test, f)
    with open('data/processed/{}/val.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(val, f)
    print('Split dataset')


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
    all_pairs, _ = read_pairs(config)
    all_pairs = filter_pairs(all_pairs, config)
    vocab = Vocabulary()
    for pair_or_s in all_pairs:
        if config['model_code'] in {'bimodal_scorer', 'clf'}:
            # pair_or_s -> a sentence
            vocab.add_sentence(pair_or_s)
            continue

        # pair_or_s -> a tuple with two sentences
        vocab.add_sentence(pair_or_s[0])
        vocab.add_sentence(pair_or_s[1])
    print('Vocab size: {}'.format(vocab.size))
    np.save(config['vocab_path'], vocab, allow_pickle=True)
    return vocab


def read_pairs(config, mode='all'):
    """
    Reads src-target sentence pairs given a mode
    """
    processed_data_path = 'data/processed/{}/'.format(config['model_code'])
    if mode == 'all':
        with open('{}combined_dataset.pkl'.format(processed_data_path), 'rb') as f:
            dataset = pickle.load(f)
    else:  # if mode == 'train' / 'val' / 'test'
        with open('{}{}.pkl'.format(processed_data_path, mode), 'rb') as f:
            dataset = pickle.load(f)

    if config['model_code'] == 'bilstm_scorer':
        return read4bilstm(dataset)
    elif config['model_code'] == 'bimodal_scorer':
        return read4bimodal(dataset)
    elif config['model_code'] in {'dae', 'vae'}:
        return read4ae(dataset)
    elif config['model_code'] == 'clf':
        return read4genre(dataset)


def read4genre(dataset):
    lines = []
    y = []
    for o in dataset:
        lines.append(normalize_string(o['lyrics']))
        y.append(o['genre_id'])
    x_y = list(zip(lines, y))
    np.random.shuffle(x_y)
    lines, y = [], []
    for p, label in x_y:
        lines.append(p)
        y.append(label)
    return lines, torch.tensor(y).long()


def read4bimodal(dataset):
    lyrics_list = []
    mel_specs = []
    for v in dataset:
        mel_spec = v['mel_spec']
        for l in v['lyrics']:
            line = normalize_string(l['text'])
            lyrics_list.append(line)
            mel_specs.append(mel_spec)
    pairs = list(zip(lyrics_list, mel_specs))
    y = [1] * len(pairs) + [-1] * len(pairs)
    # neg sampling
    np.random.shuffle(lyrics_list)
    np.random.shuffle(mel_specs)
    pairs += list(zip(lyrics_list, mel_specs))

    # Shuffle the entire dataset
    data = list(zip(pairs, y))
    np.random.shuffle(data)
    pairs = []
    y = []
    for d in data:
        pairs.append(d[0])
        y.append(d[1])
    return pairs, torch.tensor(y).long()


def read4bilstm(dataset):
    pairs = []
    lines = []
    notes = []
    for o in dataset:
        line = normalize_string(o['line'])
        lines.append(line)
        notes.append(o['notes'])
        pairs.append((normalize_string('{}'.format(o['line'])), o['notes']))
    y = [1] * len(pairs)
    # create negative samples
    print('Generating negative samples')
    np.random.shuffle(lines)
    np.random.shuffle(notes)
    neg_pairs = list(zip(lines, notes))
    pairs += neg_pairs
    y += [-1] * len(neg_pairs)
    x_y = list(zip(pairs, y))
    np.random.shuffle(x_y)
    pairs, y = [], []
    for p, label in x_y:
        pairs.append(p)
        y.append(label)
    return pairs, torch.tensor(y).long()


def read4ae(dataset):
    pairs = []
    for o in dataset:
        line = normalize_string(o)
        pairs.append([line, line])
    return pairs, []


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


def filter_pairs(pairs, config):
    """
    Filter pairs with either of the sentence > max_len tokens
    ==============
    Params:
    ==============
    pairs (list of tuples): each tuple is a src-target sentence pair
    max_len (Int): Max allowable sentence length
    """
    max_len = config['MAX_LENGTH']
    if config['model_code'] == 'bimodal_scorer':
        # No need to return those big matrices
        return [' '.join(pair[0].split()[:max_len])
                for pair in pairs if pair[0]]
    elif config['model_code'] == 'clf':
        return [' '.join(l.split()[:max_len])
                for l in pairs if l]
    return [(' '.join(pair[0].split()[:max_len]),
             ' '.join(pair[1].split()[:max_len]))
            for pair in pairs if pair[0] and pair[1]]


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


def prepare_data(config):
    data_dir = config['data_dir']
    task = config['model_code']
    max_len = config['MAX_LENGTH']

    train_pairs = filter_pairs(read_pairs(config, 'train')[0], config)
    val_pairs = filter_pairs(read_pairs(config, 'val')[0], config)
    test_pairs = filter_pairs(read_pairs(config, 'test')[0], config)

    np.random.shuffle(train_pairs)
    np.random.shuffle(val_pairs)
    np.random.shuffle(test_pairs)
    return train_pairs, val_pairs, test_pairs


def load_word_embeddings(config):
    with h5py.File(config['filtered_emb_path'], 'r') as f:
        return torch.from_numpy(np.array(f['data'])).float()


def batch_to_model_compatible_data_bilstm(vocab, pairs, device):
    """
    Returns padded source and target index sequences
    ==================
    Parameters:
    ==================
    vocab (Vocabulary object): Vocabulary built from the dataset
    pairs (list of tuples): The source line and note sequences
    """
    sos_token = vocab.word2index['<SOS>']
    pad_token = vocab.word2index['<PAD>']
    eos_token = vocab.word2index['<EOS>']
    src_indexes, src_lens, notes_seq, notes_lens = [], [], [], []
    for pair in pairs:
        sent, notes = pair
        src_indexes.append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(pair[0]) + [eos_token]
                ))
        # extra 2 for sos_token and eos_token
        src_lens.append(len(pair[0].split()) + 2)
        notes_seq.append(notes)
        notes_lens.append(len(notes))

    # pad the batches
    src_indexes = pad_sequence(src_indexes, padding_value=pad_token)
    src_lens = torch.tensor(src_lens)
    notes_seq = pad_sequence(notes_seq, padding_value=pad_token)
    notes_lens = torch.tensor(notes_lens)
    return {
        'lyrics_seq': src_indexes.to(device),
        'lyrics_lens': src_lens.to(device),
        'music_seq': notes_seq.to(device),
        'music_lens': notes_lens.to(device)
        }


def batch_to_model_compatible_data_bimodal(vocab, pairs, device):
    """
    Returns padded source and target index sequences
    ==================
    Parameters:
    ==================
    vocab (Vocabulary object): Vocabulary built from the dataset
    pairs (list of tuples): The source line and note sequences
    """
    sos_token = vocab.word2index['<SOS>']
    pad_token = vocab.word2index['<PAD>']
    eos_token = vocab.word2index['<EOS>']
    src_indexes, src_lens, mel_specs = [], [], []
    for pair in pairs:
        sent, mel_spec = pair
        src_indexes.append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(pair[0]) + [eos_token]
                ))
        # extra 2 for sos_token and eos_token
        src_lens.append(len(pair[0].split()) + 2)
        mel_specs.append(mel_spec)

    # pad the batches
    src_indexes = pad_sequence(src_indexes, padding_value=pad_token)
    src_lens = torch.tensor(src_lens)
    # make (bs, num_channels, w, h) as vgg accepts image in this format
    mel_specs = torch.tensor(mel_specs).float().permute(0, 3, 1, 2).contiguous()
    return {
        'lyrics_seq': src_indexes.to(device),
        'lyrics_lens': src_lens.to(device),
        'mel_spec': mel_specs.to(device)
        }


def batch_to_model_compatible_data_clf(vocab, lines, device):
    """
    Returns padded source and target index sequences
    ==================
    Parameters:
    ==================
    vocab (Vocabulary object): Vocabulary built from the dataset
    pairs (list of tuples): The source line and note sequences
    """
    sos_token = vocab.word2index['<SOS>']
    pad_token = vocab.word2index['<PAD>']
    eos_token = vocab.word2index['<EOS>']
    src_indexes, src_lens = [], []
    for l in lines:
        src_indexes.append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(l) + [eos_token]
                ))
        # extra 2 for sos_token and eos_token
        src_lens.append(len(l.split()) + 2)

    # pad the batches
    src_indexes = pad_sequence(src_indexes, padding_value=pad_token)
    src_lens = torch.tensor(src_lens)
    return {
        'lyrics_seq': src_indexes.to(device),
        'lyrics_lens': src_lens.to(device)
        }


def batch_to_model_compatible_data_ae(vocab, pairs, device):
    """
    Returns padded source and target index sequences
    ==================
    Parameters:
    ==================
    vocab (Vocabulary object): Vocabulary built from the dataset
    pairs (list of tuples): The source and target sentence pairs
    device (Str): Device to place the tensors in
    """
    sos_token = vocab.word2index['<SOS>']
    pad_token = vocab.word2index['<PAD>']
    eos_token = vocab.word2index['<EOS>']
    src_indexes, src_lens, target_indexes = [], [], []
    for pair in pairs:
        src_indexes.append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(pair[0]) + [eos_token]
                ))
        # extra 2 for sos_token and eos_token
        src_lens.append(len(pair[0].split()) + 2)

        target_indexes.append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(pair[1]) + [eos_token]
                ))

    # pad src and target batches
    src_indexes = pad_sequence(src_indexes, padding_value=pad_token)
    target_indexes = pad_sequence(target_indexes, padding_value=pad_token)

    src_lens = torch.tensor(src_lens)
    return src_indexes.to(device), src_lens.to(device), target_indexes.to(device)


def _btmcd(vocab, pairs, config):
    """alias for batch_to_model_compatible_data"""
    if config['model_code'] == 'bilstm_scorer':
        return batch_to_model_compatible_data_bilstm(vocab, pairs, config['device'])
    elif config['model_code'] == 'bimodal_scorer':
        return batch_to_model_compatible_data_bimodal(vocab, pairs, config['device'])
    elif config['model_code'] in {'dae', 'vae'}:
        return batch_to_model_compatible_data_ae(vocab, pairs, config['device'])
    elif config['model_code'] == 'clf':
        return batch_to_model_compatible_data_clf(vocab, pairs, config['device'])
