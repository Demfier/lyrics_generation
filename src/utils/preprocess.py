import os
import re
import sys
import h5py
import torch
import string
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
from .classes import Vocabulary
import scipy
from scipy import misc

import matplotlib.pyplot as plt


def process_raw(config):
    """
    process dali dataset and construct the train, val and test dataset
    """
    if config['model_code'] == 'bilstm_scorer':
        dataset = process_bilstm(config)
    elif config['model_code'] == 'bimodal_scorer':
        dataset = process_bimodal(config)
    elif config['model_code'] in {'dae', 'vae'}:
        dataset = process_ae(config)
    elif config['model_code'] in {'bdae', 'bvae'}:
        dataset = process_bimodal_ae(config)
    elif config['model_code'] == 'lyrics_clf':
        dataset = process_lyrics_clf(config)
    elif config['model_code'] == 'spec_clf':
        dataset = process_spec_clf(config)
    elif 'lm' in config['model_code']:
        dataset = process_lm(config)
    elif config['model_code'] == 'IC_bimodal_scorer':
        dataset = process_ic_bimodal(config)

    save_path = 'data/processed/{}/combined_dataset.pkl'.format(
        config['model_code'])
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print('Saved processed dataset at: {}'.format(save_path))
    return dataset


def make_artist_song_dict(lines):
    songs = {}

    for line in lines:
        spec_id, _ = line.strip().split('\t')
        artist, _ = spec_id.split('_', 1)
        if artist not in songs:
            songs[artist] = {}
        # get song name with id of the subsong removed
        if artist in ['DepecheMode', 'NineInchNails']:
            song = spec_id.rsplit('_', 1)[0]
        else:
            song = spec_id[:-3]

        if song not in songs[artist]:
            songs[artist][song] = []
        songs[artist][song].append(spec_id)

    print('Total unique songs per artist:')
    for k in songs:
        print('{}: {}'.format(k, len(songs[k])))
    return songs


def prepare_held_out_songs(songs):
    """Accepts artist->song dict and prepares a held out dataset"""
    ho_val = set()
    ho_test = set()
    for artist in songs:
        artist_ho = np.random.choice(list(songs[artist].keys()), size=10)
        artist_ho_val = set(np.random.choice(artist_ho, size=5))
        artist_ho_test = set(artist_ho).difference(artist_ho_val)
        ho_val = ho_val.union(artist_ho_val)
        ho_test = ho_test.union(artist_ho_test)

    all_songs = set()
    for a in songs:
        all_songs = all_songs.union(set(songs[a].keys()))
    ho_train = all_songs.difference(ho_val.union(ho_test))

    print('ho distribution')
    print(len(ho_test), len(ho_train), len(ho_val))
    return ho_train, ho_test, ho_val


def process_bimodal(config):
    """
    This step loads the lyrics data along with its spec file's name
    """
    print('Loading bimodal dataset')
    with open(config['dataset_lyrics']) as f:
        lines = f.readlines()
        # don't random shuffle here
        # np.random.shuffle(lines)
    # Set some songs aside for test and val set
    songs = make_artist_song_dict(lines)

    ho_train, ho_test, ho_val = prepare_held_out_songs(songs)

    dataset = []
    train, val, test = [], [], []

    # maintain a dict of artists and their spec ids to generate negative samples
    spec_ids_dict = {}
    print('Making mel paths dict for artists')
    for line in tqdm(lines):
        # eg: DepecheMode_waiting-for-the-night_0.png   I'm waiting for the night to fall
        spec_id, lyrics = line.strip().split('\t')
        artist = spec_id.split('_', 1)[0]
        if artist not in spec_ids_dict:
            spec_ids_dict[artist] = []
        spec_ids_dict[artist].append(spec_id)

    artists_bucket = list(spec_ids_dict.keys())

    spec_array = {}
    print('Compiling actual dataset with positive and negative samples')
    np.random.shuffle(lines)
    for line in tqdm(lines):
        # eg: DepecheMode_waiting-for-the-night_0.png   I'm waiting for the night to fall
        try:
            spec_id, lyrics = line.split('\t')
            artist = spec_id.split('_', 1)[0]
            lyrics = normalize_string(
                lyrics.translate(str.maketrans('', '', string.punctuation)))
            mel_path = '{}{}/Specs/{}'.format(config['split_spec'], artist, spec_id)
            if not os.path.exists(mel_path):  # skip if spec doesn't exist
                continue
            song = spec_id.rsplit('_', 1)[0]
            # Add a positive sample
            spec_array[spec_id] = read_spectrogram(mel_path)
            sample = {}
            sample['lyrics'] = lyrics
            sample['spec_id'] = spec_id
            sample['mel_path'] = mel_path
            sample['label'] = 1
            if song in ho_val:
                val.append(sample)
            elif song in ho_test:
                test.append(sample)
            else:
                train.append(sample)
            dataset.append(sample)
            # subsequences = get_subsequences(lyrics)
            # for l in subsequences:
            #     sample = {}
            #     sample['lyrics'] = l
            #     sample['spec_id'] = spec_id
            #     sample['mel_path'] = mel_path
            #     sample['label'] = 1
            #     if song in ho_val:
            #         val.append(sample)
            #     elif song in ho_test:
            #         test.append(sample)
            #     else:
            #         train.append(sample)
            #     dataset.append(sample)

            # Add a negative sample
            # get the other artist (works as we have just two artists)
            if np.random.random() > 0.5:
                artist = artists_bucket[1-artists_bucket.index(artist)]
            sample = {}
            sample['lyrics'] = lyrics
            sample['label'] = 0

            if song in ho_val:
                while True:  # keep sampling until we get a different song
                    song_choices = \
                        set(songs[artist].keys()).intersection(ho_val)
                    negative_song = np.random.choice(list(song_choices))
                    if negative_song != song:
                        song = negative_song
                        spec_id = np.random.choice(songs[artist][song])
                        mel_path = '{}{}/Specs/{}'.format(config['split_spec'],
                                                          artist, spec_id)
                        break
                # skip if spec doesn't exist
                if not os.path.exists(mel_path):
                    continue
                sample['spec_id'] = spec_id
                sample['mel_path'] = mel_path
                val.append(sample)
            elif song in ho_test:
                while True:
                    song_choices = \
                        set(songs[artist].keys()).intersection(ho_test)
                    negative_song = np.random.choice(list(song_choices))
                    if negative_song != song:
                        song = negative_song
                        spec_id = np.random.choice(songs[artist][song])
                        mel_path = '{}{}/Specs/{}'.format(config['split_spec'],
                                                          artist, spec_id)
                        break
                # skip if spec doesn't exist
                if not os.path.exists(mel_path):
                    continue
                sample['spec_id'] = spec_id
                sample['mel_path'] = mel_path
                test.append(sample)
            else:
                while True:
                    song_choices = \
                        set(songs[artist].keys()).intersection(ho_train)
                    negative_song = np.random.choice(list(song_choices))
                    if negative_song != song:
                        song = negative_song
                        spec_id = np.random.choice(songs[artist][song])
                        mel_path = '{}{}/Specs/{}'.format(config['split_spec'],
                                                          artist, spec_id)
                        break
                # skip if spec doesn't exist
                if not os.path.exists(mel_path):
                    continue
                sample['spec_id'] = spec_id
                sample['mel_path'] = mel_path
                train.append(sample)
            dataset.append(sample)
            # for l in subsequences:
            #     sample = {}
            #     sample['lyrics'] = lyrics
            #     sample['label'] = 0

            #     if song in ho_val:
            #         while True:  # keep sampling until we get a different song
            #             song_choices = \
            #                 set(songs[artist].keys()).intersection(ho_val)
            #             negative_song = np.random.choice(list(song_choices))
            #             if negative_song != song:
            #                 song = negative_song
            #                 spec_id = np.random.choice(songs[artist][song])
            #                 mel_path = '{}{}/Specs/{}'.format(config['split_spec'],
            #                                                   artist, spec_id)
            #                 break
            #         # skip if spec doesn't exist
            #         if not os.path.exists(mel_path):
            #             continue
            #         sample['spec_id'] = spec_id
            #         sample['mel_path'] = mel_path
            #         val.append(sample)
            #     elif song in ho_test:
            #         while True:
            #             song_choices = \
            #                 set(songs[artist].keys()).intersection(ho_test)
            #             negative_song = np.random.choice(list(song_choices))
            #             if negative_song != song:
            #                 song = negative_song
            #                 spec_id = np.random.choice(songs[artist][song])
            #                 mel_path = '{}{}/Specs/{}'.format(config['split_spec'],
            #                                                   artist, spec_id)
            #                 break
            #         # skip if spec doesn't exist
            #         if not os.path.exists(mel_path):
            #             continue
            #         sample['spec_id'] = spec_id
            #         sample['mel_path'] = mel_path
            #         test.append(sample)
            #     else:
            #         while True:
            #             song_choices = \
            #                 set(songs[artist].keys()).intersection(ho_train)
            #             negative_song = np.random.choice(list(song_choices))
            #             if negative_song != song:
            #                 song = negative_song
            #                 spec_id = np.random.choice(songs[artist][song])
            #                 mel_path = '{}{}/Specs/{}'.format(config['split_spec'],
            #                                                   artist, spec_id)
            #                 break
            #         # skip if spec doesn't exist
            #         if not os.path.exists(mel_path):
            #             continue
            #         sample['spec_id'] = spec_id
            #         sample['mel_path'] = mel_path
            #         train.append(sample)
            #     dataset.append(sample)
        except ValueError as e:
            print('skipping {}...due to value error'.format(lyrics), end='')

    print('Saving Mel Spec arrays...')
    with open('data/processed/{}/spec_array.pkl'.format(
            config['model_code']), 'wb') as f:
        pickle.dump(spec_array, f)
    with open('data/processed/{}/train.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(train, f)
    with open('data/processed/{}/test.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(test, f)
    with open('data/processed/{}/val.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(val, f)
    print('Split dataset')
    return dataset


def process_ic_bimodal(config):
    """Prepares data for training bimodal scorer on IC task"""
    print('Loading IC dataset')
    with open(config['dataset_lyrics']) as f:
        lines = f.readlines()
    # skip the first header line and always take the first comment out of 5
    # only consider half of the dataset for testing
    lines = lines[1:][::5][:16000]
    print(lines[:10])
    # list of all images (to create negative samples)
    np.random.shuffle(lines)
    imgs = [_.split('|')[0].strip() for _ in lines]
    print('Compiling actual dataset')
    spec_array = {}
    dataset = []
    for line in tqdm(lines):
        try:
            img_name, _, caption = line.split('|')
            img_name = img_name.strip()
            caption = normalize_string(caption.strip())
            img_path = '{}{}'.format(config['split_spec'], img_name)
            if not os.path.exists(img_path):
                continue
            spec_array[img_name] = read_spectrogram(img_path)
            sample = {}
            sample['caption'] = caption
            sample['img_name'] = img_name
            sample['img_path'] = img_path
            sample['label'] = 1
            dataset.append(sample)

            # Create negative sample
            sample = {}
            sample['caption'] = caption
            while True:
                negative_img = np.random.choice(imgs)
                if negative_img != img_name:
                    img_name = negative_img
                    img_path = img_path = '{}{}'.format(config['split_spec'], img_name)
                    break
            sample['img_name'] = img_name
            sample['img_path'] = img_path
            sample['label'] = 0
            dataset.append(sample)
        except ValueError as e:
            print('skipping {}...due to value error'.format(lyrics), end='')

    print('Saving img_arrays...')
    with open('data/processed/{}/spec_array.pkl'.format(
            config['model_code']), 'wb') as f:
        pickle.dump(spec_array, f)
    return dataset


def process_spec_clf(config):
    print('Loading data file for spec clf.')
    with open(config['dataset_lyrics']) as f:
        lines = f.readlines()

    # Set some songs aside for test and val set
    songs = make_artist_song_dict(lines)
    ho_train, ho_test, ho_val = prepare_held_out_songs(songs)
    artists = sorted(list(config['label_names']))
    dataset = []
    train, val, test = [], [], []
    spec_array = {}
    np.random.shuffle(lines)
    for line in tqdm(lines):
        # eg: DepecheMode_waiting-for-the-night_0.png   I'm waiting for the night to fall
        spec_id, _ = line.strip().split('\t')  # ignore lyrics
        artist = spec_id.split('_', 1)[0]
        mel_path = '{}{}/Specs/{}'.format(config['split_spec'],
                                          artist, spec_id)
        if not os.path.exists(mel_path):  # skip if spec doesn't exist
            continue
        song = spec_id.rsplit('_', 1)[0]
        sample = {}
        sample['spec_id'] = spec_id
        sample['label'] = artists.index(artist)
        if song in ho_val:
            val.append(sample)
        elif song in ho_test:
            test.append(sample)
        else:
            train.append(sample)
        dataset.append(sample)
        spec_array[spec_id] = read_spectrogram(mel_path)

    print('Saving Mel Spec arrays...')
    with open('data/processed/{}/spec_array.pkl'.format(
            config['model_code']), 'wb') as f:
        pickle.dump(spec_array, f)
    with open('data/processed/{}/train.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(train, f)
    with open('data/processed/{}/test.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(test, f)
    with open('data/processed/{}/val.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(val, f)
    print('Split dataset')
    return dataset


def process_lyrics_clf(config):
    print('Loading lyrics file for lyrics clf.')
    with open(config['dataset_lyrics']) as f:
        lines = f.readlines()
        np.random.shuffle(lines)

    artists = config['label_names']
    dataset = []
    for line in tqdm(lines):
        spec_id, lyrics = line.strip().split('\t')
        lyrics = normalize_string(
                lyrics.translate(str.maketrans('', '', string.punctuation)))
        artist = spec_id.split('_', 1)[0]

        sample = {}
        sample['lyrics'] = lyrics
        sample['label'] = artists.index(artist)
        dataset.append(sample)
    return dataset


def process_lm(config):
    print('Loading lyrics file for lyrics clf.')
    with open(config['dataset_lyrics']) as f:
        lines = f.readlines()
        np.random.shuffle(lines)

    dataset = {}
    for line in tqdm(lines):
        spec_id, lyrics = line.strip().split('\t')
        lyrics = normalize_string(
                lyrics.translate(str.maketrans('', '', string.punctuation)))
        artist = spec_id.split('_', 1)[0]
        if artist not in dataset:
            dataset[artist] = []
        dataset[artist].append(lyrics)
    return dataset


def process_ae(config):
    dataset = []
    print('Loading lyrics dataset at {}'.format(config['dataset_lyrics']))
    with open(config['dataset_lyrics'], 'r') as f:
        lyrics = f.readlines()
    for line in lyrics:
        line = line.strip()
        # remove all punctuations from the line
        dataset.append(
            line.translate(
                str.maketrans('', '', string.punctuation)
                )
            )
    return list(set(dataset))


def process_bimodal_ae(config):
    # Reads lyrics dataset file, gets the image z from a pretrained image VAE
    # and processes the lyrics
    dataset = []
    print('Loading lyrics dataset at {}'.format(config['dataset_lyrics']))
    with open(config['dataset_lyrics'], 'r') as f:
        lines = f.readlines()

    # Set some songs aside for test and val set
    songs = make_artist_song_dict(lines)

    ho_train, ho_test, ho_val = prepare_held_out_songs(songs)
    dataset = []
    train, val, test = [], [], []

    spec_array = {}
    print('Compiling dataset with held out val and test...')
    np.random.shuffle(lines)
    for line in tqdm(lines):
        line = line.strip()
        spec_id, lyrics = line.split('\t')
        artist = spec_id.split('_', 1)[0]
        # remove all punctuations from the line
        lyrics = normalize_string(
            lyrics.translate(str.maketrans('', '', string.punctuation)))

        if artist in ['DepecheMode', 'NineInchNails']:
            song = spec_id.rsplit('_', 1)[0]
        else:
            song = spec_id[:-3]

        if song in ho_val:
            val.append([lyrics, spec_id])
        elif song in ho_test:
            test.append([lyrics, spec_id])
        else:
            train.append([lyrics, spec_id])
        dataset.append([lyrics, spec_id])
    with open('data/processed/{}/train.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(train, f)
    with open('data/processed/{}/test.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(test, f)
    with open('data/processed/{}/val.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(val, f)
    print('Split dataset')
    return dataset


def read_spectrogram_batch(spectrogram_paths):
    # remove alpha dimension and resize to 224x224
    return [skimage.transform.resize(s[:, :, :3], (224, 224, 3))
            for s in skimage.io.imread_collection(spectrogram_paths)]


def read_spectrogram(spectrogram_path):
    # remove alpha dimension and resize to 224x224
    return scipy.misc.imresize(
        skimage.io.imread(spectrogram_path)[:, :, :3], (224, 224, 3)
        )


def get_subsequences(line):
    line = line.split()
    return [' '.join(line[:i]) for i in range(1, len(line))]


def uniq_dictlist(list_of_dict):
    return list({v['line']: v for v in list_of_dict}.values())


def freq2note(freq):
    return np.round(12 * np.log2(freq/440.) + 69).astype(int)


def create_train_val_split(dataset, config):
    train, testval = train_test_split(dataset, test_size=0.2)
    test, val = train_test_split(testval, test_size=0.5)

    with open('data/processed/{}/train.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(train, f)
    with open('data/processed/{}/test.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(test, f)
    with open('data/processed/{}/val.pkl'.format(config['model_code']), 'wb') as f:
        pickle.dump(val, f)
    print('Split dataset')


def build_vocab(config):
    if config['model_code'] == 'bvae':
        all_pairs, _, _ = read_pairs(config)
    else:
        all_pairs, _ = read_pairs(config)
    all_pairs = filter_pairs(all_pairs, config)
    vocab = Vocabulary()
    for pair_or_s in all_pairs:
        if config['model_code'] in {'bimodal_scorer', 'lyrics_clf'}:
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
    elif config['model_code'] in {'bvae'}:
        return read4bimodalae(dataset)
    elif config['model_code'] == 'lyrics_clf':
        return read4lyricsclf(dataset)
    elif config['model_code'] == 'spec_clf':
        return read4specsclf(dataset)
    elif config['model_code'] == 'IC_bimodal_scorer':
        return read4ICbimodal(dataset)


def read4lyricsclf(dataset):
    np.random.shuffle(dataset)
    y = []
    lines = []
    for o in dataset:
        lines.append(normalize_string(o['lyrics']))
        y.append(o['label'])
    return lines, torch.tensor(y).long()


def read4bimodal(dataset):
    """
    dataset is a list of dictionaries with two fields, lyrics and spec_ids
    and labels
    """
    np.random.shuffle(dataset)
    y = []
    spec_ids = []
    lyrics_list = []
    for v in dataset:
        lyrics_list.append(v['lyrics'])
        spec_ids.append(v['spec_id'])
        y.append(v['label'])
    pairs = list(zip(lyrics_list, spec_ids))
    return pairs, torch.tensor(y).long()


def read4ICbimodal(dataset):
    """
    dataset is a list of dictionaries with two fields, lyrics and spec_ids
    and labels
    """
    np.random.shuffle(dataset)
    y = []
    spec_ids = []
    lyrics_list = []
    for v in dataset:
        lyrics_list.append(v['caption'])
        spec_ids.append(v['img_name'])
        y.append(v['label'])
    pairs = list(zip(lyrics_list, spec_ids))
    return pairs, torch.tensor(y).long()


def read4specsclf(dataset):
    """dataset is a dict with two keys, spec_id and label"""
    np.random.shuffle(dataset)
    y = []
    spec_ids = []
    for v in dataset:
        spec_ids.append(v['spec_id'])
        y.append(v['label'])
    return spec_ids, torch.tensor(y).long()


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


def read4bimodalae(dataset):
    with open('./data/processed/bvae/spec_array.pkl', 'rb') as f:
        spec_array = pickle.load(f)
    pairs = []
    image_vectors = []
    for o in dataset:
        line = normalize_string(o[0])
        try:
            x = np.array(spec_array[o[1]])
            x = x/255.0
            x = x.astype('float32')
            x = x.transpose(2, 0, 1)
            # x = torch.from_numpy(x)
            # x = x.unsqueeze(0)
            image_vectors.append(x)
        except KeyError as e:
            print('error in read4bimodalae', e)
            continue
        pairs.append([line, line])
    return pairs, torch.from_numpy(np.array(image_vectors)), []


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
    elif config['model_code'] == 'lyrics_clf':
        return [' '.join(l.split()[:max_len])
                for l in pairs if l]
    # for seq2seq-type models
    return [(' '.join(pair[0].split()[:max_len]),
             ' '.join(pair[1].split()[:max_len]))
            for pair in pairs if pair[0] and pair[1]]


# Embeddings part
def generate_word_embeddings(vocab, config):
    # Load original (raw) embeddings
    src_embeddings = gensim.models.Word2Vec.load('data/processed/english_w2v.pkl')

    # Create filtered embeddings
    # Initialize filtered embedding matrix
    embeddings_matrix = np.zeros((vocab.size, config['embedding_dim']))
    for index, word in vocab.index2word.items():
        try:  # random normal for special and OOV tokens
            if index <= 4:
                embeddings_matrix[index] = \
                    np.random.normal(size=(config['embedding_dim'], ))
                continue  # use continue to avoid extra `else` block
            embeddings_matrix[index] = src_embeddings[word]
        except KeyError as e:
            embeddings_matrix[index] = \
                np.random.normal(size=(config['embedding_dim'], ))

    with h5py.File(config['filtered_emb_path'], 'w') as f:
        f.create_dataset('data', data=embeddings_matrix, dtype='f')
    return torch.from_numpy(embeddings_matrix).float()


def load_word_embeddings(config):
    with h5py.File(config['filtered_emb_path'], 'r') as f:
        return torch.from_numpy(np.array(f['data'])).float()


def prepare_data(config):
    if config['model_code'] in {'bdae', 'bvae'}:
        train_pairs, train_image_vec, _ = read_pairs(config, 'train')
        train_pairs = filter_pairs(train_pairs, config)
        val_pairs, val_image_vec, _ = read_pairs(config, 'val')
        val_pairs = filter_pairs(val_pairs, config)
        test_pairs, test_image_vec, _ = read_pairs(config, 'test')
        test_pairs = filter_pairs(test_pairs, config)
        return train_pairs, val_pairs, test_pairs, train_image_vec, \
            val_image_vec, test_image_vec
    else:
        train_pairs = filter_pairs(read_pairs(config, 'train')[0], config)
        val_pairs = filter_pairs(read_pairs(config, 'val')[0], config)
        test_pairs = filter_pairs(read_pairs(config, 'test')[0], config)

    np.random.shuffle(train_pairs)
    np.random.shuffle(val_pairs)
    np.random.shuffle(test_pairs)
    return train_pairs, val_pairs, test_pairs


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


def batch_to_model_compatible_data_bimodal(vocab, pairs, device, spec_array):
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
        sent, spec_id = pair
        src_indexes.append(
            torch.tensor(
                [sos_token] + vocab.sentence2index(pair[0]) + [eos_token]
                ))
        # extra 2 for sos_token and eos_token
        src_lens.append(len(pair[0].split()) + 2)
        mel_specs.append(spec_array[spec_id])

    # pad the batches
    src_indexes = pad_sequence(src_indexes, padding_value=pad_token)
    src_lens = torch.tensor(src_lens)
    # make (bs, num_channels, w, h) as vgg accepts image in this format
    mel_specs = \
        torch.tensor(mel_specs).float().permute(0, 3, 1, 2).contiguous()
    return {
        'lyrics_seq': src_indexes.to(device),
        'lyrics_lens': src_lens.to(device),
        'mel_spec': mel_specs.to(device)
        }


def batch_to_model_compatible_data_spec_clf(pairs, device, spec_array):
    mel_specs = [spec_array[spec_id] for spec_id in pairs]
    # make (bs, num_channels, w, h) as vgg accepts image in this format
    mel_specs = \
        torch.tensor(mel_specs).float().permute(0, 3, 1, 2).contiguous()
    return {'mel_spec': mel_specs.to(device)}


def batch_to_model_compatible_data_lyrics_clf(vocab, lines, device):
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


def _btmcd(vocab, pairs, config, *args):
    """alias for batch_to_model_compatible_data"""
    if config['model_code'] == 'bilstm_scorer':
        return batch_to_model_compatible_data_bilstm(vocab, pairs, config['device'])
    elif config['model_code'] in ['bimodal_scorer', 'IC_bimodal_scorer']:
        # args[0] would be spec_array for this case
        return batch_to_model_compatible_data_bimodal(vocab, pairs, config['device'], args[0])
    elif config['model_code'] in {'dae', 'vae', 'bvae'}:
        return batch_to_model_compatible_data_ae(vocab, pairs, config['device'])
    elif config['model_code'] == 'lyrics_clf':
        return batch_to_model_compatible_data_lyrics_clf(vocab, pairs, config['device'])
    elif config['model_code'] == 'spec_clf':
        return batch_to_model_compatible_data_spec_clf(pairs, config['device'], args[0])
