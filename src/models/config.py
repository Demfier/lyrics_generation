"""
Some metadata for the new artists lyrics dataset (full)
avg. sentence length = 6.8
max sentence length = 25
sentence length v/s frequency:
Counter({6: 215713, 5: 197313, 7: 183332, 4: 150900,
         8: 147375, 9: 107612, 10: 78934, 3: 76527,
         11: 51589, 12: 32846, 2: 27430, 13: 19102,
         14: 11323, 1: 7100, 15: 6727, 16: 4049,
         17: 1881, 18: 715, 19: 194, 20: 48,
         21: 14, 22: 2, 23: 2, 0: 1, 24: 1, 25: 1})


Some metadata for the 7 artiss lyrics dataset (7 artists only)
avg. sentence length = 5.49
max sentence length = 24
99.8 percentile = 15
sentence length v/s frequency:
Counter({5: 6409, 4: 5914, 6: 5637, 3: 4231, 7: 3851,
         8: 2336, 2: 1797, 9: 1419, 10: 959, 1: 855,
         11: 601, 12: 329, 13: 187, 14: 110, 15: 62,
         0: 61, 16: 27, 17: 17, 18: 5, 24: 2, 19: 1,
         23: 1})
"""


import os
import torch
model_config = {
    # hyperparameters
    'lr': 1e-3,  # 0.005
    'dropout': 0.2,
    'patience': 0,  # number of epochs to wait before decreasing lr
    'min_lr': 1e-7,  # minimum allowable value of lr
    'task': '10-second-rec',  # mt/dialog/rec/dialog-rec
    'model_code': 'bvae',  # bimodal_scorer/bilstm_scorer/dae/vae/lyrics_clf/spec_clf

    'clip': 50.0,  # values above which to clip the gradients
    'tf_ratio': 1.0,  # teacher forcing ratio

    'unit': 'lstm',
    'n_epochs': 100,
    'batch_size': 100,
    'enc_n_layers': 1,
    'dec_n_layers': 1,
    'dec_mode': 'greedy',  # type of decoding to use {greedy, beam}
    'bidirectional': True,  # make the encoder bidirectional or not
    'attn_model': None,  # None/dot/concat/general

    'hidden_dim': 256,  # 200
    'embedding_dim': 300,

    # vocab-related params
    'PAD_TOKEN': 0,
    'SOS_TOKEN': 1,
    'EOS_TOKEN': 2,
    'UNK_TOKEN': 3,
    'MAX_LENGTH': 15,  # Max length of a sentence

    # run-time conf
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # gpu_id ('x' for multiGPU mode)
    'wemb_type': 'w2v',  # type of word embedding to use: w2v/fasttext
    'lang_pair': 'en-en',  # src-target language pair
    'use_scheduler': True,
    'use_embeddings?': True,  # use word embeddings or not
    'freeze_embeddings?': False,  # keep word embeddings trainable or not
    'first_run?': True,  # True for the very first run
    'min_freq': 2,  # min frequency for the word to be a part of the vocab
    'n_layers': 1,

    # DALI-specific
    'filter_lang': 'english',
    'label_names': sorted(['DepecheMode', 'NineInchNails']),
    'max_songs': 200,  # maximum songs to consider per genre

    'embedding_dim': 300,
    'bidirectional': True,
    'use_melfeats?': False,  # whether to use already extracted img features or use calculate them on the fly while encoding
    'use_embeddings?': True,
    'generate_spectrograms': False,
    # 'pretrained_model': False,  # {'vae-1L-bilstm-11', False},
    'pretrained_scorer': False,  # {'bimodal_scorer-1L-bilstm-0', False}
    'pretrained_spec_clf': False,  # {'spec_clf_scorer-1L-bilstm-0', False}
    'pretrained_model': 'bvae-1L-bilstm-30',  # {'vae-1L-bilstm-11', False},
    # 'pretrained_scorer': 'bimodal_scorer-1L-bilstm-5',  # {'bimodal_scorer-1L-bilstm-0', False}
    'save_dir': 'saved_models/',
    'data_dir': 'data/processed/',
    # 'file_name': '/home/d35kumar/Github/lyrics_generation/data/raw/split_info.txt',
    # 'dali_path': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0',
    # 'dali_audio': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0/ogg_audio/',  # Path to store dali audio files
    # 'dali_audio_split': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0/ogg_audio_split/', # Path to store dali audio files split by lines
    # 'spectrograms': '/home/d35kumar/Github/lyrics_generation/data/processed/spectrograms_split/'
    'dataset_path': '/collection/gsahu/ae/lyrics_generation/data/raw/DALI_v1.0/',
    'dataset_audio': '/collection/gsahu/ae/lyrics_generation/data/raw/ogg_audio/',  # Path to store dali audio files
    # 'dataset_lyrics': '/collection/gsahu/ae/lyrics_generation/data/raw/Lyrics_Data/complete_lyrics_2artists.txt',  # Path to load dali lyrics
    # 'split_spec': '/collection/gsahu/ae/lyrics_generation/data/raw/Lyrics_Data/',  # Path to load dali lyrics
    'dataset_lyrics': '/collection/gsahu/ae/lyrics_generation/data/raw/7-artists-songSegments-new.txt',  # Path to load dali lyrics
    'split_spec': '/collection/gsahu/ae/lyrics_generation/data/raw/10-second/10-second-spec',  # Path to load dali lyrics
    # 'dali_path': '/home/gsahu/code/lyrics_generation/data/raw/DALI_v1.0/',
    # 'dali_audio': '/home/gsahu/code/lyrics_generation/data/raw/dali_audio/',  # Path to store dali audio files
}


def get_dependent_params(model_config):
    if model_config['dec_mode'] == 'beam':
        model_config['beam_size'] = 1
    else:
        model_config['beam_size'] = 1
    m_code = model_config['model_code']
    processed_path = 'data/processed/{}/'.format(m_code)
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    # use vae's vocab when training the scoring function
    if m_code == 'bimodal_scorer':
        model_config['vocab_path'] = 'data/processed/vae/vocab.npy'
    else:
        model_config['vocab_path'] = '{}vocab.npy'.format(processed_path, m_code)

    # VAE hyperparameters
    if m_code in {'vae', 'bvae'}:
        # model-specific hyperparams
        model_config['latent_dim'] = 100
        model_config['anneal_till'] = 1000  # 3000
        model_config['x0'] = 5500  # 4500
        model_config['k'] = 5e-3  # slope of the logistic annealing function (for vae)
        model_config['anneal_type'] = 'tanh'  # for vae {tanh, logistic, linear}
        model_config['sampling_temperature'] = 5e-3  # z_temp to be used during inference
        model_config['scorer_temp'] = 0.09
        if m_code == 'bvae':
            model_config['split_spec'] = '/collection/gsahu/ae/lyrics_generation/data/processed/bvae/spec_array.pkl'
            model_config['image_vae_path'] = '/collection/gsahu/ae/lyrics_generation/data/processed/bvae/vae-mix1e4_100.torch'
    elif m_code == 'dae':
        # there is no latent space in a dae!
        model_config['latent_dim'] = model_config['hidden_dim']

    # word-embeddings are common should be common for all types of models
    model_config['filtered_emb_path'] = 'data/processed/english_w2v_filtered.hd5'
    model_config['classes'] = [0, 1] if 'scorer' in m_code else \
        list(range(len(model_config['label_names'])))
    model_config['save_dir'] += model_config['task'] + '/'


get_dependent_params(model_config)
