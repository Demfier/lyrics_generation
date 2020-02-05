import os
import torch
model_config = {
    # hyperparameters
    'lr': 1e-3,
    'dropout': 0.3,
    'patience': 3,  # number of epochs to wait before decreasing lr
    'min_lr': 1e-7,  # minimum allowable value of lr
    'task': 'rec',  # mt/dialog/rec/dialog-rec
    'model_code': 'vae',  # bimodal_scorer/bilstm_scorer/dae/vae/clf

    # model-specific hyperparams
    'anneal_till': 350,  # for vae
    'x0': 5500,  # for vae
    'k': 5e-3,  # slope of the logistic annealing function (for vae)
    'anneal_type': 'tanh',  # for vae {tanh, logistic, linear}
    'sampling_temperature': 5e-3,  # z_temp to be used during inference
    'scorer_temp': 0.4,

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

    'latent_dim': 256,
    'hidden_dim': 256,
    'embedding_dim': 300,

    # vocab-related params
    'PAD_TOKEN': 0,
    'SOS_TOKEN': 1,
    'EOS_TOKEN': 2,
    'UNK_TOKEN': 3,
    'MAX_LENGTH': 30,  # Max length of a sentence

    # run-time conf
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # gpu_id ('x' for multiGPU mode)
    'wemb_type': 'w2v',  # type of word embedding to use: w2v/fasttext
    'lang_pair': 'en-en',  # src-target language pair
    'use_scheduler': True,  # half lr every 3 non-improving batches
    'use_embeddings?': True,  # use word embeddings or not
    'first_run?': False,  # True for the very first run
    'min_freq': 2,  # min frequency for the word to be a part of the vocab
    'n_layers': 1,

    # DALI-specific
    'filter_lang': 'english',
    'filter_genre': {'Pop', 'Rock', 'Alternative', 'Dance',
                     'Metal', 'Country', 'Electro', 'R&B'},
    'max_songs': 200,  # maximum songs to consider per genre

    'embedding_dim': 300,
    'bidirectional': True,
    'use_melfeats?': False,  # whether to use already extracted img features or use calculate them on the fly while encoding
    'use_embeddings?': True,
    'generate_spectrograms': False,
    'pretrained_model': False,  # {'vae-1L-bilstm-11', False},
    'pretrained_scorer': False,  # {'bimodal_scorer-1L-bilstm-0', False}
    # 'pretrained_model': 'vae-1L-bilstm-68',  # {'vae-1L-bilstm-11', False},
    # 'pretrained_scorer': 'bimodal_scorer-1L-bilstm-0',  # {'bimodal_scorer-1L-bilstm-0', False}
    'save_dir': 'saved_models/',
    'data_dir': 'data/processed/',
    # 'file_name': '/home/d35kumar/Github/lyrics_generation/data/raw/split_info.txt',
    # 'dali_path': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0',
    # 'dali_audio': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0/ogg_audio/',  # Path to store dali audio files
    # 'dali_audio_split': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0/ogg_audio_split/', # Path to store dali audio files split by lines
    # 'spectrograms': '/home/d35kumar/Github/lyrics_generation/data/processed/spectrograms_split/'
    'dataset_path': '/collection/gsahu/ae/lyrics_generation/data/raw/DALI_v1.0/',
    'dataset_audio': '/collection/gsahu/ae/lyrics_generation/data/raw/ogg_audio/',  # Path to store dali audio files
    'dataset_lyrics': '/collection/gsahu/ae/lyrics_generation/data/raw/final_lyrics.txt',  # Path to load dali lyrics
    'split_spec': '/collection/gsahu/ae/lyrics_generation/data/processed/spectrograms_split/',  # Path to load dali lyrics
    # 'dali_path': '/home/gsahu/code/lyrics_generation/data/raw/DALI_v1.0/',
    # 'dali_audio': '/home/gsahu/code/lyrics_generation/data/raw/dali_audio/',  # Path to store dali audio files
}


def get_dependent_params(model_config):
    if model_config['dec_mode'] == 'beam':
        model_config['beam_size'] = 3
    else:
        model_config['beam_size'] = 1
    m_code = model_config['model_code']
    processed_path = 'data/processed/{}/'.format(m_code)
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    model_config['vocab_path'] = '{}vocab.npy'.format(processed_path, m_code)
    model_config['filtered_emb_path'] = '{}english_w2v_filtered.hd5'.format(processed_path, m_code)
    model_config['classes'] = [-1, 1] if 'scorer' in m_code else \
        list(range(len(model_config['filter_genre'])))
    model_config['save_dir'] += model_config['task'] + '/'


get_dependent_params(model_config)
