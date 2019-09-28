import torch
model_config = {
    # hyperparameters
    'lr': 1e-3,
    'dropout': 0.3,
    'patience': 3,  # number of epochs to wait before decreasing lr
    'min_lr': 1e-7,  # minimum allowable value of lr
    'task': 'rec',  # mt/dialog/rec/dialog-rec
    # model-specific hyperparams
    'anneal_till': 3500,  # for vae
    'x0': 6000,  # for vae
    'k': 5e-3,  # slope of the annealing function (for vae)
    'anneal_type': 'logistic',  # for vae {tanh, logistic, linear}
    'sampling_temperature': 5e-3,  # for vae

    'clip': 50.0,  # values above which to clip the gradients
    'tf_ratio': 1.0,  # teacher forcing ratio

    'unit': 'lstm',
    'n_epochs': 100,
    'batch_size': 100,
    'enc_n_layers': 1,
    'dec_n_layers': 1,
    'disc_n_layers': 2,
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
    'device': 'cpu' if torch.cuda.is_available() else 'cpu',  # gpu_id ('x' for multiGPU mode)
    'wemb_type': 'w2v',  # type of word embedding to use: w2v/fasttext
    'lang_pair': 'en-en',  # src-target language pair
    'use_scheduler': True,  # half lr every 3 non-improving batches
    'use_embeddings?': True,  # use word embeddings or not
    'first_run?': False,  # True for the very first run
    'min_freq': 2,  # min frequency for the word to be a part of the vocab
    'n_layers': 2,
    'filter_lang': 'english',
    'filter_genre': {'Pop'},
    'embedding_dim': 300,
    'bidirectional': True,
    'use_melfeats?': False,  # whether to use already extracted img features or use calculate them on the fly while encoding
    'use_embeddings?': True,
    'pretrained_model': 'rec/vae-1L-bilstm-40',
    'save_dir': 'saved_models/',
    'data_dir': 'data/processed/',
    'model_code': 'vae',  # bimodal_scorer/bilstm_scorer/dae/vae
    'vocab_path': 'data/processed/vocab.npy',
    'filtered_emb_path': 'data/processed/english_w2v_filtered.hd5',
    'dali_path': '/collection/gsahu/ae/lyrics_generation/data/raw/DALI_v1.0/',
    'dali_audio': '/collection/gsahu/ae/lyrics_generation/data/raw/ogg_audio/',  # Path to store dali audio files
    'dali_lyrics': '/collection/gsahu/ae/lyrics_generation/data/raw/lyrics.txt',  # Path to load dali lyrics
    # 'dali_path': '/home/gsahu/code/lyrics_generation/data/raw/DALI_v1.0/',
    # 'dali_audio': '/home/gsahu/code/lyrics_generation/data/raw/dali_audio/',  # Path to store dali audio files
}
