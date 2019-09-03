import torch
model_config = {
    'clip': 50,
    'lr': 1e-4,
    'n_layers': 2,
    'unit': 'lstm',
    'dropout': 0.3,
    'n_epochs': 500,
    'MAX_LENGTH': 30,
    'hidden_dim': 256,
    'batch_size': 64,
    'use_attn?': False,
    'first_run?': True,
    'filter_lang': 'english',
    #'filter_genre': {'Pop', 'Rock', 'Alternative', 'Rock indé', 'Metal', 'Pop indé/Folk'},
    'filter_genre': {'Metal'},
    'embedding_dim': 300,
    'max_song_per_genre': 300,
    'bidirectional': True,
    'use_melfeats?': False,  # whether to use already extracted img features or use calculate them on the fly while encoding
    'use_embeddings?': True,
    'pretrained_model': False,
    'save_dir': 'saved_models/',
    'data_dir': 'data/processed/',
    'model_code': 'bimodal_scorer',  # bimodal_scorer/bilstm_scorer
    'vocab_path': 'data/processed/vocab.npy',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # gpu_id ('x' for multiGPU mode)
    'filtered_emb_path': 'data/processed/english_w2v_filtered.hd5',
    'file_name': '/home/d35kumar/Github/lyrics_generation/data/raw/split_info.txt',
    'dali_path': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0',
    'dali_audio': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0/ogg_audio/',  # Path to store dali audio files
    'dali_audio_split': '/home/d35kumar/Github/lyrics_generation/data/raw/DALI_v1.0/ogg_audio_split/', # Path to store dali audio files split by lines
    'spectrograms': '/home/d35kumar/Github/lyrics_generation/data/processed/spectrograms_split/'
    # 'dali_path': '/home/gsahu/code/lyrics_generation/data/raw/DALI_v1.0/',
    # 'dali_audio': '/home/gsahu/code/lyrics_generation/data/raw/dali_audio/',  # Path to store dali audio files
}
