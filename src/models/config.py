import torch
model_config = {
    'clip': 50,
    'lr': 1e-3,
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
    'filter_genre': {'Pop'},
    'embedding_dim': 300,
    'bidirectional': True,
    'use_melfeats?': False,  # whether to use already extracted img features or use calculate them on the fly while encoding
    'use_embeddings?': True,
    'pretrained_model': False,
    'save_dir': 'saved_models/',
    'data_dir': 'data/processed/',
    'model_code': 'bimodal_scorer',
    'vocab_path': 'data/processed/vocab.npy',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # gpu_id ('x' for multiGPU mode)
    'filtered_emb_path': 'data/processed/english_w2v_filtered.hd5',
    'dali_path': '/collection/gsahu/ae/lyrics_generation/data/raw/DALI_v1.0/',
    'dali_audio': '/collection/gsahu/ae/lyrics_generation/data/raw/ogg_audio/',  # Path to store dali audio files
    # 'dali_path': '/home/gsahu/code/lyrics_generation/data/raw/DALI_v1.0/',
    # 'dali_audio': '/home/gsahu/code/lyrics_generation/data/raw/dali_audio/',  # Path to store dali audio files
}
