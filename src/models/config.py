model_config = {
    'clip': 50,
    'lr': 0.001,
    'n_layers': 3,
    'unit': 'lstm',
    'dropout': 0.3,
    'n_epochs': 500,
    'MAX_LENGTH': 30,
    'batch_size': 10,
    'hidden_dim': 256,
    'use_attn?': True,
    'embedding_dim': 300,
    'model_code': 'bilstm_scorer',
    'vocab_path': 'data/processed/vocab.npy',
    'filtered_emb_path': 'data/processed/english_w2v_filtered.hd5',
    'dali_path': '/home/gsahu/code/lyrics_generation/data/raw/DALI_v1.0/',
    'dali_audio': '/home/gsahu/code/lyrics_generation/data/raw/dali_audio'  # Path to store dali audio files
}
