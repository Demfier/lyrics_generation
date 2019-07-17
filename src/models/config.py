import torch
model_config = {
    'clip': 50,
    'lr': 0.001,
    'n_layers': 2,
    'unit': 'lstm',
    'dropout': 0.3,
    'n_epochs': 500,
    'MAX_LENGTH': 30,
    'batch_size': 1000,
    'hidden_dim': 256,
    'use_attn?': True,
    'embedding_dim': 300,
    'bidirectional': True,
    'model_code': 'bilstm_scorer',
    'vocab_path': 'data/processed/vocab.npy',
    'filtered_emb_path': 'data/processed/english_w2v_filtered.hd5',
    'dali_path': '/home/gsahu/code/lyrics_generation/data/raw/DALI_v1.0/',
    'dali_audio': '/home/gsahu/code/lyrics_generation/data/raw/dali_audio/',  # Path to store dali audio files
    'data_dir': 'data/processed/',
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',  # gpu_id ('x' for multiGPU mode)
    'use_embeddings?': True,
    'first_run?': True,
    'bidirectional': True,
    'pretrained_model': False,
    'save_dir': 'saved_models/'
}
