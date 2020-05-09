import torch
import random
import pickle
from utils import preprocess, metrics
from models.config import model_config as conf
from train_model import load_vocabulary, get_embedding_wts
from models import scoring_functions
import h5py
import skimage
import numpy as np


def main():
    vocab = load_vocabulary()
    embedding_wts = get_embedding_wts(vocab)
    device = conf['device']
    if conf['model_code'] == 'bimodal_scorer':
        model = scoring_functions.BiModalScorer(conf, embedding_wts)
    else:
        raise ValueError('Invalid model code: {}'.format(conf['model_code']))
    model.load_state_dict(
        torch.load(
            '{}{}'.format(conf['save_dir'], 'bimodal_scorer-1L-bilstm-4'),
            map_location=device)['model'])

    model = model.to(device)

    with torch.no_grad():
        model.eval()
        spec1 = pickle.load(open('NineInchNails_1000000_23.png.pkl', 'rb'))
        spec2 = pickle.load(open('DepecheMode_never-let-me-down-again_18.png.pkl', 'rb'))
        print((spec1 == spec2).all())
        print((spec2 == spec2).all())
        print(model.img_encoder(torch.tensor(spec1).to(device)))
        print(model.img_encoder(torch.tensor(spec2).to(device)))


if __name__ == '__main__':
    main()
