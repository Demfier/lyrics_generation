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
import sys


def translate(vocab, pred_tokens):
    """
    Converts model output logits to sentences
    logits -> (max_y_len, bs, vocab_size)

    """
    pred_tokens = pred_tokens.permute(1, 0)

    generated = []
    for token_list in pred_tokens:
        sentence = []
        for t in token_list:
            sentence.append(vocab.index2word[t.item()])
            if t == conf['EOS_TOKEN']:
                break
        generated.append(' '.join(sentence))
    return generated


def get_z(x, x_lens, model):
    encoder_dict = model._encode(x, x_lens)
    mu = encoder_dict['mu']
    log_sigma = encoder_dict['log_sigma']
    return model._reparameterize(mu, log_sigma)


def show_img(data, name):
    skimage.io.imsave(name, data)


def get_specs(n=2):
    with open('data/processed/bimodal_scorer/spec_array.pkl', 'rb') as f:
        spec_array = pickle.load(f)
    # randomly choose n spec_ids
    spec_ids = list(spec_array.keys())
    # print(spec_ids)
    # for k in spec_ids:
    #     show_img(np.array(spec_array[k], dtype=np.uint8), str(k) + '.png')
    for k in spec_ids:
        yield k, torch.tensor([spec_array[k]]).float().view(-1, 224, 224, 3).permute(0, 3, 1, 2).contiguous().to(conf['device'])


def load_scorer_embeddings():
    with h5py.File('data/processed/english_w2v_filtered.hd5', 'r') as f:
        return torch.from_numpy(np.array(f['data'])).float().to(conf['device'])


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
        print('\n### Testing generation from ONLY Scoring function ###:\n')

        i = 0
        for spec_id, spec in get_specs():
            print(i, spec_id)
            tokens = model.decode(spec)
            sentences = translate(vocab, tokens)
            print('{}'.format('\n'.join(sentences)))
            # with open('reports/outputs/random_sampled_{}_with_scorer_{}_temp_{}.txt'.format(conf['pretrained_model'], conf['pretrained_scorer'], conf['scorer_temp']), 'a') as f:
            #     f.write('{}:\n{}\n'.format(spec_id, random_and_with_scorer))
            i += 1


if __name__ == '__main__':
    main()
