import torch
import random
import pickle
from utils import preprocess, metrics
from models.config import model_config as conf
from train_model import load_vocabulary, get_embedding_wts
from models import dae, vae
import h5py
import skimage
import numpy as np


def translate(vocab, logits):
    """
    Converts model output logits to sentences
    logits -> (max_y_len, bs, vocab_size)

    """
    _, pred_tokens = torch.max(logits[1:], dim=2)
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


def get_specs(n):
    with open('data/processed/bimodal_scorer/spec_array.pkl', 'rb') as f:
        spec_array = pickle.load(f)
    # randomly choose n spec_ids
    spec_ids = np.random.choice(list(spec_array.keys()), size=n)
    print(spec_ids)
    # for k in spec_ids:
    #     show_img(np.array(spec_array[k], dtype=np.uint8), str(k) + '.png')
    specs = [[spec_array[k]] for k in spec_ids]
    return torch.tensor(specs).float().view(-1, 224, 224, 3).permute(0, 3, 1, 2).contiguous().to(conf['device'])


def load_scorer_embeddings():
    with h5py.File('data/processed/vae/english_w2v_filtered.hd5', 'r') as f:
        return torch.from_numpy(np.array(f['data'])).float().to(conf['device'])


def main():
    vocab = load_vocabulary()
    embedding_wts = get_embedding_wts(vocab)
    device = conf['device']
    if conf['model_code'] == 'dae':
        model = dae.AutoEncoder(conf, vocab, embedding_wts)
    elif conf['model_code'] == 'vae':
        model = vae.VariationalAutoEncoder(conf, vocab, embedding_wts)
    elif 'ved' in conf['model_code']:
        model = vae.BimodalVED(conf, vocab, embedding_wts)
    else:
        raise ValueError('Invalid model code: {}'.format(conf['model_code']))

    model.load_state_dict(
        torch.load(
            '{}{}'.format(conf['save_dir'], conf['pretrained_model']),
            map_location=device)['model'])

    model = model.to(device)

    with torch.no_grad():
        model.eval()
        # print('\n### Random Sampling ###:\n')
        # random_sampled = translate(vocab, model._random_sample(9))
        # for s in random_sampled:
        #     print(s)
        # with open('random_sampled_sf.txt', 'w') as f:
        #     f.write('\n'.join(random_sampled))

        # _, val_pairs, _ = preprocess.prepare_data(conf)
        # s1 = random.choice(val_pairs)[0]
        # s2 = random.choice(val_pairs)[0]

        # x1, x1_lens, _ = preprocess._btmcd(vocab, [(s1, s1)], conf)
        # x2, x2_lens, _ = preprocess._btmcd(vocab, [(s2, s2)], conf)

        # print('\n### Linear Interpolation ###:\n')
        # print(s1)
        # interpolated = translate(vocab, model._interpolate(get_z(x1, x1_lens, model),
        #                                                    get_z(x2, x2_lens, model), 50))
        # for s_id in range(0, len(interpolated), conf['beam_size']):
        #     sentences = []
        #     for i in range(conf['beam_size']):
        #         sentences.append(interpolated[s_id+i])
        #     print(sentences)
        # print(s2)

        print('\n### Testing VAE+Scoring function ###:\n')
        n = 1
        scorer_emb_wts = load_scorer_embeddings()
        specs = get_specs(n)
        random_sampled = translate(vocab, model._random_sample(n, specs,
                                                               scorer_emb_wts))
        for s in random_sampled:
            print(s)


if __name__ == '__main__':
    main()
