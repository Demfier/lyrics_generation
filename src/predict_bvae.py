import torch
import random
import pickle
from utils import preprocess, metrics
from models.config import model_config as conf
from train_model import load_vocabulary, get_embedding_wts
from models import dae, vae, bvae
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


def sample_spec_ids():
    """
    Returns already sampled latent codes from the image VAE
    and some random spec ids
    """
    with open('data/processed/instrumental/spec_array.pkl', 'rb') as f:
        spec_array = pickle.load(f)
    # randomly choose a spec_id
    spec_ids = np.random.choice(list(spec_array.keys()), size=10, replace=False)
    return spec_array, spec_ids


def load_scorer_embeddings():
    with h5py.File('data/processed/english_w2v_filtered.hd5', 'r') as f:
        return torch.from_numpy(np.array(f['data'])).float().to(conf['device'])


def load_img_vae():
    img_vae = image_vae.VAE()
    img_vae.load_state_dict(
        torch.load('./saved_models/vae4_100.torch', map_location=device))
    return img_vae


def main():
    vocab = load_vocabulary()
    embedding_wts = get_embedding_wts(vocab)
    device = conf['device']
    model = bvae.BimodalVariationalAutoEncoder(conf, vocab, embedding_wts)
    model.load_state_dict(
        torch.load(
            '{}{}'.format(conf['save_dir'], conf['pretrained_model']),
            map_location=device)['model'])

    model = model.to(device)

    with torch.no_grad():
        model.eval()
        print('\n### Random Sampling ###:\n')
        spec_array, spec_ids = sample_spec_ids()
        # spec_ids for songs given by Olga
        # spec_ids = ['NineInchNails_this-isnt-the-place_0',
        #             'NineInchNails_gave-up_0',
        #             'NineInchNails_shit-mirror_0',
        #             'NineInchNails_lights-in-the-sky_0',
        #             'NineInchNails_every-day-is-exactly-the-same_0',
        #             'NineInchNails_the-background-world_0',
        #             'NineInchNails_were-in-this-together_0',
        #             'NineInchNails_the-great-below_0',
        #             'NineInchNails_no-you-dont_0',
        #             'NineInchNails_im-looking-forward-to-joining-you-finally_0',
        #             'NineInchNails_hurt_0',
        #             'DepecheMode_wrong_0',
        #             'DepecheMode_sister-of-night_0',
        #             'DepecheMode_walking-in-my-shoes_0',
        #             'DepecheMode_never-let-me-down-again_0',
        #             'DepecheMode_its-no-good_0']
        spec_ids = ['NineInchNails_various-methods-of-escape_8']
        # get spec embedding
        for k in spec_ids:
            print(k)
            print('----------')
            x = np.array(spec_array[k])
            x = x/255.0
            x = x.astype('float32')
            x = x.transpose(2, 0, 1)
            x = torch.from_numpy(x).to(conf['device'])
            x = x.unsqueeze(0)
            img_z, _, _ = model.image_vae.encode(x)
            for i in range(100):
                print(translate(vocab, model._random_sample(1, img_z)[1]))
            print('\n\n')


if __name__ == '__main__':
    main()
