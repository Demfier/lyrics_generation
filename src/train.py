import os
import h5py
import random
import argparse
import numpy as np

import torch
from torch import nn, optim

from tensorboardX import SummaryWriter

from utils import preprocess, metrics
from models.config import model_config as conf
from models.scoring_functions import RNNScorer


def load_vocabulary():
    if os.path.exists(conf['vocab_path']):

        # Trick for allow_pickle issue in np.load
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        # need to use .item() to access a class object
        vocab = np.load(conf['vocab_path']).item()
        # restore np.load for future normal usage
        np.load = np_load_old
        print('Loaded vocabulary.')
    else:
        # build a single vocab for both the languages
        print('Building vocabulary...')
        vocab = preprocess.build_vocab(conf)
        print('Built vocabulary.')
    return vocab


def get_embedding_wts(vocab):
    if conf['first_run?']:
        print('First run: GENERATING filtered embeddings.')
        embedding_wts = preprocess.generate_word_embeddings(vocab, conf)
    else:
        print('LOADING filtered embeddings.')
        embedding_wts = preprocess.load_word_embeddings(conf)
    return embedding_wts.to(conf['device'])


def save_snapshot(model, epoch_num):
    if not os.path.exists(conf['save_dir']):
            os.mkdirs(conf['save_dir'])
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch_num
        }, '{}{}-{}-{}L-{}{}-{}'.format(conf['save_dir'],
                                        conf['task'],
                                        conf['model_code'],
                                        conf['enc_n_layers'],
                                        'bi' if conf['bidirectional'] else '',
                                        conf['unit'], epoch_num))


def translate(vocab, logits, y, x, inputs, generated, ground_truth):
    """
    Converts model output logits and tokenized batch y into to sentences
    logits -> (max_y_len, bs, vocab_size)
    y -> (max_y_len, bs)

    Effects: Mutates generated, and ground_truth
    """
    inp_tokens = x[1:].permute(1, 0)
    _, pred_tokens = torch.max(logits[1:], dim=2)
    pred_tokens = pred_tokens.permute(1, 0)
    gt_tokens = y[1:].permute(1, 0)

    # Get sentences from token ids
    for token_list in inp_tokens:
        sentence = []
        for t in token_list:
            sentence.append(vocab.index2word[t.item()])
            if t == conf['EOS_TOKEN']:
                break
        inputs.append(sentence)

    for token_list in pred_tokens:
        sentence = []
        for t in token_list:
            sentence.append(vocab.index2word[t.item()])
            if t == conf['EOS_TOKEN']:
                break
        generated.append(sentence)

    for token_list in gt_tokens:
        sentence = []
        for t in token_list:
            sentence.append(vocab.index2word[t.item()])
            if t == conf['EOS_TOKEN']:
                break
        # making it a listof(listof Str) as we need to calculate the bleu score
        ground_truth.append([sentence])
    return inputs, generated, ground_truth


def main():
    # Initialize tensorboardX writer
    writer = SummaryWriter()

    vocab = load_vocabulary()
    print('Loading train, validation and test pairs.')
    train_pairs, val_pairs, test_pairs = preprocess.load_data(conf)
    # train_pairs = train_pairs[:1000]
    # val_pairs = val_pairs[:500]
    # test_pairs = test_pairs[:500]
    n_train = len(train_pairs)
    train_pairs = train_pairs[: conf['batch_size'] * (
        n_train // conf['batch_size'])]
    print(random.choice(train_pairs))
    n_val = len(val_pairs)
    n_test = len(test_pairs)
    device = conf['device'] if torch.cuda.is_available() else 'cpu'
    embedding_wts = get_embedding_wts(vocab) if conf['use_embeddings?'] else None
    print('Building model.')
    print(embedding_wts.shape)
    model = RNNScorer(conf, embedding_wts, 2)
    model = model.to(device)
    if conf['pretrained_model']:
        print('Restoring {}...'.format(conf['pretrained_model']))
        checkpoint = torch.load('{}{}'.format(conf['save_dir'],
                                              conf['pretrained_model']),
                                map_location=device)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total {} trainable parameters'.format(num_params))

    print('Training started..')
    for e in range(epoch, conf['n_epochs']):
        epoch_loss = []
        # Train for an epoch
        for iter in range(0, n_train, conf['batch_size']):
            iter_pairs = train_pairs[iter: iter + conf['batch_size']]
            if len(iter_pairs) == 0:  # handle the strange error
                continue
            x_train, x_lens, y_train = preprocess._btmcd(vocab,
                                                         iter_pairs,
                                                         conf['device'])
            # forward pass through the model
            outputs, loss = model(x_train, x_lens, y_train)
            # y_train -> (max_y_len, bs)
            epoch_loss.append(loss.item())
        # Print average batch loss
        print('Epoch [{}/{}]: Mean Train Loss: {}'.format(e, conf['n_epochs'],
                                                          np.mean(epoch_loss)))
        epoch_loss = []

        # Validate
        with torch.no_grad():
            inputs = []
            generated = []
            actual = []
            for iter in range(0, n_val, conf['batch_size']):
                iter_pairs = val_pairs[iter: iter + conf['batch_size']]
                x_val, x_lens, y_val = preprocess._btmcd(vocab,
                                                         iter_pairs,
                                                         conf['device'])

                outputs, loss = model(x_val, x_lens)
                epoch_loss.append(loss.item())
                # Get sentences from logits and token ids
                translate(vocab, outputs, y_val, x_val,
                          inputs, generated, actual)

            print('Mean Validation Loss: {}\n{}\nSamples:\n'.format(
                np.mean(epoch_loss), '-'*30))
            # Sample some val sentences randomly
            for sample_id in random.sample(list(range(len(val_pairs))), 3):
                print('I: {}\nG: {}\nA: {}\n'.format(
                    ' '.join(inputs[sample_id]),
                    ' '.join(generated[sample_id]),
                    ' '.join(actual[sample_id][0]))
                )

            # Get BLEU1-BLEU4 scores
            bleu = metrics.calculate_bleu_scores(generated, actual)
            print('Validation BLEU (1-4) scores:')
            print('{bleu1} | {bleu2} | {bleu3} | {bleu4}'.format(**bleu))
            print('{}{}\n'.format('<'*15, '>'*15))

            # Save model
            save_snapshot(model, e)

            epoch_loss = []
            inputs = []
            generated = []
            actual = []

            # Get test BLEU scores every 5 epochs
            if e > 0 and e % 5 == 0:
                for iter in range(0, n_test, conf['batch_size']):
                    iter_pairs = test_pairs[iter: iter + conf['batch_size']]
                    x_test, x_lens, y_test = preprocess._btmcd(vocab,
                                                               iter_pairs,
                                                               conf['device'])
                    outputs, loss = model(x_test, x_lens)
                    epoch_loss.append(loss.item())
                    # Get sentences
                    translate(vocab, outputs, y_test, x_test,
                              inputs, generated, actual)

                print('Mean Test Loss: {}\n{}\nSamples:\n'.format(
                    np.mean(epoch_loss), '-'*30))
                # Sample some test sentences randomly
                for sample_id in random.sample(list(range(len(test_pairs))), 3):
                    print('I: {}\nG: {}\nA: {}\n'.format(
                        ' '.join(inputs[sample_id]),
                        ' '.join(generated[sample_id]),
                        ' '.join(actual[sample_id][0]))
                    )
                bleu = metrics.calculate_bleu_scores(generated, actual)
                print('Test BLEU (1-4) scores:\n{}'.format('-'*30))
                print('{bleu1} | {bleu2} | {bleu3} | {bleu4}'.format(**bleu))
                print('{}{}\n'.format('<'*15, '>'*15))


if __name__ == '__main__':
    main()
