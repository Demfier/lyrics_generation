import os
import h5py
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim

from tensorboardX import SummaryWriter

from utils import preprocess, metrics
from models.config import model_config as conf
from models.scoring_functions import RNNScorer, BiModalScorer


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
        }, '{}-{}-{}L-{}{}-{}'.format(conf['save_dir'],
                                      conf['model_code'],
                                      conf['n_layers'],
                                      'bi' if conf['bidirectional'] else '',
                                      conf['unit'], epoch_num))


def main():
    # Initialize tensorboardX writer
    writer = SummaryWriter()

    vocab = load_vocabulary()
    print('Loading train, validation and test pairs.')
    train_pairs, y_train = preprocess.read_pairs(conf, mode='train')
    y_train = y_train.to(conf['device'])
    val_pairs, y_val = preprocess.read_pairs(conf, mode='val')
    y_val = y_train.to(conf['device'])
    test_pairs, y_test = preprocess.read_pairs(conf, mode='test')
    y_test = y_train.to(conf['device'])
    n_train = len(train_pairs)
    train_pairs = train_pairs[: conf['batch_size'] * (
        n_train // conf['batch_size'])]
    print(random.choice(train_pairs))
    n_val = len(val_pairs)
    n_test = len(test_pairs)
    device = conf['device'] if torch.cuda.is_available() else 'cpu'
    embedding_wts = get_embedding_wts(vocab) if conf['use_embeddings?'] else None
    print('Building model.')
    if conf['model_code'] == 'bilstm_scorer':
        model = RNNScorer(conf, embedding_wts, 2)
    elif conf['model_code'] == 'bimodal_scorer':
        model = BiModalScorer(conf, embedding_wts, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=conf['lr'])
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
    print('#Train: {} | #Test: {} | #Val: {} | #Params: {}'.format(
        n_train, n_test, n_val, num_params))

    print('Training started..')
    for e in range(epoch, conf['n_epochs']):
        model.train()
        optimizer.zero_grad()
        epoch_loss = []
        # Train for an epoch
        for iter in tqdm(range(0, n_train, conf['batch_size'])):
            iter_pairs = train_pairs[iter: iter + conf['batch_size']]
            if len(iter_pairs) == 0:  # handle the strange error
                continue
            x_train = preprocess._btmcd(vocab, iter_pairs, conf)
            # forward pass through the model
            predictions = model(x_train)
            # y_train -> (1, bs)
            loss = criterion(predictions, y_train[iter: iter + conf['batch_size']])
            loss.backward()
            predictions = torch.argmax(predictions, dim=1).cpu()
            nn.utils.clip_grad_norm_(model.parameters(), conf['clip'])
            optimizer.step()
            epoch_loss.append(loss.item())
            writer.add_scalar('data/train_loss', np.mean(epoch_loss), iter)

        epoch_loss = []

        # Validate
        with torch.no_grad():
            generated = []
            actual = []
            model.eval()
            for iter in tqdm(range(0, n_val, conf['batch_size'])):
                iter_pairs = val_pairs[iter: iter + conf['batch_size']]
                x_val = preprocess._btmcd(vocab, iter_pairs, conf)

                predictions = model(x_val)
                loss = criterion(predictions,
                                 y_val[iter: iter + predictions.shape[0]])
                generated += list(torch.argmax(predictions, dim=1).cpu())
                actual += list(y_val[iter: iter + predictions.shape[0]].cpu())
                epoch_loss.append(loss.item())

            performance = metrics.evaluate(actual, generated)
            writer.add_scalar('data/val_loss', np.mean(epoch_loss), e)
            writer.add_scalar('metrics/val_accuracy', performance['acc'], e)
            writer.add_scalar('metrics/val_precision', performance['precision'], e)
            writer.add_scalar('metrics/val_recall', performance['recall'], e)
            writer.add_scalar('metrics/val_f1', performance['f1'], e)
            # Save model
            save_snapshot(model, e)

            epoch_loss = []
            generated = []
            actual = []

            # Evaluate on the test set every 5 epochs
            if e > 0 and e % 5 == 0:
                for iter in range(0, n_test, conf['batch_size']):
                    iter_pairs = test_pairs[iter: iter + conf['batch_size']]
                    x_test = preprocess._btmcd(vocab, iter_pairs, conf)

                    predictions = model(x_test)
                    loss = criterion(predictions,
                                     y_test[iter: iter + predictions.shape[0]])
                    generated += list(torch.argmax(predictions, dim=1).cpu())
                    actual += list(y_test[iter: iter + predictions.shape[0]].cpu())

                performance = metrics.evaluate(actual, generated)
                writer.add_scalar('data/test_loss', np.mean(epoch_loss), e)
                writer.add_scalar('metrics/test_accuracy', performance['acc'], e)
                writer.add_scalar('metrics/test_precision', performance['precision'], e)
                writer.add_scalar('metrics/test_recall', performance['recall'], e)
                writer.add_scalar('metrics/test_f1', performance['f1'], e)

    writer.close()


if __name__ == '__main__':
    main()
