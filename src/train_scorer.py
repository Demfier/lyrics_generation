import os
import h5py
import random
import pickle
import numpy as np
from tqdm import tqdm
from pprint import pprint

import torch
from torch import nn, optim

from tensorboardX import SummaryWriter

from utils import preprocess, metrics
from models.config import model_config as conf
from models.scoring_functions import (RNNScorer,
                                      BiModalScorer,
                                      LyricsOnlyClassifier,
                                      SpecOnlyClassifier)


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
        }, '{}{}-{}L-{}{}-{}'.format(conf['save_dir'],
                                     conf['model_code'],
                                     conf['n_layers'],
                                     'bi' if conf['bidirectional'] else '',
                                     conf['unit'], epoch_num))


def main():
    # Initialize tensorboardX writer
    writer = SummaryWriter()

    pprint(conf)
    print('Loading train, validation and test pairs.')
    train_pairs, y_train = preprocess.read_pairs(conf, mode='train')
    y_train = y_train.to(conf['device'])
    val_pairs, y_val = preprocess.read_pairs(conf, mode='val')
    y_val = y_val.to(conf['device'])
    test_pairs, y_test = preprocess.read_pairs(conf, mode='test')
    y_test = y_test.to(conf['device'])
    train_pairs = train_pairs[: conf['batch_size'] * (
        len(train_pairs) // conf['batch_size'])]
    # train_pairs = train_pairs[:100]
    # val_pairs = train_pairs[:500]
    # test_pairs = train_pairs[:500]
    # y_train = y_train[:100]
    # y_val = y_val[:100]
    # y_test = y_test[:100]
    n_train = len(train_pairs)
    n_val = len(val_pairs)
    n_test = len(test_pairs)
    device = conf['device']

    if conf['model_code'] == 'spec_clf':
        vocab = None
    else:
        vocab = load_vocabulary()
        embedding_wts = get_embedding_wts(vocab) if conf['use_embeddings?'] else None

    print('Building model..')
    if conf['model_code'] == 'bilstm_scorer':
        model = RNNScorer(conf, embedding_wts)
    elif conf['model_code'] == 'bimodal_scorer':
        model = BiModalScorer(conf, embedding_wts)
        print('Loading spec_array...')
        with open('data/processed/bimodal_scorer/spec_array.pkl', 'rb') as f:
            spec_array = pickle.load(f)
    elif conf['model_code'] == 'lyrics_clf':  # classifier
        model = LyricsOnlyClassifier(conf, embedding_wts)
        print('Artists: {}'.format(', '.join(sorted(list(conf['label_names'])))))
    elif conf['model_code'] == 'spec_clf':  # artist spec classifier
        model = SpecOnlyClassifier(conf)
        with open('data/processed/spec_clf/spec_array.pkl', 'rb') as f:
            spec_array = pickle.load(f)
        print('Artists: {}'.format(', '.join(sorted(list(conf['label_names'])))))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')  # to train genre classifier
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
    rand_idx = int(random.random() * n_train)
    print(train_pairs[rand_idx], y_train[rand_idx].item())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#Train: {} | #Test: {} | #Val: {} | #Params: {}'.format(
        n_train, n_test, n_val, num_params))

    print('Training started..')
    train_iter, val_iter, test_iter = 0, 0, 0
    for e in range(epoch, conf['n_epochs']):
        model.train()
        optimizer.zero_grad()
        epoch_loss = []
        # Train for an epoch
        for iter in tqdm(range(0, n_train, conf['batch_size'])):
            iter_pairs = train_pairs[iter: iter + conf['batch_size']]
            if len(iter_pairs) == 0:  # handle the strange error
                continue
            if conf['model_code'] in ['bimodal_scorer', 'spec_clf']:
                x_train = preprocess._btmcd(vocab, iter_pairs, conf, spec_array)
            else:
                x_train = preprocess._btmcd(vocab, iter_pairs, conf)
            # forward pass through the model
            predictions = model(x_train)
            # predictions -> (bs,) for scorers and (bs, num_clases) otherwise
            loss = criterion(predictions, y_train[iter: iter + conf['batch_size']])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), conf['clip'])
            optimizer.step()
            epoch_loss.append(loss.item())
            writer.add_scalar('data/train_loss', np.mean(epoch_loss), train_iter)
            train_iter += 1
        # Print average batch loss
        print('Epoch [{}/{}]: Mean Train Loss: {}'.format(e, conf['n_epochs'],
                                                          np.mean(epoch_loss)))
        epoch_loss = []

        # Validate
        with torch.no_grad():
            model.eval()
            generated = np.array([])
            actual = y_val.cpu().numpy()
            for iter in tqdm(range(0, n_val, conf['batch_size'])):
                iter_pairs = val_pairs[iter: iter + conf['batch_size']]
                if conf['model_code'] in ['bimodal_scorer', 'spec_clf']:
                    x_val = preprocess._btmcd(vocab, iter_pairs, conf, spec_array)
                else:
                    x_val = preprocess._btmcd(vocab, iter_pairs, conf)

                predictions = model(x_val)
                preds = torch.argmax(predictions, dim=1).cpu().numpy()
                loss = criterion(predictions, y_train[iter: iter + predictions.shape[0]])
                epoch_loss.append(loss.item())
                generated = np.concatenate((generated, preds))
                epoch_loss.append(loss.item())

            metrics.plot_confusion_matrix(
                actual, generated, normalize=True,
                classes=conf['classes'],
                epoch=e, model_code=conf['model_code'])
            performance = metrics.evaluate(actual, generated)
            print('Mean Validation Loss: {}'.format(np.mean(epoch_loss), '-'*30))
            print('Val A: {acc} | P: {precision} | R: {recall} | F: {f1}\n\n'.format(**performance))
            writer.add_scalar('data/val_loss', np.mean(epoch_loss), val_iter)
            writer.add_scalars('metrics/val_metrics', performance, val_iter)
            val_iter += 1
            # Save model
            save_snapshot(model, e)

            epoch_loss = []

            # Evaluate on the test set every 5 epochs
            if e > 0 and e % 5 == 0:
                generated = np.array([])
                actual = y_test.cpu().numpy()
                for iter in range(0, n_test, conf['batch_size']):
                    iter_pairs = test_pairs[iter: iter + conf['batch_size']]
                    if conf['model_code'] in ['bimodal_scorer', 'spec_clf']:
                        x_test = preprocess._btmcd(vocab, iter_pairs, conf, spec_array)
                    else:
                        x_test = preprocess._btmcd(vocab, iter_pairs, conf)

                    predictions = model(x_test)

                    preds = torch.argmax(predictions, dim=1).cpu().numpy()
                    loss = criterion(predictions, y_train[iter: iter + predictions.shape[0]])
                    epoch_loss.append(loss.item())
                    generated = np.concatenate((generated, preds))

                metrics.plot_confusion_matrix(
                    actual, generated, normalize=True,
                    classes=conf['classes'],
                    epoch=e, model_code=conf['model_code'])
                performance = metrics.evaluate(actual, generated)
                print('Mean Test Loss: {}'.format(np.mean(epoch_loss), '-'*30))
                print('Test A: {acc} | P: {precision} | R: {recall} | F: {f1}\n\n'.format(**performance))
                writer.add_scalar('data/test_loss', np.mean(epoch_loss), test_iter)
                writer.add_scalars('metrics/test_metrics', performance, test_iter)
                test_iter += 1
    writer.close()


if __name__ == '__main__':
    main()
