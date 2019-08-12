"""
This file contains code for the scoring function (an LSTM Classifier)
"""
import sys
import torch
import pickle
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models import vgg16


class RNNScorer(nn.Module):
    """docstring for BiRNNScorer"""
    def __init__(self, config, embedding_wts, n_lables):
        super(RNNScorer, self).__init__()
        self.config = config
        self.dropout = config['dropout']
        self.embedding_wts = embedding_wts
        self.output_dim = n_lables
        self.embedding = nn.Embedding.from_pretrained(embedding_wts,
                                                      freeze=False)
        self.embedding_dropout = nn.Dropout(config['dropout'])
        self.bidirectional = config['bidirectional']
        self.unit = config['unit']
        self.pf = (2 if self.bidirectional else 1)
        self.avg_pool = True

        if self.unit == 'lstm':
            self.rnn = nn.LSTM(self.config['embedding_dim'],
                               self.config['hidden_dim'],
                               num_layers=self.config['n_layers'],
                               dropout=self.dropout,
                               bidirectional=self.bidirectional)
        elif self.unit == 'gru':
            self.rnn = nn.GRU(self.config['embedding_dim'],
                              self.config['hidden_dim'],
                              num_layers=self.config['n_layers'],
                              dropout=self.dropout,
                              bidirectional=self.bidirectional)
        else:  # basic rnn
            self.rnn = nn.RNN(self.config['embedding_dim'],
                              self.config['hidden_dim'],
                              num_layers=self.config['n_layers'],
                              dropout=self.dropout,
                              bidirectional=self.bidirectional)

        self.out = nn.Linear(4*self.pf*self.config['hidden_dim'], self.output_dim)
        self.softmax = F.softmax
        self.use_attn = config['use_attn?']

    def pool(self, rnn_output):

        if self.avg_pool:
            rnn_output = F.avg_pool1d(rnn_output, rnn_output.size(2))
        else:
            rnn_output = F.max_pool1d(rnn_output, rnn_output.size(2))

        return rnn_output.squeeze(2)

    def encoder(self, embedded):
        if self.unit == 'lstm':
            rnn_output, (hidden, _) = self.rnn(embedded)
        else:  # gru/rnn
            rnn_output, hidden = self.rnn(embedded)

        return rnn_output, hidden

    def fusion(self, p, h):
        p_h_dot = p*h
        p_h_diff = torch.abs(p-h)
        concat = torch.cat((p, p_h_dot), dim=1)
        concat = torch.cat((concat, p_h_diff), dim=1)
        concat = torch.cat((concat, h), dim=1)
        return concat

    def forward(self, x_train):

        # model taken from the paper https://arxiv.org/pdf/1605.09090.pdf
        music_embeddings = self.embedding(x_train['music_seq'])
        lyrics_embeddings = self.embedding(x_train['lyrics_seq'])
        # batch_size, max_sequence_length, embedding_length
        music_output, music_hidden = self.encoder(music_embeddings)
        lyrics_output, lyrics_hidden = self.encoder(lyrics_embeddings)

        music_pool = self.pool(music_output.permute(1, 2, 0))
        lyrics_pool = self.pool(lyrics_output.permute(1, 2, 0))

        music = self.attn(music_output.permute(1, 0, 2), music_pool)
        lyrics = self.attn(lyrics_output.permute(1, 0, 2), lyrics_pool)
        fused = self.fusion(music, lyrics)

        return self.out(fused)


class BiModalScorer(RNNScorer):
    """
    Takes inputs from two different modalities and returns a
    compatibility score
    """
    def __init__(self, config, embedding_wts, n_lables):
        super(BiModalScorer, self).__init__()
        self.config = config
        self.dropout = config['dropout']
        self.embedding_wts = embedding_wts
        self.output_dim = n_lables
        self.embedding = nn.Embedding.from_pretrained(embedding_wts,
                                                      freeze=False)
        self.embedding_dropout = nn.Dropout(config['dropout'])
        self.bidirectional = config['bidirectional']
        self.unit = config['unit']
        self.pf = (2 if self.bidirectional else 1)
        self.w = nn.Linear(self.pf*self.config['hidden_dim'], 1, bias=False)
        self.avg_pool = True

        if self.unit == 'lstm':
            self.rnn = nn.LSTM(self.config['embedding_dim'],
                               self.config['hidden_dim'],
                               num_layers=self.config['n_layers'],
                               dropout=self.dropout,
                               bidirectional=self.bidirectional)
        elif self.unit == 'gru':
            self.rnn = nn.GRU(self.config['embedding_dim'],
                              self.config['hidden_dim'],
                              num_layers=self.config['n_layers'],
                              dropout=self.dropout,
                              bidirectional=self.bidirectional)
        else:  # basic rnn
            self.rnn = nn.RNN(self.config['embedding_dim'],
                              self.config['hidden_dim'],
                              num_layers=self.config['n_layers'],
                              dropout=self.dropout,
                              bidirectional=self.bidirectional)

        self.img_encoder = vgg16(pretrained=True)
        # Keep the last layer trainable
        for p in self.img_encoder.classifier.parameters():
            p.requires_grad = True

        # We will concatenate img and lyrics features, so input size becomes
        # 1000 (from vgg16) + 2*hidden_dim (from rnn)
        self.final_in_features = \
            self.img_encoder.classifier[6].out_features + \
            self.pf*self.config['hidden_dim']

        self.out = nn.Sequential(
            nn.Linear(self.final_in_features, self.final_in_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_in_features // 2, self.output_dim))

    def pool(self, rnn_output):

        if self.avg_pool:
            rnn_output = F.avg_pool1d(rnn_output, rnn_output.size(2))
        else:
            rnn_output = F.max_pool1d(rnn_output, rnn_output.size(2))

        return rnn_output.squeeze(2)

    def fusion(self, music, lyrics):
        return torch.cat((music, lyrics), dim=-1)

    def forward(self, x_train):

        # batch_size, num_channels, width, height (bs, 3, 256, 256) when
        # use_melfeats? in False else (batch_size, 1000)
        music_melspec = self.embedding(x_train['mel_spec'])
        # batch_size, max_sequence_length, embedding_dim
        lyrics_embeddings = self.embedding(x_train['lyrics_seq'])
        if self.config['use_melfeats?']:
            # batch_size, 1000
            music_features = self.img_encoder(music_spectrogram)
        else:
            music_features = music_melspec

        lyrics_output, lyrics_hidden = self.encoder(lyrics_embeddings)
        lyrics_pool = self.pool(lyrics_output.permute(1, 2, 0))
        return self.out(self.fusion(music, lyrics))
