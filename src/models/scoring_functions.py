"""
This file contains code for the scoring function (an LSTM Classifier)
"""
import sys
import torch
import pickle
import numpy as np
from torch import nn, optim
import torch.nn.functional as F


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
        self.Wy = nn.Linear(self.pf*self.config['hidden_dim'],
                            self.pf*self.config['hidden_dim'],
                            bias=False)
        self.Wh = nn.Linear(self.pf*self.config['hidden_dim'],
                            self.pf*self.config['hidden_dim'],
                            bias=False)
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

        self.out = nn.Linear(4*self.pf*self.config['hidden_dim'], self.output_dim)
        self.softmax = F.softmax
        self.use_attn = config['use_attn?']

    def attn(self, rnn_output, pool):
        M_left = self.Wy(rnn_output)
        # (2d, 2d)(bts, n, 2d) = (bts, n, 2d)
        M_right = self.Wh(pool).unsqueeze(2)
        # (2d,2d)(bts,2d) = (bts,2d)
        # (bts, 2d, 1)
        M_right = M_right.repeat(1, 1, rnn_output.shape[1])
        # (bts, 2d, n)
        M_right = M_right.permute(0, 2, 1)
        M = torch.add(M_left, M_right)
        M = torch.tanh(M)
        attn_wts = self.w(M)
        # (2d,1)(bts,n,2d) = (bts, n, 1)
        soft_attn_wts = F.softmax(attn_wts, dim=1)  # along n
        soft_attn_wts = soft_attn_wts.permute(0, 2, 1)
        new_encoded = torch.bmm(soft_attn_wts, rnn_output)
        # (bts, 1, n)(bts, n, 2d) = (bts, 1, 2d)
        new_encoded = new_encoded.squeeze(1)
        return new_encoded

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
