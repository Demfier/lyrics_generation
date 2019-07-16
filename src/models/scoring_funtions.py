"""
This file contains code for the scoring function (an LSTM Classifier)
"""
import sys
import torch
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import model_config as config


import torch
from torch import nn, optim
import torch.nn.functional as F


class RNNScorer(nn.Module):
    """docstring for BiRNNScorer"""
    def __init__(self, config, embedding_wts, n_lables):
        super(BiRNNScorer, self).__init__()
        self.config = config
        self.dropout = config['dropout']
        self.embedding_wts = embedding_wts
        self.output_dim = n_lables
        self.embedding = nn.Embedding.from_pretrained(embedding_wts,
                                                      freeze=False)
        self.embedding_dropout = nn.Dropout(config['dropout'])
        self.bidirectional = config['bidirectional']
        self.unit = config['unit']

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

        self.out = nn.Linear(self.config['hidden_dim'], self.output_dim)
        self.softmax = F.softmax
        self.use_attn = config['use_attn?']

    def attn(self, rnn_output, final_hidden):
        """
        Returns `attended` hidden state given an rnn_output and its final
        hidden state

        attn: torch.Tensor, torch.Tensor -> torch.Tensor
        requires:
            rnn_output.shape => batch_size x max_seq_len x hidden_dim
            final_hidden.shape => batch_size x hidden_dim
        """
        attn_wts = torch.bmm(rnn_output, final_hidden.unsqueeze(2)).squeeze(2)
        soft_attn_wts = F.softmax(attn_wts, dim=1)  # bs x num_t
        # In the next step, rnn_output.shape changes as follows:
        # (bs x num_t x hidden_dim) => (bs x hidden_dim x num_t)
        # Finally, we get new_hidden of shape: (bs x hidden_dim)
        new_hidden = torch.bmm(rnn_output.transpose(1, 2),
                               soft_attn_wts.unsqueeze(2)).squeeze(2)
        return new_hidden

    def forward(self, input_seq_batch, input_lengths):
        """
        TODO: Modify the classifer for late fusion
        """
        max_seq_length, bs = input_seq_batch.size()
        embedded = self.embedding_dropout(self.embedding(input_seq_batch))

        if self.unit == 'lstm':
            rnn_output, (hidden, _) = self.rnn(embedded)
        else:  # gru/rnn
            rnn_output, hidden = self.rnn(embedded)

        if self.use_attn:
            # Do this to sum the bidirectional outputs
            rnn_output = rnn_output.view(2, max_seq_length, bs,
                                         self.config['hidden_dim'])
            rnn_output = (rnn_output[0, :, :self.config['hidden_dim']] +
                          rnn_output[1, :, :self.config['hidden_dim']])
            rnn_output = rnn_output.permute(1, 0, 2)  # bs x num_t x hidden_dim

            final_hidden = hidden.view(self.config['n_layers'], 2,
                                       bs, self.config['hidden_dim'])[-1]
            # sum (pool) forward and backward hidden states
            final_hidden = (final_hidden[0, :, :self.config['hidden_dim']] +
                            final_hidden[1, :, :self.config['hidden_dim']])

            attended_ouptut = self.attn(rnn_output, final_hidden)
            return self.out(attended_ouptut)

        # Here, rnn_output.shape becomes => (bs x hidden_dim)
        rnn_output = (rnn_output[:, :, :self.config['hidden_dim']] +
                      rnn_output[:, :, self.config['hidden_dim']:])
        return self.out(rnn_output)
