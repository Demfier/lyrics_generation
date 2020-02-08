"""
This file contains code for the scoring function (an LSTM Classifier)
"""
import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import vgg16


class RNNScorer(nn.Module):
    """docstring for BiRNNScorer"""
    def __init__(self, config, embedding_wts):
        super(RNNScorer, self).__init__()
        self.config = config
        self.dropout = config['dropout']
        self.embedding_wts = embedding_wts
        self.output_dim = 1
        self.embedding = nn.Embedding.from_pretrained(embedding_wts,
                                                      freeze=False)
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


class BiModalScorer(nn.Module):
    """
    Takes inputs from two different modalities and returns a
    compatibility score
    """
    def __init__(self, config, embedding_wts):
        super(BiModalScorer, self).__init__()
        self.config = config
        self.dropout = config['dropout']
        self.embedding_wts = embedding_wts
        self.output_dim = 2  # we want to return a score
        self.embedding = nn.Embedding.from_pretrained(embedding_wts,
                                                      freeze=False)
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
        # Keep VGG trainable
        for p in self.img_encoder.parameters():
            p.requires_grad = True

        self.img_encoder.classifier = nn.Sequential(
            nn.Linear(self.img_encoder.classifier[0].in_features, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 50))

        self.out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(50 + self.pf*self.config['hidden_dim'], self.output_dim)
            )

    def pool(self, rnn_output):
        pool_func = F.avg_pool1d if self.avg_pool else F.max_pool1d
        return pool_func(rnn_output, rnn_output.size(2)).squeeze(2)

    def fusion(self, music, lyrics):
        return torch.cat((music, lyrics), dim=-1)

    def encoder(self, embedded):
        if self.unit == 'lstm':
            rnn_output, (hidden, _) = self.rnn(embedded)
        else:  # gru/rnn
            rnn_output, hidden = self.rnn(embedded)
        return rnn_output, hidden

    def forward(self, x):
        # music_melspec -> batch_size, num_channels, width, height (bs, 3, 224, 224) when
        # use_melfeats? is False else (batch_size, 1000)
        music_melspec = x['mel_spec']
        # max_sequence_length, batch_size, embedding_dim
        lyrics_embeddings = self.embedding(x['lyrics_seq'])
        if self.config['use_melfeats?']:
            raise NotImplementedError('Using direct image features not supported yet.')
        else:
            # (batch_size, 50)
            music_features = self.img_encoder(music_melspec)

        lyrics_output, lyrics_hidden = self.encoder(lyrics_embeddings)
        lyrics_pool = self.pool(lyrics_output.permute(1, 2, 0))
        # bs, output_dim
        return self.out(self.fusion(music_features, lyrics_pool))


class SpecOnlyClassifier(nn.Module):
    """docstring for SpecOnlyClassifier"""
    def __init__(self, config):
        super(SpecOnlyClassifier, self).__init__()
        self.config = config
        self.dropout = config['dropout']
        self.output_dim = len(config['classes'])  # set of artists

        self.img_encoder = vgg16(pretrained=True)
        # Keep VGG trainable
        for p in self.img_encoder.parameters():
            p.requires_grad = True

        self.img_encoder.classifier = nn.Sequential(
            nn.Linear(self.img_encoder.classifier[0].in_features, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 50))

        self.out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(50, self.output_dim)
            )

    def forward(self, x):
        """
        x is a dict containing just mel_spec
        """
        return self.out(self.img_encoder(x['mel_spec']))


class LyricsOnlyClassifier(BiModalScorer):
    """docstring for GenreClassifier"""
    def __init__(self, config, embedding_wts):
        super(LyricsOnlyClassifier, self).__init__(config, embedding_wts)
        self.output_dim = len(self.config['classes'])
        self.out = nn.Linear(self.config['hidden_dim'], self.output_dim)

        del self.w
        del self.img_encoder

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

    def forward(self, x):
        _, bs = x['lyrics_seq'].shape
        lyrics_embeddings = self.embedding(x['lyrics_seq'])
        lyrics_output, lyrics_hidden = self.encoder(lyrics_embeddings)
        # Do this to sum the bidirectional outputs
        lyrics_output = (lyrics_output[:, :, :self.config['hidden_dim']] +
                         lyrics_output[:, :, self.config['hidden_dim']:])
        lyrics_output = lyrics_output.permute(1, 0, 2).contiguous()

        lyrics_hidden = lyrics_hidden.view(self.config['n_layers'], 2,
                                           bs, self.config['hidden_dim'])[-1]
        # sum (pool) forward and backward hidden states
        lyrics_hidden = (lyrics_hidden[0, :, :self.config['hidden_dim']] +
                         lyrics_hidden[1, :, :self.config['hidden_dim']])

        attended_ouptut = self.attn(lyrics_output, lyrics_hidden)
        return self.out(attended_ouptut)
