import torch
import random
from torch import nn, optim
from models import attn_module


class AutoEncoder(nn.Module):
    """AutoEncoder model"""
    def __init__(self, config, vocab, embedding_wts):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding_wts = embedding_wts
        self.build_model()

    def build_model(self):
        self.unit = self.config['unit']
        self.device = self.config['device']
        self.sos_idx = self.config['SOS_TOKEN']
        self.pad_idx = self.config['PAD_TOKEN']
        self.hidden_dim = self.config['hidden_dim']
        self.latent_dim = self.config['latent_dim']
        self.embedding_dim = self.config['embedding_dim']
        self.bidirectional = self.config['bidirectional']
        self.enc_dropout = nn.Dropout(self.config['dropout'])
        self.dec_dropout = nn.Dropout(self.config['dropout'])

        self.embedding = nn.Embedding.from_pretrained(self.embedding_wts) \
            if self.config['use_embeddings?'] else \
            nn.Embedding(self.vocab.size, self.embedding_dim)

        if self.unit == 'lstm':
            self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim,
                                   self.config['enc_n_layers'],
                                   bidirectional=self.bidirectional)

            self.decoder = nn.LSTM(self.embedding_dim, self.hidden_dim,
                                   self.config['dec_n_layers'])
        elif self.unit == 'gru':
            self.encoder = nn.GRU(self.embedding_dim, self.hidden_dim,
                                  self.config['enc_n_layers'],
                                  bidirectional=self.bidirectional)

            self.decoder = nn.GRU(self.embedding_dim, self.hidden_dim,
                                  self.config['dec_n_layers'])
        else:
            self.encoder = nn.RNN(self.embedding_dim, self.hidden_dim,
                                  self.config['enc_n_layers'],
                                  bidirectional=self.bidirectional)

            self.decoder = nn.RNN(self.embedding_dim, self.hidden_dim,
                                  self.config['dec_n_layers'])

        if self.config['attn_model']:
            self.attn = attn_module.Attn(self.config['attn_model'],
                                         self.hidden_dim)

        # All the projection layers
        self.pf = (2 if self.bidirectional else 1)  # project factor

        self.output2vocab = nn.Linear(self.hidden_dim + self.embedding_dim,
                                      self.vocab.size)

        self.optimizer = optim.Adam(self.parameters(), self.config['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min',
                                                              factor=0.1,
                                                              patience=2,
                                                              min_lr=self.config['min_lr'])
        # Reconstruction loss
        self.rec_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def _encode(self, x, x_lens):
        max_x_len, bs = x.shape
        # convert input sequence to embeddings
        embedded = self.enc_dropout(self.embedding(x))
        # embedded => (t x bs x embedding_dim)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, x_lens,
                                                   enforce_sorted=False)
        # Forward pass through the encoder
        # outputs => (max_seq_len, bs, hidden_dim * self.pf)
        # h_n (& c_n) => (#layers * self.pf, bs, hidden_dim)
        outputs, hidden = self.encoder(packed)
        if self.unit == 'lstm':
            hidden = hidden[1]  # ignore h_n

        # outputs => (max_seq_len x bs x hidden_dim * self.pf)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Construct z from last time_step output
        if self.bidirectional:
            outputs = outputs.view(max_x_len, bs, self.pf, self.hidden_dim)
            # concatenate forward and backward encoder outputs
            outputs = outputs[:, :, 0, :] + outputs[:, :, 1, :]

            # sum forward and backward hidden states
            hidden = hidden.view(self.config['enc_n_layers'],
                                 self.pf, bs, self.hidden_dim)
            hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        # outputs => (max_seq_len x bs x hidden_dim * self.pf)
        # hidden => (#enc_layers, bs, hidden * self.pf)
        return {'encoder_outputs': outputs, 'z': hidden}

    def _create_mask(self, tensor):
        return torch.ne(tensor, self.pad_idx)

    def _decode(self, z, y, infer, encoder_outputs=None, mask=None):
        """
        z -> (#enc_layers, batch_size x latent_dim)
        y -> (max_y_len, batch_size)

        encoder_outputs and mask will be used for attention mechanism
        TODO: Handle bidirectional decoders
        """
        max_y_len, bs = y.shape
        vocab_size = self.vocab.size

        # tensor to store decoder outputs
        decoder_outputs = torch.zeros(max_y_len, bs,
                                      vocab_size).to(self.device)

        # Reconstruct the hidden vector from z
        # if #dec_layers > #enc_layers, use currently obtained hidden
        #  to represent the last #enc_layers layers of decoder hidden
        if self.config['dec_n_layers'] > self.config['enc_n_layers']:
            dec_hidden = torch.zeros(self.config['dec_n_layers'], bs,
                                     self.latent_dim).to(self.device)
            dec_hidden[-self.config['enc_n_layers']:, :, :] = z
        else:
            dec_hidden = z[-self.config['dec_n_layers']:, :, :]

        if self.unit == 'lstm':
            # consider h_n = 0 for lstm
            h_n = torch.zeros(self.config['dec_n_layers'], bs,
                              self.latent_dim).to(self.device)
            dec_hidden = (h_n, dec_hidden)

        # initial decoder input is <sos> token
        output = y[0, :]

        # Start decoding process
        for t in range(1, max_y_len):
            # output -> (bs, vocab_size)
            # dec_hidden -> (bs, hidden * self.pf)
            output, dec_hidden = self._decode_token(output, dec_hidden, mask)
            decoder_outputs[t] = output
            do_tf = random.random() < self.config['tf_ratio']
            # always do greedy search for inference mode (y = None)
            if infer or (not do_tf):
                # output.max(1) -> (scores, tokens)
                # doing a max along `dim=1` returns logit scores and
                # token index for the most probable (max valued) token
                # scores (& tokens) -> (bs)
                output = output.max(1)[1]  # top1
            elif do_tf:
                output = y[t]
        return decoder_outputs

    def _decode_token(self, input, hidden, mask):
        """
        input -> (bs)
        hidden -> (#dec_layers * self.pf x bs x hidden_dim)
                  (c_n is zero for lstm decoder)
        mask -> (bs x max_x_len)
            mask is used for attention
        """
        input = input.unsqueeze(0)
        # input -> (1, bs)

        # embedded -> (1, bs, embedding_dim)
        embedded = self.dec_dropout(self.embedding(input))

        # output -> (1, bs, hidden_dim) (decoder is unidirectional)
        output, hidden = self.decoder(embedded, hidden)
        output = output.squeeze(0)
        embedded = embedded.squeeze(0)
        output = self.output2vocab(torch.cat((output, embedded), dim=1))
        # output -> (bs, vocab_size)
        return output, hidden

    def _attend(self, dec_output, enc_output):
        # Get attention weights
        attn_weights = self.attn(dec_output, enc_output)
        # Get weighted sum
        context = attn_weights.bmm(enc_output.transpose(0, 1))
        # Concatenate weighted context vector and rnn output using Luong eq. 5
        dec_output = dec_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((dec_output, context), 1)
        concat_output = torch.tanh(self.attn.W_cat(concat_input))
        # Predict next word (Luong eq. 6)
        dec_output = self.output2vocab(concat_output)
        return dec_output, context

    def forward(self, x, x_lens, y=None):
        """
        Performs one forward pass through the network, i.e., encodes x,
        predicts y through the decoder, calculates loss and finally,
        backprops the loss
        ==================
        Parameters:
        ==================
        x (tensor) -> padded input sequences of shape (max_x_len, batch_size)
        x_lens (tensor) -> lengths of the individual elements in x (batch_size)
        y (tensor) -> padded target sequences of shape (max_y_len, batch_size)
            y = None denotes inference mode
        """
        infer = (y is None)
        if infer:  # model in val/test mode
            self.eval()
            y = torch.zeros(
                (self.config['MAX_LENGTH'], x.shape[1])).long().fill_(
                 self.sos_idx).to(self.device)
        else:  # train mode
            self.train()
            self.optimizer.zero_grad()

        # z is the final forward and backward hidden state of all layers
        z = self._encode(x, x_lens)['z']
        # decoder_outputs -> (max_y_len, bs, vocab_size)
        decoder_outputs = self._decode(z, y, infer)

        # loss calculation and backprop
        loss = self.rec_loss(
            decoder_outputs[1:].view(-1, decoder_outputs.shape[-1]),
            y[1:].view(-1))

        if not infer:
            loss.backward()
            # Clip gradients (wt. update) (very important)
            nn.utils.clip_grad_norm_(self.parameters(), self.config['clip'])
            self.optimizer.step()
        return {'pred_outputs': decoder_outputs, 'loss': loss}
