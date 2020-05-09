import torch
import numpy as np
from models import dae, image_vae
from torch import nn, optim
import random
from pprint import pprint

from torchvision.models import vgg16


class BimodalVariationalAutoEncoder(nn.Module):
    """docstring for VariationalAutoencoder"""
    def __init__(self, config, vocab, embedding_wts):
        super(BimodalVariationalAutoEncoder, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding_wts = embedding_wts
        self.build_model()
        self.add_vae_attrs()
        self.load_image_vae()

    def build_model(self):
        self.unit = self.config['unit']
        self.device = self.config['device']
        self.sos_idx = self.config['SOS_TOKEN']
        self.pad_idx = self.config['PAD_TOKEN']
        # beam size is 1 by default
        self.beam_size = self.config['beam_size']
        self.hidden_dim = self.config['hidden_dim']
        self.latent_dim = self.config['latent_dim']
        self.embedding_dim = self.config['embedding_dim']
        self.bidirectional = self.config['bidirectional']
        self.enc_dropout = nn.Dropout(self.config['dropout'])
        self.dec_dropout = nn.Dropout(self.config['dropout'])

        if self.config['use_embeddings?']:
            self.embedding = nn.Embedding.from_pretrained(
                self.embedding_wts, freeze=self.config['freeze_embeddings?'])
        else:
            self.embedding = nn.Embedding(self.vocab.size, self.embedding_dim)

        if self.unit == 'lstm':
            self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim,
                                   self.config['enc_n_layers'],
                                   bidirectional=self.bidirectional)

            self.decoder = nn.LSTM(self.embedding_dim, self.latent_dim,
                                   self.config['dec_n_layers'])
        elif self.unit == 'gru':
            self.encoder = nn.GRU(self.embedding_dim, self.hidden_dim,
                                  self.config['enc_n_layers'],
                                  bidirectional=self.bidirectional)

            self.decoder = nn.GRU(self.embedding_dim, self.latent_dim,
                                  self.config['dec_n_layers'])
        else:
            self.encoder = nn.RNN(self.embedding_dim, self.hidden_dim,
                                  self.config['enc_n_layers'],
                                  bidirectional=self.bidirectional)

            self.decoder = nn.RNN(self.embedding_dim, self.latent_dim,
                                  self.config['dec_n_layers'])

        if self.config['attn_model']:
            self.attn = attn_module.Attn(self.config['attn_model'],
                                         self.hidden_dim)

        # All the projection layers
        self.pf = (2 if self.bidirectional else 1)  # project factor
        # 128 dim for the spec embedding
        self.output2vocab = nn.Linear(self.latent_dim + 128 +self.embedding_dim,
                                      self.vocab.size)

        self.optimizer = optim.Adam(self.parameters(), self.config['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=15,
                                                   gamma=0.5)
        # Reconstruction loss
        self.rec_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def add_vae_attrs(self):
        self.anneal_type = self.config['anneal_type']
        self.anneal_till = self.config['anneal_till']
        self.k = self.config['k']
        self.x0 = self.config['x0']
        self.z_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.z_log_sigma = nn.Linear(self.hidden_dim, self.latent_dim)

    def load_image_vae(self):
        self.image_vae = image_vae.VAE()
        self.image_vae.load_state_dict(torch.load(self.config['image_vae_path'],
                                       map_location=self.device))
        self.image_vae.eval()

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
            # sum forward and backward encoder outputs
            outputs = outputs[:, :, 0, :] + outputs[:, :, 1, :]

            # sum forward and backward hidden states
            hidden = hidden.view(self.config['enc_n_layers'],
                                 self.pf, bs, self.hidden_dim)
            hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        # outputs => (max_seq_len x bs x hidden_dim * self.pf)
        return {
                'encoder_outputs': outputs,
                'mu': self.z_mu(hidden),
                'log_sigma': self.z_log_sigma(hidden)
                }

    def _reparameterize(self, mu, log_sigma):
        """Samples a gaussian and returns N(mu, I * sigma**2)"""
        epsilon = torch.randn(mu.size()).to(self.device)
        z = mu + self.z_temp * (epsilon * torch.exp(log_sigma))
        return z.to(self.device)

    def _calculate_kl(self, mu, log_sigma):
        """
        Returns KL divergence KL(q||p) per training example
        KL = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
        """
        return -0.5*torch.sum(1 + 2*log_sigma - mu**2 - torch.exp(2*log_sigma))

    def _get_kl_weight(self, step):
        # Using min ensures that annealing happens only till anneal_till
        # number of steps. So VERY IMPORTANT!
        step = min(self.anneal_till, step)
        if step == 0:
            return 0.0
        if self.anneal_type == 'tanh':
            return np.round((np.tanh((step - self.x0)/1000) + 1)/2, decimals=6)
        elif self.anneal_type == 'logistic':
            return float(1/(1 + np.exp(-self.k*(step - self.x0))))
        else:  # linear
            return min(1, step/self.x0)

    def _random_sample(self, n, img_z, z=None):
        self.z_temp = self.config['sampling_temperature']
        if z is None:
            z = torch.randn(n, self.latent_dim).unsqueeze(0)
        y = (torch.ones(self.config['MAX_LENGTH'], n) * self.sos_idx).long()
        return z, self._decode(z.to(self.device), y.to(self.device),
                               img_z.to(self.device), infer=True, y_specs=None,
                               scorer_emb_wts=None)

    def _interpolate(self, z1, z2, steps):
        z1 = z1.cpu()
        z2 = z2.cpu()
        self.z_temp = self.config['sampling_temperature']
        y = (torch.ones(self.config['MAX_LENGTH'], steps + 2) * self.sos_idx).long()
        z = torch.tensor(np.linspace(z1, z2, steps)).squeeze().unsqueeze(0)
        z = torch.cat((z1, z), dim=1)
        z = torch.cat((z, z2), dim=1)
        return self._decode(z.to(self.device), y.to(self.device), infer=True)

    def _decode(self, z, y, img_z, infer, encoder_outputs=None,
                mask=None, y_specs=None, scorer_emb_wts=None):
        """
        z -> (#enc_layers, batch_size x latent_dim)
        y -> (max_y_len, batch_size)

        encoder_outputs and mask will be used for attention mechanism
        TODO: Handle bidirectional decoders
        """
        max_y_len, bs = y.shape
        vocab_size = self.vocab.size

        if y_specs is not None:
            # tensor to maintain the already guess subsequence for k beams
            # this tensor is intialized such that it has all vocab tokens as
            # candidates for the k beams but at each time step, we will
            # replace all of them with a single token for each beam
            # candidate_subseq => (max_y_len, bs*k*vocab)
            scorer_vocab = self.scorer_emb_wts.shape[0]
            candidate_subseq = torch.arange(scorer_vocab).repeat(
                bs*self.beam_size).unsqueeze(0).repeat(max_y_len, 1).to(
                self.device)
            candidate_subseq[0] = torch.ones(
                scorer_vocab*bs*self.beam_size)*self.sos_idx
            # tensor to store decoder outputs
            decoder_outputs = (torch.ones(max_y_len, bs*self.beam_size,
                               scorer_vocab)*self.sos_idx).to(self.device)
        else:
            # tensor to store decoder outputs
            decoder_outputs = (torch.ones(max_y_len, bs*self.beam_size,
                               vocab_size)*self.sos_idx).to(self.device)

        # Reconstruct the hidden vector from z
        # if #dec_layers > #enc_layers, use currently obtained hidden
        #  to represent the last #enc_layers layers of decoder hidden
        if self.config['dec_n_layers'] > self.config['enc_n_layers']:
            raise \
                NotImplementedError("(dec_n_layers > enc_n_layers) Decoder " +
                                    "can't have more layers than the encoder")
        else:
            if self.unit == 'lstm':
                try:
                    # this will throw an error when z comes from an encoder
                    # other DAE's (VAE, in this case)
                    dec_hidden = (
                        z[0][-self.config['dec_n_layers']:, :, :],
                        z[1][-self.config['dec_n_layers']:, :, :]
                        )
                except Exception as e:
                    # consider h_n to be zero
                    dec_hidden = (
                        torch.zeros(self.config['dec_n_layers'], bs,
                                    self.latent_dim).to(self.device),
                        z[-self.config['dec_n_layers']:, :, :]
                        )
                # doesn't make a different for beam_size = 1
                dec_hidden = (
                    dec_hidden[0].repeat(1, self.beam_size, 1),
                    dec_hidden[1].repeat(1, self.beam_size, 1)
                    )
            else:
                dec_hidden = z[-self.config['dec_n_layers']:, :, :]
                dec_hidden = dec_hidden.repeat(1, self.beam_size, 1)

            # dec_hidden => (#dec_layers, bs*beam_size, latent_dim)

        # initial decoder input is <sos> token
        # ouptut -> (bs)
        output = y[0, :]
        # We maintain a beam of k responses instead of just one
        # Note that for modes other beam search, the below doesn't make a
        # difference
        # output -> (beam_size*bs)
        output = output.repeat(self.beam_size, 1).view(-1)

        # Start decoding process
        for t in range(1, max_y_len):
            # output -> (beam_size*bs, vocab_size)
            # dec_hidden -> (beam_size*bs, hidden * self.pf)
            output, dec_hidden = self._decode_token(output, dec_hidden, img_z, mask)
            # print(output.shape, decoder_outputs.shape, t)
            decoder_outputs[t] = output
            do_tf = random.random() < self.config['tf_ratio']
            if infer or (not do_tf):
                if self.config['dec_mode'] == 'beam':
                    # split beam and bs dim from dec_outputs
                    new_logits = decoder_outputs[:t+1].view(t+1, bs,
                                                            self.beam_size,
                                                            vocab_size)
                    # run a softmax on the last dimension. No, thank you!
                    # new_logits = torch.log_softmax(new_logits, dim=-1)
                    if y_specs is not None:
                        candidates = candidate_subseq[:t+1]
                        new_logits = self._get_scores(new_logits,
                                                      candidates,
                                                      y_specs)
                    # extract probabilities of the tokens and flatten logits
                    new_logits = new_logits[-1].squeeze(0).view(bs, -1)
                    # new_logits => (bs, beam_size*vocab_size)
                    # .topk returns scores and tokens of shape (bs, k)
                    # the modulo operator is required to get real token ids
                    # as its range would be beam_size*vocab_size otherwise
                    output = new_logits.topk(k=self.beam_size, dim=-1)[1].view(
                        -1) % self.vocab.size
                    # output => (bs*beam_size)
                    if y_specs is not None:
                        # replace all the tokens at time t with the guessed
                        # token for k beams
                        candidate_subseq[t] = output.repeat(vocab_size)
                else:
                    # output.max(1) -> (scores, tokens)
                    # doing a max along `dim=1` returns logit scores and
                    # token index for the most probable (max valued) token
                    # scores (& tokens) -> (bs)
                    output = output.max(dim=1)[1]  # greedy search
            elif do_tf:
                output = y[t].repeat(self.beam_size, 1).view(-1)
        return decoder_outputs

    def _decode_token(self, input, hidden, img_z, mask):
        """
        input -> (beam_size*bs)
        hidden -> (#dec_layers * self.pf x bs x hidden_dim)
                  (c_n is zero for lstm decoder)
        mask -> (bs x max_x_len)
            mask is used for attention
        """
        input = input.unsqueeze(0)
        # input -> (1, beam_size*bs)

        # embedded -> (1, beam_size*bs, embedding_dim)
        embedded = self.dec_dropout(self.embedding(input))
        # output -> (1, beam_size*bs, hidden_dim) (decoder is unidirectional)
        # h_n (and c_n) -> (1, beam_size*bs, hidden)
        output, hidden = self.decoder(embedded, hidden)
        output = self.output2vocab(torch.cat((output, img_z.unsqueeze(0), embedded), dim=-1))
        # output -> (beam_size*bs, vocab_size)
        return output.squeeze(0), hidden

    def _attend(self, dec_output, enc_output):
        # TODO: Review for beam search compatibility
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

    def forward(self, x, x_lens, img_vec, step=None, y=None):
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
        step (int) -> current step number. Needed for dynamic kl weight
        """
        infer = (y is None)
        _, bs = x.shape
        if infer:  # model in val/test mode
            self.eval()
            self.z_temp = self.config['sampling_temperature']
            y = torch.zeros(
                (self.config['MAX_LENGTH'], x.shape[1])).long().fill_(
                 self.sos_idx).to(self.device)
        else:  # train mode
            self.train()
            self.z_temp = 1.0
            self.optimizer.zero_grad()

        encoder_dict = self._encode(x, x_lens)
        mu = encoder_dict['mu']
        log_sigma = encoder_dict['log_sigma']
        z = self._reparameterize(mu, log_sigma)

        with torch.no_grad():
            img_z, _, _ = self.image_vae.encode(img_vec)

        # decoder_outputs -> (max_y_len, bs*beam_size, vocab_size)
        decoder_outputs = self._decode(z, y, img_z, infer)

        # loss calculation and backprop
        loss = self.rec_loss(
            decoder_outputs[1:].view(-1, decoder_outputs.shape[-1]),
            y.repeat(1, self.beam_size)[1:].view(-1))

        if step is not None:  # do kl annealing only for training phase
            kl_weight = self._get_kl_weight(step)
            if kl_weight == 0:
                kl_params = {'loss': 0.0, 'weight': 0.0, 'wtd_loss': 0.0}
            else:
                kl_loss = self._calculate_kl(mu, log_sigma)
                wtd_kl_loss = kl_weight * kl_loss
                loss += wtd_kl_loss
                kl_params = {
                    'loss': kl_loss.item(),
                    'weight': kl_weight,
                    'wtd_loss': wtd_kl_loss.item()
                    }
            # pprint(kl_params)
            # print(loss)
        else:
            kl_params = {'loss': None, 'weight': None, 'wtd_loss': None}

        if not infer:
            loss.backward()
            # Clip gradients (wt. update) (very important)
            nn.utils.clip_grad_norm_(self.parameters(), self.config['clip'])
            self.optimizer.step()
        return {
                'pred_outputs': decoder_outputs,
                'loss': loss,
                'kl': kl_params
                }


class BimodalVED(dae.AutoEncoder):
    """docstring for VariationalEncoderDecoder"""
    def __init__(self, config, vocab, embedding_weights):
        super(VariationalEncoderDecoder, self).__init__(config, vocab,
                                                        embedding_weights)
        self.add_ved_attrs()

    def add_ved_attrs(self):
        self.img_encoder = vgg16(pretrained=True)
        self.img_dim = self.img_encoder.classifier[0].in_features

        # Keep the last layer trainable
        for p in self.img_encoder.parameters():
            p.requires_grad = True

        self.img_encoder.classifier = nn.Sequential(
            nn.Linear(self.img_dim, self.img_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.img_dim // 2, 1000))

        self.img2hidden = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(1000, self.hidden_dim))

    def _encode(self, x):
        max_x_len, bs = x.shape
        # convert input images to embeddings
        outputs = self.img_encoder(x)  # bs x 1000
        hidden = self.img2hidden(outputs)  # bs x hidden_dim
        # Forward pass through the encoder
        return {
                'encoder_outputs': outputs,
                'mu': self.z_mu(hidden),
                'log_sigma': self.z_log_sigma(hidden)
                }
