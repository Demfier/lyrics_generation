import torch
import numpy as np
from models import dae
from torch import nn, optim
from pprint import pprint

from torchvision.models import vgg16


class VariationalAutoEncoder(dae.AutoEncoder):
    """docstring for VariationalAutoencoder"""
    def __init__(self, config, vocab, embedding_weights):
        super(VariationalAutoEncoder, self).__init__(config, vocab,
                                                     embedding_weights)
        self.add_vae_attrs()

    def add_vae_attrs(self):
        self.anneal_type = self.config['anneal_type']
        self.anneal_till = self.config['anneal_till']
        self.k = self.config['k']
        self.x0 = self.config['x0']
        self.z_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.z_log_sigma = nn.Linear(self.hidden_dim, self.latent_dim)

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

    def _random_sample(self, n, z=None, specs=None, scorer_emb_wts=None):
        self.z_temp = self.config['sampling_temperature']
        if z is None:
            z = torch.randn(n, self.latent_dim).unsqueeze(0)
        y = (torch.ones(self.config['MAX_LENGTH'], n) * self.sos_idx).long()
        return z, self._decode(z.to(self.device), y.to(self.device), infer=True,
                               y_specs=specs, scorer_emb_wts=None)

    def _interpolate(self, z1, z2, steps):
        z1 = z1.cpu()
        z2 = z2.cpu()
        self.z_temp = self.config['sampling_temperature']
        y = (torch.ones(self.config['MAX_LENGTH'], steps + 2) * self.sos_idx).long()
        z = torch.tensor(np.linspace(z1, z2, steps)).squeeze().unsqueeze(0)
        z = torch.cat((z1, z), dim=1)
        z = torch.cat((z, z2), dim=1)
        return self._decode(z.to(self.device), y.to(self.device), infer=True)

    def forward(self, x, x_lens, step=None, y=None):
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

        # decoder_outputs -> (max_y_len, bs*beam_size, vocab_size)
        decoder_outputs = self._decode(z, y, infer)

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
