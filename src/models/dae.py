import torch
import random
from tqdm import tqdm
from torch import nn, optim
from models import attn_module
from models import scoring_functions


class AutoEncoder(nn.Module):
    """AutoEncoder model"""
    def __init__(self, config, vocab, embedding_wts):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.vocab = vocab
        self.embedding_wts = embedding_wts
        # keep them same by default. if needed, the self.scorer_emb_wts can be
        # manually updated in the concerned script. But note that we'll
        # need to call _load_scorer method there too (so that the entire scoring)
        # function is updated and not just the variable :P
        self.scorer_emb_wts = embedding_wts
        self.build_model()
        self.scorer = self._load_scorer(self.scorer_emb_wts, 'bimodal')

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

        self.output2vocab = nn.Linear(self.latent_dim + self.embedding_dim,
                                      self.vocab.size)

        self.optimizer = optim.Adam(self.parameters(), self.config['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=15,
                                                   gamma=0.5)
        # Reconstruction loss
        self.rec_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def _load_scorer(self, emb_wts, _type='rnn'):
        if not self.config['pretrained_scorer']:
            return None
        if _type == 'rnn':
            scorer = scoring_functions.RNNScorer(self.config, emb_wts)
        elif _type == 'bimodal':
            scorer = scoring_functions.BiModalScorer(self.config, emb_wts)
        else:
            raise ValueError('Invalid scoring function type given')

        scorer.load_state_dict(
            torch.load('{}{}'.format(self.config['save_dir'],
                                     self.config['pretrained_scorer']),
                       map_location=self.device)['model'])
        return scorer.to(self.device)

    def _get_scores(self, old_scores, candidates, mel_specs, scorer):
        """
        updates scores for all the candidates by measuring their compatability
        with the respective spectrograms
        candidates: (t, bs, k, vocab_size) where t is the current time step
        mel_specs: (bs, 3, 224, 224)
        """
        # t, bs, k, v = candidates.shape
        # n_candidates = bs*k*v
        t, n_candidates = candidates.shape
        k = self.beam_size
        v = self.vocab.size
        # candidates = candidates.view(t, bs*k*v).long()
        candidates = candidates.long()
        scores = torch.empty(0).to(self.device)
        # compatibility scores -> (bs, beam_size*v)
        # print('Passing candidates through {}'.format(type(scorer).__name__))
        continue_count = 0
        for iter in tqdm(range(0, n_candidates, self.config['batch_size'])):
            curr_candidates = candidates[:, iter:iter+self.config['batch_size']]
            curr_mel_specs = mel_specs[iter // (k*v)].unsqueeze(0).repeat(
                curr_candidates.shape[1], 1, 1, 1)
            curr_scores = scorer({
                'lyrics_seq': curr_candidates,
                'mel_spec': curr_mel_specs
                })[:, 1]  # get scores for class 1
            scores = torch.cat((scores, curr_scores))
        # adjust scores to make it broadcastable wrt old_scores
        scores = scores.view(-1, k, v).unsqueeze(0)
        # return updated scores -> (t, bs, k, v)
        return (old_scores * torch.exp(scores/self.config['scorer_temp']))

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

        # outputs => (max_seq_len x bs x hidden_dim * self.pf)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Construct z from last time_step output
        if self.bidirectional:
            outputs = outputs.view(max_x_len, bs, self.pf, self.hidden_dim)
            # concatenate forward and backward encoder outputs
            outputs = outputs[:, :, 0, :] + outputs[:, :, 1, :]

            if self.unit == 'lstm':
                # sum forward and backward hidden states
                c_n = hidden[0].view(self.config['enc_n_layers'],
                                     self.pf, bs, self.hidden_dim)
                c_n = c_n[:, 0, :, :] + c_n[:, 1, :, :]

                h_n = hidden[1].view(self.config['enc_n_layers'],
                                     self.pf, bs, self.hidden_dim)
                h_n = h_n[:, 0, :, :] + h_n[:, 1, :, :]
                hidden = (c_n, h_n)
            else:
                # sum forward and backward hidden states
                hidden = hidden.view(self.config['enc_n_layers'],
                                     self.pf, bs, self.hidden_dim)
                hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        # outputs => (max_seq_len x bs x hidden_dim * self.pf)
        # hidden => (#enc_layers, bs, hidden * self.pf)
        return {'encoder_outputs': outputs, 'z': hidden}

    def _create_mask(self, tensor):
        return torch.ne(tensor, self.pad_idx)

    def _decode(self, z, y, infer, encoder_outputs=None,
                mask=None, y_specs=None, scorer_emb_wts=None):
        """
        z -> (#enc_layers, batch_size x latent_dim)
        y -> (max_y_len, batch_size)

        encoder_outputs and mask will be used for attention mechanism
        TODO: Handle bidirectional decoders
        """
        max_y_len, bs = y.shape
        # max_y_len = 6
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
            output, dec_hidden = self._decode_token(output, dec_hidden, mask)
            # print(output.shape, decoder_outputs.shape, t)
            decoder_outputs[t] = output
            do_tf = random.random() < self.config['tf_ratio']
            if infer or (not do_tf):
                if self.config['dec_mode'] == 'beam':
                    # split beam and bs dim from dec_outputs
                    new_logits = decoder_outputs[:t].view(t, bs,
                                                          self.beam_size,
                                                          vocab_size)
                    # run a softmax on the last dimension. No, thank you!
                    # new_logits = torch.log_softmax(new_logits, dim=-1)
                    if y_specs is not None:
                        candidates = candidate_subseq[:t+1]
                        new_logits = self._get_scores(new_logits,
                                                      candidates,
                                                      y_specs,
                                                      self.scorer)
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

    def _decode_token(self, input, hidden, mask):
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
        output = self.output2vocab(torch.cat((output, embedded), dim=-1))
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
        # decoder_outputs -> (max_y_len, bs*beam_size, vocab_size)
        decoder_outputs = self._decode(z, y, infer)

        # loss calculation and backprop
        loss = self.rec_loss(
            decoder_outputs[1:].view(-1, decoder_outputs.shape[-1]),
            y.repeat(1, self.beam_size)[1:].view(-1))

        if not infer:
            loss.backward()
            # Clip gradients (wt. update) (very important)
            nn.utils.clip_grad_norm_(self.parameters(), self.config['clip'])
            self.optimizer.step()
        return {'pred_outputs': decoder_outputs, 'loss': loss}
