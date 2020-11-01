import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

def get_kl_temp(curr_iteration, kl_anneal_rate=1e-3, max_temp=1.0):
    temp = np.exp(kl_anneal_rate * curr_iteration) - 1.
    return float(np.minimum(temp, max_temp))

def gaussian(mean, logvar):
    '''
    .normal: sample from (0, 1)
    '''
    return mean + torch.exp(0.5 * logvar) * Variable(logvar.data.new(mean.size()).normal_())

def kl_normal2_normal2(mean1, log_var1, mean2, log_var2):
    '''
    args:
        mean_1: seq_len * dim
        log_var1: seq_len * dim
        mean2: seq_en * dim
        log_var2: seq_len * dim
    '''
    # pdb.set_trace()
    return 0.5 * log_var2 - 0.5 * log_var1 + (torch.exp(log_var1) + (mean1 - mean2) ** 2) / (2 * torch.exp(log_var2) + 1e-10) - 0.5


def compute_KL_div(mean_q, log_var_q, mean_prior, log_var_prior):
    '''
    args:
        mean_q: seq_len * dim
        log_var_q: seq_len * dim
        mean_prior: seq_en * dim
        log_var_prior: seq_len * dim
    '''
    kl_divergence = kl_normal2_normal2(mean_q, log_var_q, mean_prior, log_var_prior)
    return kl_divergence


def compute_KL_div2(mean, log_var):
    return - 0.5 * (1 + log_var - mean.pow(2) - log_var.exp())

class prior_buffer(object):
    def __init__(self, sentences, prior_type, z_dim=100, freq=1, init_path=None):

        '''
        sentences: sentences of words
        z_dim: z's dim
        freq: frequency of updating prior, default is 1
        name: mean or logvar
        experiment: the args
        init_path: where to save the prior files
        '''

        self.z_dim = z_dim
        self.freq = freq

        '''
        save prior
        '''
        if init_path is not None:
            self.path = os.path.join(init_path, prior_type + "_" + str(z_dim))
        else:
            self.path = init_path

        if self.path is None or not os.path.isfile(self.path):
            '''
            construct using standard gaussian
            '''
            self.buffer = np.asarray([np.zeros((len(sentence), z_dim)).astype('float32') for sentence in sentences])
            self.path = os.path.join('.', prior_type + "_" + str(z_dim))
        elif self.path is not None and os.path.isfile(self.path):
            '''
            construct from existing buffers
            '''
            self.buffer = self.load()
        else:
            raise ValueError("invalid initial path for prior buffer: {}".format(init_path))

        '''
        count array for each sentence
        '''
        self.count = [0] * len(sentences)

    def __len__(self):
        # number of sentences
        return len(self.buffer)

    def update_buffer(self, ixs, posterior, seq_len):
        """
        Args:
            ixs: sentence_id
            posterior: length x z_dim
            seq_len: int, sentence_len
        """

        new_i = ixs%len(self)

        assert len(self.buffer[new_i]) == seq_len

        self.buffer[new_i] = posterior.detach().cpu().numpy()
        self.count[new_i] += 1

    def __getitem__(self, ixs):
        get_buffer = self.buffer[ixs]
        # max_len = np.max([len(b) for b in get_buffer])
        # batch_size = len(ixs)

        # pad_buffer = np.zeros((batch_size, max_len, self.z_dim)).astype("float32")
        # for i, b in enumerate(get_buffer):
            # pad_buffer[i, :len(b), :] = b
        # return pad_buffer
        return get_buffer

    def save(self):
        pickle.dump(self.buffer, open(self.path, "wb+"), protocol=-1)
        print("prior saved to: {}".format(self.path))

    def load(self):
        with open(self.path, "rb+") as infile:
            priors = pickle.load(infile)
        print("prior loaded from: {}".format(self.path))
        return priors

class VAEG(nn.Module):
    def __init__(self, word_vocab_size, l_dim, embed_init, rsize, zsize, train_emb = False):
        '''
        args:
            word_vocab_size: reconstruction size
            l_dim: token_representation_dimensionality
            embed_init: reconstruction weight
            rsize: rnn input dim
            zsize: latent vector dim
        '''
        super(VAEG, self).__init__()
        self.to_latent_gaussian = gaussian_layer(input_size=rsize, latent_z_size=zsize)
        self.word_embed = nn.Embedding(word_vocab_size, l_dim)
        if embed_init is not None:
            self.word_embed.weight.data.copy_(torch.from_numpy(embed_init))
            # print("Initialized with pretrained word embedding")

        if not train_emb:
            self.word_embed.weight.requires_grad = False
            # print("Word Embedding not trainable")

        self.x2token = nn.Linear(l_dim, word_vocab_size, bias=False)
        self.x2token.weight = self.word_embed.weight

        self.z2x = nn.Sequential(
            nn.Linear(zsize, rsize),
            nn.ReLU(),
            nn.Linear(rsize, l_dim)
            )

    def forward(self, lstm_out, prior_mean, prior_logvar, kl_temp, reconstruction_labels, xvar = 1e-3):
        '''
        args:
            lstm_out: seq_len * dim
            prior_mean: seq_len * dim
            prior_logvar: seq_len * dim
            kl_temp: scalar, the annealing factor
            reconstruction_labels: seq_len
        '''

        '''
        go through stochastic layer
        '''
        batch_size, batch_len = reconstruction_labels.shape
        z, mean_qs, logvar_qs = self.to_latent_gaussian.forward(lstm_out, self.training)

        '''
        reconstruction given the latent representation
        '''
        mean_x = self.z2x.forward(z)
        varVar = Variable(mean_x.data.new(1).fill_(xvar))

        x = gaussian(mean_x, varVar)

        x_pred = self.x2token.forward(x) # sent_len * vocab_size
        # x_pred = x_pred.squeeze(1)
        # pdb.set_trace()
        log_loss = F.cross_entropy(x_pred.view(batch_size*batch_len, -1), reconstruction_labels.view(-1))
        # pdb.set_trace()
        # tmp = reconstruction_labels.cpu().numpy()
        # print tmp
        log_loss = log_loss.mean()


        if prior_mean is not None and prior_logvar is not None:
            tmp = compute_KL_div(mean_qs, logvar_qs, prior_mean, prior_logvar)
            kl_div = tmp.mean()

            loss = log_loss + kl_temp * kl_div
        else:
            kl_div = None
            loss = log_loss

        return loss.mean(), log_loss.mean(), kl_div.mean() if kl_div is not None else None, z, mean_qs, logvar_qs

    def predict(self, lstm_out):

        z, mean_qs, logvar_qs = self.to_latent_gaussian.forward(lstm_out, self.training)

        return z



class gaussian_layer(nn.Module):
    """
        h
        |
        z
        |
        x
    """
    def __init__(self, input_size, latent_z_size):
        super(gaussian_layer, self).__init__()
        self.input_size = input_size
        self.latent_z_size = latent_z_size

        '''
        nn to estimate gaussian mean
        '''
        self.q_mean2_mlp = nn.Linear(input_size, latent_z_size)
        '''
        nn to estimate gaussian var
        '''
        self.q_logvar2_mlp = nn.Linear(input_size, latent_z_size)

    def forward(self, inputs, is_to_sample):
        """
        args:
            inputs: batch x batch_len x input_size
            is_to_sample: training or testing
        """
        batch_size, batch_len, _ = inputs.size()
        mean_qs = self.q_mean2_mlp(inputs)
        logvar_qs = self.q_logvar2_mlp(inputs)

        if is_to_sample:
            z = gaussian(mean_qs, logvar_qs)
        else:
            z = mean_qs

        return z, mean_qs, logvar_qs
