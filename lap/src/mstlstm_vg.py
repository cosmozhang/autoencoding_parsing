import shutil
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Parameter
from torch.nn.init import *
from torch import optim
from utils import read_conll
from operator import itemgetter
import utils, time, random, decoder, eisner_funcs
import vg
import numpy as np
import os
import pdb

if not torch.cuda.is_available():
    print 'Using CPU'
    USE_GPU = False
else:
    print 'Using GPU'
    USE_GPU = True

get_data = (lambda x: x.data.cpu()) if USE_GPU else (lambda x: x.data)


def Variable(inner):
    return torch.autograd.Variable(inner.cuda() if USE_GPU else inner)


def Parameter(shape=None, init=xavier_uniform_):
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(torch.Tensor(init))
    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))


def scalar(f):
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    if type(f) == float:
        return Variable(torch.FloatTensor([f]))


def cat(l, dimension=-1):
    # pdb.set_trace()
    valid_l = filter(lambda x: x is not None, l)
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)


class MSTParserLSTMModel(nn.Module):
    def __init__(self, vocab, pos, rels, w2i, options, all_prior_buffers):
        super(MSTParserLSTMModel, self).__init__()
        random.seed(1)
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu}
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims

        self.zdim = options.z_dim

        self.ExtnrEmbPath = options.external_embedding

        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.external_embedding, self.edim = None, 0
        if self.ExtnrEmbPath is not None:
            external_embedding_fp = open(self.ExtnrEmbPath, 'r')
            external_embedding_fp.readline()
            '''
            self.external_embedding: a dictionary
            '''
            self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            # self.noextrn = [0.0 for _ in xrange(self.edim)]
            # self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            np_emb = np.zeros((len(vocab) + 3, self.edim), dtype = np.float32)
            for word, i in self.vocab.iteritems():
                if word in self.external_embedding:
                    np_emb[i] = self.external_embedding[word]

            self.elookup = nn.Embedding(len(vocab) + 3, self.edim)
            # pdb.set_trace()
            self.elookup.weight = torch.nn.Parameter(torch.from_numpy(np_emb))
            # self.extrnd['*PAD*'] = 1
            # self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        '''
        add lstm
        '''
        self.word_lstm = nn.LSTM(self.wdims, self.ldims, bidirectional=True)
        if self.pdims > 0:
            self.pos_lstm = nn.LSTM(self.pdims, self.ldims, bidirectional=True)
        if self.ExtnrEmbPath is not None:
            self.eword_lstm = nn.LSTM(self.edim, self.ldims, bidirectional=True)


        self.hidden_units = options.hidden_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims) # word_emb
        if self.pdims > 0:
            self.plookup = nn.Embedding(len(pos) + 3, self.pdims) # pos_emb
        self.rlookup = nn.Embedding(len(rels), self.rdims) # rel_emb

        if self.ExtnrEmbPath is not None:

            self.hidLayerFOH = Parameter((self.zdim*(3 if self.pdims > 0 else 2), self.hidden_units))
            self.hidLayerFOM = Parameter((self.zdim*(3 if self.pdims > 0 else 2), self.hidden_units))
        else:
            self.hidLayerFOH = Parameter((self.zdim*(2 if self.pdims > 0 else 1), self.hidden_units))
            self.hidLayerFOM = Parameter((self.zdim*(2 if self.pdims > 0 else 1), self.hidden_units))
        self.hidBias = Parameter((self.hidden_units))

        '''
        set up for vae
        '''
        '''
        buffers
        '''
        if all_prior_buffers:
            if self.ExtnrEmbPath is not None:
                if self.pdims > 0:
                    self.l_prior_eword_mean_buffer, self.l_prior_eword_logvar_buffer, self.l_prior_word_mean_buffer, self.l_prior_word_logvar_buffer, self.l_prior_pos_mean_buffer, self.l_prior_pos_logvar_buffer = all_prior_buffers
                else:
                    self.l_prior_eword_mean_buffer, self.l_prior_eword_logvar_buffer, self.l_prior_word_mean_buffer, self.l_prior_word_logvar_buffer = all_prior_buffers
            else:
                if self.pdims > 0:
                    self.l_prior_word_mean_buffer, self.l_prior_word_logvar_buffer, self.l_prior_pos_mean_buffer, self.l_prior_pos_logvar_buffer = all_prior_buffers
                else:
                    self.l_prior_word_mean_buffer, self.l_prior_word_logvar_buffer = all_prior_buffers

        '''
        VAE
        '''
        # OJO
        self.word_vgl = vg.VAEG(len(vocab) + 3, self.wdims, self.wlookup.weight.data.numpy(), self.ldims*2, self.zdim)
        if self.pdims > 0:
            self.pos_vgl = vg.VAEG(len(pos) + 3, self.pdims, self.plookup.weight.data.numpy(), self.ldims*2, self.zdim)
        if self.ExtnrEmbPath is not None:
            self.eword_vgl = vg.VAEG(len(vocab) + 3, self.edim, self.elookup.weight.data.numpy(), self.ldims*2, self.zdim)

        self.outLayer = Parameter((self.hidden_units, 1))

        self.esn = eisner_funcs.Eisner.apply

    def cal_scores(self, lstm_out):

        '''
        lstm_out: seq_len * out_dim
        '''
        # pdb.set_trace()
        sl, _ = lstm_out.shape

        headfov = torch.matmul(lstm_out, self.hidLayerFOH) # seq_len * hidden_dim

        modfov = torch.matmul(lstm_out, self.hidLayerFOM) # seq_len * hidden_dim

        sum_arc_out = headfov.expand(sl, -1, -1).permute(1,0,2) + modfov.expand(sl, -1, -1) + self.hidBias.squeeze()

        exprs = torch.matmul(torch.tanh(sum_arc_out), self.outLayer).squeeze(2)


        scores = get_data(exprs).numpy()
        # pdb.set_trace()
        return scores, exprs

    def predict(self, sentence):

        for entry in sentence:
            wordvec = self.wlookup(scalar(int(self.vocab.get(entry.norm, 0)))) if self.wdims > 0 else None
            # if entry.pos == '$':
                # pdb.set_trace()
            if self.pdims > 0:
                posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None

            evec = None
            if self.ExtnrEmbPath is not None:
                evec = self.elookup(scalar(int(self.vocab.get(entry.form, self.vocab.get(entry.norm,
                                                                                       0)))))

            '''
            combine three embeddings
            '''
            if self.ExtnrEmbPath is not None:
                """
                combine three embeddings
                """
                # pdb.set_trace()
                entry.wordvec = wordvec
                entry.ewordvec = evec
                if self.pdims > 0:
                    entry.posvec = posvec
            else:
                entry.wordvec = wordvec
                if self.pdims > 0:
                    entry.posvec = posvec

        '''
        add lstm
        '''
        word_lstm_input = torch.stack([entry.wordvec for entry in sentence]) #len * dim
        if self.pdims > 0:
            pos_lstm_input = torch.stack([entry.posvec for entry in sentence]) #len * dim
        if self.ExtnrEmbPath is not None:
            eword_lstm_input = torch.stack([entry.ewordvec for entry in sentence])


        word_lstm_out, _ = self.word_lstm.forward(word_lstm_input)
        if self.pdims > 0:
            pos_lstm_out, _ = self.pos_lstm.forward(pos_lstm_input)
        if self.ExtnrEmbPath is not None:
            eword_lstm_out, _ = self.eword_lstm.forward(eword_lstm_input)

        kl_temp = 1.0
        lp_eword_mean, lp_eword_logvar, lp_word_mean, lp_word_logvar, lp_pos_mean, lp_pos_logvar = None, None, None, None, None, None

        '''
        go through stocastic layer
        '''
        word_z = self.word_vgl.predict(word_lstm_out)

        if self.pdims > 0:
            pos_z = self.pos_vgl.predict(pos_lstm_out)

        if self.ExtnrEmbPath is not None:
            eword_z = self.eword_vgl.predict(eword_lstm_out)

        if self.ExtnrEmbPath is not None:
            if self.pdims > 0:
                scoring_input = cat([word_z, eword_z, pos_z])
            else:
                scoring_input = cat([word_z, eword_z])
        else:
            if self.pdims > 0:
                scoring_input = cat([word_z, pos_z])
            else:
                scoring_input = word_z

        scoring_input = scoring_input.squeeze(1)

        scores, exprs = self.cal_scores(scoring_input)
        heads = decoder.parse_proj(scores)

        for entry, head in zip(sentence, heads):
            entry.pred_parent_id = head
            entry.pred_relation = '_'

    def forward(self, sentence, isHasTree, sent_id, curr_iteration):

        word_ids = torch.LongTensor([int(self.vocab.get(entry.norm, 0)) for entry in sentence]).unsqueeze(1)
        if USE_GPU:
            word_ids = word_ids.cuda()

        pos_ids = torch.LongTensor([int(self.pos[entry.pos]) for entry in sentence]).unsqueeze(1)
        if USE_GPU:
            pos_ids = pos_ids.cuda()

        loss = 0.0
        for entry in sentence: # each token
            c = float(self.wordsCount.get(entry.norm, 0))
            '''
            whether to drop a word embedding
            '''
            dropFlag = (random.random() < (c / (0.25 + c)))
            '''
            get word embedding
            '''
            wordvec = self.wlookup(scalar(int(self.vocab.get(entry.norm, 0)) if dropFlag else 0)) if self.wdims > 0 else None

            if self.pdims > 0:
                '''
                get pos embedding
                '''
                posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
            '''
            get pretrained word embedding
            '''
            evec = None
            if self.external_embedding is not None:
                evec = self.elookup(scalar(self.vocab.get(entry.form, self.vocab.get(entry.norm, 0)) if (dropFlag or (random.random() < 0.5)) else 0))

            '''
            combine three embeddings
            '''
            if self.ExtnrEmbPath is not None:
                """
                combine three embeddings
                """
                # pdb.set_trace()
                entry.wordvec = wordvec
                entry.ewordvec = evec
                if self.pdims > 0:
                    entry.posvec = posvec
            else:
                entry.wordvec = wordvec
                if self.pdims > 0:
                    entry.posvec = posvec

        '''
        add lstm
        '''
        word_lstm_input = torch.stack([entry.wordvec for entry in sentence]) #len * dim
        if self.pdims > 0:
            pos_lstm_input = torch.stack([entry.posvec for entry in sentence]) #len * dim

        if self.ExtnrEmbPath is not None:
            eword_lstm_input = torch.stack([entry.ewordvec for entry in sentence])

        word_lstm_out, _ = self.word_lstm.forward(word_lstm_input)

        if self.pdims > 0:
            pos_lstm_out, _ = self.pos_lstm.forward(pos_lstm_input)

        if self.ExtnrEmbPath is not None:
            eword_lstm_out, _ = self.eword_lstm.forward(eword_lstm_input)

        '''
        retrieve prior
        '''

        lp_word_mean = torch.from_numpy(self.l_prior_word_mean_buffer[sent_id]) # OJO
        lp_word_logvar = torch.from_numpy(self.l_prior_word_logvar_buffer[sent_id]) # OJO

        if USE_GPU:
            lp_word_mean = lp_word_mean.double().cuda()
            lp_word_logvar = lp_word_logvar.double().cuda()

        if self.pdims > 0:
            lp_pos_mean = torch.from_numpy(self.l_prior_pos_mean_buffer[sent_id]) # OJO
            lp_pos_logvar = torch.from_numpy(self.l_prior_pos_logvar_buffer[sent_id]) # OJO

            if USE_GPU:
                lp_pos_mean = lp_pos_mean.double().cuda()
                lp_pos_logvar = lp_pos_logvar.double().cuda()

        if self.ExtnrEmbPath is not None:

            lp_eword_mean = torch.from_numpy(self.l_prior_eword_mean_buffer[sent_id]) # OJO
            lp_eword_logvar = torch.from_numpy(self.l_prior_eword_logvar_buffer[sent_id]) # OJO

            if USE_GPU:
                lp_eword_mean = lp_eword_mean.double().cuda()
                lp_eword_logvar = lp_eword_logvar.double().cuda()


        '''
        The anealling factor
        '''
        kl_temp = vg.get_kl_temp(curr_iteration) # OJO

        '''
        go through stocastic layer
        '''
        word_loss, word_log_loss, word_kl, word_z, word_mean_qs, word_logvar_qs = self.word_vgl.forward(word_lstm_out, lp_word_mean, lp_word_logvar, kl_temp, word_ids)

        if self.pdims > 0:
            pos_loss, pos_log_loss, pos_kl, pos_z, pos_mean_qs, pos_logvar_qs = self.pos_vgl.forward(pos_lstm_out, lp_pos_mean, lp_pos_logvar, kl_temp, pos_ids)

        if self.ExtnrEmbPath is not None:
            eword_loss, eword_log_loss, eword_kl, eword_z, eword_mean_qs, eword_logvar_qs = self.eword_vgl.forward(eword_lstm_out, lp_eword_mean, lp_eword_logvar, kl_temp, word_ids)


        vae_loss = word_loss

        if self.pdims > 0:
            vae_loss += pos_loss

        if self.ExtnrEmbPath is not None:
            vae_loss += eword_loss

        '''
        update prior, basically replace the old gaussian parameters of the prior by new gaussian paprameters
        '''
        self.l_prior_word_mean_buffer.update_buffer(sent_id, word_mean_qs, len(sentence))
        self.l_prior_word_logvar_buffer.update_buffer(sent_id, word_logvar_qs, len(sentence))

        if self.pdims > 0:
            self.l_prior_pos_mean_buffer.update_buffer(sent_id, pos_mean_qs, len(sentence))
            self.l_prior_pos_logvar_buffer.update_buffer(sent_id, pos_logvar_qs, len(sentence))

        if self.ExtnrEmbPath is not None:
            self.l_prior_eword_mean_buffer.update_buffer(sent_id, eword_mean_qs, len(sentence))
            self.l_prior_eword_logvar_buffer.update_buffer(sent_id, eword_logvar_qs, len(sentence))

        if isHasTree:

            if self.ExtnrEmbPath is not None:
                if self.pdims > 0:
                    scoring_input = cat([word_z, eword_z, pos_z])
                else:
                    scoring_input = cat([word_z, eword_z])
            else:
                if self.pdims > 0:
                    scoring_input = cat([word_z, pos_z])
                else:
                    scoring_input = word_z

            scoring_input = scoring_input.squeeze(1)

            scores, exprs = self.cal_scores(scoring_input)
            '''
            margin based
            '''
            # gold = [entry.parent_id for entry in sentence]
            # heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

            # e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])

            # if e > 0:
            #     for i, (h, g) in enumerate(zip(heads, gold)):
            #         if h != g:
            #             loss += (exprs[h][i] - exprs[g][i])

            '''
            crf based
            '''
            gold = torch.LongTensor([entry.parent_id for entry in sentence][1:])

            logZ, _ = self.esn(exprs) #log partition

            if USE_GPU:
                gold = gold.cuda()

            d = torch.sum(torch.gather(exprs[:,1:],
                                       0,
                                       gold.unsqueeze(0)).squeeze(0))

            loss = -d + logZ

        return loss, vae_loss



def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt == 'adam':
        return optim.Adam(parameters)


class MSTParserLSTM(object):
    def __init__(self, vocab, pos, rels, w2i, options, all_prior_buffers=None):
        model = MSTParserLSTMModel(vocab, pos, rels, w2i, options, all_prior_buffers)
        self.model = model.double().cuda() if USE_GPU else model
        self.trainer = get_optim(options.optim, self.model.parameters())
        self.options = options

    def predict(self, conll_path):
        '''
        predict and write to the conll entries
        '''
        self.model.eval()
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                self.model.predict(conll_sentence)
                yield conll_sentence

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    def train(self, conll_path, epoch):
        batch = 1
        start = time.time()

        self.model.train()
        loss_value = 0.0
        batch_loss = 0.0
        n_sent = 0
        with open(conll_path, 'r') as conllFP:
            sentencesData = list(read_conll(conllFP))
            sentence_idxls = range(len(sentencesData))

            lprop = self.options.lprop
            uprop = self.options.uprop

            max_idx_l = int(len(sentencesData) * lprop)-1
            max_idx_u = max_idx_l + int(len(sentencesData) * uprop)

            l_idxs = range(0, max_idx_l + 1)
            u_idxs = range(max_idx_l + 1, max_idx_u + 1)

            random.shuffle(l_idxs)
            random.shuffle(u_idxs)

            sentence_idxls = l_idxs + u_idxs

            l_idx_set = set(range(0, max_idx_l + 1))
            u_idx_set = set(range(max_idx_l + 1, max_idx_u + 1))

            random.shuffle(sentence_idxls)

            for iSentence, sentence_id in enumerate(sentence_idxls):
                if sentence_id in l_idx_set:
                    isHasTree = True
                elif sentence_id in u_idx_set:
                    isHasTree = False
                else:
                    continue

                sentence = sentencesData[sentence_id]

                if iSentence % 100 == 0 and iSentence != 0:
                    '''
                    print info
                    '''
                    if self.options.verbose:
                        print 'Processing sentence number:', iSentence, 'Loss:', loss_value / n_sent, 'Time', time.time() - start
                    start = time.time()
                    loss_value = 0.0
                    n_sent = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                n_sent += 1

                '''
                calculate loss
                '''
                loss, vae_loss = self.model.forward(conll_sentence, isHasTree, sentence_id, curr_iteration = epoch * len(sentencesData) + iSentence)
                batch_loss = loss + vae_loss
                # eerrors += e

                if iSentence % batch == 0:
                    if type(batch_loss) != float:
                        batch_loss.backward()
                        self.trainer.step()

                        loss_value += batch_loss.data.cpu().numpy()
                        batch_loss = 0.0

                    self.model.zero_grad()

            # last a few sentences
            if type(batch_loss) != float:
                batch_loss.backward()
                self.trainer.step()
            self.model.zero_grad()
