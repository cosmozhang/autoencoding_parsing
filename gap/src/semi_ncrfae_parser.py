import time
import shutil
import imp
import os
import importlib

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from .util import data_iterator
from .util.data_operator import read_conll

from .util.model_util import USE_GPU, get_optim

import pdb

class MSTParserLSTM(object):
    def __init__(self, vocab, pos, rels, w2i, options):

        modelmod = importlib.import_module('src.model.{}'.format(options.modelname))
        MSTParserLSTMModel = getattr(modelmod, "MSTParserLSTMModel")
        model = MSTParserLSTMModel(vocab, pos, rels, w2i, options)
        self.model = model
        if options.gpuFlag and USE_GPU:
            self.model = self.model.cuda()
        self.trainer = get_optim(options.optim, self.model.parameters())
        self.em_scale = options.emscale
        self.verbose = options.verbose

        self.init_cache()

    def init_cache(self):

        # init cache
        self.ph2pm_counts = 1e-10 * torch.ones((len(self.model.pos), len(self.model.pos)),requires_grad=False)

        if USE_GPU:
            self.ph2pm_counts = self.ph2pm_counts.cuda()

    def init_decoder(self, sentences):
        # Update real counts

        for sentence in sentences:

            phs = [int(self.model.pos[sentence[entry.parent_id].pos]) for entry in sentence[1:]]
            pms = [int(self.model.pos[entry.pos]) for entry in sentence[1:]]

            for ph, pm in zip(phs, pms):
                self.ph2pm_counts[ph, pm] += 1.0

        self.update_decoder()
        # pdb.set_trace()

    def predict(self, sentences, batch_size=50):
        '''
        predict and write to the conll entries
        '''
        self.model.eval()
        sent_len_dic = data_iterator.gen_sid_len(sentences)

        for i_batch, batched_ids in enumerate(data_iterator.data_iter(sent_len_dic, batch_size)):

            batch_sents = [sentences[idx] for idx in batched_ids]

            self.model.predict(batch_sents)

    def update_counts(self, sentences, pseudo_counts=None):

        if pseudo_counts:
            # update pseudo counts
            # pdb.set_trace()
            em_scale = self.em_scale

            for sentence, sent_pseudo_counts in zip(sentences, pseudo_counts):
                pos_ids = [int(self.model.pos[entry.pos]) for entry in sentence]

                for i, h_pos_id in enumerate(pos_ids):
                    for j, m_pos_id in enumerate(pos_ids):
                        if i != j and j != 0:
                            self.ph2pm_counts[h_pos_id, m_pos_id] += sent_pseudo_counts[i, j]**(1.0/(1-em_scale))

        else:
            # Update real counts

            for sentence in sentences:

                phs = [int(self.model.pos[sentence[entry.parent_id].pos]) for entry in sentence[1:]]
                pms = [int(self.model.pos[entry.pos]) for entry in sentence[1:]]

                for ph, pm in zip(phs, pms):
                    self.ph2pm_counts[ph, pm] += 1.0

    def update_decoder(self):

        '''
        renormalize decoder's parameters by using the cache
        '''
        # pdb.set_trace()
        self.model.log_ph2pm.data = torch.log((self.ph2pm_counts+(1e-30)) / (self.ph2pm_counts.sum(1, keepdim=True)+(1e-30)*self.ph2pm_counts.shape[1]))

        # self.model.log_ph2pm.data = F.log_softmax(self.ph2pm_counts, dim=1)

    def train(self, labeledSentences, unlabeledSentences, epoch=None, batch_size=1):
        # print(self.model.ph2pm.data)

        '''
        train one epoch
        '''
        self.model.train()
        start = time.time()

        if epoch:
            self.em_scale = (epoch-1) * 0.01

        self.init_cache()
        '''
        train on labeled
        '''
        n_sent = 0

        sent_len_dic = data_iterator.gen_sid_len(labeledSentences)

        for i_batch, batched_ids in enumerate(data_iterator.data_iter(sent_len_dic, batch_size)):

            batch_sents = [labeledSentences[idx] for idx in batched_ids]

            loss, _ = self.model.forward(batch_sents, True)

            if i_batch % 100 == 0 and i_batch != 0:
                if self.verbose:
                    print('Processing labeled batch number: {0}, Loss: {1}, Time: {2}'.format(i_batch, loss.data.cpu().numpy() / n_sent, time.time() - start))
                start = time.time()
                loss_value = 0.0
                n_sent = 0

            n_sent += 1

            loss.backward()

            # nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
            #                          max_norm=5.0)
            self.trainer.step()

            self.model.zero_grad()

            '''
            update buffer
            '''
            self.update_counts(batch_sents)

        '''
        train on unlabeled
        '''
        if len(unlabeledSentences) > 0:
            n_sent = 0

            sent_len_dic = data_iterator.gen_sid_len(unlabeledSentences)

            for i_batch, batched_ids in enumerate(data_iterator.data_iter(sent_len_dic, batch_size)):

                batch_sents = [unlabeledSentences[idx] for idx in batched_ids]

                loss, pseudo_counts = self.model.forward(batch_sents, False)
                '''
                consider viterbi-EM as well
                '''

                if i_batch % 100 == 0 and i_batch != 0:
                    if self.verbose:
                        print('Processing unlabeled batch number: {0}, Loss: {1}, Time: {2}'.format(i_batch, loss.data.cpu().numpy() / n_sent, time.time() - start))
                    start = time.time()
                    loss_value = 0.0
                    n_sent = 0

                n_sent += 1

                # loss.backward()

                # nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=5.0)
                # self.trainer.step()

                # self.model.zero_grad()
                # pdb.set_trace()

                self.update_counts(batch_sents, pseudo_counts)

        # pdb.set_trace()
        self.update_decoder()

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))
