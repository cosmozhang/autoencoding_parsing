from optparse import OptionParser
import os, time, random, pickle

import torch
import numpy as np
from src.util.data_operator import read_conll, write_conll, vocab
from src.util.config_reader import Configurable
from src import sup_parser
import pdb

if __name__ == '__main__':

    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    argparser = OptionParser()
    argparser.add_option('--config_file', type="str", default='config.cfg')
    argparser.add_option("--numthread", type="int", dest="nthread", default=4)
    argparser.add_option('--use_cuda', action='store_true', default=True)
    argparser.add_option('--parsingmodel', type="str", default='BaseParser')

    (args, extra_args) = argparser.parse_args()
    options = Configurable(args.config_file, extra_args)

    torch.set_num_threads(args.nthread)
    print("Pytorch using {} threads.".format(torch.get_num_threads()))

    if options.external_embedding:
        print('Using external embedding: {}'.format(options.external_embedding))

    if options.gpuFlag:
        print("Use GPU!")

    print('Preparing vocab')
    words, w2i, p2i, rels = vocab(options.conll_train)

    '''
    proportion setting
    '''
    with open(options.conll_train, 'r') as conllFP:
        sentencesData = list(read_conll(conllFP))

    sentence_idxls = range(len(sentencesData))

    lprop = options.lprop
    uprop = options.uprop

    assert lprop + uprop <= 1.0, "proportion of labeled and unlabeled should be less or equal to 1"

    max_idx_l = int(len(sentencesData) * lprop)-1
    l_idxs = list(range(0, max_idx_l + 1))
    random.shuffle(l_idxs)
    labeledSentences = [sentencesData[i] for i in l_idxs]


    if uprop > 0.0:
        max_idx_u = max_idx_l + int(len(sentencesData) * uprop)
        u_idxs = range(max_idx_l + 1, max_idx_u + 1)
        random.shuffle(u_idxs)
        unlabeledSentences = [sentencesData[i] for i in u_idxs]
        # unlabeledSentences = sorted(unlabeledSentences, key = lambda x:len(x))

    with open(options.conll_dev, 'r') as conllFP:
        dev_sentences = list(read_conll(conllFP))

    with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
        pickle.dump([words, w2i, p2i, rels], paramsfp)
    print('Finished collecting vocab.')

    print('Initializing parser.')
    parser = sup_parser.MSTParserLSTM(words, p2i, rels, w2i, options)

    '''
    train over epoches
    '''
    print("Training started!")
    best_uas = float("-inf")
    best_epoch = 0
    for epoch in xrange(1, options.epochs+1):
        if options.verbose:
            print('Starting epoch: {}'.format(epoch))

        '''
        train
        '''
        if epoch == 1 or uprop == 0.0:
            parser.train(labeledSentences, batch_size=options.train_batch_size)
        else:
            parser.train(labeledSentences+unlabeledSentences, batch_size=options.train_batch_size)

        if uprop > 0.0:
            parser.predict(unlabeledSentences, batch_size=options.test_batch_size, self_train=True)


        conllu = (os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
        devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch) + ('.conll' if not conllu else '.conllu'))

        parser.predict(dev_sentences, batch_size=options.test_batch_size)

        write_conll(devpath, dev_sentences)

        if not conllu:
            os.system('perl eval_utils/eval.pl -g ' + options.conll_dev + ' -s ' + devpath + ' > ' + devpath + '.txt')
            with open(devpath + '.txt', 'rb') as f:
                for l in f:
                    if l.startswith('  Unlabeled'):
                        if options.verbose:
                            print('UAS:%s' % l.strip().split()[-2])
                        cuas = float(l.strip().split()[-2])
                    # elif l.startswith('Labeled'):
                        # print('LAS:%s' % l.strip().split()[-2])
                        # pass
        else:
            # print devpath, options.conll_dev
            os.system('python eval_utils/evaluation_script/conll17_ud_eval.py -v -w eval_utils/evaluation_script/weights.clas ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
            with open(devpath + '.txt', 'rb') as f:
                for l in f:
                    if l.startswith('UAS'):
                        if options.verbose:
                            print('UAS:%s' % l.strip().split()[-1])
                        cuas = float(l.strip().split()[-1])
                    # elif l.startswith('LAS'):
                        # print('LAS:%s' % l.strip().split()[-1])
                        # pass

        if cuas > best_uas:
            best_epoch = epoch
            best_uas = cuas
            parser.save(os.path.join(options.output, os.path.basename(options.savemodel)))
    print("best_epoch: {}, best validation UAS: {}".format(best_epoch, best_uas))
