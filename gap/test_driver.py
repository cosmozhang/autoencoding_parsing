from optparse import OptionParser
import os, time, random, pickle

import torch
import numpy as np
from src.util.data_operator import read_conll, write_conll, vocab
from src.util.config_reader import Configurable
import importlib
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
        print('Using external embedding:', options.external_embedding)

    if options.gpuFlag and torch.cuda.is_available():
        print("Use GPU!")

    with open(options.conll_test, 'r') as conllFP:
        test_sentences = list(read_conll(conllFP))

    '''
    predict
    '''
    with open(os.path.join(options.output, options.params), 'r') as paramsfp:
        words, w2i, p2i, rels = pickle.load(paramsfp)

    print('Initializing parser for testing:')

    if options.modelname == "ncrfae_model":
        parsermod = importlib.import_module('src.semi_ncrfae_parser')
    else:
        parsermod = importlib.import_module('src.sup_parser')

    MSTParser = getattr(parsermod, "MSTParserLSTM")

    parser = MSTParser(words, p2i, rels, w2i, options)

    parser.load(os.path.join(options.output, os.path.basename(options.savemodel)))
    conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
    testpath = os.path.join(options.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')

    ts = time.time()
    parser.predict(test_sentences)
    te = time.time()
    print('Finished predicting test.', te - ts, 'seconds.')
    write_conll(testpath, test_sentences)

    if not conllu:
        os.system('perl eval_utils/eval.pl -g ' + options.conll_test + ' -s ' + testpath + ' > ' + testpath + '.txt')
        with open(testpath + '.txt', 'rb') as f:
            for l in f:
                if l.startswith('  Unlabeled'):
                    print('UAS:%s' % l.strip().split()[-2])
                # elif l.startswith('Labeled'):
                    # print('LAS:%s' % l.strip().split()[-2])
                    # pass
    else:
        os.system(
            'python eval_utils/evaluation_script/conll17_ud_eval.py -v -w eval_utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + testpath + ' > ' + testpath + '.txt')
        with open(testpath + '.txt', 'rb') as f:
            for l in f:
                if l.startswith('UAS'):
                    print('UAS:%s' % l.strip().split()[-1])
                # elif l.startswith('LAS'):
                    # print('LAS:%s' % l.strip().split()[-1])
                    # pass

