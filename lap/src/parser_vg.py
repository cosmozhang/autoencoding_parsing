from optparse import OptionParser
import pickle, utils, mstlstm_vg, os, os.path, time
import torch
from utils import read_conll
import vg
import numpy as np
import random
import pdb

if __name__ == '__main__':

    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="../../UD_English-GUM/en_gum-ud-train.conllu")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                      default="../../UD_English-GUM/en_gum-ud-dev.conllu")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE",
                      default="../../UD_English-GUM/en_gum-ud-test.conllu")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="best.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=60)
    parser.add_option("--numthread", type="int", dest="nthread", default=1)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=1)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_option("--verbose", action="store_true", dest="verbose", default=False)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--z_dim", type="int", dest="z_dim", default=100)
    parser.add_option('--prior_file', type=str, default=None, help='path to saved prior file (default: None)')
    parser.add_option('--ufl', type=int, dest='ufl', default=1, help='frequency of updating prior for labeled data (default: 1)')
    parser.add_option('--ufu', type=int, dest='ufu', default=1, help='frequency of updating prior for labeled data (default: 1)')
    parser.add_option('--uprop', type=float, dest='uprop', default=0.0, help='propotion of the unlabeled data (default: 0.0)')
    parser.add_option('--lprop', type=float, dest='lprop', default=1.0, help='propotion of the labeled data (default: 1.0)')

    (options, args) = parser.parse_args()
    torch.set_num_threads(options.nthread)
    print torch.get_num_threads()

    print 'Using external embedding:', options.external_embedding

    print 'pytorch version:', torch.__version__

    print 'Using {}% labeled data and {}% unlabeled data'.format(options.lprop * 100, options.uprop * 100)

    assert options.lprop + options.uprop <= 1.0, 'label propotion + unlabeled propotion is less than 1.0'

    if options.predictFlag:
        '''
        predict
        '''
        with open(options.params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        print 'Initializing lstm mstparser:'
        parser = mstlstm_vg.MSTParserLSTM(words, pos, rels, w2i, stored_opt)
        parser.load(options.model)
        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        testpath = os.path.join(options.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')

        ts = time.time()
        test_res = list(parser.predict(options.conll_test))
        te = time.time()
        print 'Finished predicting test.', te - ts, 'seconds.'
        utils.write_conll(testpath, test_res)

        if not conllu:
            os.system('perl utils/eval.pl -g ' + options.conll_test + ' -s ' + testpath + ' > ' + testpath + '.txt')
        else:
            os.system(
                'python utils/evaluation_script/conll17_ud_eval.py -v -w utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + testpath + ' > ' + testpath + '.txt')
            with open(testpath + '.txt', 'rb') as f:
                for l in f:
                    if l.startswith('UAS'):
                        print 'UAS:%s' % l.strip().split()[-1]
                    # elif l.startswith('LAS'):
                    #     print 'LAS:%s' % l.strip().split()[-1]
    else:
        '''
        train
        '''
        print 'Preparing vocab'
        words, w2i, pos, rels = utils.vocab(options.conll_train)

        with open(options.params, 'w') as paramsfp:
            pickle.dump((words, w2i, pos, rels, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing lstm mstparser:'
        '''
        init prior buffer
        '''
        with open(options.conll_train, 'r') as conllFP:
            sentencesData = list(read_conll(conllFP))

            l_prior_word_mean_buffer = vg.prior_buffer(sentencesData, "word_l_mean", options.z_dim, options.ufl, options.prior_file)
            l_prior_word_logvar_buffer = vg.prior_buffer(sentencesData, "word_l_logvar", options.z_dim, options.ufl, options.prior_file)

            if options.pembedding_dims > 0:

                l_prior_pos_mean_buffer = vg.prior_buffer(sentencesData, "pos_l_mean", options.z_dim, options.ufl, options.prior_file)
                l_prior_pos_logvar_buffer = vg.prior_buffer(sentencesData, "pos_l_logvar", options.z_dim, options.ufl, options.prior_file)

            if options.external_embedding is not None:

                l_prior_eword_mean_buffer = vg.prior_buffer(sentencesData, "eword_l_mean", options.z_dim, options.ufl, options.prior_file)
                l_prior_eword_logvar_buffer = vg.prior_buffer(sentencesData, "eword_l_logvar", options.z_dim, options.ufl, options.prior_file)

            if options.external_embedding is not None:

                if options.pembedding_dims > 0:
                    all_prior_buffers = (l_prior_eword_mean_buffer, l_prior_eword_logvar_buffer, l_prior_word_mean_buffer, l_prior_word_logvar_buffer, l_prior_pos_mean_buffer, l_prior_pos_logvar_buffer)
                else:
                    all_prior_buffers = (l_prior_eword_mean_buffer, l_prior_eword_logvar_buffer, l_prior_word_mean_buffer, l_prior_word_logvar_buffer)
            else:
                if options.pembedding_dims > 0:
                    all_prior_buffers = (l_prior_word_mean_buffer, l_prior_word_logvar_buffer, l_prior_pos_mean_buffer, l_prior_pos_logvar_buffer)
                else:
                    all_prior_buffers = (l_prior_word_mean_buffer, l_prior_word_logvar_buffer)

        parser = mstlstm_vg.MSTParserLSTM(words, pos, rels, w2i, options, all_prior_buffers) # OJO, add the prior buffer

        print "Training started!"
        best_uas = float("-inf")
        best_epoch = 0
        for epoch in xrange(1, options.epochs+1):
            if options.verbose:
                print 'Starting epoch', epoch
            '''
            train here
            '''
            parser.train(options.conll_train, epoch)

            conllu = (os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
            devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
            utils.write_conll(devpath, parser.predict(options.conll_dev))
            # parser.save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1)))

            if not conllu:
                os.system('perl utils/eval.pl -g ' + options.conll_dev + ' -s ' + devpath + ' > ' + devpath + '.txt')
                with open(devpath + '.txt', 'rb') as f:
                    for l in f:
                        if l.startswith('  Unlabeled'):
                            if options.verbose:
                                print('UAS:%s' % l.strip().split()[-2])
                            cuas = float(l.strip().split()[-2])
            else:
                os.system('python utils/evaluation_script/conll17_ud_eval.py -v -w utils/evaluation_script/weights.clas ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
                with open(devpath + '.txt', 'rb') as f:
                    for l in f:
                        if l.startswith('UAS'):
                            if options.verbose:
                                print 'UAS:%s' % l.strip().split()[-1]
                            cuas = float(l.strip().split()[-1])
                        # elif l.startswith('LAS'):
                        #     print 'LAS:%s' % l.strip().split()[-1]

            if cuas > best_uas:
                best_epoch = epoch
                best_uas = cuas
                parser.save(options.model)
        print("best_epoch: {}, best validation UAS: {}".format(best_epoch, best_uas))
