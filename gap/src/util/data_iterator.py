from collections import OrderedDict, defaultdict
import numpy as np

'''
generate a id to length dic
'''
def gen_sid_len(sentences):
    sid2len = OrderedDict()
    for i, sent in enumerate(sentences):
        sid2len[i] = len(sent)

    return sid2len

def batch_slice(data, batch_size):
    #  data is a list of sentences of the same length
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        # cur_batch_size is the end-point of the batch
        sents = data[i * batch_size: i * batch_size + cur_batch_size]

        yield sents

def data_iter(sents_id2length_dic, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for (sent_id, sent_len) in sents_id2length_dic.iteritems():
        buckets[sent_len].append(sent_id)

    batched_data = []
    for (sent_len, sent_ids_smlen) in buckets.iteritems():
        # sent_ids_smlen is a list of sentences of the same length
        if shuffle:
            np.random.shuffle(sent_ids_smlen)
        # pdb.set_trace()
        '''
        'extend' expecting a iterable finishes the iteration
        '''
        batched_data.extend(list(batch_slice(sent_ids_smlen, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        """
        sent_ids in the same batch are of the same length
        """
        yield batch
