import random

import torch
import torch.nn as nn
from torch import optim
from torch.nn.init import xavier_uniform_

import pdb

USE_GPU = True if torch.cuda.is_available() else False

get_data = (lambda x: x.data.cpu()) if USE_GPU else (lambda x: x.data)

def Parameter(shape=None, init=xavier_uniform_):
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(torch.Tensor(init))
    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))

def cat(l):
    # pdb.set_trace()

    valid_l = filter(lambda x: x is not None, l)

    return torch.cat(valid_l, 2)

def not_drop(c):

    return (random.random() < (c / (0.25 + c)))


def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt == 'adam':
        return optim.Adam(parameters)
