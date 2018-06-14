#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable


def init_embeddings(embeddings):
    init.uniform_(embeddings.weight, -0.05, 0.05)


def init_linear(linear):
    init.uniform_(linear.weight, -0.05, 0.05)
    init.constant_(linear.bias, 0.)


def init_rnn(rnn):
    if isinstance(rnn, (nn.GRUCell, nn.LSTMCell, nn.RNNCell)):
        init.xavier_uniform_(rnn.weight_hh)
        init.xavier_uniform_(rnn.weight_ih)
        init.constant_(rnn.bias_hh, 0.)
        init.constant_(rnn.bias_ih, 0.)

    else:
        for layer in range(rnn.num_layers):
            init.xavier_uniform_(getattr(rnn, 'weight_hh_l{layer}'.format(layer=layer)))
            init.xavier_uniform_(getattr(rnn, 'weight_ih_l{layer}'.format(layer=layer)))
            init.constant_(getattr(rnn, 'bias_hh_l{layer}'.format(layer=layer)), 0.)
            init.constant_(getattr(rnn, 'bias_ih_l{layer}'.format(layer=layer)), 0.)


def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    init.constant_(conv.bias, 0.)


def init_sequential_linear(sequential):
    for child in sequential.children():
        if isinstance(child, nn.Linear):
            init_linear(child)


def multiple_index_select(t, index):
    """
    Permute columns of a given tensor for each row independently of the others
    :param t: tensor with at least 2 dimensions
    :param index: 2D LongTensor with size corresponding to the first two dims

    >>> a, b, c = list(range(0, 2)), list(range(2, 4)), list(range(4, 6))
    >>> t = torch.LongTensor([[a, b, c], [c, b, c]])
    >>> index = torch.LongTensor([[1, 2, 0], [0, 1, 1]])
    >>> multiple_index_select(t, index).tolist()
    [[[2, 3], [4, 5], [0, 1]], [[4, 5], [2, 3], [2, 3]]]
    """
    # check dimensions
    for dim, (tdim, indexdim) in enumerate(zip(t.size(), index.size())):
        if tdim != indexdim:
            raise ValueError("Mismatch {dim}, {tdim} != {indexdim}".format(
                dim=dim, tdim=tdim, indexdim=indexdim
            ))
    offset = (torch.arange(0, index.size(0)) * index.size(1)).long()
    if t.is_cuda:
        offset = offset.cuda()
    size = [-1] + list(t.size()[2:])
    return t.view(*size)[(index + offset[:, None]).view(-1)].view(*t.size())


def broadcast(t, outsize, dim):
    """
    General function that broadcasts a tensor (copying the data) along
    a given dimension `dim` to a desired size `outsize`, if the current
    size of that dimension is a divisor of the desired size `outsize`.

    >>> x = broadcast(torch.randn(5, 2, 6), 6, 1)
    >>> y = broadcast(torch.randn(5, 6, 6), 6, 1)
    >>> x.size(1)
    6
    >>> y.size(1)
    6
    """
    tsize = t.size(dim)
    # do we need to broadcast?
    if tsize != outsize:
        div, mod = divmod(outsize, tsize)
        if mod != 0:
            raise ValueError(
                "Cannot broadcast {} -> {}".format(outsize, tsize))
        size = [1 if d != dim else div for d in range(len(t.size()))]
        return t.repeat(*size)
    return t


def batchify(d, batch_size):
    batched = {k: torch.split(d[k], batch_size, 0) for k in d}
    for splits in zip(*batched.values()):
        assert len(set([len(split) for split in splits])) == 1, \
            "Unequal nb of batches across input datasets"
    batches, nb_batches = [], len(list(batched.values())[0])
    for batch in range(nb_batches):
        batches.append({k: v[batch] for k, v in batched.items()})
    return batches


def from_numpy(array):
    if array.dtype == np.int32:
        array = array.astype(np.int64, copy=False)
    return torch.from_numpy(array)


class BatchIterator(object):
    def __init__(self, batch_size, src, trg=None, gpu=False, dev=False):
        self.batch_size = batch_size
        self.gpu = gpu
        self.dev = dev
        self.src = batchify(src, batch_size)
        self.trg = batchify(trg, batch_size) if trg is not None else None

    def __getitem__(self, idx):
        src = {k: Variable(v.cuda() if self.gpu else v, volatile=self.dev)
               for k, v in self.src[idx].items()}
        if self.trg is None:
            trg = None
        else:
            trg = {k: Variable(v.cuda() if self.gpu else v, volatile=self.dev)
                   for k, v in self.trg[idx].items()}

        return src, trg

    def __len__(self):
        return len(self.src)

    @classmethod
    def from_numpy(cls, batch_size, src, trg=None, **kwargs):
        src = {k: from_numpy(v) for k, v in src.items()}
        if trg is not None:
            # TODO: currently transforming all trgs to long
            # but it should happen in the preprocessor
            trg = {k: from_numpy(v).long() for k, v in trg.items()}
        return cls(batch_size, src, trg=trg, **kwargs)


class Optimizer(object):
    def __init__(self, params, method, max_norm=10., **kwargs):
        self.params = list(params)
        self.method = method
        self.optim = getattr(optim, method)(self.params, **kwargs)
        self.max_norm = max_norm

    def step(self, norm_type=2):
        if self.max_norm is not None:
            clip_grad_norm(self.params, self.max_norm, norm_type=norm_type)
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()


# Kindly taken from here:
# https://github.com/fchollet/keras/blob/{commit}/keras/utils/generic_utils.py
# {commit}: 268672df65941fcc2e1727b877aa457c05acfc45
class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval and current < self.target:
                return

            prev_total_width = self.total_width
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * ' ')

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write('\n')

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)
