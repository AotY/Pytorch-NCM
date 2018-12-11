#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 taoqing <taoqing@gpu3>
#
# Distributed under terms of the MIT license.

"""
Utils
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# orthogonal initialization
def init_gru_orth(model, gain=1):
    model.reset_parameters()
    # orthogonal initialization of gru weights
    for _, hh, _, _ in model.all_weights:
        for i in range(0, hh.size(0), model.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + model.hidden_size], gain=gain)

def init_lstm_orth(model, gain=1):
    init_gru_orth(model, gain)

    #positive forget gate bias (Jozefowicz es at. 2015)
    for _, _, ih_b, hh_b in model.all_weights:
        l = len(ih_b)
        ih_b[l // 4 : l // 2].data.fill_(1.0)
        hh_b[l // 4 : l // 2].data.fill_(1.0)

def init_linear_wt(linear):
    init_wt_normal(linear.weight, linear.in_features)
    if linear.bias is not None:
        init_wt_normal(linear.bias, linear.in_features)

def init_wt_normal(weight, dim=512):
    weight.data.normal_(mean=0, std=np.sqrt(2.0 / dim))

def init_wt_unif(weight, dim=512):
    weight.data.uniform_(-np.sqrt(3.0 / dim), np.sqrt(3.0 / dim))

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel() # elements number
    max_len = max_len or lengths.max() # max_len
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat((batch_size, 1))
            .lt(lengths.unsqueeze(1)))


def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    if rnn_type in ['RNN', 'LSTM', 'GRU']:
        return getattr(nn, rnn_type)(**kwargs)
    else:
        raise ValueError("{} is not valid, ".format(rnn_type))



