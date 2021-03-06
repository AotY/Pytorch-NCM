#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Encoder
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 config,
                 embedding):

        super(Encoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.bidirection_num = 2 if config.bidirectional else 1
        self.hidden_size = config.hidden_size // self.bidirection_num

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # rnn
        self.rnn = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
            batch_first=True
        )

    def forward(self, inputs, lengths=None):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [seq_len, batch_size, num_directions * hidden_size]
            max_output: [1, batch_size, hidden_size * num_directions]
            hidden_state: (h_n, c_n)
        '''
        #  print(lengths)
        if lengths is not None:
            # sort lengths
            lengths, sorted_indexes = torch.sort(lengths, dim=0, descending=True)
            # restore to original indexes
            _, restore_indexes = torch.sort(sorted_indexes, dim=0)

            inputs = inputs[sorted_indexes]

        # embedded
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)

        outputs, hidden_state = self.rnn(embedded)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[restore_indexes].contiguous()
            hidden_state = hidden_state.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        return outputs, hidden_state
