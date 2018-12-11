#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Decoder
"""

import torch
import torch.nn as nn

from modules.attention import Attention


class Decoder(nn.Module):
    def __init__(self, config, embedding):

        super(Decoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # attn
        self.attn = Attention(config.hidden_size)

        self.rnn = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
			batch_first=True
        )

        self.linear = nn.Linear(config.hidden_size * 2, config.vocab_size)

        init_linear_wt(self.linear)

    def forward(self,
                dec_input,
                dec_hidden,
                enc_outputs,
                enc_length):
        '''
        Args:
            dec_input: [batch_size, 1] or [batch_size, max_len]
            dec_hidden: [num_layers, batch_size, hidden_size]
            inputs_length: [batch_size, ]
        '''
        # embedded
        embedded = self.embedding(dec_input)  
        embedded = self.dropout(embedded)

        rnn_input = torch.cat((embedded, dec_context), dim=2) 
        output, dec_hidden = self.rnn(rnn_input, dec_hidden)

        context, attn_weights = self.attn(output, enc_outputs, enc_length)

        output = torch.cat(output_list, dim=2)

        # [batch_size, 1, vocab_size]
        output = self.linear(output)

        return output, dec_hidden , attn_weights
