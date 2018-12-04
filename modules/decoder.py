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
            dropout=config.dropout
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
            dec_input: [1, batch_size] or [max_len, batch_size]
            dec_hidden: [num_layers, batch_size, hidden_size]
            inputs_length: [batch_size, ] or [1, ]
            h_encoder_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            h_encoder_lengths: [batch_size]
            f_enc_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            f_enc_length: [batch_size]
        '''
        # embedded
        embedded = self.embedding(dec_input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        rnn_input = torch.cat((embedded, dec_context), dim=2) # [1, batch_size, embedding_size + hidden_size]
        output, dec_hidden = self.rnn(rnn_input, dec_hidden)

        # output: [1, batch_size, 1 * hidden_size]
        context, attn_weights = self.attn(output, enc_outputs, enc_length)

        output = torch.cat(output_list, dim=2)

        # [1, batch_size, vocab_size]
        output = self.linear(output)

        return output, dec_hidden , attn_weights
