# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn


from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.reduce_state import ReduceState
from modules.beam import Beam

from misc.vocab import PAD_ID, SOS_ID, EOS_ID

"""
Neural Conversation Model
"""


class NCModel(nn.Module):
    '''
    generating responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''
    def __init__(self,
                 config,
                 device='cuda'):
        super(NCModel, self).__init__()

        self.config = config
        self.device = device

        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.forward_step = 0

        enc_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            PAD_ID
        )

        dec_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            PAD_ID
        )

        # q encoder
        self.encoder = Encoder(
            config,
            enc_embedding
        )

        # encoder hidden_state -> decoder hidden_state
        self.reduce_state = ReduceState(config.rnn_type)

        # decoder
        self.decoder = Decoder(config, dec_embedding)

        # encoder, decode embedding share
        if config.share_embedding:
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self,
                enc_inputs,
                enc_length,
                dec_inputs,
                dec_length,
                evaluate=False):
        '''
        Args:
            enc_inputs: [batch_size, max_len]
            en_length: [batch_size]
        '''
        # [max_len, batch_size, hidden_size]
        enc_outputs, enc_hidden = self.encoder(
            enc_inputs,
            enc_length
        )

        # init decoder hidden
        dec_hidden = self.reduce_state(enc_hidden)

        # decoder
        dec_outputs = []
        dec_output = None
        self.update_teacher_forcing_ratio()
        for i in range(0, enc_inputs.size(1)):
            if i == 0:
                input = dec_inputs[i].view(1, -1)
            else:
                if evaluate:
                    input = dec_inputs[i].view(1, -1)
                else:
                    use_teacher_forcing = True \
                        if random.random() < self.teacher_forcing_ratio else False
                    if use_teacher_forcing:
                        input = dec_inputs[i].view(1, -1)
                    else:
                        input = torch.argmax(
                            dec_output, dim=2).detach().view(1, -1)

            dec_output, dec_hidden, _ = self.decoder(
                input,
                dec_hidden,
                enc_outputs,
                enc_length,
            )

            dec_outputs.append(dec_output)

        dec_outputs = torch.cat(dec_outputs, dim=0)
        dec_outputs = dec_outputs.view(-1, self.config.vocab_size)
        return dec_outputs

    def reset_teacher_forcing_ratio(self):
        self.forward_step = 0
        self.teacher_forcing_ratio = 1.0

    def update_teacher_forcing_ratio(self, eplison=0.0001, min_t=0.8):
        self.forward_step += 1
        if (self.teacher_forcing_ratio == min_t):
            return
        update_t = self.teacher_forcing_ratio - \
            eplison * (self.forward_step)
        self.teacher_forcing_ratio = max(update_t, min_t)

    '''decode'''

    def decode(self,
               enc_inputs,
               enc_length):
        enc_outputs, enc_hidden = self.encoder(
            enc_inputs,
            enc_length
        )

        # init decoder hidden
        dec_hidden = self.reduce_state(enc_hidden)

        # decoder
        beam_outputs, beam_score, beam_length = self.beam_decode(
            dec_hidden,
            enc_outputs,
            enc_length,
        )

        greedy_outputs = self.greedy_decode(
            dec_hidden,
            enc_outputs,
            enc_length,
        )

        return greedy_outputs, beam_outputs, beam_length

    def greedy_decode(self,
                      dec_hidden,
                      enc_outputs,
                      enc_length):

        greedy_outputs = []
        input = torch.ones((1, self.config.batch_size),
                               dtype=torch.long, device=self.device) * SOS_ID
        for i in range(self.config.max_len):
            output, dec_hidden, _ = self.decoder(
                input,
                dec_hidden,
                enc_outputs,
                enc_length,
            )
            input = torch.argmax(output, dim=2).detach().view(
                1, -1)  # [1, batch_size]
            greedy_outputs.append(input)

            # eos problem
            #  if input[0][0].item() == EOS_ID
            #  break
            #  eos_index = input[0].eq(EOS_ID)

        # [len, batch_size]  -> [batch_size, len]
        greedy_outputs = torch.cat(greedy_outputs, dim=0).transpose(0, 1)

        return greedy_outputs

    def beam_decode(self,
                    dec_hidden,
                    enc_outputs,
                    enc_length):
        '''
        Args:
            dec_hidden : [num_layers, batch_size, hidden_size] (optional)
        Return:
            prediction: [batch_size, beam, max_len]
        '''
        batch_size, beam_size = self.config.batch_size, self.config.beam_size

        # [1, batch_size x beam_size]
        input = torch.ones(1, batch_size * beam_size,
                               dtype=torch.long,
                               device=self.device) * SOS_ID

        # [num_layers, batch_size * beam_size, hidden_size]
        dec_hidden = dec_hidden.repeat(1, beam_size, 1)
        # [1, batch_size * beam_size, hidden_size]

        enc_outputs = enc_outputs.repeat(1, beam_size, 1)
        enc_length = enc_length.repeat(beam_size)

        # [batch_size] [0, beam_size * 1, ..., beam_size * (batch_size - 1)]
        batch_position = torch.arange(
            0, batch_size, dtype=torch.long, device=self.device) * beam_size

        score = torch.ones(batch_size * beam_size,
                           device=self.device) * -float('inf')
        score.index_fill_(0, torch.arange(
            0, batch_size, dtype=torch.long, device=self.device) * beam_size, 0.0)

        # Initialize Beam that stores decisions for backtracking
        beam = Beam(
            batch_size,
            beam_size,
            self.config.max_len,
            batch_position,
            EOS_ID
        )

        for i in range(self.config.max_len):
            output, dec_hidden, _ = self.decoder(
                input.view(1, -1),
                dec_hidden,
                enc_outputs,
                enc_length,
            )

            # output: [1, batch_size * beam_size, vocab_size]
            # -> [batch_size * beam_size, vocab_size]
            log_prob = output.squeeze(0)
            #  print('log_prob: ', log_prob.shape)

            # score: [batch_size * beam_size, vocab_size]
            score = score.view(-1, 1) + log_prob

            # score [batch_size, beam_size]
            score, top_k_idx = score.view(
                batch_size, -1).topk(beam_size, dim=1)

            # input: [batch_size x beam_size]
            input = (top_k_idx % self.config.vocab_size).view(-1)

            # beam_idx: [batch_size, beam_size]
            # [batch_size, beam_size]
            beam_idx = top_k_idx / self.config.vocab_size

            # top_k_pointer: [batch_size * beam_size]
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            # [num_layers, batch_size * beam_size, hidden_size]
            dec_hidden = dec_hidden.index_select(1, top_k_pointer)

            # Update sequence scores at beam
            beam.update(score.clone(), top_k_pointer, input)

            # Erase scores for EOS so that they are not expanded
            # [batch_size, beam_size]
            eos_idx = input.data.eq(EOS_ID).view(
                batch_size, beam_size)

            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length

