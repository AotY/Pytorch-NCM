#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.
import pickle

PAD = '__pad__'
UNK = '__unk__'
SOS = '__sos__'
EOS = '__eos__'

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

class Vocab(object):
    def __init__(self):
        self.init_vocab()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}

        self.word2idx[PAD] = 0
        self.word2idx[UNK] = 1
        self.word2idx[SOS] = 2
        self.word2idx[EOS] = 3

    @property
    def size(self):
        return len(self.word2idx)

    '''word to id '''

    def word_to_id(self, word):
        return self.word2idx.get(word, self.unkid)

    def words_to_id(self, words):
        word_ids = [self.word_to_id(cur_word) for cur_word in words]
        return word_ids

    '''id to  word'''

    def id_to_word(self, id):
        return self.idx2word.get(id, self.unk)

    '''ids to word'''
    def ids_to_word(self, ids):
        words = [self.id_to_word(id) for id in ids]
        return words

    def build_from_freq(self, freq_list):
        cur_id = 4  # because of the unk, pad, sos, and eos tokens.
        for word, _ in freq_list:
            self.word2idx[word] = cur_id
            cur_id += 1

        # init idx2word
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def ids_to_text(self, ids):
        final_ids = []
        for id in ids:
            if id in [self.padid, self.sosid]:
                continue
            elif id == self.eosid:
                break
            else:
                final_ids.append(id)
        words = self.ids_to_word(final_ids)
        text = ' '.join(words)
        return text

