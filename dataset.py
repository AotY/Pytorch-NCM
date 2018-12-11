#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Dataset
"""
import os
import yaml
import numpy as np
import torch
import torch.utils.data as data
from collections import Counter
from vocab import Vocab
from vocab import PAD_ID, SOS_ID, EOS_ID
from tokenizer import Tokenizer

class Dataset(data.Dataset):
    def __init__(self, datas, vocab):
        self._datas = datas
        self.vocab = vocab

    def __len__(self):
        return len(self._datas)

    def __getitem__(self, idx):
        q_tokens, r_tokens = self._datas[idx]
        q_ids = self.vocab.words_to_id(q_tokens)
        r_ids = self.vocab.words_to_id(r_tokens)
        return q_ids, r_ids


def collate_fn(batch_pair):
    ''' Pad the instance to the max seq length in batch '''
    batch_q, batch_r = list(), list()
    q_max_len, r_max_len = 0, 0
    batch_q_len, batch_r_len = list(), list()
    for q_ids, r_ids in batch_pair:
        q_max_len = max(q_max_len, len(q_ids))
        r_max_len = max(r_max_len, len(r_ids))

        batch_q.append(q_ids)
        batch_r.append(r_ids)

        batch_q_len.append(len(q_ids))
        batch_r_len.append(len(r_ids) + 1)

    batch_q = np.array([
        ids + [PAD_ID] * (q_max_len - len(ids))
        for ids in batch_q
    ])

    batch_r = np.array([
        [SOS_ID] + ids + [EOS_ID] + [PAD_ID] * (r_max_len - len(ids))
        for ids in batch_r
    ])

    batch_q = torch.LongTensor(batch_q)
    batch_r = torch.LongTensor(batch_r)

    batch_q_len = torch.LongTensor(batch_q_len)
    batch_r_len = torch.LongTensor(batch_r_len)

    return batch_q, batch_r, batch_q_len, batch_r_len


def prepare_datas_vocab(data_dir):
    tokenizer = Tokenizer()
    vocab = Vocab()

    datas = []
    freq_dict = Counter()
    for filename in os.listdir(data_dir):
        if not filename.endswith('.yml'):
            continue
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r') as f:
            yaml_dict = yaml.load(f)
            categories = yaml_dict['categories']
            conversations = yaml_dict['conversations']
            for conversation in conversations:
                #  print('categories: ', categories)
                #  print('conversation: ', conversation)
                q = conversation[0]
                q_tokens = tokenizer.tokenize(q)
                freq_dict.update(q_tokens)
                for r in conversation[1:]:
                    r_tokens = tokenizer.tokenize(r)
                    freq_dict.update(r_tokens)
                    datas.append((q_tokens, r_tokens))

    #  print(freq_dict)
    freq_list = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)
    vocab.build_from_freq(freq_list)
    np.random.shuffle(datas)

    return datas, vocab
