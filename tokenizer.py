#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Tokenizer
"""

import re
from nltk.tokenize import TweetTokenizer


class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)

        tokens = self.clean_str(text).split()
        tokens = [token for token in tokens if len(token.split()) > 0]
        return tokens

    def clean_str(self, text):
        text = text.lower()

        # contraction
        add_space = ["'s", "'m", "'re", "n't", "'ll","'ve","'d","'em"]
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)
        text = ' ' + ' '.join(tokenizer.tokenize(text)) + ' '
        text = text.replace(" won't ", " will n't ")
        text = text.replace(" can't ", " can n't ")
        for a in add_space:
            text = text.replace(a+' ', ' '+a+' ')

        text = re.sub(r'^\s+', '', text)
        text = re.sub(r'\s+$', '', text)
        text = re.sub(r'\s+', ' ', text) # remove extra spaces

        return text

