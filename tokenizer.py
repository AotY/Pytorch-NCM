#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Tokenizer
"""

import re
import string
from nltk.tokenize import TweetTokenizer


class Tokenizer:
    def __init__(self):
        number_regex_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'  # numbers
        self.number_re = re.compile(number_regex_str, re.VERBOSE | re.IGNORECASE)

    ''' replace number by NUMBER_TAG'''
    def replace_number(self, tokens):
        return ['<number>' if self.number_re.search(token) else token for token in tokens]

    def tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)

        tokens = self.replace_number(tokens)
        tokens = [token for token in tokens if len(token.split()) > 0]
        return tokens

    def clean_str(self, text):
        text = text.lower()
        text = re.sub('^',' ', text)
        text = re.sub('$',' ', text)
        text = re.sub(r"[^A-Za-z0-9,!?\'\`\.]", " ", text)
        text = re.sub(r"\.{3}", " ...", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\s{2,}", " ", text)

        # url
        words = []
        for word in text.split():
            i = -1
            if word.find('http') != -1:
                i = word.find('http')
            elif word.find('www') != -1:
                i = word.find('www')
            elif word.find('.com') != -1:
                i = word.find('.com')

            if i >= 0:
                word = word[:i] + ' ' + '<url>'
            words.append(word.strip())
        text = ' '.join(words)

        # remove markdown url
        text = re.sub(r'\[([^\]]*)\] \( *<url> *\)', r'\1', text)

        # remove illegal char
        text = re.sub('<url>', 'url', text)
        text = re.sub(r"[^a-za-z0-9():,.!?\"\']", " ", text)
        text = re.sub('url', '<url>',text)

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

