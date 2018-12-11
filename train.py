#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import time
import math
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from nc_model import NCModel

from dataset import Dataset, collate_fn, prepare_datas_vocab
from vocab import PAD_ID

# Parse argument for language to train
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data dir')
parser.add_argument('--vocab_size', type=int, help='')
parser.add_argument('--max_len', type=int, help='') # decode
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--bidirectional', type='store_true')
parser.add_argument('--num_layers', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--tied', action='store_true')
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--epochs', type=int)
parser.add_argument('--device', type=str, help='cpu or cuda')
parser.add_argument('--save_model', type=str, help='save model.')
parser.add_argument('--log', type=str, help='save log.')
parser.add_argument('--seed', type=str, help='random seed')
args = parser.parse_args()

torch.random.manual_seed(args.seed)
device = torch.device(args.device)

# load data
datas, vocab = prepare_datas_vocab(args.data_dir)

args.vocab_size = int(vocab.size)

# dataset
validation_split = int(args.val_split * len(datas))
training_dataset = Dataset(datas[validation_split:])
validation_dataset = Dataset(datas[:validation_split])

# data loader
training_data = data.DataLoader(
    training_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

validation_data = data.DataLoader(
    validation_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

# model
model = NCModel(
    args,
    device
)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train

def train(epoch):
    ''' Epoch operation in training phase'''
    model.train()
    model.reset_teacher_forcing_ratio()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc=' (Training: %d) ' % epoch, leave=False):

        # prepare data
        batch_q, batch_r, batch_q_len, batch_r_len = map(lambda x: x.to(device), batch)

        batch_r_input = batch_r[:, :-1]
        batch_r_target = batch_r[:, 1:]

        # forward
        optimizer.zero_grad()

        pred = model(
            batch_q,
            batch_r_input,
            batch_r_len,
            batch_q_len
        )

        # backward
        loss, n_correct = cal_performance(pred, batch_r_target, smoothing=True)

        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = batch_r_target.ne(PAD_ID)

        n_word = non_pad_mask.sum().item()

        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval(epoch):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc=' (Validation: %d) ' % epoch, leave=False):

            batch_q, batch_r, batch_q_len, batch_r_len = map(lambda x: x.to(device), batch)

            batch_r_input = batch_r[:, :-1]
            batch_r_target = batch_r[:, 1:]

            pred = model(
                batch_q,
                batch_q_len
            )

            # backward
            loss, n_correct = cal_performance(pred, batch_r_target, smoothing=True)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = batch_r_target.ne(PAD_ID)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train_epochs():
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if args.log:
        log_train_file = args.log + '.train.log'
        log_valid_file = args.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch in range(args.epochs):
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_loss, train_accu = train(epoch)

        print(' (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval(epoch)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch
        }

        if args.save_model:
            if args.save_mode == 'all':
                model_name = args.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = args.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(PAD_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(PAD_ID)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=PAD_ID, reduction='sum')

    return loss


if __name__ == '__main__':
    train_epochs()
