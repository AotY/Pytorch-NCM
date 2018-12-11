#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#
export CUDA_VISIBLE_DEVICES=5

python train.py \
    --data_dir data/english \
    --log log/ \
    --embedding_size 256 \
    --hidden_size 256 \
    --num_layers 2 \
    --bidirectional \
    --share_embedding \
    --dropout 0.1 \
    --teacher_forcing_ratio 1.0 \
    --clip 5.0 \
    --lr 0.0005 \
    --epochs 10 \
    --device cpu \
    --seed 19 \

/
