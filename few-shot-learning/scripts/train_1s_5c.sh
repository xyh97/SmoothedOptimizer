#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --mode train \
               --lamb 1.0 \
               --eps 1.0 \
               --n-shot 1 \
               --n-eval 15 \
               --n-class 5 \
               --input-size 4 \
               --hidden-size 20 \
               --lr 1e-4 \
               --episode 50000 \
               --episode-val 100 \
               --epoch 10 \
               --batch-size 5 \
               --image-size 84 \
               --seed 1 \
               --grad-clip 0.25 \
               --bn-momentum 0.95 \
               --bn-eps 1e-3 \
               --data miniImagenet \
               --data-root ../miniImagenet \
               --pin-mem True \
               --log-freq 50 \
               --val-freq 1000 \
               --perturb