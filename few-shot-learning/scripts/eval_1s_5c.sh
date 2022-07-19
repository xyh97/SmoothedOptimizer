#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --mode test \
               --resume mini-1/logs-1-perturb-eps-1.0-lamb-1.0/ckpts/meta-learner-49000.pth.tar \
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
               --grad-clip 0.25 \
               --bn-momentum 0.95 \
               --bn-eps 1e-3 \
               --data miniimagenet \
               --data-root ../miniImagenet \
               --pin-mem True \
               --log-freq 100
