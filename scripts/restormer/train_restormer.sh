#!/bin/bash

CONFIG=$1
export PYTHONPATH="./Restormer"

torchrun --standalone --nnodes=1 --nproc-per-node=1 Restormer/basicsr/train.py -opt $CONFIG --launcher pytorch