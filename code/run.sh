#!/bin/sh

RANDOM_SEED=57924
LOG_FILE=log/resnet18_splitEven_2fc_4layer2222_60epoch_rotate15_cropPad_valNocrop_normalize2_batchsize64.txt
python data/split_train_val.py --random_seed=$RANDOM_SEED > $LOG_FILE
CUDA_VISIBLE_DEVICES=4,5 python code/train.py --random_seed=$RANDOM_SEED >> $LOG_FILE
