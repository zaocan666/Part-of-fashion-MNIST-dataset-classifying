人智大作业2
=====================================
## Requirements
- torch 1.2.0
- python 3.5.2
- Cuda compilation tools, release 10.1, V10.1.105
- cudnn 6

## 文件夹
- bagging投票：使用bagging投票方法时的csv文件以及生成投票结果的代码。
- code：训练、测试所用代码，包括曾用代码，如尝试过的各种模型，数据分析所用代码等。
- 训练日志log：每次训练记录的loss，准确率等的变化。

## 代码
### train.py
训练主要使用的代码，包括训练主循环、准确率计算、结果测试等。

### model.py
resnet模型的定义，其他模型的定义在“曾用代码”文件夹中。

### utilis.py
一些功能函数，如存储模型、提取模型等.

### dataset.py
提取训练数据。

### data/split_train_val.py
划分训练集和测试集。

### data/convert2pic.py
将npy转化为图片文件，方便查看。

## 训练方法
首先需要将训练数据放在同目录data文件夹中，然后运行run.sh
###run.sh 样例
\#!/bin/sh

RANDOM_SEED=57924
LOG_FILE=resnet18_splitEven_2fc_4layer2222_60epoch_rotate15_cropPad_valNocrop_normalize2_batchsize64.txt
python data/split_train_val.py --random_seed=$RANDOM_SEED > $LOG_FILE
CUDA_VISIBLE_DEVICES=4,5 python train.py --random_seed=$RANDOM_SEED >> $LOG_FILE

### training parameters
- **FC_LR = 1e-3**
- **NET_LR = 1e-3**
- **BATCH_SIZE = 64**
- **OPTIMIZER = 'adam'**
- **WEIGHT_DECAY = 1e-4**
- **MOMENTUM = 0.9**
- **DECAY_RATE = 0.1**
