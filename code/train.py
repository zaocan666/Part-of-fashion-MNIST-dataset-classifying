import argparse
from torch.autograd import Variable
import os
import random
import numpy as np
import time
import torchvision.transforms as transforms
import csv
import torch.nn as nn
import torch
import copy

from model import resnet18
from utils import *
from dataset import load_data, mean_std
from googlenet import googleNet

dtype = torch.float32
USE_GPU = True
EPOCH = 60
BATCH_SIZE = 64
print_every = int(50 / BATCH_SIZE * 64)
Load_model = False
NAME = 'CLOTH'

NET_LR = 1e-3
FC_LR = 1e-3
OPTIMIZER = 'adam'
LR_DECAY_EPOCH = [] if OPTIMIZER == 'adam' else [15, 30]
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
DECAY_RATE = 0.1
RANDOM_SEED = 'in args'

START_EPOCH = 0

loader_train = None
loader_val = None

def set_device():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('using device:', device)
    return device

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    if True:
        maxk = max(topk)
        batch_size = target.size(0)

        target_int = target.to(dtype=torch.long)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target_int.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc_k = correct_k.mul_(100.0 / batch_size)
            acc_k = acc_k.cpu().data.numpy()[0]
            res.append(acc_k)

        return res

#训练
def train(model, optimizer, criterion, device, epochs=1, start=0):
    global loader_train, loader_val
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    bestACC_ever = 0

    if not os.path.isdir(NAME + '_save'):
        os.mkdir(NAME + '_save')

    acc_val=0

    for e in range(start, epochs):
        print('epoch: %d'%e)

        losses = AverageMeter()
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if e in LR_DECAY_EPOCH:
            adjust_learning_rate(optimizer, decay_rate=DECAY_RATE)

        end_time = time.time()
        for t, (x, y, _) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            losses.update(loss.item())
            top1.update(prec1)
            top5.update(prec5)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if t % print_every == 0:
                print('Train: [%d/%d]\t'
                      'Time %.3f (%.3f)\t'
                      'Loss %.4f (%.4f)\t'
                      'Prec@1 %.3f (%.3f)\t'
                      'Prec@5 %.3f (%.3f)'
                      % (t, len(loader_train), batch_time.val, batch_time.avg,losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg))

        #using val dataloader
        if type(loader_val)==torch.utils.data.dataloader.DataLoader:
            acc_val = test_epoch(model, criterion, loader_val, device, e, epochs)
        else:
            acc_val = acc_val+0.001

        if acc_val > bestACC_ever:
            bestACC_ever = acc_val
            save_model_optimizer_history(model, optimizer, filepath=NAME + '_save' + '/epoch%d_ACC_%.3f' % (e, acc_val),
                                    device=device)

    print("best ACC:", bestACC_ever)

def tensor_showImg(a):
    a=a.cpu()
    image_PIL = transforms.ToPILImage()(a)
    image_PIL.show()

#在validation集上测试
def test_epoch(model, criterion, loader_val, device, epoch, end_epoch, verbo=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    total = 0
    correct = 0
    end_time = time.time()
    for batch_idx, (x, targets, _) in enumerate(loader_val):
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        targets = targets.to(device=device, dtype=torch.long)
        x, targets = Variable(x), Variable(targets)

        output = model(x)
        output = output.to(device=device)

        loss = criterion(output, targets)

        _, predicted = output.max(1)
        total += targets.size(0)
        target_int = targets.to(device=device, dtype=torch.long)
        correct += predicted.eq(target_int).sum().cpu().data.numpy()

        prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.cpu().data.numpy())
        top1.update(prec1)
        top5.update(prec5)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_idx % 20 == 0 and verbo == True:
            print('Test: [%d/%d]\t'
                  'Time %.3f (%.3f)\t'
                  
                  'Loss %.4f (%.4f)\t'
                  'Prec@1 %.3f (%.3f)\t'
                  'Prec@5 %.3f (%.3f)' % (batch_idx, len(loader_val),
                                       batch_time.val, batch_time.avg,
                                       losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg))

    acc = 100. * correct / total
    print('Test: [%d/%d] Acc %.3f' % (epoch, end_epoch, acc))
    return acc

def is_fc(para_name):
    split_name = para_name.split('.')
    if split_name[0]=='module':
      split_name=split_name[1:]

    if split_name[0] == 'fc':
        return True
    else:
        return False

def net_lr(model, fc_lr, lr):
    params = []
    for keys, param_value in model.named_parameters():
        if (is_fc(keys)):
            print(' fc:',keys)
            params += [{'params': [param_value], 'lr': fc_lr}]
        else:
            print('~fc:', keys)
            params += [{'params': [param_value], 'lr': lr}]

    print('fc learning rate:', fc_lr)
    print('not fc learning rate:', lr)
    return params

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

#训练主逻辑
def train_main(args):
    global loader_train, loader_val
    loader_train, loader_val, loader_test = load_data(train_bath_size=BATCH_SIZE, args=args,RANDOM_SEED=RANDOM_SEED, val_batch_size=BATCH_SIZE)

    device = set_device()
    setup_seed(RANDOM_SEED) #随机种子

    #model = googleNet()
    model = resnet18()
    #model = load_model(model, args.pretrained_model_path, device=device)
    model = nn.DataParallel(model) #多gpu

    criterion = nn.CrossEntropyLoss()

    params = net_lr(model, FC_LR, NET_LR)

    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999), weight_decay=0, eps=1e-08)
    else:
        optimizer = torch.optim.SGD(params, momentum=MOMENTUM, nesterov=True,
                                    weight_decay=WEIGHT_DECAY)

    print(model)
    start_epoch = 0
    if Load_model:
        start_epoch = 25
        filepath = 'load_model_path'
        model = load_model(model, filepath, device=device)
        model = model.to(device=device)
        optimizer = load_optimizer(optimizer, filepath, device=device)

    train(model, optimizer, criterion, device=device, epochs=EPOCH, start=start_epoch)

#在测试集上测试，获得csv
def predict_main(args):
    global loader_train, loader_val
    loader_train, loader_val, loader_test = load_data(train_bath_size=BATCH_SIZE, args=args,RANDOM_SEED=RANDOM_SEED, val_batch_size=BATCH_SIZE)

    device = set_device()
    setup_seed(RANDOM_SEED) #random seed

    model = resnet18()
    model = load_model(model, args.load_model_path, device=device)
    model = model.to(device=device)

    test_epoch(model, nn.CrossEntropyLoss(), loader_val, device, 0, 1)

    model.eval()
    result = []
    for batch_idx, (x, label, idx) in enumerate(loader_test):

        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        x = Variable(x)

        output = model(x)
        _, predicted = output.max(1)

        predicted = predicted.cpu()

        if len(idx.shape)==1:
          idx = idx.unsqueeze(1)
        index_predicted = torch.cat([idx, predicted.unsqueeze(1)], dim=1)
        index_predicted = index_predicted.cpu().data.numpy()
        result.extend(index_predicted)

    headers = ['image_id', 'label']

    with open(args.predict_output_root, 'w', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(result)

def predict_bagging(args):
    global loader_train, loader_val
    loader_train, loader_val, loader_test = load_data(train_bath_size=BATCH_SIZE, args=args,RANDOM_SEED=RANDOM_SEED, val_batch_size=BATCH_SIZE)

    device = set_device()
    setup_seed(RANDOM_SEED) #random seed

    model = resnet18()
    model_list=[]
    for maindir, subdir, file_name_list in os.walk(args.bagging_root):
      for filename in file_name_list:
        model_path = os.path.join(maindir, filename)
        print(model_path)
        model_i = load_model(model, model_path, device=device)
        model_i = model_i.to(device=device)
        model_i.eval()
        model_list.append(copy.deepcopy(model_i))

    result = []
    for batch_idx, (x, label, idx) in enumerate(loader_test):

        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        x = Variable(x)

        oneHot_result_sum = torch.zeros(x.size(0), 10)
        for model_i in model_list:
          output = model_i(x)
          _, predicted = output.max(1)
          predicted_cpu = predicted.cpu()
          oneHot_predicted = torch.zeros(output.size(0), output.size(1)).scatter_(1, predicted_cpu.unsqueeze(1), 1)
          oneHot_result_sum += oneHot_predicted
        
        _, bagging_result = oneHot_result_sum.max(1)

        bagging_result = bagging_result.cpu()

        if len(idx.shape)==1:
          idx = idx.unsqueeze(1)
        index_predicted = torch.cat([idx, bagging_result.unsqueeze(1)], dim=1)
        index_predicted = index_predicted.cpu().data.numpy()
        result.extend(index_predicted)

    headers = ['image_id', 'label']

    with open(args.predict_output_root, 'w', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(result)

#获得多个模型的csv文件
def predict_getCSV(args):
    global loader_train, loader_val
    loader_train, loader_val, loader_test = load_data(train_bath_size=BATCH_SIZE, args=args,RANDOM_SEED=RANDOM_SEED, val_batch_size=BATCH_SIZE)

    device = set_device()
    setup_seed(RANDOM_SEED) #random seed

    model = resnet18()
    for maindir, subdir, file_name_list in os.walk(args.bagging_root):
      for filename in file_name_list:
        model_path = os.path.join(maindir, filename)
        print(model_path)
        model_i = load_model(model, model_path, device=device)
        model_i = model_i.to(device=device)
        model_i.eval()
        
        result = []
        for batch_idx, (x, label, idx) in enumerate(loader_test):
          x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
          x = Variable(x)

          output = model_i(x)
          _, predicted = output.max(1)

          predicted = predicted.cpu()

          if len(idx.shape)==1:
            idx = idx.unsqueeze(1)
          index_predicted = torch.cat([idx, predicted.unsqueeze(1)], dim=1)
          index_predicted = index_predicted.cpu().data.numpy()
          result.extend(index_predicted)

        headers = ['image_id', 'label']

        with open('predict_result/'+filename+'.csv', 'w', newline='')as f:
          f_csv = csv.writer(f)
          f_csv.writerow(headers)
          f_csv.writerows(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch AI big homework')
    #parser.add_argument('--pretrained_vgg_path', type=str, help='path of pretrained model')
    #parser.add_argument('--pretrained_resnet_path', type=str, help='path of pretrained resnet model')
    parser.add_argument('--train_list_root', type=str, help='path of train image list', default='data/split_train.csv')
    parser.add_argument('--val_list_root', type=str, help='path of train image list', default='data/split_val.csv')
    parser.add_argument('--train_pic_root', type=str, help='path of train image npy', default='data/train.npy')
    parser.add_argument('--test_pic_root', type=str, help='path of test image npy', default='data/test.npy')
    parser.add_argument('--load_model_path', type=str, help='path of test image npy', default='CLOTH_save/save/resnet18_splitEven_2fc_4layer2222_60epoch_rotate15_cropPad_valNocrop_normalize2_batchsize64_91.87_epoch57_ACC_91.867')
    parser.add_argument('--predict_output_root', type=str, help='path of test image npy', default='predict_result.csv')
    parser.add_argument('--random_seed', type=str, help='path of test image npy')
    parser.add_argument('--pretrained_model_path', type=str, help='path of test image npy', default='pretrained_model/fashionMNIST_resnet18_4layer2222_rotate15_cropPad_valNocrop_normalize2_93.69_epoch39_ACC_93.690')
    parser.add_argument('--bagging_root', type=str, help='path of bagging model', default='CLOTH_save/bagging_save_test90.8')

    args = parser.parse_args()

    RANDOM_SEED = int(args.random_seed)
    print("RANDOM SEED:", RANDOM_SEED)

    train_main(args)
    #predict_main(args)
    #predict_bagging(args)
    #predict_getCSV(args)
