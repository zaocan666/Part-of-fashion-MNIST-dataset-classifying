import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import numpy as np

def split_train_val(info_file, ratio, random_seed):
    train = pd.read_csv(info_file)
    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=ratio, stratify=Y_train, random_state=random_seed)
    #X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=ratio, random_state=random_seed)

    train_label = []
    val_label = []

    for i, index in enumerate(X_train.values):
        train_label.append(int(Y_train.values[i]))

    for i, index in enumerate(X_val.values):
        val_label.append(int(Y_val.values[i]))

    return X_train.values, X_val.values, train_label, val_label

def write_to_file(X, label, file_name):
    label=np.expand_dims(np.array(label),1)
    index_label = np.hstack([X, label])
    headers = ['image_id', 'label']

    with open(file_name, 'w', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(index_label)

def test_accuracy(groud_truth_csv, to_check_csv):
    groud_truth = pd.read_csv(groud_truth_csv)
    label_ground = groud_truth["label"]
    idx_ground = groud_truth.drop(labels=["label"], axis=1)

    toCheck = pd.read_csv(to_check_csv)
    label_toCheck = toCheck["label"]
    idx_toCheck = toCheck.drop(labels=["label"], axis=1)

    if idx_toCheck.__len__() != idx_toCheck.__len__():
      print("num error!")
      return

    all_num = idx_toCheck.__len__()
    correct = 0

    for i in range(all_num):
      if idx_ground.values[i] != idx_toCheck.values[i]:
        print("idx error!")
        break
      if label_ground.values[i] == label_toCheck.values[i]:
        correct+=1

    print("acc: %.2f"%(float(correct*100)/float(all_num)))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch AI big homework split')
    parser.add_argument('--train_list_root', type=str, help='path of train image list', default='data/train.csv')
    parser.add_argument('--val_ratio', type=str, help='ratio of train data among all labeled data', default='0.1')
    parser.add_argument('--random_seed', type=str, help='path of test image npy')
    parser.add_argument('--output_file', type=str, help='path of test image npy', default='data/split_')

    args = parser.parse_args()
   
    print("split train val random seed:%d"%(int(args.random_seed)))

    X_train, X_val, train_label, val_label = split_train_val(info_file=args.train_list_root, ratio=float(args.val_ratio), random_seed=int(args.random_seed))
    write_to_file(X_train, train_label, args.output_file+"train.csv")
    write_to_file(X_val, val_label, args.output_file + "val.csv")
    
    #test_accuracy('split_val.csv', 'predict_result.csv')
