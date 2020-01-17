import os
import pandas as pd
import numpy as np
import csv

result_name = 'bagging_result.csv'
num_classes=10
count = 5000
oneHot_result_sum = np.zeros([count, num_classes])
for maindir, subdir, file_name_list in os.walk('./'):
  for filename in file_name_list:
    if filename.split('.')[-1]!='csv' and filename!=result_name:
      continue

    print(filename)
    train = pd.read_csv(filename)
    labels = train["label"].values
    labels_oneHot = np.eye(num_classes)[labels]
    oneHot_result_sum += labels_oneHot

bagging_result = np.argmax(oneHot_result_sum, 1)
indexes = np.array([[i] for i in range(count)])
index_predicted = np.concatenate([indexes, np.reshape(bagging_result, [bagging_result.shape[0], 1])], axis=1)

headers = ['image_id', 'label']

with open('bagging_result.csv', 'w', newline='')as f:
  f_csv = csv.writer(f)
  f_csv.writerow(headers)
  f_csv.writerows(index_predicted)

