import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

MED_FILTER=False
PIC_SIZE = 28

class AgeEstimationDataset(Dataset):
    """Face dataset."""

    def __init__(self, labels_csv, pics, category, transform=None):
        if labels_csv != None:
            train = pd.read_csv(labels_csv)
            labels = train["label"]
            indexes = train.drop(labels=["label"], axis=1).values
            self.data_num = labels.__len__()
            labels = labels.values
        else:
            self.data_num = pics.__len__()
            labels = [-1 for i in range(self.data_num)]
            indexes = [i for i in range(self.data_num)]

        self.all_pics = pics
        self.img_indexes = indexes
        self.labels = labels
        self.transform = transform
        self.category = category

        print('data catagory:', category)
        print('len of img:', self.data_num)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        img_index = self.img_indexes[idx]
        image = torch.reshape(self.all_pics[img_index], (PIC_SIZE, PIC_SIZE))

        if MED_FILTER==True:
          image_med = cv2.medianBlur(np.array(image), 3) # medfilter, 
          image_PIL = transforms.ToPILImage()(image_med)
        else:
          image_PIL = transforms.ToPILImage()(image)

        label = self.labels[idx]

        image_t=self.transform(image_PIL)

        return image_t, label, img_index

# compute the mean and std of data
def mean_std(pic_root):
  data = np.load(pic_root)/255.0
  mean = data.mean()
  std = data.std()
  print("mean:", mean)
  print("std:", std)

def load_data(train_bath_size, args, RANDOM_SEED, val_batch_size):
    train_list_root=args.train_list_root
    val_list_root=args.val_list_root
    train_pic_root=args.train_pic_root
    test_pic_root=args.test_pic_root

    all_trainVal_pics = torch.Tensor(np.load(train_pic_root))

    test_pics = torch.Tensor(np.load(test_pic_root))
    transform_train=transforms.Compose([#transforms.Resize([128,128]),
                                      #transforms.RandomCrop([28,28]),
                                      transforms.RandomCrop(28, padding=2),
                                      transforms.RandomRotation(15),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.1307],[0.3081])
                                      transforms.Normalize([0.2926],[0.3343])
                                                     ])
    transform_test_val = transforms.Compose([#transforms.Resize([128,128]),
                                      #transforms.CenterCrop([28,28]),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.1307],[0.3081])
                                      transforms.Normalize([0.2926],[0.3343])
     ])

    transformed_train_dataset = AgeEstimationDataset(labels_csv=train_list_root,
                                                     pics=all_trainVal_pics,
                                                     category='train',
                                                     transform=transform_train)

    transformed_val_dataset = AgeEstimationDataset(labels_csv=val_list_root,
                                                   pics=all_trainVal_pics,
                                                   category='val',
                                                   transform=transform_test_val)

    transformed_test_dataset = AgeEstimationDataset(labels_csv=None,
                                                     pics=test_pics,
                                                     category='test',
                                                     transform=transform_test_val)

    # Loading dataset into dataloader
    train_loader = DataLoader(transformed_train_dataset, batch_size=train_bath_size,
                              shuffle=True, num_workers=4, worker_init_fn=np.random.seed(RANDOM_SEED))

    

    test_loader = DataLoader(transformed_test_dataset, batch_size=val_batch_size,
                            shuffle=False, num_workers=4, worker_init_fn=np.random.seed(RANDOM_SEED))


    val_loader = DataLoader(transformed_val_dataset, batch_size=val_batch_size,
                             shuffle=False, num_workers=4, worker_init_fn=np.random.seed(RANDOM_SEED))

    return train_loader, val_loader, test_loader
