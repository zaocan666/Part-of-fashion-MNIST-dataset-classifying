import torch
import numpy as np
import torchvision.transforms as transforms
import csv
import cv2

PIC_SIZE = 28

def convert2pic(path, out_path, info_file):
    pics = torch.Tensor(np.load(path))

    with open(info_file, 'r') as fin:
        f_csv = csv.reader(fin)
        for i, line in enumerate(f_csv):
            if i==0: #列首
                continue
            pic = pics[int(line[0])]
            label = int((line[1]))

            image = torch.reshape(pic, (PIC_SIZE, PIC_SIZE))
            med_filt_img = cv2.medianBlur(np.array(image), 3)

            #image_PIL = transforms.ToPILImage()(image)
            image_med_PIL = transforms.ToPILImage()(torch.Tensor(med_filt_img))
            #image_PIL.show()
            #image_med_PIL.show()
            image_med_PIL.save(out_path + '/%d_%d.jpg' % (label, i))

def convert2pic_noLabel(path, out_path):
    pics = torch.Tensor(np.load(path))

    for i in range(pics.shape[0]):
        pic = pics[i]

        image = torch.reshape(pic, (PIC_SIZE, PIC_SIZE))
        med_filt_img = cv2.medianBlur(np.array(image), 3)
        #image_PIL = transforms.ToPILImage()(image)
        image_med_PIL = transforms.ToPILImage()(torch.Tensor(med_filt_img))
        image_med_PIL.save(out_path + '/%d.jpg' % (i))

if __name__ == '__main__':
    #convert2pic('train.npy', 'train_med_image', 'train.csv')
    convert2pic_noLabel('test.npy', 'test_med_image')