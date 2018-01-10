# encoding:utf-8

# 加载自己的数据集
# 数据集含有自己标签文件

import os
from torch.utils.data import Dataset
from PIL import Image


# 用PIL读取图片
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


class MyImageFolder(Dataset):
    def __init__(self, img_root_path, label_path, dataset='', data_transforms=None, loader=default_loader):
        with open(label_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_root_path, line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label

