from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path):
        self.images_path = images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        print("self.images_path[item] = {}".format(self.images_path[item]))
        return self.images_path[item]

    @staticmethod
    def collate_fn(batch):
        # batch 就是继承__getitem__的return
        print('batch 就是继承__getitem__的return \nbatch = {}'.format(batch))
        '''
        collate_fn的用处:
        1、自定义数据堆叠过程
        2、自定义batch数据的输出形式
        3、输入输出分别域getitem函数和loader调用时对应
        '''
        real_batch_array = np.array(batch)
        real_batch_torch = torch.from_numpy(real_batch_array)
        return real_batch_array, real_batch_torch

a = np.random.rand(4,3)
print("a = \n{}".format(a))
dataset = MyDataSet(a)
print("\nlen(dataset) \n{}".format(len(dataset)))

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
images = iter(dataloader)
print("\n next(images) \n{}".format(next(images)))
print("源$ 【猛】 $仔 - "*5)
# print(next(images))