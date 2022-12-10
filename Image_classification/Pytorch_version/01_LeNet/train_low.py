import torch
import numpy as np
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
                               )

# ============================================= 【cifar10数据集下载与导入】 ============================================= #
# 50000张训练图像
trainset = torchvision.datasets.CIFAR10(root='E:/Deep_learning/My_code/Dataset',
                                        download=True, train=True, transform=transform)
# 10000张测试图像
testset = torchvision.datasets.CIFAR10(root='E:/Deep_learning/My_code/Dataset',
                                       download=False, train=False, transform=transform)

# ========================================== 【数据分batch_size\num_workers】 ========================================== #
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size=10000,
                                         shuffle=True, num_workers=0)

# ==================================================== 【数据集种类】 ================================================== #
classes = ('plane', 'autom', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ============================================= 【获取一些train\test的图像】 ============================================ #
train_data_iter = iter(trainloader)
train_images, train_labels = next(train_data_iter)

test_data_iter = iter(testloader)
test_images, test_labels = next(test_data_iter)

# =============================================== 【去归一化、转换通道数】 =============================================== #
def imshow(img):
    img = img*(0.5) + 0.5  # 乘标准差，除以均值
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 显示图片
imshow(torchvision.utils.make_grid(train_images,padding=2))
# 打印label
print(' -- '.join(classes[train_labels[i]] for i in range(32)))

# =========================================== 【定义网络模型、损失函数、优化器】 =========================================== #
net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
'''
注意：Pytorchz中的交叉熵损失函数自带softmax，所以书写网络模型的输出时，不需要加上softmax操作
'''

# =================================================== 【开始训练网络】 ================================================== #
for epoch in range(5):
    running_loss = 0
    for step, data in enumerate(trainloader, start=0):
        inputs_img, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        '''
        如果不清楚历史梯度，就会对计算的历史梯度进行累加（通过这个特性你能够变相实现一个很大的batch数值的训练）
        参考链接：https://www.zhihu.com/question/303070254
        '''

        # forward + backward + optimize
        outputs = net(inputs_img)
        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batch
            with torch.no_grad():
                outputs = net(test_images)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1] # [0]指的是batch这一维度；[1]指的是标签
                accuracy = (predict_y == test_labels).sum().item()/test_labels.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss/500, accuracy))
                running_loss = 0.0

print('Bushi - Finished Training')

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)


