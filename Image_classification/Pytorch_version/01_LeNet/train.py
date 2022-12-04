import torch
import torchvision
import numpy as np
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# ================================================== 【数据预处理操作】 ================================================= #
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

# ============================================== 【cifar10数据集下载与导入】 ============================================= #
trainset = torchvision.datasets.CIFAR10(root="E:/Deep_learning/My_code/Image_classification/Pytorch_version/01_LeNet/data",
                                        download=True,train=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="E:/Deep_learning/My_code/Image_classification/Pytorch_version/01_LeNet/data",
                                        download=False,train=False, transform=transform)

# ==================================================== 【加载数据集】 =================================================== #
trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,
                                           shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,
                                         shuffle=False,num_workers=0)

# ==================================================== 【数据集种类】 =================================================== #
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ============================================= 【获取一些train\test的图像】 ============================================= #
train_data_iter = iter(trainloader)
train_images, train_labels = next(train_data_iter)

test_data_iter = iter(testloader)
test_images, test_labels = next(test_data_iter)

# ============================================== 【tensorboard界面可视化】 ============================================== #
writer = SummaryWriter('E:/Deep_learning/My_code/Image_classification/Pytorch_version/01_LeNet/summary/fashion_cifar10_experiment_1')
# create grid of images
img_grid = torchvision.utils.make_grid(train_images)
# write to tensorboard
writer.add_image('batch_fashion_cifar10_images', img_grid)
writer.close()
'''
在终端输入：tensorboard --logdir=E:/Deep_learning/My_code/Image_classification/Pytorch_version/01_LeNet/summary
记住从这里打开tensorboard网页，下面关于在tensorboard中显示的操作，运行到后会自动在打开的网页更新出来。
TensorboardX必须writer.close() 才能把缓存中保存的数据写到目标events文件中，一旦训练中断没有close，则你的保存目录中不会有相应的数据
'''

# ====================================== 【helper function to show an image】 ======================================== #
# ===================================== 【由Tensor(B,C,H,W)转化为Numpy(B,H,W,C)】 ====================================== #
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize(乘以标准差加上均值)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")  # 开’（注释标号，使用时，标号相同的都要Ctrl+？）
        # plt.show() # 开’
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 开‘
        # plt.show() # 开‘

# show images
# print(''.join('%10s' % classes[test_labels[j]] for j in range(4)))  # 开’
# matplotlib_imshow(img_grid, one_channel=False)  # 开‘

# ==================================================== 【网络模型】 ==================================================== #
net = LeNet()

# =============================================== 【定义损失函数和优化器】 ================================================ #
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ================================================= 【显示网络整体结构】 ================================================= #
writer.add_graph(net, train_images)

# =============================================== 【定义损失函数和优化器】 ================================================ #
# get the class labels for each image
class_labels = [classes[lab] for lab in train_labels]

# log embeddings
features = train_images.view(-1, 3*32*32)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=train_images)
writer.close()


# ========================================== 【使用 TensorBoard 跟踪模型训练】 =========================================== #
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

# ================================================== 【开始训练模型】 =================================================== #
running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)  # 自带求softmax
        loss.backward()
        optimizer.step()   # 更新所有参数

        running_loss += loss.item()
        if i % 100 == 99:    # every 100 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')

# ========================================== 【使用 TensorBoard 评估经过训练的模型】 =========================================== #
# 1. gets the probability predictions in a test_size x num_classes Tensor
# 2. gets the preds in a test_size Tensor
# takes ~10 seconds to run
class_probs = []
class_label = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]

        class_probs.append(class_probs_batch)
        class_label.append(labels)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_label = torch.cat(class_label)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    # Adds precision recall curve
    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step)
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_label)

labels = np.random.randint(2, size=100)  # binary label
predictions = np.random.rand(100)
writer = SummaryWriter()
writer.add_pr_curve('pr_curve_1', labels, predictions, 0)
writer.close()
print("结束")
