import json
import os
import torch
import time
import torch
import  torch.nn as nn
import numpy as np
from torch.utils import data
from model import Alexnet
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, datasets,utils



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.229, 0.224, 0.225),(0.485, 0.456, 0.406))]),
    "val":   transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.229, 0.224, 0.225),(0.485, 0.456, 0.406))])
                }

data_root = os.path.abspath(os.path.join(os.getcwd(),"../../.."))
print(data_root)
image_path = (data_root + "/Dataset/flower_data").replace("\\",'/')   # replace("\\",'/')为了把路径中的 \ 换成 /
print(image_path)

train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())  # items():以列表返回可遍历的(键, 值) 元组数组。

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices,json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=0)
validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = next(test_data_iter)
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = Alexnet(num_classes=5, init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())  # For debugging and viewing model parameters
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
best_acc = 0.0
for epoch in range(10):
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}-->{}]{:.3f}".format(int(rate*100),a,b,loss))
    print()
    print(time.perf_counter()-t1)

    # validate
    net.eval()
    predict_true_nums = 0.0   # accumulate accurate number / epoch
    with torch.no_grad():   # Disable model parameter updates
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            predict_true_nums += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = predict_true_nums / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[Epoch %d] train_loss: %.3f test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, accurate_test))

print('Finished Train')
