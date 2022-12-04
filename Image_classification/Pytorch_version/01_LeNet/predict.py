import matplotlib.pyplot as plt
import torch
from PIL import Image
from model import  LeNet
import  torchvision.transforms as transforms

transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

classes = ('plane', 'autom', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

img = Image.open('./data/cat.jpg')
plt.imshow(img)
plt.show()

img = transform(img)   # [C, H, W]
img = torch.unsqueeze(img, dim=0)  # [B, C, H, W] = [1, C, H, W]

# with torch.no_grad():
#     outputs = net(img)  # [N, 10]
#     predict = torch.max(outputs, dim=1)[1].data.numpy()   # 返回最大元素的索引，并由tensor转化为array类型
# print(classes[int(predict)])

with torch.no_grad():
    outputs = net(img)  # [N, 10]
    predict = torch.softmax(outputs, dim=1)
print(predict)