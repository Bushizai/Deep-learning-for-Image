import torch
from model import  Alexnet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.229, 0.224, 0.225),(0.485, 0.456, 0.406))])

# load image
img = Image.open("sunflowers.jpg")
plt.imshow(img)
# [B, C, H, W]
img = data_transform(img)
# Expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices,json','r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = Alexnet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))      # output[32, 5]
    predict = torch.softmax(output, dim=0)  # 按行进行softmax
    predict_index = torch.argmax(predict).numpy()
print(class_indict[str(predict_index)], predict[predict_index].item())
plt.show()

