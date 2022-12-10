import torch
import torch.nn as nn
import torch.nn.functional as F

class Alexnet(nn.Module):
    def __init__(self,num_classes=1000, init_weights=False):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(

            # two kernel:3*3 = one kernel:11*11
            # nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=0),      # input[3,224,224] output[48,55,55]
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # output[48,27,27]

            # two kernel:3*3 = one kernel:5*5
            # nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),  # output[128,27,27]
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # output[128,13,13]

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),   # output[193,13,13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),   # output[128,13,13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                      # output[128,6,6]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():   # 这里的self.modules()是继承了nn.Module这个父类
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)    # 或写成m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


