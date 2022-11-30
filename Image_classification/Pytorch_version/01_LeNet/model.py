import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def init(self):
        super(LeNet, self).__init__()
