import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):

        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU     # SiLU alias Swish (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return  result

class SE(nn.Module):
    def __init__(self,
                 input_c: int,  # MBConv block input channel
                 expand_c: int, # block expand channel
                 se_ratio: float = 0.25
                 ):
        super(SE, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)    # 全局平均池化操作。input[B, C, H, W] output[B, C, 1, 1]
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x
'''
SE模块：为什么用两个卷积层替代全连接层。因为因为使用全连接层和卷积上没有区别，所以这里可能是卷积底层的优化比较好。
'''
class MBConv(nn.Module):
    def __int__(self,
                kernel_size: int,
                input_c: int,
                out_c: int,
                expand_ratio: int,
                stride: int,
                se_ratio: float,
                drop_rate: float,
                norm_layer: Callable[..., nn.Module]):
        super(MBConv,self).__init__()

        # stride只有1和2两种
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中， MBConv中不存在expansion=1情况，所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SE(input_c,expanded_c,se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity) # 注意MBConv中第三个1*1卷积后面是没有激活函数，所以传入Identity
        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用DropPath层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x):
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result

class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio:float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1,2]
        assert  se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution(expand_ratio = 1时)
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数

        else:
            # 当只有project_conv时的情况(expand_ratio ≠ 1时)
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=1,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意没有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时，才使用dropout
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x):
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result

class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 0

        # partial(): 把nn.BatchNorm2d函数中的eps、momentum两个参数固定住，返回一个新的函数norm_layer
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,   # drop path的rate是随着block_id的增加而成线性变换的
                                 norm_layer=norm_layer))
                block_id += 1

        self.blocks = nn.Sequential(*blocks)

        head_input_c = model_cnf[-1][-3]
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            # 这里指的是正常的dropout舍弃全连接层一些元素的那种，并不是前面的Drop paths (Stochastic Depth)
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features,num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x

def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator:0代表Fused-MBConv；1代表MBConv, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2)
    return model

def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model

'''
nn.Sequential()介绍参考如下链接：
https://blog.csdn.net/Just_do_myself/article/details/124195393?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167126018716800182787837%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167126018716800182787837&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124195393-null-null.142^v68^control,201^v4^add_ask,213^v2^t3_control1&utm_term=nn.Sequential&spm=1018.2226.3001.4187

一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。
除此之外，一个包含神经网络模块的OrderedDict也可以被传入nn.Sequential()容器中。利用nn.Sequential()搭建好模型架构，
模型前向传播时调用forward()方法，模型接收的输入首先被传入nn.Sequential()包含的第一个网络模块中。
然后，第一个网络模块的输出传入第二个网络模块作为输入，按照顺序依次计算并传播，直到nn.Sequential()里的最后一个模块输出结果。
'''