import torch.nn as nn
import torch

def drop_path(x, drop_prob: float = 0.2, training: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    # 如果drop_prob == 0. 以及training = False时：则直接舍弃带有卷积的通道（直接等返回恒等映射x，也就是输入）
    # 如果drop_prob ≠ 0以及training = True时，
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor  # x.div(keep_prob) :x中的每个元素除以keep_prob
    return output



x = torch.rand((3, 4, 4))
drop_prob = 0.2
train = True
a = drop_path(x)
print(a)
