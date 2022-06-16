# https://openreview.net/forum?id=TVHS5Y4dNvM

import torch.nn as nn
from module import ActFn, Conv2d, Linear

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
 nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
 nn.GELU(),
 nn.BatchNorm2d(dim),
 *[nn.Sequential(
 Residual(nn.Sequential(
 Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
 nn.GELU(),
 nn.BatchNorm2d(dim)
 )),
 Conv2d(dim, dim, kernel_size=1),
 nn.GELU(),
 nn.BatchNorm2d(dim)
 ) for i in range(depth)],
 nn.AdaptiveAvgPool2d((1,1)),
 nn.Flatten(),
 Linear(dim, n_classes)
)