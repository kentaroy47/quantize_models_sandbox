'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from module import ActFn, Conv2d, Linear
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet34', "resnet50"]

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k=8, expansion=1, snr=0.1, inference=False, res50=False):
        super(BasicBlock, self).__init__()
        self.k = k
        self.expansion = expansion
        self.res50 = res50

        if not res50:
            self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bitwidth = k, noise=snr, inference=inference)
            self.bn1 = nn.BatchNorm2d(planes)           
            self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, bitwidth = k, noise=snr, inference=inference)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            # 1x1, 3x3, 1x1 block
            outchannel = planes
            inchannel = in_planes
            channel = outchannel//4            
            # 1x1 conv
            self.conv1 = nn.Conv2d(inchannel, channel, 1)
            self.bn1 = nn.BatchNorm2d(channel)
            self.relu1 = nn.ReLU(True)
            # 3x3 conv
            self.conv2 = nn.Conv2d(channel, channel, 3, padding=1, stride=stride,)
            self.bn2 = nn.BatchNorm2d(channel)
            self.relu2 = nn.ReLU(True)
            # 1x1 conv
            self.conv3 = nn.Conv2d(channel, outchannel, 1)
            self.bn3 = nn.BatchNorm2d(outchannel)
            self.relu3 = nn.ReLU(True)

        # PACT
        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.alpha2 = nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply
        self.snr = snr

        if stride != 1 or in_planes != planes:
              # original resnet shortcut
              self.shortcut = nn.Sequential(
                    # nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
              )
        else: # nothing done if stride or inplanes do not differ
          self.shortcut = nn.Sequential()

    def forward(self, x):
        if not self.res50:
            out = self.ActFn(self.bn1(self.conv1(x)), self.alpha1, self.k)
            out = self.bn2(self.conv2(out))
        else:
            # 1x1
            out = self.relu1(self.bn1(self.conv1(x)))
            # 3x3
            out = self.relu2(self.bn2(self.conv2(out)))
            # 1x1
            out = self.bn3(self.conv3(out))
        # residue
        out += self.shortcut(x)
        out = self.ActFn(out, self.alpha2, self.k)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, K=8, snr=0, inference=False, conv1_noise=True, linear_noise=True, res50=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.k = K
        self.snr = snr
        self.inference = inference
        self.res50 = res50

        # 1st layers
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, bitwidth = 8, noise=snr*float(conv1_noise), inference=inference)
        self.bn1 = nn.BatchNorm2d(64)
        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply

        # Blocks
        if not res50:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, expansion=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, expansion=1)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, expansion=1)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, expansion=1)
        else:
            self.layer1 = self._make_layer(block, 256, num_blocks[0], stride=1, expansion=1)
            self.layer2 = self._make_layer(block, 512, num_blocks[1], stride=2, expansion=1)
            self.layer3 = self._make_layer(block, 1024, num_blocks[2], stride=2, expansion=1)
            self.layer4 = self._make_layer(block, 2048, num_blocks[3], stride=2, expansion=1)

        # FCs
        if not res50:
            self.linear = Linear(512, num_classes, bitwidth = 8, noise=snr*float(linear_noise), inference=inference)
        else:
            self.linear = Linear(2048, num_classes, bitwidth = 8, noise=snr*float(linear_noise), inference=inference)
        self.apply(_weights_init)       

    def _make_layer(self, block, planes, num_blocks, stride, expansion):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k, expansion, self.snr, inference=self.inference, res50=self.res50))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.ActFn(self.bn1(self.conv1(x)), self.alpha1, self.k)
        # layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(k=8, snr=0, inference=False, conv1_noise=True, linear_noise=True):
    print("bit width:", k)
    return ResNet(BasicBlock, [2, 2, 2, 2], K=k, snr=snr, inference=inference, conv1_noise=conv1_noise, linear_noise=linear_noise)

def resnet34(k=8, snr=0, inference=False, conv1_noise=True, linear_noise=True):
    print("bit width:", k)
    return ResNet(BasicBlock, [3, 4, 6, 3], K=k, snr=snr, inference=inference, conv1_noise=conv1_noise, linear_noise=linear_noise)

def resnet50(k=8, snr=0, inference=False, conv1_noise=True, linear_noise=True):
    print("bit width:", k)
    return ResNet(BasicBlock, [3, 4, 6, 3], K=k, snr=snr, inference=inference, conv1_noise=conv1_noise, linear_noise=linear_noise, res50=True)

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
