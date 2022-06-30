'''
PACT and uniform quantized resnets
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from module import ActFn, Conv2d, Linear
from torch.autograd import Variable
from pact_utils import QuantizedConv2d, PactReLU, QuantizedLinear

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockPACT(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, wbits=8, abits=8, expansion=1):
        super(BasicBlockPACT, self).__init__()
        self.abits = abits
        self.expansion = expansion

        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bitwidth = wbits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.alpha1 = nn.Parameter(torch.tensor(10.))

        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, bitwidth = wbits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.alpha2 = nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply
        
        if stride != 1 or in_planes != planes:
              # original resnet shortcut
              self.shortcut = nn.Sequential(
                    Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, bitwidth = wbits),
                    nn.BatchNorm2d(self.expansion * planes)
              )
        else: # nothing done if stride or inplanes do not differ
          self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.ActFn(self.bn1(self.conv1(x)), self.alpha1, self.abits)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.ActFn(out, self.alpha2, self.abits)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, wbits=4, abits=4, pact=False, expansion=1):
        super(BasicBlock, self).__init__()
        
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = PactReLU() if pact else nn.ReLU()        
        
        if stride != 1 or in_planes != planes:
              # original resnet shortcut
              self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,),
                    nn.BatchNorm2d(self.expansion * planes)
              )
        else: # nothing done if stride or inplanes do not differ
          self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BasicBlockUQ(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, wbits=4, abits=4, pact=False, expansion=1):
        super(BasicBlockUQ, self).__init__()
        
        self.expansion = expansion
        self.conv1 = QuantizedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, wbits=wbits, abits=abits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, wbits=wbits, abits=abits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = PactReLU() if pact else nn.ReLU()        
        
        if stride != 1 or in_planes != planes:
              # original resnet shortcut
              self.shortcut = nn.Sequential(
                    QuantizedConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, wbits=wbits, abits=abits),
                    nn.BatchNorm2d(self.expansion * planes)
              )
        else: # nothing done if stride or inplanes do not differ
          self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, wbits=4, abits=4, pact=False, ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class BottleneckUQ(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, wbits=4, abits=4, pact=False):
        super(BottleneckUQ, self).__init__()
        self.conv1 = QuantizedConv2d(in_planes, planes, kernel_size=1, bias=False, wbits=wbits, abits=abits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, wbits=wbits, abits=abits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuantizedConv2d(planes, self.expansion*planes, kernel_size=1, bias=False, wbits=wbits, abits=abits)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuantizedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, wbits=wbits, abits=abits),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, abits=8, wbits=8, pact=False, shallow=True, quant=True, expansion=1):
        """
        shallow = True follows the resnet in pact and various works.
        shallow = False is the standard resnet implementation.
        """
        super(ResNet, self).__init__()
        self.abits = abits
        self.wbits = wbits
        self.shallow = shallow
        self.relu = nn.ReLU()

        if shallow:
            self.in_planes = 16
            if quant:
                self.conv1 = QuantizedConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, abits=8, wbits=8)
            else:
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, expansion=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, expansion=1)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, expansion=1)
            if quant:
                self.linear = QuantizedLinear(64*expansion, num_classes, abits=abits, wbits=wbits)
            else:
                self.linear = nn.Linear(64, num_classes)
        else:
            self.in_planes = 64
            if quant:
                self.conv1 = QuantizedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, abits=8, wbits=8)
            else:
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, expansion=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, expansion=1)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, expansion=1)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, expansion=1)
            if quant:
                self.linear = QuantizedLinear(512*expansion, num_classes, abits=abits, wbits=wbits)
            else:
                self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, expansion):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.wbits, self.abits, expansion))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if not self.shallow:
            out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(abits, wbits, pact=False, shallow=True, noquant=False):
    print("abit/wbit:", abits, wbits)
    if shallow:
        blocks = [3,3,3]
    else:
        blocks = [2,2,2,2]
    if noquant:
        return ResNet(BasicBlock, blocks, abits=abits, wbits=wbits, shallow=shallow, quant=False)
    else:
        if not pact:
            return ResNet(BasicBlockUQ, blocks, abits=abits, wbits=wbits, shallow=shallow)
        else:
            return ResNet(BasicBlockPACT, blocks, abits=abits, wbits=wbits, shallow=shallow)

def resnet34(abits, wbits, pact=False, shallow=True, noquant=False):
    print("abit/wbit:", abits, wbits)
    if shallow:
        blocks = [6,6,6]
    else:
        blocks = [3,4,6,3]
    if noquant:
        return ResNet(BasicBlock, blocks, abits=abits, wbits=wbits, shallow=shallow, quant=False)
    else:
        if not pact:
            return ResNet(BasicBlockUQ, blocks, abits=abits, wbits=wbits, shallow=shallow)
        else:
            return ResNet(BasicBlockPACT, blocks, abits=abits, wbits=wbits, shallow=shallow)

def resnet50(abits, wbits, pact=False, shallow=True, noquant=False):
    print("abit/wbit:", abits, wbits)
    if shallow:
        blocks = [6,6,6]
    else:
        blocks = [3,4,6,3]
    if noquant:
        return ResNet(Bottleneck, blocks, abits=abits, wbits=wbits, shallow=shallow, quant=False, expansion=4)
    else:
        if not pact:
            return ResNet(BottleneckUQ, blocks, abits=abits, wbits=wbits, shallow=shallow, expansion=4)
        else:
            return ResNet(BottleneckUQ, blocks, abits=abits, wbits=wbits, shallow=shallow, expansion=4)

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