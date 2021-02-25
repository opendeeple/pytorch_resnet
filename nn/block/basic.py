import torch.nn as nn
from nn.layers import conv3x3

class BasicBlock(nn.Module):
  def __init__(self, inplanes, planes, stride=1, 
      downsample=None, norm_layer=nn.BatchNorm2d):
    super(BasicBlock, self).__init__()
    self.conv1      = conv3x3(inplanes, planes, stride)
    self.bn1        = norm_layer(planes)
    self.relu       = nn.ReLu(inplane=True)
    self.conv2      = conv3x3(planes, planes)
    self.downsample = downsample
    self.stride     = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    
    out += identity
    return self.relu(out)
