import torch.nn as nn
from nn.layers import conv3x3, conv1x1

class BottleneckBlock(nn.Module):
  expansion = 4
  def __init__(self, inplanes, planes, stride=1, downsample=None,
      groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d):
    super(BottleneckBlock, self).__init__()
    width           = int(planes * (base_width / 64.)) * groups
    self.conv1      = conv1x1(inplanes, width)
    self.bn1        = norm_layer(width)
    self.conv2      = conv3x3(width, width, stride, groups, dilation)
    self.bn2        = norm_layer(width)
    self.conv3      = conv1x1(width, planes * self.expansion)
    self.bn3        = norm_layer(planes * self.expansion)
    self.relu       = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride     = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    
    out += identity
    return self.relu(out)
