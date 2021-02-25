import torch.nn as nn

def conv3x3(in_planes, out_planes, stride, groups, dilation):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
