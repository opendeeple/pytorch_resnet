from nn.resnet import ResNet
from nn.block import BasicBlock, BottleneckBlock as Bottleneck

layers = {
  'resnet18' : [2, 2,  2, 2],
  'resnet34' : [3, 4,  6, 3],
  'resnet50' : [3, 4,  6, 3],
  'resnet101': [3, 4, 23, 3],
  'resnet152': [3, 8, 36, 3]
}

def build(arch, pretrained=False, progress=True, **kwargs):
  if arch in ['resnet18', 'resnet34']:
    block = BasicBlock
  else:
    block = Bottleneck
  layers  = layers.get(arch)
  model   = ResNet(block, layers, **kwargs)
  if pretrained:
    pass
  return model

def ResNet18(pretrained=False, progress=True, **kwargs):
  return build('resnet18', pretrained=pretrained, progress=progress, **kwargs)

def ResNet34(pretrained=False, progress=True, **kwargs):
  return build('resnet34', pretrained=pretrained, progress=progress, **kwargs)

def ResNet50(pretrained=False, progress=True, **kwargs):
  return build('resnet50', pretrained=pretrained, progress=progress, **kwargs)

def ResNet101(pretrained=False, progress=True, **kwargs):
  return build('resnet101', pretrained=pretrained, progress=progress, **kwargs)

def ResNet152(pretrained=False, progress=True, **kwargs):
  return build('resnet152', pretrained=pretrained, progress=progress, **kwargs)
