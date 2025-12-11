import os
import sys
from pathlib import Path
import torch
import time
import torchvision
import math
import glob
import cv2
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch import Tensor
from tinygrad import nn as tiny_nn, Tensor as tiny_Tensor
from tinygrad.helpers import fetch

TORCH_1_10 = False
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np


class tiny_Sequential():
    def __init__(self): self.list = []
    def append(self,x): self.list.append(x)
    def __call__(self, x):
      for i in range(len(self.list)):
        if type(self.list[i]) == tiny_nn.Conv2d: x = tiny_Tensor(x.detach().numpy())
        x = self.list[i](x)
        if type(self.list[i]) == tiny_nn.Conv2d: x = Tensor(x.numpy())
      return x
    def __len__(self): return len(self.list)
    def __setitem__(self, key, value): self.list[key] = value
    def __getitem__(self, idx):
      return self.list[idx]


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True): super().__init__()
    def forward_fuse(self, x): return

class tiny_Conv():
    def __init__(self):
        super().__init__()
        return
    def __call__(self, x):
      tiny_x = x
      if type(tiny_x) != tiny_Tensor: tiny_x = tiny_Tensor(tiny_x.detach().numpy())
      tiny_x = self.tiny_conv(tiny_x)
      tiny_x = tiny_x.silu()
      return Tensor(tiny_x.numpy())
    
class ADown(nn.Module): pass

class tiny_ADown():
    def __init__(self, c1=1, c2=1): super().__init__()
    def __call__(self, x):
      if type(x) != tiny_Tensor: x = tiny_Tensor(x.detach().numpy())
      x = tiny_Tensor.avg_pool2d(x, 2, 1, 1, 0, False, True)
      x1,x2 = x.chunk(2, 1)
      x1 = self.cv1(x1)
      x1 = tiny_Tensor(x1.detach().numpy())
      x2 = tiny_Tensor.max_pool2d(x2, kernel_size=3, stride=2, dilation=1, padding=1)
      x2 = self.cv2(x2)
      x2 = tiny_Tensor(x2.detach().numpy())
      x = tiny_Tensor.cat(x1, x2, dim=1)
      return Tensor(x.numpy())

class AConv(nn.Module): pass

class tiny_AConv():
    def __init__(self):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()

    def __call__(self, x):
        x = tiny_Tensor(x.detach().numpy())
        x = tiny_Tensor.avg_pool2d(x, kernel_size=2, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        return self.cv1(x)

class ELAN1(nn.Module): pass

class tiny_ELAN1():
    def __init__(self, c1=1, c2=1, c3=1, c4=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()

    def __call__(self, x):
      if type(x) != tiny_Tensor: x = tiny_Tensor(x.detach().numpy())
      y = self.cv1(x)
      y = tiny_Tensor(y.detach().numpy())
      y = y.chunk(2,1)
      y = list(y)
      y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
      y[-1] = tiny_Tensor(y[-1].detach().numpy())
      y[-2] = tiny_Tensor(y[-2].detach().numpy())
      y = tiny_Tensor.cat(y[0], y[1], y[2], y[3], dim=1)
      y = self.cv4(y)
      #y = Tensor(y.numpy())
      return y

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class RepNBottleneck(nn.Module): pass
    
class tiny_RepNBottleneck():
    # Standard bottleneck
    def __init__(self):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()

    def __call__(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class RepConvN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_fuse(self, x):
        exit()

    def forward(self, x):
        exit()


class RepNCSP(nn.Module): pass

class tiny_RepNCSP():
    # CSP Bottleneck with 3 convolutions
    def __init__(self):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()

    def __call__(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class RepNCSPELAN4(nn.Module): pass

class tiny_RepNCSPELAN4():
    # csp-elan
    def __init__(self, c1=1, c2=1, c3=1, c4=1, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()

    def __call__(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class SP(nn.Module): pass

class tiny_SP():
    def __init__(self, k=3, s=1):
        super(tiny_SP, self).__init__()
        self.k = k
        self.s = s

    def __call__(self, x):
        x = tiny_Tensor(x.detach().numpy())
        x = tiny_Tensor.max_pool2d(x, self.k, self.s, dilation=1, padding=self.k//2)
        return Tensor(x.numpy())

class SPPELAN(nn.Module): pass

class tiny_SPPELAN():
    # spp-elan
    def __init__(self):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()

    def __call__(self, x):
        if type(x) != tiny_Tensor: x = tiny_Tensor(x.detach().numpy())
        y = [self.cv1(x)]
        y.append(self.cv2(y[-1]))
        y.append(self.cv3(y[-1]))
        y.append(self.cv4(y[-1]))
        for i in range(len(y)): y[i] = tiny_Tensor(y[i].detach().numpy())
        y = tiny_Tensor.cat(*y, dim=1)
        y = self.cv5(y)
        return y

class Concat(nn.Module): pass

class tiny_Concat():
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()

    def __call__(self, x):
      if type(x[0]) != tiny_Tensor: x[0] = tiny_Tensor(x[0].detach().numpy())
      if type(x[1]) != tiny_Tensor: x[1] = tiny_Tensor(x[1].detach().numpy())
      y = tiny_Tensor.cat(x[0],x[1],dim=self.d)
      return Tensor(y.numpy())


class DDetect(nn.Module): pass

class tiny_DDetect():
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        return
    
    def __call__(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

class DFL(nn.Module): pass

class CBLinear(nn.Module): pass

class tiny_CBLinear():
    def __init__(self):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(tiny_CBLinear, self).__init__()
        return

    def __call__(self, x):
        if type(x) != tiny_Tensor: x = tiny_Tensor(x.detach().numpy())
        x = self.conv(x)
        outs = x.split(self.c2s, dim=1)
        outs = list(outs)
        for i in range(len(outs)): outs[i] = Tensor(outs[i].numpy())
        return tuple(outs)

class CBFuse(nn.Module): pass

class tiny_CBFuse():
    def __init__(self):
        super(tiny_CBFuse, self).__init__()

    def __call__(self, xs):
        for i in range(len(xs)):
          if type(xs[i]) == tuple:
            xs[i] = list(xs[i])
            for j in range(len(xs[i])):
              if type(xs[i][j]) != tiny_Tensor: xs[i][j] = tiny_Tensor(xs[i][j].detach().numpy())
            xs[i] = tuple(xs[i])
          else:
            xs[i] = tiny_Tensor(xs[i].detach().numpy())
        target_size = xs[-1].shape[2:]
        res = []
        for i, x in enumerate(xs[:-1]):
            tensor_to_upsample = x[self.idx[i]]
            upsampled = tiny_Tensor.interpolate(tiny_Tensor(tensor_to_upsample.detach().numpy()), size=target_size, mode='nearest')
            res.append(upsampled)
        
        res += xs[-1:]

        y = tiny_Tensor.stack(*res)
        out = y.sum(0)
        return Tensor(out.numpy())

  #def interpolate(self, size:tuple[int, ...], mode:str="linear", align_corners:bool=False) -> Tensor:

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class DFL(nn.Module): pass

class tiny_DFL():
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        # self.bn = nn.BatchNorm2d(4)

    def __call__(self, x):
        if type(x) != tiny_Tensor: x = tiny_Tensor(x.detach().numpy())
        b, _, a = x.shape  # batch, channels, anchors
        x = x.view(b, 4, self.c1, a)
        x = self.conv(x.transpose(2, 1).softmax(1)).view(b, 4, a)
        return Tensor(x.numpy())


class tiny_Upsample(): # nearest for now
  def __init__(self):
      super().__init__()
  
  def __call__(self, x):
    if type(x) != tiny_Tensor: x = tiny_Tensor(x.detach().numpy())
    N, C, H, W = x.shape
    s = self.scale_factor
    x = x.repeat_interleave(s, dim=2).repeat_interleave(s, dim=3)
    return Tensor(x.numpy())

class Silence():
    def __init__(self):
        super(Silence, self).__init__()
    def __call__(self, x):    
        return x

class DetectionModel(nn.Module): pass

class tiny_DetectionModel():
  def convert(self):
    for i in range(len(self.model)):
      m = self.model[i]
      if type(m) == Conv:
        tiny = tiny_Conv()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks


        tiny.conv = tiny_nn.Conv2d(m.conv.in_channels, m.conv.out_channels, m.conv.kernel_size, m.conv.stride, m.conv.padding, m.conv.dilation, m.conv.groups, True if m.conv.bias is not None else False)
        tiny.conv.weight = m.conv.weight
        tiny.conv.bias = m.conv.bias

        tiny.tiny_conv = tiny_nn.Conv2d(m.conv.in_channels, m.conv.out_channels, m.conv.kernel_size, m.conv.stride, m.conv.padding, m.conv.dilation, m.conv.groups, True if m.conv.bias is not None else False)
        tiny.tiny_conv.weight = tiny_Tensor(m.conv.weight.detach().numpy().copy())
        tiny.tiny_conv.bias = tiny_Tensor(m.conv.bias.detach().numpy().copy())
        self.model[i] = tiny
        m = self.model[i]
      elif type(m) == AConv:
        tiny = tiny_AConv()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks

        tiny_cv1 = tiny_Conv()
        tiny_cv1.tiny_conv = tiny_nn.Conv2d(m.cv1.conv.in_channels, m.cv1.conv.out_channels, m.cv1.conv.kernel_size, m.cv1.conv.stride, m.cv1.conv.padding, m.cv1.conv.dilation, m.cv1.conv.groups, True if m.cv1.conv.bias is not None else False)
        tiny_cv1.tiny_conv.weight = tiny_Tensor(m.cv1.conv.weight.detach().numpy().copy())
        tiny_cv1.tiny_conv.bias = tiny_Tensor(m.cv1.conv.bias.detach().numpy().copy())
        tiny.cv1 = tiny_cv1
        self.model[i] = tiny
        m = self.model[i]
      elif type(m) == ADown:
        tiny = tiny_ADown()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks

        tiny_cv1 = tiny_Conv()
        tiny_cv1.tiny_conv = tiny_nn.Conv2d(m.cv1.conv.in_channels, m.cv1.conv.out_channels, m.cv1.conv.kernel_size, m.cv1.conv.stride, m.cv1.conv.padding, m.cv1.conv.dilation, m.cv1.conv.groups, True if m.cv1.conv.bias is not None else False)
        tiny_cv1.tiny_conv.weight = tiny_Tensor(m.cv1.conv.weight.detach().numpy().copy())
        tiny_cv1.tiny_conv.bias = tiny_Tensor(m.cv1.conv.bias.detach().numpy().copy())
        tiny.cv1 = tiny_cv1

        tiny_cv2 = tiny_Conv()
        tiny_cv2.tiny_conv = tiny_nn.Conv2d(m.cv2.conv.in_channels, m.cv2.conv.out_channels, m.cv2.conv.kernel_size, m.cv2.conv.stride, m.cv2.conv.padding, m.cv2.conv.dilation, m.cv2.conv.groups, True if m.cv2.conv.bias is not None else False)
        tiny_cv2.tiny_conv.weight = tiny_Tensor(m.cv2.conv.weight.detach().numpy().copy())
        tiny_cv2.tiny_conv.bias = tiny_Tensor(m.cv2.conv.bias.detach().numpy().copy())
        tiny.cv2 = tiny_cv2

        self.model[i] = tiny
        m = tiny
        
      elif type(m) == ELAN1:
        tiny = tiny_ELAN1()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks


        tiny_cv1 = tiny_Conv()
        tiny_cv1.tiny_conv = tiny_nn.Conv2d(m.cv1.conv.in_channels, m.cv1.conv.out_channels, m.cv1.conv.kernel_size, m.cv1.conv.stride, m.cv1.conv.padding, m.cv1.conv.dilation, m.cv1.conv.groups, True if m.cv1.conv.bias is not None else False)
        tiny_cv1.tiny_conv.weight = tiny_Tensor(m.cv1.conv.weight.detach().numpy().copy())
        tiny_cv1.tiny_conv.bias = tiny_Tensor(m.cv1.conv.bias.detach().numpy().copy())
        tiny.cv1 = tiny_cv1

        tiny_cv2 = tiny_Conv()
        tiny_cv2.tiny_conv = tiny_nn.Conv2d(m.cv2.conv.in_channels, m.cv2.conv.out_channels, m.cv2.conv.kernel_size, m.cv2.conv.stride, m.cv2.conv.padding, m.cv2.conv.dilation, m.cv2.conv.groups, True if m.cv2.conv.bias is not None else False)
        tiny_cv2.tiny_conv.weight = tiny_Tensor(m.cv2.conv.weight.detach().numpy().copy())
        tiny_cv2.tiny_conv.bias = tiny_Tensor(m.cv2.conv.bias.detach().numpy().copy())
        tiny.cv2 = tiny_cv2

        tiny_cv3 = tiny_Conv()
        tiny_cv3.tiny_conv = tiny_nn.Conv2d(m.cv3.conv.in_channels, m.cv3.conv.out_channels, m.cv3.conv.kernel_size, m.cv3.conv.stride, m.cv3.conv.padding, m.cv3.conv.dilation, m.cv3.conv.groups, True if m.cv3.conv.bias is not None else False)
        tiny_cv3.tiny_conv.weight = tiny_Tensor(m.cv3.conv.weight.detach().numpy().copy())
        tiny_cv3.tiny_conv.bias = tiny_Tensor(m.cv3.conv.bias.detach().numpy().copy())
        tiny.cv3 = tiny_cv3

        tiny_cv4 = tiny_Conv()
        tiny_cv4.tiny_conv = tiny_nn.Conv2d(m.cv4.conv.in_channels, m.cv4.conv.out_channels, m.cv4.conv.kernel_size, m.cv4.conv.stride, m.cv4.conv.padding, m.cv4.conv.dilation, m.cv4.conv.groups, True if m.cv4.conv.bias is not None else False)
        tiny_cv4.tiny_conv.weight = tiny_Tensor(m.cv4.conv.weight.detach().numpy().copy())
        tiny_cv4.tiny_conv.bias = tiny_Tensor(m.cv4.conv.bias.detach().numpy().copy())
        tiny.cv4 = tiny_cv4

        self.model[i] = tiny
        m = tiny
      elif type(m) == CBLinear:
        tiny = tiny_CBLinear()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks

        tiny.conv = tiny_nn.Conv2d(m.conv.in_channels, m.conv.out_channels, m.conv.kernel_size, m.conv.stride, m.conv.padding, m.conv.dilation, m.conv.groups, True if m.conv.bias is not None else False)
        tiny.conv.weight = tiny_Tensor(m.conv.weight.detach().numpy().copy())
        tiny.conv.bias = tiny_Tensor(m.conv.bias.detach().numpy().copy())
        tiny.c2s = m.c2s

        self.model[i] = tiny
        m = tiny
      elif type(m) == Concat:
        tiny = tiny_Concat()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks

        tiny.d = m.d

        self.model[i] = tiny
        m = tiny
      elif type(m) == CBFuse:
        tiny = tiny_CBFuse()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks

        tiny.idx = m.idx
        self.model[i] = tiny
        m = tiny
      elif type(m) == DDetect: # todo ...too much stuff
        tiny = tiny_DDetect()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks

        tiny.nl = m.nl
        tiny.cv2 = tiny_Sequential()
        tiny_seq = tiny_Sequential()
        for j in range(len(m.cv3)):
          tiny_seq_2 = tiny_Sequential()
          for k in range(len(m.cv3[j])):
            if type(m.cv3[j][k]) == Conv:
              tiny_conv = tiny_Conv()
              tiny_conv.tiny_conv = tiny_nn.Conv2d(m.cv3[j][k].conv.in_channels, m.cv3[j][k].conv.out_channels, m.cv3[j][k].conv.kernel_size, m.cv3[j][k].conv.stride, m.cv3[j][k].conv.padding, m.cv3[j][k].conv.dilation, m.cv3[j][k].conv.groups, True if m.cv3[j][k].conv.bias is not None else False)
              tiny_conv.tiny_conv.weight = tiny_Tensor(m.cv3[j][k].conv.weight.detach().numpy().copy())
              tiny_conv.tiny_conv.bias = tiny_Tensor(m.cv3[j][k].conv.bias.detach().numpy().copy())
              tiny_seq_2.append(tiny_conv)
            else:
              tiny_conv = tiny_nn.Conv2d(m.cv3[j][k].in_channels, m.cv3[j][k].out_channels, m.cv3[j][k].kernel_size, m.cv3[j][k].stride, m.cv3[j][k].padding, m.cv3[j][k].dilation, m.cv3[j][k].groups, True if m.cv3[j][k].bias is not None else False)
              tiny_conv.weight = tiny_Tensor(m.cv3[j][k].weight.detach().numpy().copy())
              tiny_conv.bias = tiny_Tensor(m.cv3[j][k].bias.detach().numpy().copy())
              tiny_seq_2.append(tiny_conv)
          tiny_seq.append(tiny_seq_2)
        

        tiny.cv3 = tiny_seq
        tiny.stride = m.stride
        tiny.no = m.no
        tiny.reg_max = m.reg_max
        tiny.nc = m.nc


        tiny.dfl = tiny_DFL()
        tiny.dfl.c1 = m.dfl.c1
        tiny.dfl.conv = tiny_nn.Conv2d(tiny.dfl.c1, 1, 1, bias=False)
        tiny.dfl.conv.weight = tiny_Tensor(m.dfl.conv.weight.detach().numpy().copy())

        tiny_cv2 = tiny_Sequential()

        tiny_seq = tiny_Sequential()          
        tiny_conv = tiny_Conv()
        tiny_conv.tiny_conv = tiny_nn.Conv2d(m.cv2[0][0].conv.in_channels, m.cv2[0][0].conv.out_channels, m.cv2[0][0].conv.kernel_size, m.cv2[0][0].conv.stride, m.cv2[0][0].conv.padding, m.cv2[0][0].conv.dilation, m.cv2[0][0].conv.groups, True if m.cv2[0][0].conv.bias is not None else False)
        tiny_conv.tiny_conv.weight = tiny_Tensor(m.cv2[0][0].conv.weight.detach().numpy().copy())
        tiny_conv.tiny_conv.bias = tiny_Tensor(m.cv2[0][0].conv.bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)

        tiny_conv = tiny_Conv()
        tiny_conv.tiny_conv = tiny_nn.Conv2d(m.cv2[0][1].conv.in_channels, m.cv2[0][1].conv.out_channels, m.cv2[0][1].conv.kernel_size, m.cv2[0][1].conv.stride, m.cv2[0][1].conv.padding, m.cv2[0][1].conv.dilation, m.cv2[0][1].conv.groups, True if m.cv2[0][1].conv.bias is not None else False)
        tiny_conv.tiny_conv.weight = tiny_Tensor(m.cv2[0][1].conv.weight.detach().numpy().copy())
        tiny_conv.tiny_conv.bias = tiny_Tensor(m.cv2[0][1].conv.bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)

        # todo switch?
        tiny_conv = tiny_nn.Conv2d(m.cv2[0][2].in_channels, m.cv2[0][2].out_channels, m.cv2[0][2].kernel_size, m.cv2[0][2].stride, m.cv2[0][2].padding, m.cv2[0][2].dilation, m.cv2[0][2].groups, True if m.cv2[0][2].bias is not None else False)
        tiny_conv.weight = tiny_Tensor(m.cv2[0][2].weight.detach().numpy().copy())
        tiny_conv.bias = tiny_Tensor(m.cv2[0][2].bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)
        tiny_cv2.append(tiny_seq)

        
        tiny_seq = tiny_Sequential()          
        tiny_conv = tiny_Conv()
        tiny_conv.tiny_conv = tiny_nn.Conv2d(m.cv2[1][0].conv.in_channels, m.cv2[1][0].conv.out_channels, m.cv2[1][0].conv.kernel_size, m.cv2[1][0].conv.stride, m.cv2[1][0].conv.padding, m.cv2[1][0].conv.dilation, m.cv2[1][0].conv.groups, True if m.cv2[1][0].conv.bias is not None else False)
        tiny_conv.tiny_conv.weight = tiny_Tensor(m.cv2[1][0].conv.weight.detach().numpy().copy())
        tiny_conv.tiny_conv.bias = tiny_Tensor(m.cv2[1][0].conv.bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)

        tiny_conv = tiny_Conv()
        tiny_conv.tiny_conv = tiny_nn.Conv2d(m.cv2[1][1].conv.in_channels, m.cv2[1][1].conv.out_channels, m.cv2[1][1].conv.kernel_size, m.cv2[1][1].conv.stride, m.cv2[1][1].conv.padding, m.cv2[1][1].conv.dilation, m.cv2[1][1].conv.groups, True if m.cv2[1][1].conv.bias is not None else False)
        tiny_conv.tiny_conv.weight = tiny_Tensor(m.cv2[1][1].conv.weight.detach().numpy().copy())
        tiny_conv.tiny_conv.bias = tiny_Tensor(m.cv2[1][1].conv.bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)

        tiny_conv = tiny_nn.Conv2d(m.cv2[1][2].in_channels, m.cv2[1][2].out_channels, m.cv2[1][2].kernel_size, m.cv2[1][2].stride, m.cv2[1][2].padding, m.cv2[1][2].dilation, m.cv2[1][2].groups, True if m.cv2[1][2].bias is not None else False)
        tiny_conv.weight = tiny_Tensor(m.cv2[1][2].weight.detach().numpy().copy())
        tiny_conv.bias = tiny_Tensor(m.cv2[1][2].bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)
        tiny_cv2.append(tiny_seq)


        tiny_seq = tiny_Sequential()          
        tiny_conv = tiny_Conv()
        tiny_conv.tiny_conv = tiny_nn.Conv2d(m.cv2[2][0].conv.in_channels, m.cv2[2][0].conv.out_channels, m.cv2[2][0].conv.kernel_size, m.cv2[2][0].conv.stride, m.cv2[2][0].conv.padding, m.cv2[2][0].conv.dilation, m.cv2[2][0].conv.groups, True if m.cv2[2][0].conv.bias is not None else False)
        tiny_conv.tiny_conv.weight = tiny_Tensor(m.cv2[2][0].conv.weight.detach().numpy().copy())
        tiny_conv.tiny_conv.bias = tiny_Tensor(m.cv2[2][0].conv.bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)

        tiny_conv = tiny_Conv()
        tiny_conv.tiny_conv = tiny_nn.Conv2d(m.cv2[2][1].conv.in_channels, m.cv2[2][1].conv.out_channels, m.cv2[2][1].conv.kernel_size, m.cv2[2][1].conv.stride, m.cv2[2][1].conv.padding, m.cv2[2][1].conv.dilation, m.cv2[2][1].conv.groups, True if m.cv2[2][1].conv.bias is not None else False)
        tiny_conv.tiny_conv.weight = tiny_Tensor(m.cv2[2][1].conv.weight.detach().numpy().copy())
        tiny_conv.tiny_conv.bias = tiny_Tensor(m.cv2[2][1].conv.bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)

        tiny_conv = tiny_nn.Conv2d(m.cv2[2][2].in_channels, m.cv2[2][2].out_channels, m.cv2[2][2].kernel_size, m.cv2[2][2].stride, m.cv2[2][2].padding, m.cv2[2][2].dilation, m.cv2[2][2].groups, True if m.cv2[1][2].bias is not None else False)
        tiny_conv.weight = tiny_Tensor(m.cv2[2][2].weight.detach().numpy().copy())
        tiny_conv.bias = tiny_Tensor(m.cv2[2][2].bias.detach().numpy().copy())
        tiny_seq.append(tiny_conv)
        tiny_cv2.append(tiny_seq)

        tiny.cv2 = tiny_cv2

        self.model[i] = tiny
        m = tiny
      elif type(m) == SPPELAN:
        tiny = tiny_SPPELAN()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks


        tiny_cv1 = tiny_Conv()
        tiny_cv1.tiny_conv = tiny_nn.Conv2d(m.cv1.conv.in_channels, m.cv1.conv.out_channels, m.cv1.conv.kernel_size, m.cv1.conv.stride, m.cv1.conv.padding, m.cv1.conv.dilation, m.cv1.conv.groups, True if m.cv1.conv.bias is not None else False)
        tiny_cv1.tiny_conv.weight = tiny_Tensor(m.cv1.conv.weight.detach().numpy().copy())
        tiny_cv1.tiny_conv.bias = tiny_Tensor(m.cv1.conv.bias.detach().numpy().copy())
        tiny.cv1 = tiny_cv1
        tiny.cv2 = tiny_SP(m.cv2.m.kernel_size, m.cv2.m.stride)
        tiny.cv3 = tiny_SP(m.cv3.m.kernel_size, m.cv3.m.stride)
        tiny.cv4 = tiny_SP(m.cv4.m.kernel_size, m.cv4.m.stride)
        tiny.cv5 = tiny_Conv()
        tiny.cv5.tiny_conv = tiny_nn.Conv2d(m.cv5.conv.in_channels, m.cv5.conv.out_channels, m.cv5.conv.kernel_size, m.cv5.conv.stride, m.cv5.conv.padding, m.cv5.conv.dilation, m.cv5.conv.groups, True if m.cv5.conv.bias is not None else False)
        tiny.cv5.tiny_conv.weight = tiny_Tensor(m.cv5.conv.weight.detach().numpy())
        tiny.cv5.tiny_conv.bias = tiny_Tensor(m.cv5.conv.bias.detach().numpy())

        self.model[i] = tiny
        m = tiny
      elif type(m) == RepNCSPELAN4:
        tiny = tiny_RepNCSPELAN4()

        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks

        tiny_cv1 = tiny_Conv()
        tiny_cv1.tiny_conv = tiny_nn.Conv2d(m.cv1.conv.in_channels, m.cv1.conv.out_channels, m.cv1.conv.kernel_size, m.cv1.conv.stride, m.cv1.conv.padding, m.cv1.conv.dilation, m.cv1.conv.groups, True if m.cv1.conv.bias is not None else False)
        tiny_cv1.tiny_conv.weight = tiny_Tensor(m.cv1.conv.weight.detach().numpy().copy())
        tiny_cv1.tiny_conv.bias = tiny_Tensor(m.cv1.conv.bias.detach().numpy().copy())
        tiny.cv1 = tiny_cv1

        tiny_seq = tiny_Sequential() # todo
        tiny_rn = tiny_RepNCSP()
        tiny_rn.cv1 = tiny_Conv()
        tiny_rn.cv1.tiny_conv = tiny_nn.Conv2d(m.cv2[0].cv1.conv.in_channels, m.cv2[0].cv1.conv.out_channels, m.cv2[0].cv1.conv.kernel_size, m.cv2[0].cv1.conv.stride, m.cv2[0].cv1.conv.padding, m.cv2[0].cv1.conv.dilation, m.cv2[0].cv1.conv.groups, True if m.cv2[0].cv1.conv.bias is not None else False)
        tiny_rn.cv1.tiny_conv.weight = tiny_Tensor(m.cv2[0].cv1.conv.weight.detach().numpy().copy())
        tiny_rn.cv1.tiny_conv.bias = tiny_Tensor(m.cv2[0].cv1.conv.bias.detach().numpy().copy())
        tiny_rn.cv2 = tiny_Conv()
        tiny_rn.cv2.tiny_conv = tiny_nn.Conv2d(m.cv2[0].cv2.conv.in_channels, m.cv2[0].cv2.conv.out_channels, m.cv2[0].cv2.conv.kernel_size, m.cv2[0].cv2.conv.stride, m.cv2[0].cv2.conv.padding, m.cv2[0].cv2.conv.dilation, m.cv2[0].cv2.conv.groups, True if m.cv2[0].cv2.conv.bias is not None else False)
        tiny_rn.cv2.tiny_conv.weight = tiny_Tensor(m.cv2[0].cv2.conv.weight.detach().numpy().copy())
        tiny_rn.cv2.tiny_conv.bias = tiny_Tensor(m.cv2[0].cv2.conv.bias.detach().numpy().copy())
        tiny_rn.cv3 = tiny_Conv()
        tiny_rn.cv3.tiny_conv = tiny_nn.Conv2d(m.cv2[0].cv3.conv.in_channels, m.cv2[0].cv3.conv.out_channels, m.cv2[0].cv3.conv.kernel_size, m.cv2[0].cv3.conv.stride, m.cv2[0].cv3.conv.padding, m.cv2[0].cv3.conv.dilation, m.cv2[0].cv3.conv.groups, True if m.cv2[0].cv3.conv.bias is not None else False)
        tiny_rn.cv3.tiny_conv.weight = tiny_Tensor(m.cv2[0].cv3.conv.weight.detach().numpy().copy())
        tiny_rn.cv3.tiny_conv.bias = tiny_Tensor(m.cv2[0].cv3.conv.bias.detach().numpy().copy())

        tiny_seq_2 = tiny_Sequential()
        for j in range(len(m.cv2[0].m)):
          tiny_btl = tiny_RepNBottleneck()
          tiny_btl.cv1 = tiny_Conv()
          tiny_btl.cv1.tiny_conv = tiny_nn.Conv2d(m.cv2[0].m[j].cv1.conv.in_channels, m.cv2[0].m[j].cv1.conv.out_channels, m.cv2[0].m[j].cv1.conv.kernel_size, m.cv2[0].m[j].cv1.conv.stride, m.cv2[0].m[j].cv1.conv.padding, m.cv2[0].m[j].cv1.conv.dilation, m.cv2[0].m[j].cv1.conv.groups, True if m.cv2[0].m[j].cv1.conv.bias is not None else False)
          tiny_btl.cv1.tiny_conv.weight = tiny_Tensor(m.cv2[0].m[j].cv1.conv.weight.detach().numpy().copy())
          tiny_btl.cv1.tiny_conv.bias = tiny_Tensor(m.cv2[0].m[j].cv1.conv.bias.detach().numpy().copy())
          tiny_btl.cv2 = tiny_Conv()
          tiny_btl.cv2.tiny_conv = tiny_nn.Conv2d(m.cv2[0].m[j].cv2.conv.in_channels, m.cv2[0].m[j].cv2.conv.out_channels, m.cv2[0].m[j].cv2.conv.kernel_size, m.cv2[0].m[j].cv2.conv.stride, m.cv2[0].m[j].cv2.conv.padding, m.cv2[0].m[j].cv2.conv.dilation, m.cv2[0].m[j].cv1.conv.groups, True if m.cv2[0].m[j].cv2.conv.bias is not None else False)
          tiny_btl.cv2.tiny_conv.weight = tiny_Tensor(m.cv2[0].m[j].cv2.conv.weight.detach().numpy().copy())
          tiny_btl.cv2.tiny_conv.bias = tiny_Tensor(m.cv2[0].m[j].cv2.conv.bias.detach().numpy().copy())

          tiny_btl.add = m.cv2[0].m[j].add
          tiny_seq_2.append(tiny_btl) # todo
        tiny_rn.m = tiny_seq_2
        tiny_seq.append(tiny_rn)

        tiny_seq_cv2 = tiny_Conv()
        tiny_seq_cv2.tiny_conv = tiny_nn.Conv2d(m.cv2[1].conv.in_channels, m.cv2[1].conv.out_channels, m.cv2[1].conv.kernel_size, m.cv2[1].conv.stride, m.cv2[1].conv.padding, m.cv2[1].conv.dilation, m.cv2[1].conv.groups, True if m.cv2[1].conv.bias is not None else False)
        tiny_seq_cv2.tiny_conv.weight = tiny_Tensor(m.cv2[1].conv.weight.detach().numpy().copy())
        tiny_seq_cv2.tiny_conv.bias = tiny_Tensor(m.cv2[1].conv.bias.detach().numpy().copy())
        tiny_seq.append(tiny_seq_cv2)
        tiny.cv2 = tiny_seq

        tiny_seq = tiny_Sequential() # todo

        tiny_rn = tiny_RepNCSP()
        tiny_rn.cv1 = tiny_Conv()
        tiny_rn.cv1.tiny_conv = tiny_nn.Conv2d(m.cv3[0].cv1.conv.in_channels, m.cv3[0].cv1.conv.out_channels, m.cv3[0].cv1.conv.kernel_size, m.cv3[0].cv1.conv.stride, m.cv3[0].cv1.conv.padding, m.cv3[0].cv1.conv.dilation, m.cv3[0].cv1.conv.groups, True if m.cv3[0].cv1.conv.bias is not None else False)
        tiny_rn.cv1.tiny_conv.weight = tiny_Tensor(m.cv3[0].cv1.conv.weight.detach().numpy().copy())
        tiny_rn.cv1.tiny_conv.bias = tiny_Tensor(m.cv3[0].cv1.conv.bias.detach().numpy().copy())
        tiny_rn.cv2 = tiny_Conv()
        tiny_rn.cv2.tiny_conv = tiny_nn.Conv2d(m.cv3[0].cv2.conv.in_channels, m.cv3[0].cv2.conv.out_channels, m.cv3[0].cv2.conv.kernel_size, m.cv3[0].cv2.conv.stride, m.cv3[0].cv2.conv.padding, m.cv3[0].cv2.conv.dilation, m.cv3[0].cv2.conv.groups, True if m.cv3[0].cv2.conv.bias is not None else False)
        tiny_rn.cv2.tiny_conv.weight = tiny_Tensor(m.cv3[0].cv2.conv.weight.detach().numpy().copy())
        tiny_rn.cv2.tiny_conv.bias = tiny_Tensor(m.cv3[0].cv2.conv.bias.detach().numpy().copy())
        tiny_rn.cv3 = tiny_Conv()
        tiny_rn.cv3.tiny_conv = tiny_nn.Conv2d(m.cv3[0].cv3.conv.in_channels, m.cv3[0].cv3.conv.out_channels, m.cv3[0].cv3.conv.kernel_size, m.cv3[0].cv3.conv.stride, m.cv3[0].cv3.conv.padding, m.cv3[0].cv3.conv.dilation, m.cv3[0].cv3.conv.groups, True if m.cv3[0].cv3.conv.bias is not None else False)
        tiny_rn.cv3.tiny_conv.weight = tiny_Tensor(m.cv3[0].cv3.conv.weight.detach().numpy().copy())
        tiny_rn.cv3.tiny_conv.bias = tiny_Tensor(m.cv3[0].cv3.conv.bias.detach().numpy().copy())
        
        tiny_seq_2 = tiny_Sequential()
        for j in range(len(m.cv3[0].m)):
          tiny_btl = tiny_RepNBottleneck()
          tiny_btl.cv1 = tiny_Conv()
          tiny_btl.cv1.tiny_conv = tiny_nn.Conv2d(m.cv3[0].m[j].cv1.conv.in_channels, m.cv3[0].m[j].cv1.conv.out_channels, m.cv3[0].m[j].cv1.conv.kernel_size, m.cv3[0].m[j].cv1.conv.stride, m.cv3[0].m[j].cv1.conv.padding, m.cv3[0].m[j].cv1.conv.dilation, m.cv3[0].m[j].cv1.conv.groups, True if m.cv3[0].m[j].cv1.conv.bias is not None else False)
          tiny_btl.cv1.tiny_conv.weight = tiny_Tensor(m.cv3[0].m[j].cv1.conv.weight.detach().numpy().copy())
          tiny_btl.cv1.tiny_conv.bias = tiny_Tensor(m.cv3[0].m[j].cv1.conv.bias.detach().numpy().copy())
          tiny_btl.cv2 = tiny_Conv()
          tiny_btl.cv2.tiny_conv = tiny_nn.Conv2d(m.cv3[0].m[j].cv2.conv.in_channels, m.cv3[0].m[j].cv2.conv.out_channels, m.cv3[0].m[j].cv2.conv.kernel_size, m.cv3[0].m[j].cv2.conv.stride, m.cv3[0].m[j].cv2.conv.padding, m.cv3[0].m[j].cv2.conv.dilation, m.cv3[0].m[j].cv1.conv.groups, True if m.cv3[0].m[j].cv2.conv.bias is not None else False)
          tiny_btl.cv2.tiny_conv.weight = tiny_Tensor(m.cv3[0].m[j].cv2.conv.weight.detach().numpy().copy())
          tiny_btl.cv2.tiny_conv.bias = tiny_Tensor(m.cv3[0].m[j].cv2.conv.bias.detach().numpy().copy())

          tiny_btl.add = m.cv3[0].m[j].add
          tiny_seq_2.append(tiny_btl) # todo
        tiny_rn.m = tiny_seq_2
        tiny_seq.append(tiny_rn)

        tiny_seq_cv3 = tiny_Conv()
        tiny_seq_cv3.tiny_conv = tiny_nn.Conv2d(m.cv3[1].conv.in_channels, m.cv3[1].conv.out_channels, m.cv3[1].conv.kernel_size, m.cv3[1].conv.stride, m.cv3[1].conv.padding, m.cv3[1].conv.dilation, m.cv3[1].conv.groups, True if m.cv3[1].conv.bias is not None else False)
        tiny_seq_cv3.tiny_conv.weight = tiny_Tensor(m.cv3[1].conv.weight.detach().numpy().copy())
        tiny_seq_cv3.tiny_conv.bias = tiny_Tensor(m.cv3[1].conv.bias.detach().numpy().copy())
        tiny_seq.append(tiny_seq_cv3)
        tiny.cv3 = tiny_seq

        tiny_seq_cv4 = tiny_Conv()
        tiny_seq_cv4.tiny_conv = tiny_nn.Conv2d(m.cv4.conv.in_channels, m.cv4.conv.out_channels, m.cv4.conv.kernel_size, m.cv4.conv.stride, m.cv4.conv.padding, m.cv4.conv.dilation, m.cv4.conv.groups, True if m.cv4.conv.bias is not None else False)
        tiny_seq_cv4.tiny_conv.weight = tiny_Tensor(m.cv4.conv.weight.detach().numpy().copy())
        tiny_seq_cv4.tiny_conv.bias = tiny_Tensor(m.cv4.conv.bias.detach().numpy().copy())
        tiny.cv4 = tiny_seq_cv4

        self.model[i] = tiny
        m = tiny
      elif type(m) == nn.Upsample:
        tiny = tiny_Upsample()
        tiny.f = m.f
        tiny._backward_hooks = m._backward_hooks
        tiny._backward_pre_hooks = m._backward_pre_hooks
        tiny._forward_hooks = m._forward_hooks
        tiny._forward_pre_hooks = m._forward_pre_hooks

        tiny.scale_factor = int(m.scale_factor)

        self.model[i] = tiny
        m = tiny
          
  def __call__(self, x):
    y = []  # outputs
    for i in range(len(self.model)):
      m = self.model[i]
      if m.f != -1:
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
      x = m(x)
      y.append(x)
    return x

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493



def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    return new_size

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLO model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output


    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - nm - 4  # number of classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.T[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]


    return output

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

# python detect.py --source './data/images/football.webp' --img 1280 --device cpu --name yolov9_c_c_640_detect

expected = {}

expected["t"] = [[118.48889,186.87097,490.29224,780.10376,0.93842876,0.0,],
[289.68475,84.81915,707.49225,737.02356,0.9069821,0.0,],
[1010.8401,114.85803,1245.0735,782.2656,0.894862,0.0,],
[817.91797,196.67847,1073.5225,791.2289,0.85889053,0.0,],
[724.6112,231.13603,811.25964,315.62823,0.7664335,32.0,],
[679.993,497.45862,803.5401,650.10535,0.71624434,0.0,],
[19.570282,440.98154,120.16864,650.69934,0.6812207,0.0,],
[1144.8735,96.08271,1222.6304,227.3689,0.5918,0.0,],
[608.77686,522.07214,690.4009,648.5106,0.57779104,0.0,],
[613.6139,299.13907,653.77405,410.4597,0.5654181,35.0,],
[348.43942,409.34723,419.53134,525.9255,0.5540223,0.0,],
[295.30267,213.90076,362.94757,288.6588,0.50664437,0.0,],
[1204.2368,444.46072,1279.854,654.7822,0.4638907,0.0,],
[474.15613,522.90265,568.84875,651.9123,0.45667136,0.0,],
[383.23853,27.169552,447.8485,141.24829,0.3796056,0.0,],
[632.66943,102.471565,697.4557,204.47845,0.3763075,0.0,],
[943.87646,102.48796,1050.704,247.25853,0.32999045,0.0,],
[421.2162,67.35925,529.47986,230.03897,0.31452325,0.0,],
[1216.606,174.5149,1279.7205,360.19037,0.31061673,0.0,],
[47.871307,126.34314,135.86726,287.8742,0.29735684,0.0,],
[609.82007,658.92847,736.8911,744.7439,0.29662183,0.0,],
[371.5387,456.5051,476.0901,628.5676,0.28850546,0.0,],
[1121.1753,493.20435,1206.5063,647.27295,0.28599265,0.0,],
[1019.1849,16.87535,1124.2117,139.78244,0.27446195,0.0,],
[741.343,349.93536,824.43555,472.80652,0.27320808,0.0,],
[668.13586,399.45288,758.0482,581.0128,0.26745877,0.0,],
[90.71086,201.64069,194.29411,324.84863,0.26404554,0.0,],]

expected["s"] = [[117.691574,186.87549,481.98724,777.3633,0.94480765,0.0,],
[293.3302,85.02185,706.69885,790.9646,0.9355096,0.0,],
[1018.44415,112.981384,1243.446,772.97754,0.9258939,0.0,],
[815.60986,194.09521,1080.2942,786.17957,0.90948486,0.0,],
[725.4081,230.6186,810.3927,313.67657,0.8910456,32.0,],
[20.390778,441.57764,124.01999,649.26807,0.7890667,0.0,],
[681.3502,496.86386,804.75867,649.2582,0.70027196,0.0,],
[943.07654,102.96634,1051.0062,248.9498,0.5589638,0.0,],
[613.88995,299.4808,653.95135,409.4313,0.551256,35.0,],
[1145.6532,95.29636,1220.9867,223.71481,0.5508369,0.0,],
[608.3568,522.6909,685.2451,648.83203,0.53190607,0.0,],
[472.7083,524.12244,568.46735,652.5436,0.50567526,0.0,],
[295.37073,213.88959,357.98315,283.62296,0.48159686,0.0,],
[370.9068,456.7616,477.37958,650.7324,0.4751682,0.0,],
[109.96647,536.96436,191.80283,651.8696,0.4648949,0.0,],
[743.9175,349.90747,824.59607,459.01703,0.45445773,0.0,],
[646.484,398.2118,756.5325,574.2148,0.39138803,0.0,],
[1219.3749,444.47748,1279.7882,650.2165,0.3909197,0.0,],
[1223.5648,353.19305,1279.7916,484.95282,0.38625097,0.0,],
[89.38068,202.01147,194.82486,322.28748,0.3682094,0.0,],
[381.3059,26.800003,448.4911,140.15681,0.36068103,0.0,],
[0.21734619,273.7856,102.57646,446.3063,0.3556079,0.0,],
[278.594,61.971497,365.0326,229.31952,0.30487975,0.0,],
[1035.801,81.406525,1113.3376,208.52734,0.27518255,0.0,],
[834.80774,123.468,905.06213,215.84737,0.2574972,0.0,],]

expected["m"] = [[117.34009,189.81927,479.09924,783.0599,0.9438101,0.0,],
[816.08057,196.10352,1077.5957,789.5227,0.9203351,0.0,],
[1014.5182,113.038574,1244.4349,786.60876,0.9176292,0.0,],
[725.22815,230.3343,810.9481,315.1541,0.90587795,32.0,],
[290.60394,86.51221,711.9071,801.2782,0.887479,0.0,],
[680.63916,501.5403,803.2905,649.6194,0.7672667,0.0,],
[18.929394,443.29602,125.42543,650.18774,0.7547232,0.0,],
[612.72,299.77414,653.71265,410.16013,0.69090587,35.0,],
[264.15485,727.1196,310.48492,785.6472,0.6092315,35.0,],
[608.8002,523.90576,688.4347,648.5874,0.5705397,0.0,],
[107.19,535.11316,192.12415,653.0707,0.5412098,0.0,],
[295.4754,214.22916,366.02667,284.95938,0.488116,0.0,],
[472.33865,523.7383,567.677,652.3862,0.4832986,0.0,],
[547.23816,12.899979,655.46497,164.50172,0.47911665,0.0,],
[372.34235,458.5536,477.4571,647.43054,0.47786063,0.0,],
[419.76874,61.08551,529.637,227.59033,0.43891892,0.0,],
[1147.0884,96.09647,1221.9414,224.45914,0.43541908,0.0,],
[742.2915,351.67172,824.66016,462.13028,0.41485325,0.0,],
[609.28796,658.4569,744.47473,741.58167,0.40127435,0.0,],
[944.5514,103.05133,1050.9742,246.6127,0.386276,0.0,],
[647.1035,396.06665,780.86035,574.61395,0.31369215,0.0,],
[293.6776,214.2187,380.75488,513.5096,0.26827645,0.0,],
[1220.947,443.83496,1279.74,578.26935,0.26436853,0.0,],
[0.097229004,273.80762,114.38226,485.50806,0.2605467,0.0,],
[0.037929535,484.65192,50.289692,650.3603,0.25725868,0.0,],]

expected["c"] = [[725.2341,230.70872,811.0788,315.21094,0.9546069,32.0,],
[117.90036,187.97607,478.69913,780.22156,0.944186,0.0,],
[299.46942,86.74158,710.71204,801.4336,0.9244813,0.0,],
[815.171,197.49829,1078.7058,790.5238,0.91612774,0.0,],
[1012.4674,120.719604,1243.5284,785.41296,0.9081284,0.0,],
[679.8407,500.44278,804.6222,650.1498,0.8142025,0.0,],
[21.291092,440.879,123.3313,648.44763,0.74982446,0.0,],
[473.753,523.1787,568.8349,651.8861,0.6673066,0.0,],
[608.3159,523.3396,689.572,648.905,0.64557374,0.0,],
[374.16895,457.90585,480.55817,652.1283,0.5980868,0.0,],
[296.13043,214.36687,376.294,498.58234,0.55707,0.0,],
[943.8263,103.312744,1052.1256,242.16962,0.55068827,0.0,],
[95.44034,534.64197,199.2861,653.0082,0.48393574,0.0,],
[608.75415,658.06445,763.3115,742.5748,0.48314795,0.0,],
[420.8114,68.96379,530.1542,227.1894,0.46491036,0.0,],
[296.0612,214.51318,366.38058,287.47125,0.43723086,0.0,],
[1122.6321,488.88773,1215.7766,650.17334,0.43220803,0.0,],
[0.05958557,274.7494,119.55137,498.05957,0.39801753,0.0,],
[645.58167,399.74298,767.8002,576.14056,0.3704799,0.0,],
[0.0066375732,486.67712,65.7234,650.8165,0.3591559,0.0,],
[1146.9055,95.90793,1221.5461,227.08972,0.34007654,0.0,],
[548.7862,11.946671,659.1894,161.7699,0.33652493,0.0,],
[742.81836,351.54636,822.03735,455.0455,0.28838772,0.0,],
[1205.561,507.64404,1279.7339,654.1824,0.28583172,0.0,],
[1220.5984,353.64185,1279.7351,490.05103,0.2595313,0.0,],
[0.0838089,211.67484,71.78677,348.92593,0.25199288,0.0,],]

expected["e"] = [[118.6095,186.78137,479.75702,778.1138,0.9503603,0.0,],
[816.7417,195.57068,1079.2681,788.6847,0.94166964,0.0,],
[1017.5192,112.72449,1243.4812,781.58826,0.9370377,0.0,],
[293.3199,86.687805,710.7832,793.20215,0.9351896,0.0,],
[725.2562,230.41527,811.56213,315.45837,0.9234742,32.0,],
[678.6655,497.56955,803.42163,649.6846,0.8086615,0.0,],
[20.615784,442.1587,124.11772,650.20996,0.79137737,0.0,],
[610.2068,658.1646,743.36804,741.20636,0.7558812,0.0,],
[474.35535,523.82263,568.5758,652.55383,0.67917824,0.0,],
[608.0458,524.05994,686.04626,648.8014,0.668596,0.0,],
[945.42334,103.39756,1051.6908,243.69931,0.6150132,0.0,],
[103.49456,534.89,201.45349,653.5941,0.6048614,0.0,],
[295.5619,214.24454,360.89148,284.42078,0.5917732,0.0,],
[647.00635,397.90393,773.1504,574.0868,0.56694686,0.0,],
[0.055675507,488.6075,60.569878,650.4669,0.5474956,0.0,],
[1121.6663,487.93317,1229.6025,650.1055,0.4617893,0.0,],
[1145.1206,96.41137,1221.2607,225.25604,0.43339804,0.0,],
[371.36176,465.5744,478.67395,649.03107,0.4306733,0.0,],
[548.5515,12.337021,656.72314,161.74568,0.41807055,0.0,],
[1221.0321,444.45862,1279.7272,560.0082,0.411652,0.0,],
[743.5349,350.9657,821.5581,456.61682,0.40063515,0.0,],
[420.97638,68.083145,534.1436,231.17815,0.32892612,0.0,],
[710.27454,25.734863,787.7479,140.57143,0.32494408,0.0,],
[381.78247,29.136017,448.31702,139.27228,0.32442042,0.0,],
[1221.7955,353.96008,1279.8656,455.0799,0.31737176,0.0,],
[49.255463,165.62592,122.33687,281.20215,0.30858666,0.0,],
[91.172134,200.8421,191.30974,315.91217,0.28115842,0.0,],
[778.28625,41.40837,885.3248,160.98529,0.27413738,0.0,],
[629.4326,102.30394,701.3318,204.90413,0.26948094,0.0,],
[0.030563354,276.41248,102.32032,448.27612,0.2668721,0.0,],
[1173.237,450.7884,1230.4148,563.08185,0.26479527,0.0,],
[295.18512,214.09583,377.89227,501.78497,0.2601459,0.0,],
[130.26685,557.0753,379.346,768.1344,0.2538286,0.0,],]

def run():
    
    for size in ["t", "s", "m", "c", "e"]:
        weights = f'./yolov9-{size}-tiny.pt'
        weights = f'./yolov9-{size}-converted.pt'

        source = "data/images/football.webp"
        imgsz = (1280,1280)
        model = pickle.load(open(weights, 'rb'))

        tiny_model = tiny_DetectionModel()
        tiny_model.model = model.model
        for key, value in vars(model).items(): setattr(tiny_model, key, value)
        model = tiny_model

        tiny_seq = tiny_Sequential()
        for x in model.model: tiny_seq.append(x)
        model.model = tiny_seq

        model.device = "cpu"
        model.fp16 = False
        stride, pt = 32, True
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        for path, im, im0s, vid_cap, s in dataset:
          im = torch.from_numpy(im).to(model.device)
          im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
          im /= 255  # 0 - 255 to 0.0 - 1.0
          if len(im.shape) == 3: im = im[None]  # expand for batch dim

          model.convert()
          #pickle.dump(model, open(f'yolov9-{size}-tiny.pt', 'wb'))

          pred = model(im)
          pred = non_max_suppression(pred, 0.25, 0.45, None, False, 1000)
          pred = pred[0].detach().numpy()
          np.testing.assert_allclose(pred, expected[size], atol=1e-4, rtol=1e-3)

          class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
          pred = rescale_bounding_boxes(pred)
          draw_bounding_boxes_and_save(source, f"out_{size}.jpg", pred, class_labels)
    print("passed")

def rescale_bounding_boxes(predictions, from_size=(1280, 853), to_size=(3020, 1986)):
    from_w, from_h = from_size
    to_w, to_h = to_size
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    
    rescaled_predictions = []
    for pred in predictions:
        x1, y1, x2, y2, conf, class_id = pred
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        rescaled_predictions.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, conf, class_id])
    return rescaled_predictions

from collections import defaultdict
def draw_bounding_boxes_and_save(orig_img_path, output_img_path, predictions, class_labels):
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  font = cv2.FONT_HERSHEY_SIMPLEX
  def is_bright_color(color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127

  orig_img = cv2.imread(orig_img_path) if not isinstance(orig_img_path, np.ndarray) else cv2.imdecode(orig_img_path, 1)
  height, width, _ = orig_img.shape
  box_thickness = int((height + width) / 400)
  font_scale = (height + width) / 2500
  object_count = defaultdict(int)

  for pred in predictions:
    x1, y1, x2, y2, conf, class_id = pred
    if conf == 0: continue
    x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
    color = color_dict[class_labels[class_id]]
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, box_thickness)
    label = f"{class_labels[class_id]} {conf:.2f}"
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    label_y, bg_y = (y1 - 4, y1 - text_size[1] - 4) if y1 - text_size[1] - 4 > 0 else (y1 + text_size[1], y1)
    cv2.rectangle(orig_img, (x1, bg_y), (x1 + text_size[0], bg_y + text_size[1]), color, -1)
    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    cv2.putText(orig_img, label, (x1, label_y), font, font_scale, font_color, 1, cv2.LINE_AA)
    object_count[class_labels[class_id]] += 1

  print("Objects detected:")
  for obj, count in object_count.items():
    print(f"- {obj}: {count}")

  cv2.imwrite(output_img_path, orig_img)
  print(f'saved detections at {output_img_path}')


def main():
  run()

if __name__ == "__main__":
  main()




