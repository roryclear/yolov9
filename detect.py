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
from urllib.parse import urlparse
from copy import deepcopy
import pickle
import torch.nn.functional as F

TORCH_1_10 = False
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode


    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ADown(nn.Module):
    def __init__(self, c1=1, c2=1):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class AConv(nn.Module):
    def __init__(self, c1=1, c2=1):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)

class ELAN1(nn.Module):

    def __init__(self, c1=1, c2=1, c3=1, c4=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3//2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1=1, c2=1, c3=1, c4=1, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)

class SPPELAN(nn.Module):
    # spp-elan
    def __init__(self, c1=1, c2=1, c3=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4*c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class DDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        return
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
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

class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        return
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)) # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class CBLinear(nn.Module):
    def __init__(self, c1=1, c2s=1, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        return
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs

class CBFuse(nn.Module):
    def __init__(self):
        return
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

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


class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)) # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    def forward(self, x):    
        return x

class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class BaseModel(nn.Module):
    # YOLO base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if isinstance(m, (RepConvN)) and hasattr(m, 'fuse_convs'):
                m.fuse_convs()
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self


class DetectionModel(BaseModel): pass

class DetectMultiBackend(nn.Module):
    # YOLO MultiBackend class for python inference on various backends
    def __init__(self, weights='yolo.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        super().__init__()
        self.pt = True
        self.device = "cpu"
        self.fp16 = False
        model = pickle.load(open(weights, 'rb'))
        self.stride = max(int(model.stride.max()), 32) 
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

    def forward(self, im, augment=False, visualize=False): return self.model(im)

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

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files

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

    device = prediction.device

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
    time_limit = 2.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

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
        weights = f'./yolov9-{size}-converted.pt'

        source = "data/images/football.webp"
        imgsz = (1280,1280)
        device = "cpu"
        model = DetectMultiBackend(weights, device=device, dnn=False, data="ROOT", fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        for path, im, im0s, vid_cap, s in dataset:
          im = torch.from_numpy(im).to(model.device)
          im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
          im /= 255  # 0 - 255 to 0.0 - 1.0
          if len(im.shape) == 3: im = im[None]  # expand for batch dim

          pred = model(im, augment=False, visualize=False)
          pred = non_max_suppression(pred, 0.25, 0.45, None, False, 1000)
          pred = pred[0].detach().numpy()

          
          np.testing.assert_allclose(pred, expected[size])
    print("passed")

def main():
  run()

if __name__ == "__main__":

  main()
