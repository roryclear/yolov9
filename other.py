import torch.nn as nn
class Conv(nn.Module):
    def __init__(self): super().__init__()
    def forward_fuse(self, x): return
class RepConvN(nn.Module):
    def __init__(self):
        super().__init__()
    def forward_fuse(self, x): exit()

class ADown(nn.Module): pass
class AConv(nn.Module): pass
class ELAN1(nn.Module): pass
class RepNBottleneck(nn.Module): pass
class RepNCSP(nn.Module): pass
class RepNCSPELAN4(nn.Module): pass
class SP(nn.Module): pass
class SPPELAN(nn.Module): pass
class Concat(nn.Module): pass
class DDetect(nn.Module): pass
class DFL(nn.Module): pass
class CBLinear(nn.Module): pass
class CBFuse(nn.Module): pass
class DetectionModel(nn.Module): pass
