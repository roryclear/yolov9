from yolov9 import DetectionModel, SIZES, safe_load, load_state_dict, Sequential, Silence, Conv, RepNCSPELAN4, AConv,\
ADown, CBLinear, CBFuse, SPPELAN, Upsample, Concat, DDetect, postprocess, fetch, rescale_bounding_boxes, draw_bounding_boxes_and_save
import cv2
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes
import numpy as np
import time
import sys

@TinyJit
def do_inf(model, im): return model(im)

if __name__ == "__main__":
  size = "s"
  res = 640
  if len(sys.argv) > 1: size = sys.argv[1]
  if len(sys.argv) > 2: res = int(sys.argv[2])

  if size in ["t", "s", "m", "c"]:
    model = DetectionModel(*SIZES[size])
    state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/yolov9/resolve/main/yolov9-{size}.safetensors'))
    load_state_dict(model, state_dict)
  else:
    model = DetectionModel()
    model.model = Sequential(size=43)
    model.model[0] = Silence()
    model.model[1] = Conv(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),  groups=1, bias=True)
    model.model[2] = Conv(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),  groups=1, bias=True)
    model.model[3] = RepNCSPELAN4(128, 32, 256, n=2)
    model.model[4] = ADown(ch0=128)
    model.model[5] = RepNCSPELAN4(256, 64, 512, n=2)
    model.model[6] = ADown(ch0=256)
    model.model[7] = RepNCSPELAN4(512, 128, 1024, n=2)
    model.model[8] = ADown(ch0=512)
    model.model[9] = RepNCSPELAN4(1024, 128, 1024, n=2)
    model.model[10] = CBLinear()
    model.model[11] = CBLinear(ch0=256, ch1=192, c2s=[64, 128], f=3)
    model.model[12] = CBLinear(ch0=512, ch1=448, c2s=[64, 128, 256], f=5)
    model.model[13] = CBLinear(ch0=1024, ch1=960, c2s=[64, 128, 256, 512], f=7)
    model.model[14] = CBLinear(ch0=1024, ch1=1984, c2s=[64, 128, 256, 512, 1024], f=9)
    model.model[15] = Conv(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),  groups=1, bias=True, f=0)
    model.model[16] = CBFuse(f=[10, 11, 12, 13, 14, -1], idx=[0, 0, 0, 0, 0])
    model.model[17] = Conv(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1),  groups=1, bias=True)
    model.model[18] = CBFuse(f=[11, 12, 13, 14, -1], idx=[1, 1, 1, 1])
    model.model[19] = RepNCSPELAN4(128, 32, 256, n=2)
    model.model[20] = ADown(ch0=128)
    model.model[21] = CBFuse(f=[12, 13, 14, -1], idx=[2, 2, 2])
    model.model[22] = RepNCSPELAN4(256, 64, 512, n=2)
    model.model[23] = ADown(ch0=256)
    model.model[24] = CBFuse(f=[13, 14, -1], idx=[3, 3])
    model.model[25] = RepNCSPELAN4(512, 128, 1024, n=2)
    model.model[26] = ADown(ch0=512)
    model.model[27] = CBFuse(f=[14, -1], idx=[4])
    model.model[28] = RepNCSPELAN4(1024, 128, 1024, n=2)
    model.model[29] = SPPELAN(ch0=1024, ch1=256, ch2=1024, ch3=512, f=28)
    model.model[30] = Upsample()
    model.model[31] = Concat(f=[-1, 25])
    model.model[32] = RepNCSPELAN4(1536, 128, 512, n=2)
    model.model[33] = Upsample()
    model.model[34] = Concat(f=[-1, 22])
    model.model[35] = RepNCSPELAN4(1024, 64, 256, n=2)
    model.model[36] = ADown(ch0=128)
    model.model[37] = Concat(f=[-1, 32]) 
    model.model[38] = RepNCSPELAN4(768, 128, 512, n=2)
    model.model[39] = ADown(ch0=256)
    model.model[40] = Concat(f=[-1, 29])
    model.model[41] = RepNCSPELAN4(1024, 256, 512, n=2)
    model.model[42] = DDetect(a=256, b=512, c=512, d=256, f=[35, 38, 41])
    state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/yolov9/resolve/main/yolov9-{size}.safetensors'))
    load_state_dict(model, state_dict)

  im = Tensor(np.random.rand(1, 3, res, res).astype(np.float32))
  # capture JIT + BEAM
  non_jit_out = None
  for _ in range(2):
    pred = do_inf(model, im)
    non_jit_out = pred.numpy()

  total_time = 0
  for i in range(10):
    t = time.time()
    pred = do_inf(model, im)
    jit_out = pred.numpy()
    total_time += (time.time() - t)
    fps = (i + 1) / total_time
    print(f"FPS: {fps:.2f}", end="\r", flush=True)
    np.testing.assert_allclose(jit_out, non_jit_out)
  print(f"FPS for model {size} res {res}x{res}:\t {fps:.2f}")
  with open("perf_results.md", "a") as f: f.write(f"| {size} | {res} | {fps:.2f} |\n")