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

  model = DetectionModel(*SIZES[size]) if size in SIZES else DetectionModel()
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