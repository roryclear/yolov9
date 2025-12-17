import subprocess
import sys
script = "jit_yolov9.py"
sizes = ["t", "s", "m", "c", "e"]
resolutions = [320 ,640, 960, 1280]
for size in sizes:
  for res in resolutions:
    subprocess.run([sys.executable, script, size, str(res)], check=True)