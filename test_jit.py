import subprocess
import sys
import os
if os.path.exists("perf_results.md"): os.remove("perf_results.md")
with open("perf_results.md", "a") as f:
  f.write("| Model | Resolution | FPS |\n")
  f.write("|-------|------------|-----|\n")
script = "jit_yolov9.py"
sizes = ["t", "s", "m", "c", "e"]
resolutions = [320 ,640, 960, 1280, 1536]
for size in sizes:
  for res in resolutions:
    subprocess.run([sys.executable, script, size, str(res)], check=True)