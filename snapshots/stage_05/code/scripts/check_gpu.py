"""GPU check: torch/CUDA versions, device, memory and a timed 4096x4096 matmul on cuda.
Exits with code 1 if CUDA is unavailable (training must never fall back to CPU).
Usage: .venv/bin/python scripts/check_gpu.py"""
import subprocess
import sys
import time

import torch

print("python", sys.version.split()[0], "|", sys.executable)
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("torch.backends.cudnn.version() =", torch.backends.cudnn.version())
print("torch.cuda.is_available() =", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available - stop, do not train on CPU.")
    sys.exit(1)

r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True)
print("nvidia driver =", r.stdout.strip() or "UNKNOWN")
props = torch.cuda.get_device_properties(0)
print("device name =", torch.cuda.get_device_name(0))
print(f"total memory = {props.total_memory / 2**30:.3f} GiB")

torch.manual_seed(42)
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
_ = a @ b  # warm-up
torch.cuda.synchronize()
t = time.perf_counter()
c = a @ b
torch.cuda.synchronize()
dt = time.perf_counter() - t
print(f"4096x4096 fp32 matmul on cuda: {dt * 1000:.2f} ms ({2 * 4096 ** 3 / dt / 1e12:.2f} TFLOP/s)")
print(f"max_memory_allocated = {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
print("GPU CHECK: OK")
