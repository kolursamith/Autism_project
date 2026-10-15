"""Stage 1 - environment verification. Prints torch/CUDA info, a GPU matmul timing and package versions."""
import importlib, platform, sys, time
import torch

torch.manual_seed(42)
print("python", sys.version.replace("\n", " "), "| executable", sys.executable)
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("cudnn =", torch.backends.cudnn.version())
print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))
print("total GPU memory (GiB) =", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3))
a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); _ = a @ b; torch.cuda.synchronize()  # warm-up
t = time.perf_counter(); c = a @ b; torch.cuda.synchronize(); dt = time.perf_counter() - t
print(f"4096x4096 fp32 matmul on GPU: {dt*1000:.2f} ms ({2*4096**3/dt/1e12:.2f} TFLOP/s), checksum {c.float().sum().item():.4f}")
print("\npackage versions (import name -> version, module file)")
for pip_name, mod in [("torch","torch"),("torchvision","torchvision"),("timm","timm"),("numpy","numpy"),
                      ("pandas","pandas"),("scikit-learn","sklearn"),("matplotlib","matplotlib"),("pillow","PIL"),
                      ("opencv-python","cv2"),("pyyaml","yaml"),("pytest","pytest"),("scipy","scipy"),("imagehash","imagehash")]:
    try:
        m = importlib.import_module(mod)
        print(f"  {pip_name:14s} OK   {getattr(m,'__version__','?'):12s} {m.__file__}")
    except Exception as e:
        print(f"  {pip_name:14s} MISSING {e!r}")
