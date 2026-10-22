"""
GPU smoke test for the SRCSM port, at a realistic nnU-Net patch size.
Runs on CUDA if available; reports device, correctness, timing and peak memory.
"""
import time, random
import torch
from sem_rand_conv_3d import SemRandConv3D
from srcsm_transform import SemRandConvTransform

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device={dev} | "
      f"cuda_available={torch.cuda.is_available()}")
if dev == "cuda":
    print("GPU:", torch.cuda.get_device_name(0),
          f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

torch.manual_seed(0); random.seed(0)

# realistic single-modality abdominal-ish patch and label count
B, C = 2, 1
D, H, W = 160, 128, 160          # their SRCSM patch size
L = 5                             # e.g. CHAOS MR: 4 organs + background

img = torch.randn(B, C, D, H, W, device=dev)
lab = torch.randint(0, L, (B, C, D, H, W), device=dev)

op = SemRandConv3D(num_labels=L, per_label=True, smoothing=True)

# correctness on-device
def energy(x): return (x.float()**2).reshape(x.shape[0], -1).sum(1)
a1 = op(img, lab); a2 = op(img, lab)
print("output device:", a1.device, "| dtype:", a1.dtype)
print("fresh weights (calls differ):", not torch.allclose(a1, a2))
print("finite:", torch.isfinite(a1).all().item(), "| requires_grad:", a1.requires_grad)

# timing: warmup + timed loop (per-label loop is the cost driver)
def sync():
    if dev == "cuda": torch.cuda.synchronize()
for _ in range(3):
    op(img, lab); sync()
if dev == "cuda": torch.cuda.reset_peak_memory_stats()
N = 20
t0 = time.time()
for _ in range(N):
    _ = op(img, lab)
sync()
dt = (time.time() - t0) / N
print(f"per-call (per_label, B={B}, {D}x{H}x{W}, L={L}): {dt*1000:.1f} ms")
if dev == "cuda":
    print(f"peak GPU mem this op: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

# global mode timing for reference
opg = SemRandConv3D(num_labels=L, per_label=False)
for _ in range(3): opg(img); sync()
t0 = time.time()
for _ in range(N): _ = opg(img)
sync()
print(f"per-call (global): {(time.time()-t0)/N*1000:.1f} ms")

# fp16 / autocast path (nnU-Net trains under autocast)
if dev == "cuda":
    with torch.autocast("cuda", dtype=torch.float16):
        ah = op(img, lab)
    print("autocast fp16 finite:", torch.isfinite(ah).all().item(),
          "| dtype:", ah.dtype)

# transform wrapper with auto device (should pick cuda)
t = SemRandConvTransform(num_labels=L, per_label=True)  # device=None -> auto
print("transform auto device:", t.device)
dd = {"image": img[0].clone(), "segmentation": lab[0].clone()}
out = t(**dd)
print("wrapper output device:", out["image"].device,
      "| changed:", not torch.allclose(out["image"], img[0]))

print("SMOKE TEST OK")
