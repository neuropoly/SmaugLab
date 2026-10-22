import numpy as np
import torch
from srcsm_transform import SemRandConvTransform

L = 4

# --- v1 convention: batch dict, numpy, keys data/seg, (B,C,X,Y,Z) ---------- #
B, C, D, H, W = 2, 1, 16, 24, 24
d1 = {
    "data": np.random.randn(B, C, D, H, W).astype(np.float32),
    "seg": np.random.randint(0, L, (B, C, D, H, W)).astype(np.float32),
}
t = SemRandConvTransform(num_labels=L, p=1.0, per_label=True)
before = d1["data"].copy()
out1 = t(**d1)
print("v1 numpy/batch:",
      "dtype", out1["data"].dtype,
      "shape", out1["data"].shape,
      "changed", not np.allclose(out1["data"], before),
      "seg untouched", np.array_equal(out1["seg"], d1["seg"]))

# --- v2 convention: per-sample, torch, keys image/segmentation, (C,X,Y,Z) -- #
d2 = {
    "image": torch.randn(C, D, H, W),
    "segmentation": torch.randint(0, L, (C, D, H, W)),
}
before2 = d2["image"].clone()
out2 = t(**d2)
print("v2 torch/sample:",
      "dtype", out2["image"].dtype,
      "shape", tuple(out2["image"].shape),
      "changed", not torch.allclose(out2["image"], before2))

# --- ignore label (-1) folded into background ------------------------------ #
seg_ig = torch.randint(0, L, (C, D, H, W))
seg_ig[0, 0, 0, 0] = -1
d3 = {"image": torch.randn(C, D, H, W), "segmentation": seg_ig}
out3 = t(**d3)
print("ignore-label ok, finite:", torch.isfinite(out3["image"]).all().item())

# --- p=0 leaves data untouched --------------------------------------------- #
t0 = SemRandConvTransform(num_labels=L, p=0.0)
d4 = {"image": torch.randn(C, D, H, W), "segmentation": torch.randint(0, L, (C, D, H, W))}
b4 = d4["image"].clone()
out4 = t0(**d4)
print("p=0 unchanged:", torch.allclose(out4["image"], b4))

# --- multi-channel rejected ------------------------------------------------ #
try:
    tbad = SemRandConvTransform(num_labels=L)
    tbad(image=torch.randn(2, D, H, W), segmentation=torch.randint(0, L, (2, D, H, W)))
    print("multichannel: NOT rejected (BUG)")
except ValueError as e:
    print("multichannel rejected:", "C==1" in str(e))
