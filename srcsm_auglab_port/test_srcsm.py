import torch, random
import torch.nn.functional as F
from sem_rand_conv_3d import SemRandConv3D, _smooth_onehot

torch.manual_seed(1); random.seed(1)
B, D, H, W, L = 2, 16, 24, 24, 4
img = torch.randn(B, 1, D, H, W) * 3.0 + 1.0
lab = torch.randint(0, L, (B, 1, D, H, W))

opg = SemRandConv3D(num_labels=L, per_label=False)
ag = opg(img)
ein = (img ** 2).reshape(B, -1).sum(1)
eout = (ag ** 2).reshape(B, -1).sum(1)
print("GLOBAL ratio (should be ~1):", [round(o / i, 4) for o, i in zip(eout.tolist(), ein.tolist())])

oh = F.one_hot(lab.squeeze(1), num_classes=L).permute(0, 4, 1, 2, 3).float()
sm = _smooth_onehot(oh)
print("hard mask unique count:", len(torch.unique(oh)))
print("soft mask min/max:", round(sm.min().item(), 3), round(sm.max().item(), 3))
print("soft mask has intermediate values:", ((sm > 0.01) & (sm < 0.99)).any().item())

op16 = SemRandConv3D(num_labels=L, per_label=True)
a16 = op16(img.half(), lab)
print("float16 output dtype:", a16.dtype, "finite:", torch.isfinite(a16).all().item())

lab2 = lab.clone(); lab2[lab2 == 3] = 0
a2 = op16(img, lab2)
print("missing-label patch ok, finite:", torch.isfinite(a2).all().item())
