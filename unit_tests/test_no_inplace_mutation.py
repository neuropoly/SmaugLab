"""A transform must not write into the tensor its caller handed over.

kornia passes the caller's own tensor to `apply_transform` when every sample of
the batch applies, which is the normal case. The intensity transforms in
`gpu/contrast.py` and the two in `gpu/fromSeg.py` ended with
`input[:, c] = checked; return input`, so `batch["data"]` was modified under any
caller still holding a reference to it.

Every transform in `gpu/spatial.py` already cloned, as do the palette and
domain-transfer transforms -- `transforms_list.py` even has a comment about
exactly this ("A clone, not `out = input` ... Every sibling transform in
gpu/spatial.py clones"). So the file was internally inconsistent, not
deliberately in-place.
"""

from __future__ import annotations

import torch

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from unit_tests.helpers import SmaugLabTestCase
from unit_tests.test_transforms_gpu import DISCOVERED, build_kwargs, skip_reason


class TestTransformsDoNotMutateTheirInput(SmaugLabTestCase):
    def test_the_callers_tensor_is_untouched(self):
        for label, cls, signature in DISCOVERED:
            with self.subTest(transform=label):
                reason = skip_reason(cls)
                if reason:
                    self.skipTest(reason)

                image, seg = self.tiny_volume(), self.tiny_seg()
                reference = image.clone()
                pipeline = AugmentationSequentialCustom(
                    cls(**build_kwargs(cls, signature)), data_keys=["input", "mask"], same_on_batch=True
                )

                pipeline(image, seg)

                self.assertTrue(
                    bool(torch.equal(image, reference)),
                    f"{cls.__name__} wrote into the tensor it was given",
                )

    def test_the_callers_mask_is_untouched(self):
        for label, cls, signature in DISCOVERED:
            with self.subTest(transform=label):
                reason = skip_reason(cls)
                if reason:
                    self.skipTest(reason)

                image, seg = self.tiny_volume(), self.tiny_seg()
                reference = seg.clone()
                pipeline = AugmentationSequentialCustom(
                    cls(**build_kwargs(cls, signature)), data_keys=["input", "mask"], same_on_batch=True
                )

                pipeline(image, seg)

                self.assertTrue(
                    bool(torch.equal(seg, reference)),
                    f"{cls.__name__} wrote into the mask it was given",
                )
