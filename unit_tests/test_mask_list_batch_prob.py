"""`MaskSequentialOpsCustom.transform_list` must hand kornia a 1-element batch.

The per-mask loop sliced the application mask with `params["batch_prob"][i]`,
which gives a **0-dim** tensor. Kornia then does

    to_apply = batch_prob > 0.5
    if to_apply.all(): ... elif not to_apply.any(): ... else: ...

and for a 0-dim tensor `.all()` and `.any()` are always equal, so the
"some samples apply" branch is unreachable -- and `in_tensor[to_apply]` with a
0-dim boolean indexes along the wrong axis instead of selecting rows.

This path is latent today: `AugmentationSequentialOpsCustom.transform` asserts
the mask is a Tensor before the list branch is reached, so nothing in SmaugLab
can get here. The slice is still wrong, and the assert is the only thing
standing between it and a caller who passes a list of masks -- which is the
signature's whole reason to exist.
"""

from __future__ import annotations

import inspect

import torch

from smauglab.transforms.gpu import base
from unit_tests.helpers import SmaugLabTestCase


class TestListMaskBatchProb(SmaugLabTestCase):
    def test_the_slice_keeps_the_batch_axis(self):
        """What kornia needs, and what the two forms actually give."""
        probs = torch.tensor([1.0, 0.0, 1.0])

        self.assertEqual(probs[0:1].dim(), 1, "a 1-element slice keeps the batch axis")
        self.assertEqual(probs[0].dim(), 0, "a scalar index drops it")

    def test_a_zero_dim_mask_collapses_kornias_branching(self):
        """Why the 0-dim form is not merely untidy."""
        scalar = torch.tensor([1.0, 0.0])[1] > 0.5

        self.assertEqual(bool(scalar.all()), bool(scalar.any()), "all() and any() coincide, so the partial branch is dead")

    def test_the_loop_slices_rather_than_indexes(self):
        """Pinned on the source, because the path is unreachable from the pipeline.

        `AugmentationSequentialOpsCustom.transform` asserts the mask is a Tensor,
        so a list never reaches `transform_list`. Asserting on the text is worth
        more than a test that cannot run the code.
        """
        source = inspect.getsource(base.MaskSequentialOpsCustom.transform_list)

        self.assertNotIn('params["batch_prob"][i]', source, "the scalar index is back")
        self.assertEqual(
            source.count('params["batch_prob"][i : i + 1]'),
            2,
            "both per-mask loops must slice the application mask",
        )
