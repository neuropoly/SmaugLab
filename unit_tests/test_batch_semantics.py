"""What the GPU pipeline does when a transform fires on part of the batch.

kornia's `_AugmentationBase.transform_inputs` calls

    self.apply_transform(in_tensor[to_apply], params, flags, ...)

when `0 < to_apply.sum() < B` -- the image is sliced down to the rows that drew
below `p`, but `params` is handed over untouched. `AugmentationSequentialOpsCustom`
injects the segmentation into exactly those `params`, so a consumer reading
`params["seg"]` used to get a B-row mask alongside a B'-row image.

That is a shape error for anything that broadcasts (`RandomBrightnessGPU` with
`in_seg`), and something worse for anything that loops `for b in range(N)` with
`N = input.shape[0]`: `RandomRedistributeSegGPU`, `RandomDomainTransferGPU` and
`RandomSynthSegGPU` all ran without complaint while pairing image row `b` with
the segmentation of whichever sample happened to sit at index `b` of the *full*
batch.

Nothing caught it because `helpers.VOLUME_SHAPE` is batch size 1, where
`to_apply` is all-true or all-false and the partial branch is unreachable.
"""

from __future__ import annotations

import torch

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.contrast import RandomBrightnessGPU
from unit_tests.helpers import SmaugLabTestCase

BATCH = 6
SHAPE = (BATCH, 1, 16, 16, 16)


def batched_seg() -> torch.Tensor:
    """One distinguishable blob per sample, so a mis-paired row is visible."""
    seg = torch.zeros(*SHAPE, dtype=torch.float32)
    for b in range(BATCH):
        seg[b, :, b : b + 4, b : b + 4, b : b + 4] = 1.0
    return seg


class TestPartialBatchSelection(SmaugLabTestCase):
    def _pipeline(self, transform):
        # same_on_batch=False is what makes the selection partial; with the
        # sequential's default the whole batch applies or none of it does.
        return AugmentationSequentialCustom(transform, data_keys=["input", "mask"], same_on_batch=False)

    def test_a_partially_applied_transform_does_not_raise(self):
        """`p=0.5` at batch size 6 lands in the partial branch on most seeds."""
        image, seg = torch.rand(*SHAPE), batched_seg()

        for seed in range(8):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                pipeline = self._pipeline(RandomBrightnessGPU(p=0.5, in_seg=1.0, same_on_batch=False))

                out, _ = pipeline(image.clone(), seg.clone())

                self.assertEqual(out.shape, image.shape)
                self.assertTrue(bool(torch.isfinite(out).all()))

    def test_the_injected_segmentation_matches_the_rows_that_were_selected(self):
        """The silent half of the bug, asserted on the contract itself.

        Checking the *output* cannot separate the two cases: the transforms that
        loop over the batch also touch voxels outside the segmentation for
        unrelated reasons. So record what `apply_transform` was actually handed
        and compare it to the rows kornia selected -- row `i` of `params["seg"]`
        must be the segmentation of the `i`-th selected sample, not of the `i`-th
        sample of the full batch.
        """
        image, seg = torch.rand(*SHAPE), batched_seg()
        seen: list[tuple[int, torch.Tensor]] = []

        class SpyBrightness(RandomBrightnessGPU):
            def apply_transform(self, input, params, flags, transform=None):
                seen.append((input.shape[0], params["seg"].detach().clone()))
                return super().apply_transform(input, params, flags, transform)

        torch.manual_seed(0)
        pipeline = self._pipeline(SpyBrightness(p=0.5, in_seg=1.0, same_on_batch=False))
        out, _ = pipeline(image.clone(), seg.clone())

        self.assertTrue(seen, "apply_transform was never called; pick a seed where the transform fires")
        rows, injected = seen[0]
        selected = [b for b in range(BATCH) if bool((out[b] != image[b]).any())]

        self.assertEqual(rows, len(selected), "kornia selected a different number of rows than the output shows")
        self.assertEqual(injected.shape[0], rows, "the injected segmentation does not have one row per selected sample")
        for i, b in enumerate(selected):
            with self.subTest(selected_row=i, batch_row=b):
                self.assertTrue(
                    bool(torch.equal(injected[i], seg[b])),
                    f"row {i} of the injected seg is not sample {b}'s segmentation",
                )


class TestSameOnBatchIsConfigurable(SmaugLabTestCase):
    """`pipeline.same_on_batch` decides whether `p` is per-sample or per-batch.

    `AugTransformsGPU` used to pass `same_on_batch=True` unconditionally. kornia's
    `SequentialBase.update_attribute` then writes that onto every child, so the
    `"same_on_batch": false` that every shipped config sets per transform was
    discarded -- and since `_BasicAugmentationBase.__batch_prob_generator__` draws
    the application mask with `_adapted_sampling((B,), p, same_on_batch)`, a `p`
    of 0.5 stopped meaning "half the samples" and started meaning "half the
    batches, all of it".

    The default stays True so the published configs reproduce.
    """

    def _write(self, tmp_path, same_on_batch):
        import json

        payload = {
            "GPU": {"RandomBrightnessGPU": {"p": 0.5, "same_on_batch": False}},
            "pipeline": {} if same_on_batch is None else {"same_on_batch": same_on_batch},
        }
        tmp_path.write_text(json.dumps(payload))
        return str(tmp_path)

    def _pipeline(self, path):
        from smauglab.transforms.gpu.transforms import AugTransformsGPU

        return AugTransformsGPU(json_path=path)

    def _applied_per_batch(self, pipeline, seeds=8):
        image, seg = torch.rand(*SHAPE), batched_seg()
        counts = []
        for seed in range(seeds):
            torch.manual_seed(seed)
            out, _ = pipeline(image.clone(), seg.clone())
            counts.append(sum(bool((out[b] != image[b]).any()) for b in range(BATCH)))
        return counts

    def test_the_default_is_unchanged(self):
        """No `pipeline.same_on_batch` key must behave exactly as before."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp) / "transform_params_gpu_default.json", None)
            pipeline = self._pipeline(path)

            child = next(iter(pipeline.children()))
            self.assertTrue(child.same_on_batch, "the default must still force same_on_batch onto every child")

            counts = self._applied_per_batch(pipeline)
            self.assertTrue(
                all(c in (0, BATCH) for c in counts),
                f"the default must stay all-or-nothing per batch, got {counts}",
            )

    def test_false_restores_the_configured_value_and_per_sample_p(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp) / "transform_params_gpu_persample.json", False)
            pipeline = self._pipeline(path)

            child = next(iter(pipeline.children()))
            self.assertFalse(child.same_on_batch, "the config's per-transform same_on_batch was still overwritten")

            counts = self._applied_per_batch(pipeline)
            self.assertTrue(
                any(0 < c < BATCH for c in counts),
                f"p should select part of the batch at least once over 8 seeds, got {counts}",
            )
