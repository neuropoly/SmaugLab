"""SynthSeg deforms the label map, then crops it -- not the other way round.

The module docstring states the order as

    spatial deform (affine + diffeomorphic SVF, on labels, nearest)
      -> [optional random crop]
      -> left/right flip ...

and lab2im's `RandomSpatialDeformation(output_shape=...)` resamples the *full*
label map onto the output grid. The code ran `_random_crop` first.

That is not merely a stale comment. `warp_volume` pads with `zeros`, so
deforming a volume that has already been cropped pulls background in from
outside the crop box, where the reference implementation pulls in the real
anatomy that sits there.

No shipped config is affected: `output_shape` defaults to None and is not among
`RandomSynthSegGPU`'s accepted parameters, so the crop only runs for a caller
constructing `SynthSegGenerator` directly. Both facts are asserted below so the
claim is checked rather than believed.
"""

from __future__ import annotations

import torch

from smauglab import registry
from smauglab.registry import Backend
from smauglab.transforms.synthseg.generator import SynthSegGenerator
from unit_tests.helpers import SmaugLabTestCase

FULL = (24, 24, 24)
CROPPED = (16, 16, 16)


def dense_labels(shape=FULL) -> torch.Tensor:
    """A label map with no background at all, so any zero is imported padding."""
    labels = torch.ones(1, 1, *shape, dtype=torch.long)
    labels[:, :, : shape[0] // 2] = 2
    return labels


class TestCropHappensAfterTheDeformation(SmaugLabTestCase):
    def _generator(self, **kwargs):
        return SynthSegGenerator(
            generation_labels=[0, 1, 2],
            output_labels=[0, 1, 2],
            output_shape=CROPPED,
            apply_affine=True,
            apply_nonlinear=False,
            flipping=False,
            **kwargs,
        )

    def test_the_output_has_the_requested_shape(self):
        torch.manual_seed(0)

        _, labels = self._generator()(dense_labels())

        self.assertEqual(tuple(labels.shape[2:]), CROPPED)

    def test_far_less_background_is_imported(self):
        """The behavioural difference, measured rather than asserted in prose.

        Every voxel of the input carries a label, so a background voxel in the
        output can only be padding that `warp_volume` invented. Deforming an
        already-cropped box pulls that in constantly, because the box is small and
        the padding starts immediately outside it; deforming the full map first
        only reaches padding when the crop lands against the volume edge.

        Measured over 40 seeds on a 24-cube input cropped to 16:

            crop -> deform:  mean 13.5% background, 0/40 draws with none
            deform -> crop:  mean  3.3% background, 7/40 draws with none

        The thresholds sit between the two, so this fails on the old order.
        """
        background = []
        for seed in range(40):
            torch.manual_seed(seed)

            _, labels = self._generator()(dense_labels())

            background.append(float((labels == 0).float().mean()))

        mean_background = sum(background) / len(background)
        self.assertLess(mean_background, 0.08, f"too much padding imported: mean {mean_background:.4f}")
        self.assertTrue(
            any(value == 0.0 for value in background),
            "no draw came through without imported padding, which the deform-first order should allow",
        )

    def test_the_order_is_the_one_the_docstring_states(self):
        import inspect

        source = inspect.getsource(SynthSegGenerator.forward)

        self.assertLess(
            source.index("spatial deformation of the LABEL MAP"),
            source.index("random crop to output_shape"),
            "the crop runs before the deformation again",
        )


class TestTheCropIsUnreachableFromAConfig(SmaugLabTestCase):
    """Why no shipped pipeline changes."""

    def test_output_shape_is_not_a_config_parameter(self):
        registry.load_all()
        entry = registry.get("RandomSynthSegGPU", Backend.GPU)

        self.assertNotIn("output_shape", registry.accepted_params(entry))

    def test_the_generator_defaults_to_no_crop(self):
        generator = SynthSegGenerator(generation_labels=[0, 1], output_labels=[0, 1])

        self.assertIsNone(generator.output_shape)
