"""Resampling must not translate the volume.

Torch's ``F.interpolate(mode="nearest")`` is not half-pixel centred: it maps
``src = floor(dst * scale)``, so it samples the *left edge* of each output voxel
rather than its centre. One such call drifts the content about half a voxel
toward higher indices; a down-then-up pair drops ~0.5 voxels in each step and
lands on ``out[i] = in[i - 1]`` almost independently of the scale factor.

Nothing about that is visible in a foreground count -- the volume barely changes
-- but the image path next to it is trilinear and does *not* move, so the label
and the intensities come apart. ``"nearest-exact"`` is the centred variant and
is what these sites use.

The probe is a soft step edge: the sub-voxel location of its 0.5 crossing before
and after the op gives the translation directly, with none of the dependence on
object size that a centroid has.
"""

from __future__ import annotations

import torch

from smauglab.transforms.gpu.spatial import RandomLowResTransformGPU
from smauglab.transforms.synthseg.functional import mimic_acquisition
from unit_tests.helpers import SmaugLabTestCase

N = 96

# The nearest-neighbour tie-break alone is worth up to half a voxel per call and
# has no preferred direction, so the mean over positions and factors has to clear
# it comfortably. Plain "nearest" scores +1.16 (low-res) and +0.76 (mimic).
MAX_MEAN_DRIFT = 0.25


def _step(position: float) -> torch.Tensor:
    """A [1, 1, N, 4, 4] volume that ramps 0 -> 1 across one voxel at `position`."""
    line = torch.clamp(torch.arange(N).float() - position + 0.5, 0.0, 1.0)
    return line.view(1, 1, N, 1, 1).expand(1, 1, N, 4, 4).contiguous()


def _edge(volume: torch.Tensor) -> float:
    """Sub-voxel index where the profile crosses 0.5."""
    line = volume[0, 0, :, 0, 0]
    above = (line >= 0.5).nonzero()
    index = int(above[0])
    lo, hi = float(line[index - 1]), float(line[index])
    return index - 1 + (0.5 - lo) / (hi - lo)


def _mean_drift(op, factors) -> float:
    drifts = []
    for factor in factors:
        down = max(1, round(factor * N))
        if down == N:
            continue
        # Irrational-ish spacing so the edge does not sit on the same phase of the
        # sampling grid every time, which is exactly where the bias hides.
        for position in torch.arange(38.0, 58.0, 0.37).tolist():
            volume = _step(position)
            drifts.append(_edge(op(volume, down)) - _edge(volume))
    return float(torch.tensor(drifts).mean())


class TestResamplingDoesNotTranslate(SmaugLabTestCase):
    def test_low_res_mask_branch_stays_put(self):
        """`RandomLowResTransformGPU` down- and up-samples a mask with nearest."""
        transform = RandomLowResTransformGPU()

        def op(volume, down):
            from kornia.constants import DataKey

            scale = down / N
            params = {"scale": torch.tensor([[scale, scale, scale]])}
            return transform.apply_transform(volume, params, {"data_keys": [DataKey.MASK]})

        drift = _mean_drift(op, torch.linspace(0.5, 0.98, 12).tolist())

        self.assertLess(abs(drift), MAX_MEAN_DRIFT, f"the mask drifted {drift:+.3f} voxels")

    def test_mimic_acquisition_stays_put(self):
        """SynthSeg's partial-volume step; the label map it pairs with is not resampled."""

        def op(volume, down):
            current = torch.ones(3)
            return mimic_acquisition(volume, current, current * (N / down), (N, N, N))

        drift = _mean_drift(op, torch.linspace(0.25, 0.9, 12).tolist())

        self.assertLess(abs(drift), MAX_MEAN_DRIFT, f"the image drifted {drift:+.3f} voxels")

    def test_the_trilinear_image_path_is_the_reference(self):
        """It was always centred -- this is what the nearest paths have to match."""
        transform = RandomLowResTransformGPU()

        def op(volume, down):
            scale = down / N
            params = {"scale": torch.tensor([[scale, scale, scale]])}
            return transform.apply_transform(volume, params, {})

        drift = _mean_drift(op, torch.linspace(0.5, 0.98, 12).tolist())

        self.assertLess(abs(drift), MAX_MEAN_DRIFT, f"the image drifted {drift:+.3f} voxels")
