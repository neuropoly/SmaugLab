"""GIN, the augmentation and the two things about it that are easy to get wrong.

`RandomGINGPU` draws a fresh convolutional network on every call from raw `N(0, 1)`
weights, with no fan-in scaling. Two consequences carry most of the tests here.

* **The Frobenius renormalisation is what makes that safe**, and it is exact rather than
  statistical: the per-sample output RMS equals the input's for every depth and kernel
  size, measured to 4.1e-07. `RandomRandConvGPU` needs its `1/sqrt(k**3)` because it has
  no such step, and `test_randconv_gain.py` pins that. The invariant here is on RMS, not
  on the standard deviation: the biases and the leaky_relu move energy into the DC term,
  so the standard deviation still varies across settings.
* **The same unscaled weights overflow float16**, and the failure is silent. Simulating
  the stack without `apply_transform`'s float32 forcing, on a 32^3 patch with
  `n_layers=6, kernel_sizes=(3, 5)`, the norm exceeds float16's 65504 in **19 of 20**
  draws; `1 / (inf + 1e-5)` is then exactly 0 and the output is an all-zero patch that
  `torch.isfinite(...).all()` reports as fine. So neither `_select_and_check` nor the
  trainer's non-finite-loss guard can see it, and asserting finiteness alone would not
  catch a regression. `test_a_float16_input_stays_finite_and_energy_matched` asserts the
  energy match instead.

Ouyang, C., et al. (2022). Causality-inspired single-source domain generalization for
medical image segmentation. IEEE TMI 42(4), 1095-1106. DOI 10.1109/TMI.2022.3224067
"""

import torch

from smauglab.transforms.gpu.gin import RandomGINGPU
from unit_tests.helpers import SmaugLabTestCase

#: Depth and kernel-size combinations the scale invariance has to hold across. The last
#: two are deliberately far outside the paper's defaults, where an unnormalised cascade
#: amplifies by four orders of magnitude.
CONFIGS = (
    (1, (1,)),
    (1, (3,)),
    (2, (1, 3)),
    (4, (1, 3)),
    (8, (1, 3)),
    (4, (1, 3, 5, 7)),
    (4, (7,)),
)

DRAWS = 5


def rms(volume: torch.Tensor) -> torch.Tensor:
    """Per-sample root mean square, over channels and space together.

    The quantity GIN's Frobenius renormalisation preserves -- `||x||_F` divided by the
    square root of the per-sample element count, which is fixed within a comparison.
    """
    return volume.reshape(volume.shape[0], -1).pow(2).mean(dim=1).sqrt()


class GinTestCase(SmaugLabTestCase):
    """Shared driver. `apply_transform` is called directly, as the sibling tests do."""

    def apply(self, transform: RandomGINGPU, volume: torch.Tensor, seed: int = 0, seg: torch.Tensor | None = None) -> torch.Tensor:
        torch.manual_seed(seed)
        params = {} if seg is None else {"seg": seg}
        return transform.apply_transform(volume.clone(), params, transform.flags)

    def batch_of_identical_rows(self, rows: int = 3) -> torch.Tensor:
        """A batch whose samples are byte-identical, so any difference in the output is the draw."""
        return torch.rand(1, 1, 16, 16, 16).repeat(rows, 1, 1, 1, 1)


class TestTheOutputScaleIsConfigurationIndependent(GinTestCase):
    def test_the_output_rms_matches_the_input_rms_for_every_depth_and_kernel_size(self):
        """`out_norm="frob"` pins the output energy to the input's, whatever the cascade.

        This is why the kernels are drawn from raw `N(0, 1)` rather than scaled by
        `1/sqrt(k**3)` the way `_RandomConvBaseGPU.get_kernel` scales RandConv's.
        """
        volume = self.tiny_volume()
        reference = rms(volume)
        for n_layers, kernel_sizes in CONFIGS:
            transform = RandomGINGPU(p=1.0, n_layers=n_layers, kernel_sizes=kernel_sizes)
            for seed in range(DRAWS):
                with self.subTest(n_layers=n_layers, kernel_sizes=kernel_sizes, seed=seed):
                    error = float(((rms(self.apply(transform, volume, seed)) - reference) / reference).abs().max())
                    self.assertLess(error, 1e-4, f"output RMS drifted by {error:.2e} relative")

    def test_without_the_frobenius_norm_the_scale_does_track_the_configuration(self):
        """The control for the test above: it measures the renormalisation, not a tautology.

        Turning it off makes the output scale a function of the cascade -- measured spread
        19197x across `CONFIGS` -- which is exactly what an unscaled random net does and
        what the renormalisation exists to undo.
        """
        volume = self.tiny_volume()
        scales = []
        for n_layers, kernel_sizes in CONFIGS:
            transform = RandomGINGPU(p=1.0, n_layers=n_layers, kernel_sizes=kernel_sizes, out_norm="none")
            draws = [float(rms(self.apply(transform, volume, seed))[0]) for seed in range(DRAWS)]
            scales.append(sum(d * d for d in draws) ** 0.5)

        spread = max(scales) / min(scales)
        self.assertGreater(spread, 10.0, f"out_norm='none' should let the scale run away, spread was only {spread:.1f}x")


class TestTheDrawIsSharedExactlyWhenAsked(GinTestCase):
    def test_same_on_batch_true_shares_every_draw(self):
        """One network, one kernel size per layer and one alpha for the whole batch.

        Asserted bitwise: `same_on_batch` repeats the drawn weights across the groups
        rather than switching to `groups=1`, so the rows are not merely close.
        """
        rows = self.batch_of_identical_rows()
        out = self.apply(RandomGINGPU(p=1.0, same_on_batch=True), rows)
        for row in range(1, out.shape[0]):
            with self.subTest(row=row):
                self.assertTrue(bool(torch.equal(out[0], out[row])), "same_on_batch=True gave a sample its own draw")

    def test_same_on_batch_false_gives_every_sample_its_own_network(self):
        """The `groups=B` fold really does hand group `b` its own weights.

        Nothing else in the suite reaches this branch: `test_transforms_gpu.py` and its
        relatives all build `AugmentationSequentialCustom(..., same_on_batch=True)`.
        """
        rows = self.batch_of_identical_rows()
        out = self.apply(RandomGINGPU(p=1.0, same_on_batch=False), rows)
        for row in range(1, out.shape[0]):
            with self.subTest(row=row):
                self.assertFalse(bool(torch.allclose(out[0], out[row])), "identical samples came out identical without same_on_batch")

    def test_a_fixed_seed_reproduces_the_output(self):
        """Every draw goes through torch's RNG, including the per-layer kernel size.

        `_draw_kernel_sizes` uses `smauglab.transforms.rng`, not `random.choice`, so
        `torch.manual_seed` reaches it and DDP ranks agree on it.
        """
        volume = self.tiny_volume()
        transform = RandomGINGPU(p=1.0)
        self.assertTrue(bool(torch.equal(self.apply(transform, volume, 7), self.apply(transform, volume, 7))))


class TestTheBlend(GinTestCase):
    def test_alpha_range_zero_is_the_identity(self):
        """`alpha=0` keeps the original, up to the renormalisation's own epsilon.

        Not bitwise: with `mixed == work` the rescaling still multiplies by
        `||x|| / (||x|| + 1e-5)`, which is 1.19e-07 away from 1 on this patch. Asserted on
        a random volume rather than `constant_volume()`, where that factor happens to
        round to exactly 1.0 and a `torch.equal` would pass by luck.
        """
        volume = self.tiny_volume()
        out = self.apply(RandomGINGPU(p=1.0, alpha_range=(0.0, 0.0)), volume)
        self.assertTrue(bool(torch.allclose(out, volume, atol=1e-6)), f"max deviation {float((out - volume).abs().max()):.2e}")

    def test_alpha_range_one_never_keeps_the_original(self):
        """`alpha=1` takes the network's output whole, so the blend is not silently clamped."""
        volume = self.tiny_volume()
        out = self.apply(RandomGINGPU(p=1.0, alpha_range=(1.0, 1.0)), volume)
        self.assertGreater(float((out - volume).abs().max()), 0.1)


class TestPrecision(GinTestCase):
    def test_a_float16_input_stays_finite_and_energy_matched(self):
        """float32 is forced inside the transform, so the float16 norm cannot overflow.

        The configuration is the one measured to overflow in 19 of 20 draws without the
        forcing. Energy is asserted, not just finiteness: the overflow produces an
        all-zero patch that `isfinite` calls healthy.
        """
        volume = torch.rand(1, 1, 32, 32, 32, dtype=torch.float16)
        reference = float(rms(volume.float())[0])
        transform = RandomGINGPU(p=1.0, n_layers=6, kernel_sizes=(3, 5))
        for seed in range(DRAWS):
            with self.subTest(seed=seed):
                out = self.apply(transform, volume, seed)
                self.assertEqual(out.dtype, torch.float16, "the input dtype was not restored")
                self.assertTrue(bool(torch.isfinite(out).all()), "non-finite output")
                self.assertTrue(bool(out.any()), "the output was zeroed, which is what a float16 norm overflow looks like")
                got = float(rms(out.float())[0])
                self.assertAlmostEqual(got / reference, 1.0, delta=1e-3, msg=f"energy drifted to {got:.4f} from {reference:.4f}")

    def test_an_open_autocast_region_does_not_change_the_result(self):
        """The `autocast_active()`-guarded `enabled=False` region keeps the cascade in float32.

        Bitwise, because the guard means the same kernels run the same convolutions
        whether or not a caller has autocast open -- which is how the trainer calls it.
        """
        volume = self.tiny_volume()
        transform = RandomGINGPU(p=1.0)
        plain = self.apply(transform, volume, 3)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
            under_autocast = self.apply(transform, volume, 3)
        self.assertTrue(bool(torch.equal(plain, under_autocast)), "autocast changed the result, so the cascade ran in reduced precision")


class TestDegeneratePatches(GinTestCase):
    def test_a_constant_patch_stays_constant(self):
        """GIN cannot manufacture structure out of an air patch.

        A constant input gives a constant output whose magnitude the renormalisation
        restores to the input's. The sign may flip -- the network's output can be
        negative where the input was not -- so only the magnitude is pinned.
        """
        volume = self.constant_volume()
        expected = abs(float(volume.flatten()[0]))
        transform = RandomGINGPU(p=1.0)
        for seed in range(DRAWS):
            with self.subTest(seed=seed):
                out = self.apply(transform, volume, seed)
                unique = torch.unique(out)
                self.assertEqual(unique.numel(), 1, f"a constant patch came out with {unique.numel()} values")
                self.assertAlmostEqual(abs(float(unique[0])), expected, delta=1e-3)

    def test_an_all_zero_patch_stays_all_zero(self):
        """`||x||_F = 0` makes the rescaling return zero rather than NaN.

        The network's biases give it a non-zero output even here, so this is the epsilon's
        placement being right: the zero numerator wins over the non-zero denominator.
        """
        volume = torch.zeros_like(self.tiny_volume())
        transform = RandomGINGPU(p=1.0)
        for seed in range(DRAWS):
            with self.subTest(seed=seed):
                out = self.apply(transform, volume, seed)
                self.assertTrue(bool((out == 0).all()), "an empty patch came back non-zero")


class TestChannels(GinTestCase):
    def test_multi_channel_input_is_transformed_jointly(self):
        """One network over all of `apply_to_channel`, with a single joint energy match.

        The selected channels mix -- which is the published method -- and share one
        rescaling factor, so their *joint* RMS is what is preserved, not each channel's
        own. An unselected channel is left byte-identical.
        """
        volume = torch.rand(2, 3, 16, 16, 16)
        out = self.apply(RandomGINGPU(p=1.0, apply_to_channel=(0, 1)), volume)

        self.assertTrue(bool(torch.equal(out[:, 2], volume[:, 2])), "an unselected channel was modified")
        for channel in (0, 1):
            with self.subTest(channel=channel):
                self.assertFalse(bool(torch.equal(out[:, channel], volume[:, channel])), "a selected channel was left alone")

        before, after = rms(volume[:, :2]), rms(out[:, :2])
        self.assertTrue(bool(torch.allclose(after, before, rtol=1e-4)), f"joint RMS moved from {before.tolist()} to {after.tolist()}")

    def test_a_duplicate_apply_to_channel_is_rejected(self):
        """A repeated channel is harmless on the siblings and changes the network here.

        `len(apply_to_channel)` is the cascade's input width, so `(0, 0)` would build a
        two-input net and scatter both its outputs into channel 0.
        """
        with self.assertRaises(ValueError):
            RandomGINGPU(p=1.0, apply_to_channel=(0, 0))

    def test_an_out_of_range_channel_names_the_config_key(self):
        """Checked before the gather, so the message names the key and not a stray index."""
        transform = RandomGINGPU(p=1.0, apply_to_channel=(1,))
        with self.assertRaises(IndexError) as caught:
            self.apply(transform, self.tiny_volume())
        self.assertIn("apply_to_channel", str(caught.exception))


class TestConstructorValidation(GinTestCase):
    def test_invalid_constructor_arguments_are_rejected(self):
        """A config typo fails at construction rather than falling through to a default."""
        cases = {
            "n_layers below one": {"n_layers": 0},
            "interm_channels below one": {"interm_channels": 0},
            "no kernel sizes to draw from": {"kernel_sizes": ()},
            "an even kernel size": {"kernel_sizes": (2,)},
            "a non-positive kernel size": {"kernel_sizes": (0,)},
            "an unknown out_norm": {"out_norm": "nope"},
            "an unknown padding_mode": {"padding_mode": "nope"},
            "an inverted alpha_range": {"alpha_range": (1.0, 0.0)},
        }
        for label, kwargs in cases.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                RandomGINGPU(p=1.0, **kwargs)


class TestPadding(GinTestCase):
    def test_zero_padding_changes_only_the_border(self):
        """`padding_mode` is wired through, and reaches exactly as far as the cascade does.

        Four layers of `k=3` propagate the padded edge one voxel per layer, so the two
        modes differ over the outer four voxels and leave everything inside them
        byte-identical. Measured on the network's raw output (`alpha=1`, `out_norm="none"`)
        so the global rescaling does not smear the border difference over the whole patch.
        """
        volume = self.tiny_volume()
        shared = {"p": 1.0, "n_layers": 4, "kernel_sizes": (3,), "alpha_range": (1.0, 1.0), "out_norm": "none"}
        reflected = self.apply(RandomGINGPU(padding_mode="reflect", **shared), volume, 1)
        zeroed = self.apply(RandomGINGPU(padding_mode="zeros", **shared), volume, 1)

        interior = (slice(None), slice(None), slice(4, -4), slice(4, -4), slice(4, -4))
        self.assertTrue(bool(torch.equal(reflected[interior], zeroed[interior])), "the padding reached further in than the cascade can")
        self.assertGreater(float((reflected - zeroed).abs().max()), 1.0, "the two padding modes produced the same border")


class TestTheCallContract(GinTestCase):
    def test_it_runs_with_an_empty_params_dict(self):
        """`params={}` -- no segmentation, no `batch_prob` -- is how it is called directly.

        `RandomChooseXTransformsGPU` dispatches to `apply_transform` that way, and so does
        every test here.
        """
        volume = self.tiny_volume()
        transform = RandomGINGPU(p=1.0)
        self.assertIsImageLike(transform.apply_transform(volume.clone(), {}, transform.flags), volume, "RandomGINGPU")

    def test_it_does_not_write_into_the_callers_tensor(self):
        """Pinned here as well as in the sweep: the gather and scatter is where this is lost."""
        volume = self.tiny_volume()
        keep = volume.clone()
        transform = RandomGINGPU(p=1.0)
        transform.apply_transform(volume, {}, transform.flags)
        self.assertTrue(bool(torch.equal(volume, keep)), "apply_transform modified its argument")

    def test_region_selection_keeps_the_complement_of_the_segmentation(self):
        """`in_seg=1.0` confines the result to the labelled voxels and nothing else moves."""
        volume = self.tiny_volume()
        seg = self.tiny_seg()
        out = self.apply(RandomGINGPU(p=1.0, in_seg=1.0), volume, seg=seg)
        outside = seg == 0
        self.assertTrue(bool(torch.equal(out[outside], volume[outside])), "the transform leaked outside the segmentation")
        self.assertFalse(bool(torch.equal(out, volume)), "in_seg=1.0 left the whole patch alone")
