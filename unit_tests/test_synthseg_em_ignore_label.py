"""EM label completion on a label map that carries nnU-Net's ignore label.

nnU-Net writes -1 for voxels outside the annotated region, and the preprocessed
volumes of a partially-annotated dataset are full of them. `em_subdivide_labels`
took each label's voxel count from a histogram of `label_map.clamp_min(0)`, which
counts every -1 as background, and then selected that label's voxels with
`label_map == p`, which does not. The count was larger than the selection, so the
`randint(0, count)` subsample used to fit the background GMM could draw an index
past the end of the gathered values.

On CPU that is an `IndexError`. On CUDA it is an asynchronous `index out of bounds`
device-side assert, attributed to whichever kernel happens to run next, and it
only fires when an oversized index is actually drawn -- for a 128^3 patch with six
ignore voxels and 100 000 fit draws, about two runs in five. The symptom was an
occasional crash that pointed at innocent code.

The first test is the invariant itself and is deterministic; the rest exercise the
call that used to raise, and the all-ignore patch that the clamp made
accidentally representable.
"""

from __future__ import annotations

import torch

from smauglab.transforms.synthseg import functional as FN
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (16, 16, 16)
VOXELS = SHAPE[0] * SHAPE[1] * SHAPE[2]
FOREGROUND = 8 * 8 * 8
#: Smaller than the number of background voxels, so the subsample path is the one
#: under test: with `max_fit_voxels >= count` the function uses every voxel and
#: never draws an index at all.
MAX_FIT = 64
#: Enough ignored voxels that an out-of-range draw is a near-certainty rather than a
#: coin flip: the old histogram would report 3584 background voxels where only 1584
#: can be selected, so P(no bad index in 64 draws) = (1584/3584)^64 < 1e-23.
N_IGNORE = 2000


def pair_with_ignore(n_ignore: int = N_IGNORE, batch: int = 2):
    """A label map with background, one structure and `n_ignore` ignored voxels."""
    depth, height, width = SHAPE
    labels = torch.zeros(batch, 1, depth, height, width, dtype=torch.long)
    labels[..., 4:12, 4:12, 4:12] = 1
    labels.reshape(batch, -1)[:, :n_ignore] = -1
    image = torch.rand(batch, 1, depth, height, width)
    return labels, image


class EmIgnoreLabelTests(SmaugLabTestCase):
    def test_counts_match_the_selection_they_are_used_for(self):
        """The histogram must count exactly the voxels `label_map == p` selects.

        This is the invariant the crash violated, checked directly so a future
        rewrite of the histogram cannot reintroduce it without failing here.
        """
        labels, _ = pair_with_ignore()
        counts = FN._label_counts_per_sample(labels)
        for b in range(labels.shape[0]):
            for p in counts.sum(dim=0).nonzero().flatten().tolist():
                with self.subTest(sample=b, label=p):
                    self.assertEqual(int(counts[b][p]), int((labels[b, 0] == p).sum()))

    def test_ignore_voxels_are_not_counted_as_background(self):
        labels, _ = pair_with_ignore()
        counts = FN._label_counts_per_sample(labels)
        # The structure is partly overwritten by the ignore block, so the foreground
        # count is read back rather than assumed.
        foreground = int((labels[0, 0] == 1).sum())
        self.assertEqual(int(counts[0][0]), VOXELS - foreground - N_IGNORE)
        self.assertEqual(int(counts.sum()), labels.numel() - N_IGNORE * labels.shape[0])

    def test_subdivide_survives_the_ignore_label(self):
        """The call that used to raise. `n_iters=1` keeps it quick -- the index is
        drawn before the first iteration, so the path under test is covered."""
        labels, image = pair_with_ignore()
        fine, gen, out = FN.em_subdivide_labels(
            image,
            labels,
            n_foreground_clusters=2,
            background_clusters_range=(3, 3),
            background_label=0,
            n_iters=1,
            max_fit_voxels=MAX_FIT,
        )
        self.assertEqual(fine.shape, labels.shape)
        self.assertEqual(len(gen), len(out))
        self.assertTrue(set(out).issubset({0, 1}), out)
        # The ignored voxels keep generation label 0, which merges back to the
        # background parent: they are treated as background, as they were before.
        self.assertEqual(int(fine.reshape(labels.shape[0], -1)[0, :N_IGNORE].abs().sum()), 0)

    def test_all_ignore_patch_returns_one_region(self):
        """A patch with no annotated voxel at all resolves instead of raising."""
        labels = torch.full((1, 1, *SHAPE), -1, dtype=torch.long)
        image = torch.rand(1, 1, *SHAPE)
        fine, gen, out = FN.em_subdivide_labels(image, labels, background_label=0, n_iters=1, max_fit_voxels=MAX_FIT)
        self.assertEqual(gen, [0])
        self.assertEqual(out, [0])
        self.assertEqual(int(fine.abs().sum()), 0)
