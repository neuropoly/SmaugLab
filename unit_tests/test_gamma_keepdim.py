"""`keepdim` is kornia's output-shape flag, not a reduction argument.

`_RandomGammaBaseGPU.apply_transform` computed its per-sample min and max with
`flat_data.min(dim=1, keepdim=self.keepdim)`. `self.keepdim` is kornia's
"keep the output shape the same as the input" flag; passing it here made the
intermediate shape depend on an unrelated setting.

It was inert -- both `[N]` and `[N, 1]` have `numel == N` and survive the
`view(reshape_dims)` that follows -- so what this test pins is that the gamma
transform gives the same answer whichever way `keepdim` is set, which is the
property that was true only by accident.
"""

import torch

from smauglab.transforms.gpu.contrast import RandomGammaGPU, RandomInvGammaGPU
from unit_tests.helpers import SmaugLabTestCase


class TestGammaIgnoresKeepdim(SmaugLabTestCase):
    def test_the_result_does_not_depend_on_keepdim(self):
        for cls in (RandomGammaGPU, RandomInvGammaGPU):
            for batch in (1, 4):
                with self.subTest(transform=cls.__name__, batch=batch):
                    image = torch.rand(batch, 1, 8, 8, 8)

                    outputs = []
                    for keepdim in (False, True):
                        torch.manual_seed(0)
                        transform = cls(p=1.0, keepdim=keepdim)
                        outputs.append(transform.apply_transform(image.clone(), {}, transform.flags))

                    torch.testing.assert_close(outputs[0], outputs[1])

    def test_it_still_changes_the_image(self):
        image = torch.rand(2, 1, 8, 8, 8)
        torch.manual_seed(0)
        transform = RandomGammaGPU(p=1.0)

        out = transform.apply_transform(image.clone(), {}, transform.flags)

        self.assertFalse(bool(torch.equal(out, image)))
        self.assertTrue(bool(torch.isfinite(out).all()))
