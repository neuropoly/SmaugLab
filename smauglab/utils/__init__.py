"""NIfTI image handling.

Only `image.py` lives here. `utils.py` used to sit alongside it -- MONAI training-loop
helpers, argparse tuple parsers, a Dice function -- but nothing under `smauglab/`
imported any of it once the `__main__` demo blocks moved out to `scripts/`, so it
shipped in every wheel for the benefit of two standalone scripts. It is now
`scripts/_common.py`.

`image.py` stayed despite having no in-package consumer either: five modules in the
sibling segtransferaug repository import `smauglab.utils.image.Image`, so it is part
of the public API in practice. It is a vendored subset of spinalcordtoolbox's
`image.py` -- see the class docstrings for the upstream links.
"""
