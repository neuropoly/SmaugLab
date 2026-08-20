"""Augmentation transforms, split by execution backend.

`cpu` wraps batchgeneratorsv2 transforms for the dataloader worker; `gpu` wraps kornia
ones for the training step; `synthseg` holds the generative label-to-image augmentation.

This is deliberately empty of imports for now. It becomes the point that populates
`smauglab.registry` -- importing it runs every `@register(...)` decorator -- once the
transform classes carry those decorators, which is the next change in this series.
"""
