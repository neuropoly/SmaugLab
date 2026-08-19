"""Random draws that `torch.manual_seed` actually reaches, and that DDP ranks agree on.

Several GPU transforms reached for Python's `random.choice` to pick a blur sigma or a
kernel size, inside an `apply_transform` that was otherwise entirely `torch.rand`
driven. Two consequences:

* `torch.manual_seed(...)` does not seed Python's `random`, so a "seeded" run was not
  reproducible. The test suite hid this -- `unit_tests/helpers.py::seed_everything`
  seeds torch, numpy *and* random -- but training does not call that.
* Under DistributedDataParallel each rank has its own `random` state, so ranks picked
  different sigmas for the same batch.

`gpu/fromSeg.py` already contained `_next_shared_seed` / `_shared_rand` written for
exactly this, and never called them. That machinery lives here now, with the `choice`
helper the call sites actually needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")

_SHARED_RNG_COUNTER = 0


def next_shared_seed() -> int:
    """A seed every rank agrees on, different on each call."""
    global _SHARED_RNG_COUNTER  # noqa: PLW0603 -- module-level counter is the point: it makes successive seeds distinct
    _SHARED_RNG_COUNTER += 1
    seed = (int(torch.initial_seed()) + _SHARED_RNG_COUNTER) % (2**63 - 1)
    if dist.is_available() and dist.is_initialized():
        seed_tensor = torch.tensor([seed], dtype=torch.long)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    return seed


def shared_cpu_generator() -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(next_shared_seed())
    return generator


def shared_rand(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Uniform [0, 1) draws; identical across ranks when running under DDP.

    Outside DDP this is just `torch.rand`, so it stays on whatever device and generator
    the caller has already seeded.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return torch.rand(shape, device=device, dtype=dtype)
    rand_cpu = torch.rand(shape, generator=shared_cpu_generator(), device="cpu", dtype=dtype)
    return rand_cpu.to(device=device, dtype=dtype)


def shared_choice(options: Sequence[T]) -> T:
    """Pick one element of `options`, using torch's RNG rather than Python's.

    The drop-in replacement for `random.choice` in a transform.
    """
    if len(options) == 0:
        raise ValueError("cannot choose from an empty sequence")
    draw = float(shared_rand((1,), torch.device("cpu")).item())
    # torch.rand is [0, 1), so the index is already in range; the clamp is belt-and-braces.
    return options[min(int(draw * len(options)), len(options) - 1)]
