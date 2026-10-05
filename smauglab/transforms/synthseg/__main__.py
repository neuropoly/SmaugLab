"""Smoke test for the SynthSeg modules: `python -m smauglab.transforms.synthseg`.

The smoke tests used to be `if __name__ == "__main__"` blocks inside `generator.py`
and `transforms.py`, run as `python -m smauglab.transforms.synthseg.generator` and
`... .transforms`. The second of those had not worked since the registry landed:

    RegistryError: GPU augmentation 'RandomSynthSegGPU' is already registered
                   (as smauglab.transforms.synthseg.transforms)

`__init__.py` imports both modules, so `python -m` on either one puts its file in
`sys.modules` twice -- once under its own name, once as `__main__` -- and executes it
twice. Python warns about this by itself ("found in sys.modules ... may result in
unpredictable behaviour"); for the registering module it is fatal, because `@register`
fires a second time and the duplicate-name guard rejects it.

Running the *package* loads each module exactly once: runpy imports
`smauglab.transforms.synthseg`, then executes this file, whose imports are already
satisfied from `sys.modules`. So there is one copy of every class, the registry sees
one registration, and the smoke tests exercise the same objects importers get --
rather than near-identical duplicates that `isinstance` would disagree about.
"""

from __future__ import annotations

from smauglab.transforms.synthseg import generator, transforms


def main() -> int:
    for name, module in (("generator", generator), ("transforms", transforms)):
        print(f"--- {name}")
        module.smoke_test()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
