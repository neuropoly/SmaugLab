"""Smoke test for the SynthSeg modules: `python -m smauglab.transforms.synthseg`.

The package, not a module inside it. `__init__.py` imports both `generator` and
`transforms`, so `python -m` on either one puts its file in `sys.modules` twice -- once
under its own name, once as `__main__` -- and executes it twice. CPython warns about that
on its own ("found in sys.modules ... may result in unpredictable behaviour"); for the
registering module it is fatal, because `@register` fires again:

    RegistryError: GPU augmentation 'RandomSynthSegGPU' is already registered
                   (as smauglab.transforms.synthseg.transforms)

Running the package loads each module exactly once, so the registry sees one registration
and the smoke tests exercise the same objects importers get.
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
