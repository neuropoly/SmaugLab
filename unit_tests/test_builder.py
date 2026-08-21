"""The registry-driven builder, against what the `if` ladders used to produce.

`fixtures/legacy_effective_kwargs.json` records, per config and pipeline mode, the
class and effective constructor kwargs of every transform the hand-written ladders
built. It was captured from the commit before they were deleted, by instrumenting each
transform's `__init__`. This file is the evidence that replacing ~900 lines of dispatch
changed nothing about what gets built.

Three differences are normalised away, because they are the change itself rather than a
difference in behaviour:

* the ladders built one class parameterised by `kernel_type` / `func` / `invert_image`;
  the builder builds the leaf class that replaced it;
* that shared class therefore bound kernel-specific parameters its leaves do not
  declare -- a Scharr transform carried `sigma`, which only the blur reads;
* `p_batch` is new and defaults to kornia's own 1.0 either way.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import re
from pathlib import Path
from typing import ClassVar

import torch

from smauglab.config import OrderSource, PipelineMode, SmaugConfig
from smauglab.registry import Backend, InvalidConfigError
from smauglab.transforms.build import build_gpu_pipeline, build_transforms
from unit_tests.helpers import SmaugLabTestCase

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_effective_kwargs.json"

#: kernel_type value -> the leaf class that replaced it.
KERNEL_LEAF = {
    "'Laplace'": "RandomLaplaceGPU",
    "'Scharr'": "RandomScharrGPU",
    "'GaussianBlur'": "RandomGaussianBlurGPU",
    "'UnsharpMask'": "RandomUnsharpMaskGPU",
    "'RandConv'": "RandomRandConvGPU",
}
#: which kernel-specific parameters each leaf actually reads
LEAF_PARAMS = {
    "RandomLaplaceGPU": set(),
    "RandomScharrGPU": {"absolute"},
    "RandomGaussianBlurGPU": {"sigma"},
    "RandomUnsharpMaskGPU": {"sigma", "unsharp_amount"},
    "RandomRandConvGPU": {"kernel_sizes"},
}
KERNEL_SPECIFIC = {"absolute", "sigma", "unsharp_amount", "kernel_sizes"}
IGNORED_PARAMS = {"p_batch"}


def _function_fingerprints() -> dict[str, str]:
    """Behavioural fingerprint -> leaf class, for the elementwise function transforms.

    The ladder passed lambdas, whose repr carries an address, so the fixture records
    what the function does to a probe rather than what it is called.
    """
    probe = torch.tensor([0.25, 0.5, 0.75])
    leaves = {
        "RandomLog1pGPU": torch.log1p,
        "RandomSqrtGPU": torch.sqrt,
        "RandomSinGPU": torch.sin,
        "RandomExpGPU": torch.exp,
        "RandomSigmoidGPU": torch.sigmoid,
    }
    return {"fn:" + ",".join(f"{x:.6f}" for x in fn(probe).tolist()): name for name, fn in leaves.items()}


FUNCTION_LEAF = _function_fingerprints()


@contextlib.contextmanager
def record_constructions():
    """Record (class name, effective kwargs) for every transform built inside the block.

    The same instrumentation the fixture was captured with, so the two sides are
    measured identically rather than one being re-derived from instance attributes --
    which do not reliably mirror the constructor's parameter names.
    """
    import inspect

    modules = [
        importlib.import_module(f"smauglab.transforms.{name}")
        for name in ("gpu.contrast", "gpu.spatial", "gpu.fromSeg", "gpu.domain_transfer", "synthseg.transforms")
    ]
    records: list[tuple[str, dict]] = []
    patched: list[tuple[type, object]] = []
    probe = torch.tensor([0.25, 0.5, 0.75])

    def describe(value):
        if callable(value):
            try:
                return "fn:" + ",".join(f"{x:.6f}" for x in value(probe).tolist())
            except Exception:
                return "fn:<unevaluable>"
        return repr(value)

    seen: set[type] = set()
    for module in modules:
        for name, obj in list(vars(module).items()):
            if not inspect.isclass(obj) or obj in seen or not issubclass(obj, torch.nn.Module):
                continue
            if not name.startswith(("Random", "_Random", "Zscore")):
                continue
            seen.add(obj)
            original = obj.__init__
            patched.append((obj, original))

            def make(original=original):
                @functools.wraps(original)
                def __init__(self, *args, **kwargs):  # noqa: N807 -- it *is* a dunder; we are replacing one
                    # Leaves inherit __init__ from their base and both are patched, so
                    # only the outermost call counts.
                    outermost = not getattr(self, "_recorded", False)
                    self._recorded = True
                    try:
                        bound = inspect.signature(original).bind(self, *args, **kwargs)
                        bound.apply_defaults()
                        captured = {k: describe(v) for k, v in list(bound.arguments.items())[1:] if k != "kwargs"}
                    except Exception:
                        captured = {}
                    if outermost:
                        records.append((type(self).__name__, captured))
                    return original(self, *args, **kwargs)

                # the registry reads accepted parameters off the signature
                __init__.__signature__ = inspect.signature(original)
                return __init__

            obj.__init__ = make()
    try:
        yield records
    finally:
        for cls, original in patched:
            cls.__init__ = original


def normalise(name: str, kwargs: dict) -> tuple[str, tuple[tuple[str, str], ...]]:
    kwargs = dict(kwargs)
    if name == "_RandomConvBaseGPU":
        name = KERNEL_LEAF[kwargs.pop("kernel_type")]
        kwargs = {k: v for k, v in kwargs.items() if k not in KERNEL_SPECIFIC or k in LEAF_PARAMS[name]}
    elif name == "_RandomFunctionBaseGPU":
        name = FUNCTION_LEAF[kwargs.pop("func")]
    elif name == "_RandomGammaBaseGPU":
        name = "RandomInvGammaGPU" if kwargs.pop("invert_image", "False") == "True" else "RandomGammaGPU"
    kwargs.pop("func", None)
    # Sorted, because dict repr is insertion-ordered and the two sides insert in
    # different orders. tuple-vs-list is a spelling change in the defaults, not a
    # change of value.
    return name, tuple(sorted((k, re.sub(r"^\((.*?),?\)$", r"[\1]", v)) for k, v in kwargs.items() if k not in IGNORED_PARAMS))


class TestBuilderMatchesTheOldLadders(SmaugLabTestCase):
    def setUp(self):
        super().setUp()
        self.legacy = json.loads(FIXTURE.read_text())

    def test_the_fixture_covers_the_shipped_configs(self):
        configs = {key.split("|")[0] for key in self.legacy}
        self.assertGreaterEqual(len(configs), 20, "the fixture should cover most shipped configs")

    def test_every_recorded_pipeline_builds_the_same_transforms(self):
        for key, recorded in self.legacy.items():
            config_name, mode = key.split("|")
            with self.subTest(config=config_name, mode=mode):
                path = Path("smauglab/configs") / config_name
                if not path.is_file():
                    self.skipTest(f"{config_name} is not shipped")
                config = SmaugConfig.from_path(path)
                with record_constructions() as built:
                    build_gpu_pipeline(
                        config.section(Backend.GPU),
                        mode=PipelineMode(mode),
                        options=config.pipeline_options("random_choose"),
                        source=config.source,
                    )
                expected = sorted(str(normalise(n, kw)) for n, kw in recorded)
                actual = sorted(str(normalise(n, kw)) for n, kw in built)
                self.assertEqual(expected, actual)


class TestOrderSource(SmaugLabTestCase):
    """Registry order by default; config key order when the config asks for it."""

    #: Deliberately in the opposite order to PIPELINE_ORDER, so the two sources disagree.
    SECTION: ClassVar[dict] = {
        "ZscoreNormalizationGPU": {"p": 1.0},
        "RandomFlipTransformGPU": {"p": 1.0},
    }

    def test_registry_order_ignores_the_order_of_the_keys(self):
        built = build_transforms(self.SECTION, Backend.GPU)
        self.assertEqual([type(t).__name__ for t, _ in built], ["RandomFlipTransformGPU", "ZscoreNormalizationGPU"])

    def test_config_order_keeps_the_order_of_the_keys(self):
        built = build_transforms(self.SECTION, Backend.GPU, order_source=OrderSource.CONFIG)
        self.assertEqual([type(t).__name__ for t, _ in built], ["ZscoreNormalizationGPU", "RandomFlipTransformGPU"])

    def test_the_pipeline_honours_the_config_setting(self):
        payload = {"GPU": dict(self.SECTION), "pipeline": {"order": "config"}}
        config = SmaugConfig(payload)
        built = build_gpu_pipeline(config.section(Backend.GPU), order_source=config.order_source())
        self.assertEqual([type(t).__name__ for t in built], ["ZscoreNormalizationGPU", "RandomFlipTransformGPU"])


class TestValidation(SmaugLabTestCase):
    def test_an_unknown_augmentation_stops_the_build(self):
        with self.assertRaises(InvalidConfigError):
            build_transforms({"NotAThing": {}}, Backend.GPU)

    def test_an_unknown_parameter_stops_the_build(self):
        with self.assertRaises(InvalidConfigError):
            build_transforms({"RandomFlipTransformGPU": {"nope": 1}}, Backend.GPU)

    def test_comment_keys_are_skipped(self):
        built = build_transforms({"_note": "hi", "RandomFlipTransformGPU": {"p": 1.0}}, Backend.GPU)
        self.assertEqual(len(built), 1)

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(InvalidConfigError) as caught:
            build_transforms({"Nope": {}, "RandomFlipTransformGPU": {"bad": 1}}, Backend.GPU)
        self.assertEqual(len(caught.exception.problems), 2)
