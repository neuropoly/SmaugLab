# Contributing to AugLab

Thanks for contributing. This page covers the development setup, the checks CI
runs, and how versioning and releases work.

## Development setup

```bash
git clone git@github.com:neuropoly/AugLab.git
cd AugLab

python3 -m venv venv
source venv/bin/activate

# PyTorch first, matching your CUDA version (see https://pytorch.org).
# For development and running the tests, the CPU build is enough:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` is the important step: it wires the same Ruff lint and
format hooks CI enforces into your local `git commit`, so you find problems
before pushing.

## Running the checks

```bash
pytest                       # the full suite, ~10 seconds
pytest -m "not slow"         # skip the wheel-building packaging tests
pre-commit run --all-files   # everything CI's lint job runs
ruff check .                 # lint only
ruff format .                # format in place
```

## The test suite

`unit_tests/` runs entirely on CPU with 24×24×24 volumes and needs no image
data on disk, so it is fast enough to gate every pull request.

| File | What it covers |
| --- | --- |
| `helpers.py` | `AugLabTestCase` base class (RNG seeding, test volumes) and config lookup |
| `test_imports.py` | Every module under `auglab/` imports cleanly |
| `test_configs.py` | Every shipped config parses, builds a pipeline, runs a forward pass, and is reproducible under a fixed seed |
| `test_transforms_gpu.py` | Each GPU transform in isolation |
| `test_packaging.py` | Builds the real wheel and checks its contents |

Tests are `unittest.TestCase` subclasses, so they run under either runner:

```bash
pytest                                        # what CI uses
python -m unittest discover -s unit_tests -t .
```

Cases that vary over configs or transforms use `subTest`, so one bad config
does not hide the rest and the failure names the offending item — look for
`SUBFAILED(config=...)` in the output. Derive new test classes from
`AugLabTestCase` to get seeded RNGs and the shared `tiny_volume()` /
`tiny_seg()` helpers.

Transforms in `test_transforms_gpu.py` are discovered by introspection, so a
new transform class is covered as soon as it lands — as long as it can be built
with default arguments. If yours needs configuration, cover it by adding a
config JSON under `auglab/configs/`, which `test_configs.py` picks up
automatically.

Note that these are smoke and contract tests: they check that transforms run,
preserve shape, stay finite, and do not corrupt the segmentation labels. They
do not verify that an augmentation is *visually* or *statistically* correct.

## Style

Ruff handles both linting and formatting; the configuration lives in
`pyproject.toml`. Line length is 140.

If a rule genuinely fights a deliberate choice, add a narrow `# noqa: RULE`
with a short reason on the line rather than widening the global ignore list.

`git blame` is configured to skip the bulk reformatting commit:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Pull requests

1. Branch off `main` (`yourinitials/short-description`).
2. Make the change, with tests.
3. Make sure `pytest` and `pre-commit run --all-files` pass.
4. Open a PR. CODEOWNERS requests reviewers automatically.
5. One approval and green checks are required before merge.

## Versioning

The version comes from the git tag via
[poetry-dynamic-versioning](https://github.com/mtkennerly/poetry-dynamic-versioning). The `version = "0.0.0"` in `pyproject.toml` is a placeholder — **never bump it by
hand**; it is substituted at build time.

To release, tag a commit and publish a GitHub release; `publish.yml` does the
rest.

## Dependency pins

`kornia` is capped at `>=0.7.3,<0.9`. AugLab subclasses kornia's *private*
augmentation internals (`_AugmentationBase`, `RigidAffineAugmentationBase3D`,
`augmentation.container.ops`, `_adapted_rsampling`, `_tuple_range_reader`),
which move between minor releases — 0.8.3 removed `kornia.core.Module` and the
whole `kornia.utils.helpers` module. The `kornia-compat` CI job runs the suite
against both ends of the supported range, so a break shows up here rather than
in a user's training run.

`auglab/transforms/gpu/contrast.py` imports the private
`torchvision.transforms._functional_tensor`. It still exists as of torchvision
0.28, but carries the same risk.
