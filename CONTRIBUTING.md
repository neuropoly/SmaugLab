# Contributing to SmaugLab

Thanks for contributing. This page covers the development setup, the checks CI
runs, and how versioning and releases work.

## Development setup

```bash
git clone git@github.com:neuropoly/SmaugLab.git
cd SmaugLab

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
mypy smauglab/               # CI's typecheck job
smauglab matrix --check      # are the generated README matrix and template current?
```

`smauglab matrix --check` is a CI step rather than a pre-commit hook: populating
the registry imports torch and kornia, which is far too slow to pay on every
commit. Run `smauglab matrix --write` when you add an augmentation.

Some tests need the domain-transfer LUT bank, which is built offline and not
shipped. They skip without it; export `SMAUGLAB_DOMAIN_BANK=/path/to/bank.npz`
to run them.

## The test suite

`unit_tests/` runs entirely on CPU with 24×24×24 volumes and needs no image
data on disk, so it is fast enough to gate every pull request.

| File | What it covers |
| --- | --- |
| `helpers.py` | `SmaugLabTestCase` base class (RNG seeding, test volumes) and config lookup |
| `test_imports.py` | Every module under `smauglab/` imports cleanly |
| `test_registry.py` | Registry mechanics, against synthetic classes |
| `test_registered_augmentations.py` | The real registry, and that the generated matrix and template are current |
| `test_variant_leaves.py` | Each variant leaf fixes its variant and hides it from the config surface |
| `test_builder.py` | Pipeline order, bucketing, runtime context, and strict validation |
| `test_migration.py` | Migrated configs reproduce the pre-registry builder exactly |
| `test_configs.py` | Every shipped config parses, builds a pipeline, runs a forward pass, and is reproducible under a fixed seed |
| `test_trainers.py` | The nnU-Net trainer builds what each config asks for, and puts deep supervision in the right place |
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
`SmaugLabTestCase` to get seeded RNGs and the shared `tiny_volume()` /
`tiny_seg()` helpers.

Transforms in `test_transforms_gpu.py` come from the registry, so a new
augmentation is covered the moment it is registered — no list to update, and no
denylist of helper classes. Transforms with a required constructor argument are
covered through `test_configs.py` instead, which picks up any config JSON added
under `smauglab/configs/`.

`test_migration.py` replays `fixtures/legacy_effective_kwargs.json`: a record of
every constructor call the pre-registry builder made, captured while it still
existed. It is what proves the config rename changed no behaviour, so treat it as
append-only — regenerating it from current code would make it prove nothing.

Note that these are smoke and contract tests: they check that transforms run,
preserve shape, stay finite, and do not corrupt the segmentation labels. They
do not verify that an augmentation is *visually* or *statistically* correct.

## The nnU-Net trainer

There is one trainer class, `nnUNetTrainerDAExtGPU`. It reads a config and builds
whatever the config names: the `CPU` section runs in the dataloader worker, the `GPU`
section runs on the batch in `train_step`. Three classes used to encode that split in
their names; the config already carried it.

The name is load-bearing and must not change: nnU-Net writes the trainer class name
into every checkpoint and resolves the class from it at inference, so renaming it
would make every previously trained model unloadable.

Two behaviours are decided from the config rather than hardcoded:

* **Deep supervision** is downsampled in `train_step` when there are GPU
  augmentations (the mask is deformed there) and in the dataloader when there are
  not. Getting this backwards trains against targets that no longer match the image.
* **Pipeline arrangement** comes from `pipeline.mode` -- `sequential`,
  `random_order` or `random_order_ta`.

## Adding an augmentation

Config keys are class names, exactly, and parameters are constructor arguments,
exactly. Both are checked against the registry, so an augmentation is reachable
from a config only once it is registered.

1. **Write the class.** No `**kwargs` — parameter validation reads the signature,
   so anything hidden behind it is invisible to a config and to `smauglab show`.
   Declare `p` and `p_batch` explicitly on GPU transforms. Give every parameter a
   default that is the value you actually want, not a `None` sentinel you resolve
   in the body: the signature is what the generated template advertises.

2. **Register it.**

   ```python
   @register(
       aug_id=AugId.SCHARR,       # add a member if the concept is new
       backend=Backend.GPU,
       group=AugType.TA,          # GEO / GE / TA, used for random-order bucketing
       order=90,                  # pipeline position; 10-spaced, unique per backend
   )
   class RandomScharrGPU(_RandomConvBaseGPU): ...
   ```

   Third-party classes cannot be decorated; register those in
   `smauglab/transforms/cpu/external.py` instead.

   Less common fields: `forwards_to` when the constructor genuinely passes kwargs
   on to another class, `context_params` for values the trainer supplies at
   runtime, `param_adapters` when a value must be wrapped before use, and
   `external_asset` when it needs a file the wheel does not ship.

3. **Regenerate and commit the artefacts.**

   ```bash
   smauglab matrix --write
   ```

4. **Check it.**

   ```bash
   smauglab show YourTransform
   pytest unit_tests/test_registry.py unit_tests/test_registered_augmentations.py
   ```

If a variant differs only by one fixed argument — a kernel, a function, an
inversion flag — give it its own thin subclass rather than exposing the argument.
One class per config key is what keeps a config from expressing the same
augmentation two different ways.

### Do not write your own kernel, blur or random draw

Four separate 3-D Gaussian blurs, three bias fields and two copies of the
Laplace/Scharr tables accumulated in this repository, each "kept local so the module
stays self-contained". They drifted, and two of them were wrong for a long time — an
uncentred Gaussian that translated the image as well as blurring it, and a 2-D Scharr
kernel that summed to −20. Nothing caught either, because there was nothing to compare
them against.

* Convolution kernels, separable blurs and smooth random fields:
  `smauglab/transforms/kernels.py`.
* Random draws: `torch.rand` / `torch.randint`, or
  `smauglab.transforms.rng.shared_choice` when picking from a sequence. **Never
  `random.choice` or `numpy.random`** — `torch.manual_seed` does not reach them, so a
  seeded run stops being reproducible and DDP ranks silently diverge. The test suite
  will not catch this: `unit_tests/helpers.py::seed_everything` seeds all three
  generators, and training does not.
* Sampling that must vary per batch belongs in a generator's `forward`, not in
  `make_samplers`. kornia calls `make_samplers` **once** and caches what it builds, so
  a value drawn there is fixed for the transform's whole lifetime.
* Read the input channel as `input[:, c].clone()` before writing into it. A bare
  `input[:, c]` is a view, and assigning into it defeats the non-finite guard at the
  end of the loop — the values are already in the batch by then.

The per-channel scaffolding every intensity transform shares — `_channel_stats` /
`_restore_stats` for `retain_stats`, and `_select_and_check` for `in_seg`/`out_seg`
plus the non-finite guard — lives at the top of `smauglab/transforms/gpu/contrast.py`.
Use it rather than writing the block out again.

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

`kornia` is capped at `>=0.7.3,<0.9`. SmaugLab subclasses kornia's *private*
augmentation internals (`_AugmentationBase`, `RigidAffineAugmentationBase3D`,
`augmentation.container.ops`, `_adapted_rsampling`, `_tuple_range_reader`),
which move between minor releases — 0.8.3 removed `kornia.core.Module` and the
whole `kornia.utils.helpers` module. The `kornia-compat` CI job runs the suite
against both ends of the supported range, so a break shows up here rather than
in a user's training run.

`smauglab/transforms/gpu/contrast.py` imports the private
`torchvision.transforms._functional_tensor`. It still exists as of torchvision
0.28, but carries the same risk.
