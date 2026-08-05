# Contributing to AugLab

Thanks for contributing. This page covers the development setup, the checks CI
runs, and the repository settings an admin needs to configure.

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
| `test_imports.py` | Every module under `auglab/` imports cleanly |
| `test_configs.py` | Every shipped config parses, builds a pipeline, runs a forward pass, and is reproducible under a fixed seed |
| `test_transforms_gpu.py` | Each GPU transform in isolation |
| `test_packaging.py` | Builds the real wheel and checks its contents |

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

## Releasing

`project.version` in `pyproject.toml` is a manual date string (e.g. `20260109`).

1. Bump `project.version` and merge that to `main`.
2. Create a GitHub release tagged `v<version>` (e.g. `v20260109`).
3. `publish.yml` builds and uploads to PyPI.

The workflow refuses to publish if the tag disagrees with `project.version`, or
if the wheel is missing its config JSONs. To rehearse without touching PyPI,
run the workflow manually via **Actions → publish to PyPI → Run workflow** and
pick `testpypi`.

## Repository settings (admin only)

These cannot be set from files in the repository.

### Secrets

**Settings → Secrets and variables → Actions:**

| Secret | Needed by |
| --- | --- |
| `PYPI_API_TOKEN` | `publish.yml` — without it, releases cannot upload |
| `TEST_PYPI_API_TOKEN` | `publish.yml` TestPyPI dry runs (optional) |
| `CODECOV_TOKEN` | coverage upload in `tests.yml` (optional) |

`publish.yml` uses a `pypi` environment; create it under
**Settings → Environments** and consider adding a required reviewer so a
release upload needs a human to approve it.

### Branch protection

This is the "Branch protection, requiring code-review" part of issue #34.
Via **Settings → Rules → Rulesets**, or with the `gh` CLI:

```bash
gh api -X PUT repos/neuropoly/AugLab/branches/main/protection \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "pre-commit",
      "test (python 3.10)",
      "test (python 3.11)",
      "test (python 3.12)",
      "build distribution"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

Set this up *after* the first PR has run, so the status check names exist and
GitHub can match them.
