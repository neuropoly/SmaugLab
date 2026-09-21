# docs

## `hash_migration.json`

`segtransferaug/run_trainings.py` names each experiment directory

    Dataset{id}-{trainer}-aug-{transform_hash}-c-{config_hash}

where `config_hash` is a sha256 over the canonical JSON of the config and
`transform_hash` is a sha256 over the text of `smauglab/transforms/gpu/transforms.py`.

The registry migration moved both: every config was rewritten to class-name keys, and
the GPU builder that file contained was replaced by a registry-driven one. Runs made
before the migration keep their old directory names, so anything that looks a run up
by hash needs this table. The canonicalisation itself is unchanged and is asserted
byte-for-byte against the original implementation, so a config that did not change
would still hash the same.

Two old hashes map to configs that are now byte-identical:

* `0b639cf1` covered `transform_params_gpu_default01-23.json` and
  `transform_params_one-sequence-to-segment-them-all.json`, which had the same content
  all along under different names.
* `transform_params_gpu_default01-23_ICGT_plus.json` had its own hash only because of
  the dead `ImageContrastGPUTransform` block; with that gone it is a duplicate of
  `default01-23`.

Known consumer, updated to accept both: `segtransferaug/refinement/config.py`, whose
`DOMAIN_TRANSFER_HASHES` selects the fusion source pool.

## The paper configs

Two shipped configs are the migrated form of a run that went into the paper's tables
(`segtransferaug/paper_results/setups.py`). Neither carries a `_comment`, deliberately:
each is exactly what `migrate.py` emits for the corresponding frozen config, so the
equivalence is a one-command check rather than a claim.

| shipped config | new hash | old hash | paper setup | migrated from |
| --- | --- | --- | --- | --- |
| `transform_params_one-sequence-to-segment-them-all.json` | `c69b9872` | `0b639cf1` | Ours | any `Dataset80x-nnUNetTrainerDAExtGPU-aug-ac8f2a27-c-0b639cf1` |
| `transform_params_paper-synthseg.json` | `0d5d802f` | `77e8ea61` | SynthSeg | `trainings/Dataset803-nnUNetTrainerDAExtGPU-aug-a611732f-c-77e8ea61` |

`-c-77e8ea61` is the reason the second file exists at all: it never had a source on the
old stack. It is `..._Synthseg.json` with `SynthSeg.em_label_completion` flipped to
`true`, and that edit was only ever made in memory, so the sole surviving copy is the
`fold_0/transform_params_gpu_used_for_training.json` frozen inside each of its three run
folders. To re-derive either file:

    python migration/migrate.py <run>/<Dataset…>/<trainer>__<plans>__<setting>/fold_0/transform_params_gpu_used_for_training.json -o /tmp/check.json

Despite the name, "SynthSeg" is not "Ours plus SynthSeg": it is Ours with all six
transfer augmentations (`RandomScharrGPU`, `RandomUnsharpMaskGPU`, `RandomRandConvGPU`,
`RandomRedistributeSegGPU`, `RandomInverseGPU`, `RandomHistogramEqualizationGPU`) set to
`p=0` and `RandomSynthSegGPU` at `p=1.0` in their place. The general-enhancement half is
identical between the two.
