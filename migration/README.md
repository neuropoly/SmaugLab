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
