from smauglab.config import PipelineMode, load_config
from smauglab.registry import Backend
from smauglab.transforms.build import build_gpu_pipeline
from smauglab.transforms.gpu.base import AugmentationSequentialCustom


class AugTransformsGPU(AugmentationSequentialCustom):
    """GPU augmentation pipeline, built from a config section via the registry.

    The ~370-line `if` ladder this replaces decided the class, the parameters and
    the pipeline position of every augmentation inline; all three now come from the
    registry, and `smauglab.transforms.build` does the dispatch once for all three
    GPU pipeline modes.
    """

    mode: PipelineMode = PipelineMode.SEQUENTIAL

    def __init__(self, json_path: str):
        config = load_config(str(json_path))
        self.transform_params = config.section(Backend.GPU)
        transforms = build_gpu_pipeline(
            self.transform_params,
            mode=self.mode,
            options=config.pipeline_options("random_choose"),
            source=config.source,
            order_source=config.order_source(),
        )
        # None, not False: kornia only overwrites a child's own `same_on_batch`
        # when the argument is not None, and every config sets the flag per
        # transform. Passing True discards all of those -- and, because kornia
        # draws the per-sample application mask with the same flag, turns every
        # configured `p` into a single batch-wide coin flip.
        #
        # The mask stays aligned with the image either way: that comes from
        # AugmentationSequential replaying the same ParamItem over both, not
        # from this flag.
        super().__init__(
            *transforms,
            data_keys=["input", "mask"],
            same_on_batch=True if config.same_on_batch() else None,
        )
