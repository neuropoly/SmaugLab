from smauglab.config import load_config
from smauglab.registry import Backend
from smauglab.transforms.build import PipelineMode, build_gpu_pipeline
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
        )
        # same_on_batch keeps the mask aligned with the image; see
        # AugmentationSequentialOpsCustom in base.py.
        super().__init__(*transforms, data_keys=["input", "mask"], same_on_batch=True)
