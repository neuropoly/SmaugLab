import argparse
import importlib.resources
import shutil
from pathlib import Path

import nnunetv2

from smauglab import trainers

#: Module to copy -> the trainer classes it provides.
#:
#: The choice is a *module* name, but `nnUNetv2_train -tr` wants a *class* name, and
#: the two differ: nnUNetTrainerDAExt.py holds nnUNetTrainerDAExtGPU, and
#: nnUNetTrainerTest.py holds two classes. Following the old help text and passing
#: the same string to nnU-Net got "trainer class not found", so the help says which
#: class each module provides.
TRAINER_CLASSES = {
    "nnUNetTrainerDAExt": ("nnUNetTrainerDAExtGPU",),
    "nnUNetTrainerTest": ("nnUNetTrainerTest", "nnUNetTrainerTestGPU"),
}


def _trainer_help() -> str:
    provides = "; ".join(f"{module} provides {', '.join(classes)}" for module, classes in TRAINER_CLASSES.items())
    return f"Trainer module to copy into nnU-Net. Pass the class name, not this one, to `nnUNetv2_train -tr`: {provides}."


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="This script copies a smauglab nnUNetTrainer inside the nnunet folder.")
    parser.add_argument(
        "-t",
        "--trainer",
        choices=sorted(TRAINER_CLASSES),
        type=str,
        required=True,
        help=_trainer_help(),
    )
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing trainer.")
    args = parser.parse_args(argv)

    # Get trainer name
    trainer_name = args.trainer
    overwrite = args.overwrite

    # Add trainer
    add_trainer(trainer_name, overwrite=overwrite)


def add_trainer(trainer_name: str, overwrite: bool = False):

    # Find trainer path.
    # importlib.resources returns a Traversable, which only promises open()/read_bytes()
    # -- not .exists(), and not something shutil.copy accepts. Copying *into* the
    # installed nnunetv2 package needs a real directory on disk regardless (nnU-Net
    # cannot run from a zipped install), so resolve both ends to concrete paths here.
    # Same idiom as unit_tests/helpers.py.
    trainers_path = Path(str(importlib.resources.files(trainers)))
    if trainer_name not in TRAINER_CLASSES:
        raise ValueError(f"Trainer {trainer_name} not recognized. Choices are: {', '.join(sorted(TRAINER_CLASSES))}.")
    source_trainer = trainers_path / f"{trainer_name}.py"

    # Find nnUNet path
    nnunetv2_path = Path(str(importlib.resources.files(nnunetv2)))
    nnunet_trainers_path = nnunetv2_path / "training" / "nnUNetTrainer"

    # Copy trainer
    output_path = nnunet_trainers_path / source_trainer.name
    if not output_path.exists() or overwrite:
        shutil.copy(source_trainer, output_path)

        # Confirmation message, naming what to pass to nnU-Net -- which is the
        # class, not the module copied here.
        classes = ", ".join(TRAINER_CLASSES[trainer_name])
        print(f"Trainer {trainer_name} was added to {output_path}")
        print(f"Train with: nnUNetv2_train ... -tr {classes}")
    else:
        print(f"Trainer {trainer_name} already exists at {output_path}. Use --overwrite to replace it.")


if __name__ == "__main__":
    main()
