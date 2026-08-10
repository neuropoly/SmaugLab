import argparse
import importlib.resources
import shutil
from pathlib import Path

import nnunetv2

from smauglab import trainers


def main():
    parser = argparse.ArgumentParser(description="This script copies a smauglab nnUNetTrainer inside the nnunet folder.")
    parser.add_argument(
        "-t",
        "--trainer",
        choices=["nnUNetTrainerDAExt", "nnUNetTrainerTest"],
        type=str,
        required=True,
        help="nnUNetTrainer to be copied. Choices are: nnUNetTrainerDAExt and nnUNetTrainerTest",
    )
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing trainer.")
    args = parser.parse_args()

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
    if trainer_name == "nnUNetTrainerDAExt":
        source_trainer = trainers_path / "nnUNetTrainerDAExt.py"
    elif trainer_name == "nnUNetTrainerTest":
        source_trainer = trainers_path / "nnUNetTrainerTest.py"
    else:
        raise ValueError(f"Trainer {trainer_name} not recognized.")

    # Find nnUNet path
    nnunetv2_path = Path(str(importlib.resources.files(nnunetv2)))
    nnunet_trainers_path = nnunetv2_path / "training" / "nnUNetTrainer"

    # Copy trainer
    output_path = nnunet_trainers_path / source_trainer.name
    if not output_path.exists() or overwrite:
        shutil.copy(source_trainer, output_path)

        # Confirmation message
        print(f"Trainer {trainer_name} was added to {output_path}")
    else:
        print(f"Trainer {trainer_name} already exists at {output_path}. Use --overwrite to replace it.")


if __name__ == "__main__":
    main()
