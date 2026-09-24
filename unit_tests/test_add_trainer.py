"""`smauglab_add_nnunettrainer -t ...` names a module; nnU-Net wants a class.

The `--trainer` choices are module names (`nnUNetTrainerDAExt`), and the help
said "nnUNetTrainer to be copied" -- but the class inside that module is
`nnUNetTrainerDAExtGPU`, and `nnUNetTrainerTest.py` holds two classes. Anyone
who followed the help and passed the same string to `nnUNetv2_train -tr` got
"trainer class not found".

The class name is load-bearing: nnU-Net writes it into every checkpoint and
resolves the class from it at inference.
"""

from __future__ import annotations

import importlib
import unittest

from smauglab.add_trainer import TRAINER_CLASSES, _trainer_help, main


class TestTrainerClassesAreReal(unittest.TestCase):
    def test_every_declared_class_exists_in_its_module(self):
        for module_name, classes in TRAINER_CLASSES.items():
            module = importlib.import_module(f"smauglab.trainers.{module_name}")
            for class_name in classes:
                with self.subTest(module=module_name, cls=class_name):
                    self.assertTrue(hasattr(module, class_name), f"{module_name}.py does not define {class_name}")

    def test_every_trainer_class_in_the_package_is_declared(self):
        """So a new trainer cannot be shipped without the help mentioning it."""
        import inspect

        for module_name, classes in TRAINER_CLASSES.items():
            module = importlib.import_module(f"smauglab.trainers.{module_name}")
            defined = {
                name
                for name, obj in vars(module).items()
                if inspect.isclass(obj) and obj.__module__ == module.__name__ and name.startswith("nnUNetTrainer")
            }
            with self.subTest(module=module_name):
                self.assertEqual(defined, set(classes))

    def test_the_help_names_the_classes(self):
        help_text = _trainer_help()

        for classes in TRAINER_CLASSES.values():
            for class_name in classes:
                with self.subTest(cls=class_name):
                    self.assertIn(class_name, help_text)

    def test_an_unknown_trainer_is_rejected(self):
        with self.assertRaises(SystemExit):
            main(["-t", "nnUNetTrainerDAExtGPU"])
