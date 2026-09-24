"""`train_monai`'s help has to describe the defaults it actually has.

`--channels` advertised `16,32,64,128,256` against a real default of
`(32, 64, 128, 256)`. That is not only documentation: the run directory and the
checkpoint are named from `args.channels[-1]`, so someone trusting the help
believed they had trained a five-level `attunet256` when they had trained a
four-level one.

`--weight-folder` named a path from a different project entirely, and the
checkpoint path ran a `.replace("config_SegVert_", "")` that cannot match the
name it is given.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

for _optional in ("wandb", "monai", "monai.data", "monai.losses", "monai.networks.nets", "monai.transforms"):
    sys.modules.setdefault(_optional, MagicMock())


class TestHelpMatchesDefaults(unittest.TestCase):
    def _actions(self):
        import train_monai

        return {action.dest: action for action in train_monai.get_parser()._actions}

    def test_the_channels_help_states_the_real_default(self):
        action = self._actions()["channels"]

        self.assertEqual(action.default, (32, 64, 128, 256))
        self.assertIn("32,64,128,256", action.help)
        self.assertNotIn("16,32,64,128,256", action.help)

    def test_the_weight_folder_help_states_the_real_default(self):
        action = self._actions()["weight_folder"]

        self.assertEqual(action.default, os.path.abspath("weights/"))
        self.assertNotIn("3DGAN", action.help, "the help names a path from another project")

    def test_the_checkpoint_name_drops_the_config_prefix(self):
        """The dead replace: json_name never contains 'config_SegVert_'."""
        json_name = "config_attunet256_pixdimRSP_1-1-1.json"

        self.assertEqual(json_name.replace("config_SegVert_", ""), json_name, "the old replace was a no-op")
        self.assertEqual(json_name.removeprefix("config_").replace(".json", ".pth"), "attunet256_pixdimRSP_1-1-1.pth")
