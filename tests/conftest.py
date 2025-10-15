"""Test configuration for ai4kitchen."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

if "datasets" not in sys.modules:
    stub = ModuleType("datasets")

    def _unconfigured_load_dataset(*args, **kwargs):
        raise RuntimeError("tests expected to patch datasets.load_dataset")

    stub.load_dataset = _unconfigured_load_dataset
    sys.modules["datasets"] = stub

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
