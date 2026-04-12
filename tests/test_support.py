from __future__ import annotations

import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


class WorkspaceTestCase(unittest.TestCase):
    workspace_root = WORKSPACE_ROOT

    @staticmethod
    @contextmanager
    def temp_dir(prefix: str = "rkvoice_tts_test_") -> Iterator[Path]:
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            yield Path(temp_dir)