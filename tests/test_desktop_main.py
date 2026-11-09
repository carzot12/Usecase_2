from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from desktop import main as desktop_main


class DesktopResourcePathTests(unittest.TestCase):
    def test_frozen_app_prefers_resources_beside_launched_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "Selected.app/Contents/MacOS/KI-Bildanalyse"
            selected = root / "Selected.app/Contents/Resources/frontend/dist/index.html"
            stale = root / "Older.app/Contents/Resources/frontend/dist/index.html"
            executable.parent.mkdir(parents=True)
            selected.parent.mkdir(parents=True)
            stale.parent.mkdir(parents=True)
            executable.write_bytes(b"")
            selected.write_text("selected", encoding="utf-8")
            stale.write_text("stale", encoding="utf-8")

            with patch.object(desktop_main.sys, "frozen", True, create=True), patch.object(
                desktop_main.sys, "executable", str(executable)
            ), patch.object(
                desktop_main.sys, "_MEIPASS", str(stale.parents[2]), create=True
            ):
                resolved = desktop_main.resource_path("frontend/dist/index.html")

            self.assertEqual(resolved, selected.resolve())


if __name__ == "__main__":
    unittest.main()
