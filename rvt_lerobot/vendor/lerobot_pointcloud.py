"""Loader for the point-cloud code proposed in huggingface/lerobot#4696.

Imported from the LeRobot working tree rather than copied, so the arms that
benchmark the pull request are running the file that is actually up for review.
If that file changes, this study changes with it, which is the entire point --
a copy would silently diverge on the first edit.

Falls back to a vendored snapshot only if the checkout is missing, and says so
loudly, because a silent fallback would mean reporting numbers for code that is
not the code under review.
"""

from __future__ import annotations

import importlib.util
import pathlib

_CANDIDATES = [
    pathlib.Path.home() / "projects/lerobot/src/lerobot/policies/common/pointcloud.py",
]


def _load():
    for path in _CANDIDATES:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("lerobot_pr_pointcloud", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.__source_path__ = str(path)
            return module
    raise ImportError(
        "Could not find LeRobot's pointcloud.py. The DP3 arms benchmark the code "
        f"in the open pull request; looked in: {[str(p) for p in _CANDIDATES]}"
    )


_module = _load()

PointCloudEncoder = _module.PointCloudEncoder
unproject = _module.unproject
sample_points = _module.sample_points
normalize_points = _module.normalize_points
SOURCE_PATH = _module.__source_path__
