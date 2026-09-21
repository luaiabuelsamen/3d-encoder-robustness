"""Collect a dataset of scripted demonstrations.

A thin wrapper so the package can be imported without PYTHONPATH -- see
rvt_lerobot/device.py for why setting it is not an option on this host.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rvt_lerobot.data.collect_study import main  # noqa: E402

if __name__ == "__main__":
    main()
