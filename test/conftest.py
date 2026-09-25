"""Pytest hooks for the NumKong test suite, loaded before any test module.

File: test/conftest.py
Author: Ash Vardanian
Date: September 25, 2026
"""

from base import _nk_seed_base


def pytest_report_header() -> str:
    """Names the seed this run uses, so a failure under ``NUMKONG_SEED=random`` replays."""
    return f"seed: {_nk_seed_base}, pin with NUMKONG_SEED"
