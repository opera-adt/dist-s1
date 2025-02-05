from __future__ import annotations

import importlib.metadata

import dist_s1


def test_version() -> None:
    assert importlib.metadata.version('dist_s1') == dist_s1.__version__
