"""Fixtures that import waste_classification with TensorFlow replaced by test doubles.

The tests exercise path handling, label mapping, evaluation selection and import
behaviour only. They need no TensorFlow install, no dataset and no network.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TF_MODULES = [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.models",
    "tensorflow.keras.callbacks",
    "tensorflow.keras.layers",
    "tensorflow.keras.preprocessing",
    "tensorflow.keras.preprocessing.image",
    "tensorflow.keras.applications",
]


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """No test may reach the real dataset URL; tests that need a download fake requests.get."""
    import requests

    def _blocked(*args, **kwargs):
        raise AssertionError("tests must not use the network")

    monkeypatch.setattr(requests, "get", _blocked)


@pytest.fixture
def tf_stubs(monkeypatch):
    stubs = {name: MagicMock(name=name) for name in TF_MODULES}
    stubs["tensorflow"].__version__ = "stub"
    stubs["tensorflow"].keras.callbacks.Callback = type("Callback", (), {})
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    return stubs


@pytest.fixture
def load_wc(tf_stubs, tmp_path, monkeypatch):
    """Return a function that freshly imports waste_classification inside an empty cwd."""
    monkeypatch.chdir(tmp_path)

    def _load():
        sys.modules.pop("waste_classification", None)
        import waste_classification

        return waste_classification

    yield _load
    sys.modules.pop("waste_classification", None)


@pytest.fixture
def wc(load_wc):
    return load_wc()
