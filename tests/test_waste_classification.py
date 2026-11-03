import glob as real_glob_module
import io
import os
import zipfile

import numpy as np
import pytest

real_glob = real_glob_module.glob


def make_images(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")


def zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, payload, fail_after=None):
        self.payload = payload
        self.fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.payload), 16):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("connection dropped")
            yield self.payload[i:i + 16]


VALID_ARCHIVE = {
    "o-vs-r-split/train/O/a.jpg": b"x",
    "o-vs-r-split/train/R/b.jpg": b"x",
    "o-vs-r-split/test/O/c.jpg": b"x",
    "o-vs-r-split/test/R/d.jpg": b"x",
}


# --- import side effects -------------------------------------------------------------

def test_import_has_no_side_effects(load_wc, tf_stubs, tmp_path, monkeypatch):
    """Importing used to download, extract, train and plot at module level."""
    import matplotlib.pyplot as plt
    import requests

    def boom(*a, **k):
        raise AssertionError("module import must not touch the network or plots")

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(plt, "show", boom)
    wc = load_wc()
    assert callable(wc.main)
    assert list(tmp_path.iterdir()) == []
    tf_stubs["tensorflow.keras.preprocessing.image"].ImageDataGenerator.assert_not_called()
    tf_stubs["tensorflow.keras.applications"].vgg16.VGG16.assert_not_called()


# --- evaluation selection ------------------------------------------------------------

def test_select_eval_files_ignores_directory_listing_order(wc, tmp_path, monkeypatch):
    """glob order is filesystem dependent; the evaluated subset must not be."""
    root = tmp_path / "test"
    names = [f"img_{i:03d}.jpg" for i in range(60)]
    make_images(root / "O", reversed(names))
    make_images(root / "R", names)

    expected_O = [str(root / "O" / n) for n in names[:50]]
    expected_R = [str(root / "R" / n) for n in names[:50]]

    for order in (lambda x: x, lambda x: x[::-1], lambda x: x[1::2] + x[::2]):
        monkeypatch.setattr(wc.glob, "glob", lambda pattern, order=order: order(real_glob(pattern)))
        files, available = wc.select_eval_files(str(root))
        assert files == expected_O + expected_R
        assert available == 120


def test_select_eval_files_caps_per_class_and_keeps_labels_aligned(wc, tmp_path):
    root = tmp_path / "test"
    make_images(root / "O", [f"o{i}.png" for i in range(70)])
    make_images(root / "R", [f"r{i}.png" for i in range(3)])
    files, available = wc.select_eval_files(str(root), per_class=50)
    assert len(files) == 53
    assert available == 73
    labels = [os.path.basename(os.path.dirname(f)) for f in files]
    assert labels == ["O"] * 50 + ["R"] * 3


def test_select_eval_files_skips_non_images(wc, tmp_path):
    root = tmp_path / "test"
    make_images(root / "O", ["a.jpg", "Thumbs.db", "notes.txt"])
    (root / "O" / "nested").mkdir()
    make_images(root / "R", ["b.JPG"])
    files, available = wc.select_eval_files(str(root))
    assert [os.path.basename(f) for f in files] == ["a.jpg", "b.JPG"]
    assert available == 2


# --- label mapping -------------------------------------------------------------------

def test_class_encoding_matches_keras_alphabetical_indices(wc):
    # flow_from_directory sorts class folders: O -> 0, R -> 1.
    assert wc.class2num_lt(["O", "R", "R", "O"]) == [0, 1, 1, 0]


@pytest.mark.parametrize(
    "probs",
    [
        np.array([[0.1], [0.5], [0.9], [0.4999]]),  # model.predict output, shape (N, 1)
        [np.array([0.1]), np.array([0.5]), np.array([0.9]), np.array([0.4999])],
        [0.1, 0.5, 0.9, 0.4999],
    ],
)
def test_num2class_threshold_and_length(wc, probs):
    # P(R) < 0.5 -> 'O'; exactly 0.5 -> 'R'.
    assert wc.num2class_lt(probs) == ["O", "R", "R", "O"]


def test_labels_round_trip(wc):
    labels = ["O", "R", "O", "R", "R"]
    assert wc.num2class_lt(wc.class2num_lt(labels)) == labels


# --- dataset handling ----------------------------------------------------------------

def test_ensure_dataset_skips_download_when_present(wc, tmp_path, monkeypatch):
    import requests

    make_images(tmp_path / "o-vs-r-split" / "train" / "O", ["a.jpg"])
    make_images(tmp_path / "o-vs-r-split" / "test" / "O", ["a.jpg"])
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("must not re-download"))
    wc.ensure_dataset(str(tmp_path))


def test_ensure_dataset_downloads_extracts_and_removes_zip(wc, tmp_path, monkeypatch):
    import requests

    calls = []
    payload = zip_bytes(VALID_ARCHIVE)

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)
    wc.ensure_dataset(str(tmp_path))

    assert len(calls) == 1 and calls[0][1].get("timeout")
    assert (tmp_path / "o-vs-r-split" / "test" / "R" / "d.jpg").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["o-vs-r-split"]


def test_ensure_dataset_failed_download_leaves_no_partial_file(wc, tmp_path, monkeypatch):
    import requests

    payload = zip_bytes(VALID_ARCHIVE)
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(payload, fail_after=32))
    with pytest.raises(ConnectionError):
        wc.ensure_dataset(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_ensure_dataset_rejects_unexpected_archive_layout(wc, tmp_path, monkeypatch):
    import requests

    payload = zip_bytes({"something-else/readme.txt": b"x"})
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(payload))
    with pytest.raises(RuntimeError, match="not found"):
        wc.ensure_dataset(str(tmp_path))
