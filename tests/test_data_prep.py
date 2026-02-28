"""
Unit tests for src/data_prep.py.

All network calls are mocked so tests run offline without a GPU.
"""
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from PIL import Image

from src.data_prep import download_image, prepare_dataset, PROMPT_STRATEGIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(size=(10, 10), color=(255, 0, 0)):
    """Return bytes of a minimal valid JPEG image."""
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _fake_download(url, save_path):
    """Drop-in replacement for download_image that writes a tiny JPEG."""
    Image.new("RGB", (10, 10)).save(save_path, format="JPEG")
    return True


def _make_csv(tmp_path, rows):
    path = str(tmp_path / "test.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# download_image
# ---------------------------------------------------------------------------

class TestDownloadImage:
    def test_success_returns_true_and_writes_file(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = _make_jpeg_bytes()
        mock_resp.raise_for_status = MagicMock()

        save_path = str(tmp_path / "out.jpg")
        with patch("src.data_prep.requests.get", return_value=mock_resp):
            result = download_image("http://example.com/img.jpg", save_path)

        assert result is True
        assert os.path.exists(save_path)

    def test_http_error_returns_false(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")

        save_path = str(tmp_path / "out.jpg")
        with patch("src.data_prep.requests.get", return_value=mock_resp):
            result = download_image("http://example.com/missing.jpg", save_path)

        assert result is False
        assert not os.path.exists(save_path)

    def test_connection_error_returns_false(self, tmp_path):
        save_path = str(tmp_path / "out.jpg")
        with patch("src.data_prep.requests.get", side_effect=requests.ConnectionError("timeout")):
            result = download_image("http://unreachable.example.com/img.jpg", save_path)

        assert result is False

    def test_corrupt_content_returns_false(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = b"this is not an image"
        mock_resp.raise_for_status = MagicMock()

        save_path = str(tmp_path / "out.jpg")
        with patch("src.data_prep.requests.get", return_value=mock_resp):
            result = download_image("http://example.com/corrupt.jpg", save_path)

        assert result is False

    def test_saved_image_is_rgb(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = _make_jpeg_bytes()
        mock_resp.raise_for_status = MagicMock()

        save_path = str(tmp_path / "out.jpg")
        with patch("src.data_prep.requests.get", return_value=mock_resp):
            download_image("http://example.com/img.jpg", save_path)

        img = Image.open(save_path)
        assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------------------

class TestPrepareDataset:

    def test_valid_rows_produce_dataset(self, tmp_path):
        rows = [
            {"image_url": "http://x.com/1.jpg", "latitude": 48.8, "longitude": 2.3,
             "city": "Paris", "country": "France"},
            {"image_url": "http://x.com/2.jpg", "latitude": 40.7, "longitude": -74.0,
             "city": "New York", "country": "USA"},
        ]
        csv = _make_csv(tmp_path, rows)
        with patch("src.data_prep.download_image", side_effect=_fake_download):
            ds = prepare_dataset(csv, str(tmp_path))

        assert ds is not None
        assert len(ds) == 2

    def test_nan_city_replaced_with_unknown(self, tmp_path):
        rows = [{"image_url": "http://x.com/1.jpg", "latitude": 0.0, "longitude": 0.0,
                 "city": float("nan"), "country": "France"}]
        csv = _make_csv(tmp_path, rows)
        with patch("src.data_prep.download_image", side_effect=_fake_download):
            ds = prepare_dataset(csv, str(tmp_path))

        assert ds is not None
        assert "Unknown" in ds[0]["target"]
        assert "nan" not in ds[0]["target"]

    def test_nan_country_replaced_with_unknown(self, tmp_path):
        rows = [{"image_url": "http://x.com/1.jpg", "latitude": 0.0, "longitude": 0.0,
                 "city": "Paris", "country": float("nan")}]
        csv = _make_csv(tmp_path, rows)
        with patch("src.data_prep.download_image", side_effect=_fake_download):
            ds = prepare_dataset(csv, str(tmp_path))

        assert ds is not None
        assert "Unknown" in ds[0]["target"]

    def test_all_downloads_fail_returns_none(self, tmp_path):
        rows = [{"image_url": "http://bad.com/1.jpg", "latitude": 0.0, "longitude": 0.0,
                 "city": "X", "country": "Y"}]
        csv = _make_csv(tmp_path, rows)
        with patch("src.data_prep.download_image", return_value=False):
            ds = prepare_dataset(csv, str(tmp_path))

        assert ds is None

    def test_failed_rows_skipped(self, tmp_path):
        rows = [
            {"image_url": "http://x.com/good.jpg", "latitude": 48.8, "longitude": 2.3,
             "city": "Paris", "country": "France"},
            {"image_url": "http://x.com/bad.jpg", "latitude": 0.0, "longitude": 0.0,
             "city": "X", "country": "Y"},
        ]
        csv = _make_csv(tmp_path, rows)

        def partial(url, save_path):
            if "good" in url:
                return _fake_download(url, save_path)
            return False

        with patch("src.data_prep.download_image", side_effect=partial):
            ds = prepare_dataset(csv, str(tmp_path))

        assert ds is not None
        assert len(ds) == 1

    def test_dataset_has_expected_columns(self, tmp_path):
        rows = [{"image_url": "http://x.com/1.jpg", "latitude": 48.8, "longitude": 2.3,
                 "city": "Paris", "country": "France"}]
        csv = _make_csv(tmp_path, rows)
        with patch("src.data_prep.download_image", side_effect=_fake_download):
            ds = prepare_dataset(csv, str(tmp_path))

        assert "prompt" in ds.column_names
        assert "target" in ds.column_names
        assert "coordinates" in ds.column_names

    def test_saved_to_disk(self, tmp_path):
        rows = [{"image_url": "http://x.com/1.jpg", "latitude": 48.8, "longitude": 2.3,
                 "city": "Paris", "country": "France"}]
        csv = _make_csv(tmp_path, rows)
        with patch("src.data_prep.download_image", side_effect=_fake_download):
            prepare_dataset(csv, str(tmp_path))

        assert os.path.isdir(str(tmp_path / "processed_dataset"))

    # --- Prompt strategy tests (Phase 3.1) ---

    def test_coordinates_strategy_target_has_decimal_coords(self, tmp_path):
        rows = [{"image_url": "http://x.com/1.jpg", "latitude": 48.8584, "longitude": 2.2945,
                 "city": "Paris", "country": "France"}]
        csv = _make_csv(tmp_path, rows)
        with patch("src.data_prep.download_image", side_effect=_fake_download):
            ds = prepare_dataset(csv, str(tmp_path), strategy="coordinates")

        assert ds is not None
        # Target should be "48.8584, 2.2945" (4 decimal places)
        assert "48.8584" in ds[0]["target"]
        assert "2.2945" in ds[0]["target"]

    def test_combined_strategy_target_has_both_city_and_coords(self, tmp_path):
        rows = [{"image_url": "http://x.com/1.jpg", "latitude": 48.8584, "longitude": 2.2945,
                 "city": "Paris", "country": "France"}]
        csv = _make_csv(tmp_path, rows)
        with patch("src.data_prep.download_image", side_effect=_fake_download):
            ds = prepare_dataset(csv, str(tmp_path), strategy="combined")

        assert ds is not None
        target = ds[0]["target"]
        assert "Paris" in target
        assert "France" in target
        assert "48.8584" in target

    def test_invalid_strategy_raises_value_error(self, tmp_path):
        rows = [{"image_url": "http://x.com/1.jpg", "latitude": 0.0, "longitude": 0.0,
                 "city": "X", "country": "Y"}]
        csv = _make_csv(tmp_path, rows)
        with pytest.raises(ValueError, match="Unknown strategy"):
            prepare_dataset(csv, str(tmp_path), strategy="nonexistent")

    def test_all_strategy_keys_present(self):
        assert set(PROMPT_STRATEGIES.keys()) == {"city_country", "coordinates", "combined"}

    def test_each_strategy_has_prompt_and_callable(self):
        for key, (prompt, fn) in PROMPT_STRATEGIES.items():
            assert isinstance(prompt, str) and prompt
            assert callable(fn)

    def test_existing_image_not_redownloaded(self, tmp_path):
        rows = [{"image_url": "http://x.com/1.jpg", "latitude": 0.0, "longitude": 0.0,
                 "city": "A", "country": "B"}]
        csv = _make_csv(tmp_path, rows)

        # Pre-create the expected image file so download_image should NOT be called
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (10, 10)).save(str(img_dir / "0.jpg"))

        mock_dl = MagicMock(return_value=True)
        with patch("src.data_prep.download_image", mock_dl):
            prepare_dataset(csv, str(tmp_path))

        mock_dl.assert_not_called()
