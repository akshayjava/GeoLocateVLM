"""
Unit tests for src/inference.py.

All HuggingFace model and processor calls are mocked so tests run without
a GPU or network access.
"""
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from src.inference import GeoLocator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(tmp_path, name="test.jpg", size=(224, 224)):
    path = str(tmp_path / name)
    Image.new("RGB", size).save(path)
    return path


def _make_locator(tmp_path, input_len=10, extra_tokens=3, decoded="Paris, France"):
    """
    Build a GeoLocator bypassing __init__, with all heavy attributes mocked.

    input_len    : number of tokens in the mocked processor output (simulates
                   image + prompt tokens)
    extra_tokens : tokens that the model "generates" beyond the input
    decoded      : string returned by batch_decode
    """
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    # processor(text=..., images=...) returns a dict-like object
    mock_inputs = {
        "input_ids": torch.zeros(1, input_len, dtype=torch.long),
        "attention_mask": torch.ones(1, input_len, dtype=torch.long),
    }
    mock_processor.return_value = mock_inputs

    # generate() returns full sequence: input + generated tokens
    full_ids = torch.zeros(1, input_len + extra_tokens, dtype=torch.long)
    mock_model.generate.return_value = full_ids

    mock_processor.batch_decode.return_value = [decoded]

    locator = GeoLocator.__new__(GeoLocator)
    locator.processor = mock_processor
    locator.model = mock_model
    return locator


# ---------------------------------------------------------------------------
# GeoLocator.predict
# ---------------------------------------------------------------------------

class TestGeoLocatorPredict:
    def test_returns_string(self, tmp_path):
        img = _write_image(tmp_path)
        locator = _make_locator(tmp_path, decoded="Paris, France")
        result = locator.predict(img)
        assert isinstance(result, str)

    def test_returns_decoded_text(self, tmp_path):
        img = _write_image(tmp_path)
        locator = _make_locator(tmp_path, decoded="Tokyo, Japan")
        assert locator.predict(img) == "Tokyo, Japan"

    def test_only_generated_tokens_decoded(self, tmp_path):
        """
        generate() returns input_len + extra_tokens tokens.
        batch_decode must receive only the extra_tokens slice.
        """
        img = _write_image(tmp_path)
        input_len = 10
        extra = 4
        locator = _make_locator(tmp_path, input_len=input_len, extra_tokens=extra)

        locator.predict(img)

        decoded_arg = locator.processor.batch_decode.call_args[0][0]
        assert decoded_arg.shape[-1] == extra, (
            f"Expected only {extra} generated tokens to be decoded, "
            f"got {decoded_arg.shape[-1]}"
        )

    def test_no_grad_context_used(self, tmp_path):
        """Model should be called inside torch.no_grad()."""
        img = _write_image(tmp_path)
        locator = _make_locator(tmp_path)

        grad_enabled_during_generate = []

        def mock_generate(**kwargs):
            grad_enabled_during_generate.append(torch.is_grad_enabled())
            input_len = kwargs.get("input_ids", torch.zeros(1, 10)).shape[-1]
            return torch.zeros(1, input_len + 3, dtype=torch.long)

        locator.model.generate.side_effect = mock_generate
        locator.predict(img)

        assert grad_enabled_during_generate == [False], (
            "generate() should be called with gradients disabled"
        )

    def test_image_converted_to_rgb(self, tmp_path):
        """Even if the image on disk is RGBA, it must be passed as RGB."""
        img_path = str(tmp_path / "rgba.png")
        Image.new("RGBA", (224, 224)).save(img_path)

        locator = _make_locator(tmp_path)
        locator.predict(img_path)

        # The image passed to the processor should have mode RGB
        call_kwargs = locator.processor.call_args
        passed_image = call_kwargs[1].get("images") or call_kwargs[0][1]
        assert passed_image.mode == "RGB"

    def test_custom_prompt_used(self, tmp_path):
        img = _write_image(tmp_path)
        locator = _make_locator(tmp_path)
        locator.predict(img, prompt="Estimate the GPS coordinates.")
        call_kwargs = locator.processor.call_args
        text_arg = call_kwargs[1].get("text") or call_kwargs[0][0]
        assert text_arg == "Estimate the GPS coordinates."


# ---------------------------------------------------------------------------
# GeoLocator.__init__ path selection (no GPU required — all HF calls mocked)
# ---------------------------------------------------------------------------

class TestGeoLocatorInit:
    def _patch_hf(self, model_path_exists=False):
        """Return a context that patches all HuggingFace loading calls."""
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        patches = [
            patch("src.inference.PaliGemmaProcessor.from_pretrained", return_value=mock_proc),
            patch(
                "src.inference.PaliGemmaForConditionalGeneration.from_pretrained",
                return_value=mock_model,
            ),
            patch("src.inference.PeftModel.from_pretrained", return_value=mock_model),
            patch("os.path.exists", return_value=model_path_exists),
        ]
        return patches, mock_proc, mock_model

    def test_falls_back_to_base_model_when_path_missing(self, tmp_path):
        patches, mock_proc, mock_model = self._patch_hf(model_path_exists=False)
        [p.start() for p in patches]
        try:
            locator = GeoLocator(model_path="models/nonexistent")
        finally:
            for p in patches:
                p.stop()
        # No exception should have been raised
        assert locator is not None

    def test_loads_adapters_when_path_exists(self, tmp_path):
        patches, mock_proc, mock_model = self._patch_hf(model_path_exists=True)
        with patch("src.inference.PeftModel.from_pretrained", return_value=mock_model):
            [p.start() for p in patches]
            try:
                locator = GeoLocator(model_path="models/geolocate_vlm")
            finally:
                for p in patches:
                    p.stop()
        assert locator is not None
