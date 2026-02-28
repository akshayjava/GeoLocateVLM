"""
Unit tests for the collate_fn produced by src/train.train().

We extract collate_fn by calling train() with all HuggingFace objects mocked,
then verify:
  - Prompt tokens are masked with -100 in the labels tensor.
  - At least one non-masked target token exists per sample.
  - Data augmentation is applied when model.training == True.
  - Data augmentation is skipped when model.training == False.
"""
from unittest.mock import MagicMock, patch

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake processor
# ---------------------------------------------------------------------------

def _make_mock_processor(prompt_ids, full_ids, pad_id=0):
    """
    Return a mock processor whose behaviour mirrors PaliGemmaProcessor.

    prompt_ids : list[int]  — token ids for the prompt-only encoding
    full_ids   : list[int]  — token ids for the full (prompt + target) encoding
    pad_id     : int        — padding token id
    """
    mock_proc = MagicMock()

    # processor.tokenizer.pad_token_id
    mock_proc.tokenizer.pad_token_id = pad_id

    # processor.tokenizer(targets, add_special_tokens=False)
    # returns an object with ["input_ids"] = list of lists
    target_ids = full_ids[len(prompt_ids):]   # synthetic target portion
    mock_proc.tokenizer.return_value = {"input_ids": [target_ids]}

    # processor(text=full_texts, images=...) returns model_inputs dict
    input_tensor = torch.tensor([full_ids], dtype=torch.long)
    attention = torch.ones_like(input_tensor)
    mock_proc.return_value = {
        "input_ids": input_tensor,
        "attention_mask": attention,
        "pixel_values": torch.zeros(1, 3, 224, 224),
    }

    return mock_proc


def _make_mock_model(training=True):
    mock_model = MagicMock()
    mock_model.training = training
    return mock_model


def _make_image(tmp_path, name="img.jpg"):
    path = str(tmp_path / name)
    Image.new("RGB", (224, 224)).save(path)
    return path


def _extract_collate_fn(mock_processor, mock_model, augment=False):
    """
    Import and call train() with mocked HF objects, capturing the collate_fn
    that Trainer would receive.
    """
    captured = {}

    def fake_trainer(model, args, train_dataset, eval_dataset, data_collator):
        captured["collate_fn"] = data_collator
        return MagicMock()

    bnb_mock = MagicMock()
    bnb_mock.return_value = MagicMock()

    with patch("src.train.BitsAndBytesConfig", return_value=MagicMock()), \
         patch("src.train.PaliGemmaForConditionalGeneration.from_pretrained",
               return_value=mock_model), \
         patch("src.train.PaliGemmaProcessor.from_pretrained",
               return_value=mock_processor), \
         patch("src.train.get_peft_model", return_value=mock_model), \
         patch("src.train.LoraConfig"), \
         patch("src.train.load_from_disk") as mock_load, \
         patch("src.train.TrainingArguments", return_value=MagicMock()), \
         patch("src.train.Trainer", side_effect=fake_trainer):

        mock_ds = MagicMock()
        mock_ds.__contains__ = lambda self, key: key in ("train", "test")
        mock_ds.__getitem__ = lambda self, key: MagicMock()
        mock_load.return_value = mock_ds

        from src.train import train
        train(dataset_path="/fake", output_dir="/fake_out", max_steps=1, augment=augment)

    return captured.get("collate_fn")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollateLabels:
    def test_prompt_tokens_masked_with_minus100(self, tmp_path):
        """
        Tokens in the prompt region (before the target) must all be -100
        in the labels tensor.
        """
        # Prompt occupies positions 0–4, target occupies positions 5–7
        prompt_ids = [1, 2, 3, 4, 5]        # len = 5
        full_ids   = [1, 2, 3, 4, 5, 10, 11, 12]  # prompt + target

        mock_proc  = _make_mock_processor(prompt_ids, full_ids)
        mock_model = _make_mock_model(training=False)  # augment=False for simplicity

        collate_fn = _extract_collate_fn(mock_proc, mock_model, augment=False)
        assert collate_fn is not None

        example = {
            "image": Image.new("RGB", (224, 224)),
            "prompt": "Where was this photo taken?",
            "target": "Paris, France",
        }
        batch = collate_fn([example])

        labels = batch["labels"]
        # Target tokens: full_ids has 8 tokens, target is 3 tokens → prompt is 5
        target_len = len(full_ids) - len(prompt_ids)  # 3
        prompt_len = len(full_ids) - target_len        # 5

        # Prompt positions must be -100
        assert (labels[0, :prompt_len] == -100).all(), (
            f"Expected first {prompt_len} label positions to be -100, got {labels[0, :prompt_len]}"
        )

    def test_at_least_one_non_masked_target_token(self, tmp_path):
        """The label tensor must contain at least one non-(-100) position."""
        prompt_ids = [1, 2, 3]
        full_ids   = [1, 2, 3, 10, 11]

        mock_proc  = _make_mock_processor(prompt_ids, full_ids)
        mock_model = _make_mock_model(training=False)
        collate_fn = _extract_collate_fn(mock_proc, mock_model, augment=False)

        example = {
            "image": Image.new("RGB", (224, 224)),
            "prompt": "prompt text",
            "target": "answer",
        }
        batch = collate_fn([example])
        assert (batch["labels"] != -100).any(), "At least one target token must not be masked"

    def test_padding_tokens_masked_with_minus100(self, tmp_path):
        """
        Padding positions (pad_token_id) must also be -100 in the labels.
        We simulate padding by appending 0s at the end of full_ids.
        """
        pad_id    = 0
        full_ids  = [1, 2, 3, 10, 11, pad_id, pad_id]  # last 2 are padding
        prompt_ids = [1, 2, 3]

        mock_proc  = _make_mock_processor(prompt_ids, full_ids, pad_id=pad_id)
        mock_model = _make_mock_model(training=False)

        # Adjust processor return to include padding
        input_tensor = torch.tensor([full_ids], dtype=torch.long)
        attention    = (input_tensor != pad_id).long()
        mock_proc.return_value = {
            "input_ids": input_tensor,
            "attention_mask": attention,
            "pixel_values": torch.zeros(1, 3, 224, 224),
        }

        collate_fn = _extract_collate_fn(mock_proc, mock_model, augment=False)
        example = {
            "image": Image.new("RGB", (224, 224)),
            "prompt": "p",
            "target": "t",
        }
        batch = collate_fn([example])
        # The two trailing padding positions must be -100
        assert batch["labels"][0, -2] == -100
        assert batch["labels"][0, -1] == -100


class TestCollateAugmentation:
    def test_augment_called_during_training(self, tmp_path):
        """When augment=True and model.training=True, AUGMENT transform is applied."""
        prompt_ids = [1, 2]
        full_ids   = [1, 2, 10]

        mock_proc  = _make_mock_processor(prompt_ids, full_ids)
        mock_model = _make_mock_model(training=True)

        mock_aug = MagicMock(side_effect=lambda img: img)

        with patch("src.train.AUGMENT", mock_aug):
            collate_fn = _extract_collate_fn(mock_proc, mock_model, augment=True)
            assert collate_fn is not None

            example = {
                "image": Image.new("RGB", (224, 224)),
                "prompt": "p",
                "target": "t",
            }
            collate_fn([example])

        mock_aug.assert_called_once()

    def test_augment_skipped_during_eval(self, tmp_path):
        """When model.training=False, AUGMENT transform must NOT be called."""
        prompt_ids = [1, 2]
        full_ids   = [1, 2, 10]

        mock_proc  = _make_mock_processor(prompt_ids, full_ids)
        mock_model = _make_mock_model(training=False)   # eval mode

        mock_aug = MagicMock(side_effect=lambda img: img)

        with patch("src.train.AUGMENT", mock_aug):
            collate_fn = _extract_collate_fn(mock_proc, mock_model, augment=True)
            assert collate_fn is not None

            example = {
                "image": Image.new("RGB", (224, 224)),
                "prompt": "p",
                "target": "t",
            }
            collate_fn([example])

        mock_aug.assert_not_called()

    def test_augment_disabled_by_flag(self, tmp_path):
        """When augment=False, AUGMENT transform is never called even in train mode."""
        prompt_ids = [1, 2]
        full_ids   = [1, 2, 10]

        mock_proc  = _make_mock_processor(prompt_ids, full_ids)
        mock_model = _make_mock_model(training=True)

        mock_aug = MagicMock(side_effect=lambda img: img)

        with patch("src.train.AUGMENT", mock_aug):
            collate_fn = _extract_collate_fn(mock_proc, mock_model, augment=False)
            assert collate_fn is not None

            example = {
                "image": Image.new("RGB", (224, 224)),
                "prompt": "p",
                "target": "t",
            }
            collate_fn([example])

        mock_aug.assert_not_called()
