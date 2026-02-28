import os
import torch
from datasets import load_from_disk
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType

# ---------------------------------------------------------------------------
# Optional data augmentation (Phase 3.2)
# torchvision is an optional dependency; augmentation is skipped if absent.
# ---------------------------------------------------------------------------
try:
    from torchvision import transforms as T

    AUGMENT = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ])
except ImportError:  # pragma: no cover
    AUGMENT = None


def train(
    dataset_path="data/processed_dataset",
    output_dir="models/geolocate_vlm",
    base_model_id="google/paligemma-3b-pt-224",
    max_steps=500,
    batch_size=2,
    lora_rank=16,
    lora_alpha=64,
    augment=True,
    include_vision_lora=False,
):
    """
    Fine-tune PaliGemma with QLoRA.

    Parameters
    ----------
    dataset_path      : Path to a HuggingFace dataset saved with save_to_disk().
    output_dir        : Directory to save trained adapters.
    base_model_id     : HuggingFace model ID for PaliGemma.
    max_steps         : Total training steps (default 500 for better convergence).
    batch_size        : Per-device training batch size.
    lora_rank         : LoRA rank r (Phase 3.3). Higher = more capacity, more VRAM.
    lora_alpha        : LoRA scaling factor (typically 4 × rank).
    augment           : Apply random data augmentation during training (Phase 3.2).
    include_vision_lora : Also adapt vision encoder attention layers (Phase 3.4).
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    if "train" not in dataset or "test" not in dataset:
        dataset = dataset.train_test_split(test_size=0.1)

    print(f"Loading model {base_model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    processor = PaliGemmaProcessor.from_pretrained(base_model_id)

    # LoRA target modules (Phase 3.4)
    # q/k/v/o_proj match BOTH the language decoder and the SigLIP vision encoder
    # inside PaliGemma, so vision encoder is automatically included.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    if include_vision_lora:
        # Also adapt the SigLIP MLP feed-forward layers inside the vision tower.
        target_modules += ["fc1", "fc2"]

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def collate_fn(examples):
        # Apply augmentation only during training (model.training is True then)
        images = [example["image"].convert("RGB") for example in examples]
        if augment and AUGMENT is not None and model.training:
            images = [AUGMENT(img) for img in images]

        prompts = [example["prompt"] for example in examples]
        targets = [example["target"] for example in examples]

        # Full sequence: prompt + target
        full_texts = [f"{p} {t}" for p, t in zip(prompts, targets)]
        model_inputs = processor(
            text=full_texts, images=images, return_tensors="pt", padding=True
        )

        # Build labels: compute loss only on answer tokens, mask prompt + padding.
        label_encodings = processor.tokenizer(targets, add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        labels_tensor = input_ids.clone()

        pad_id = processor.tokenizer.pad_token_id
        for i, label_ids in enumerate(label_encodings["input_ids"]):
            label_len = len(label_ids)
            non_pad_len = (input_ids[i] != pad_id).sum().item()
            prompt_len = non_pad_len - label_len
            labels_tensor[i, :prompt_len] = -100    # mask image + prompt tokens
            labels_tensor[i, non_pad_len:] = -100   # mask padding tokens

        model_inputs["labels"] = labels_tensor
        return model_inputs

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate_fn,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune PaliGemma for geolocation with QLoRA")
    parser.add_argument("--dataset_path", default="data/processed_dataset")
    parser.add_argument("--output_dir", default="models/geolocate_vlm")
    parser.add_argument("--base_model_id", default="google/paligemma-3b-pt-224")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Total training steps (default: 500)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank r. Higher = more capacity (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha scaling factor (default: 64)")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--include_vision_lora", action="store_true",
                        help="Also adapt vision encoder MLP layers")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        base_model_id=args.base_model_id,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        augment=not args.no_augment,
        include_vision_lora=args.include_vision_lora,
    )
