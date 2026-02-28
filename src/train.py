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

def train(
    dataset_path="data/processed_dataset",
    output_dir="models/geolocate_vlm",
    base_model_id="google/paligemma-3b-pt-224",
    max_steps=100,
    batch_size=2
):
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Split dataset if no train/test split exists
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
    
    # Freeze vision encoder and projector? 
    # Usually for QLoRA we freeze everything and add adapters.
    # PEFT handles this.
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    def collate_fn(examples):
        images = [example["image"].convert("RGB") for example in examples]
        prompts = [example["prompt"] for example in examples]
        labels = [example["target"] for example in examples]

        # Full sequence: prompt + label
        full_texts = [f"{p} {l}" for p, l in zip(prompts, labels)]
        model_inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)

        # Build labels tensor: only compute loss on answer tokens, not on prompt or padding.
        # Tokenize labels alone (no special tokens) to determine their lengths.
        label_encodings = processor.tokenizer(labels, add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        labels_tensor = input_ids.clone()

        pad_id = processor.tokenizer.pad_token_id
        for i, label_ids in enumerate(label_encodings["input_ids"]):
            label_len = len(label_ids)
            non_pad_len = (input_ids[i] != pad_id).sum().item()
            prompt_len = non_pad_len - label_len
            labels_tensor[i, :prompt_len] = -100   # mask image + prompt tokens
            labels_tensor[i, non_pad_len:] = -100  # mask padding tokens

        model_inputs["labels"] = labels_tensor
        return model_inputs

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=max_steps,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        remove_unused_columns=False 
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate_fn
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    train()
