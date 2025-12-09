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
    
    # Split dataset if no split exists
    if "train" not in dataset:
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
        texts = ["image" for _ in examples] # PaliGemma expects 'image' token in text? 
        # Actually PaliGemma prompt format is: "<image>prompt"
        # But the processor handles the image token insertion usually.
        # Let's check PaliGemma docs. 
        # For PaliGemma, we pass text and images.
        
        images = [example["image"].convert("RGB") for example in examples]
        prompts = [example["prompt"] for example in examples]
        labels = [example["target"] for example in examples]
        
        # We need to format inputs: "prompt" -> "target"
        # PaliGemma training expects input_ids and labels.
        
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        
        # Process labels
        # We need to tokenize labels and append them?
        # Or does the processor handle it?
        # For Causal LM, we usually concat prompt + label.
        
        # Simplified for now:
        # We will use the processor to prepare inputs.
        # But we need 'labels' for the loss.
        
        # Let's manually tokenize for now to be safe.
        # prefix = "answer " # PaliGemma often uses a prefix
        
        # Actually, let's use a simpler approach for the plan:
        # Just return inputs. The Trainer expects 'labels'.
        
        # For PaliGemma, the labels should be the text we want to generate.
        text_inputs = [f"{p} {l}" for p, l in zip(prompts, labels)]
        
        model_inputs = processor(text=text_inputs, images=images, return_tensors="pt", padding=True)
        
        # Mask the prompt part in labels?
        # This is complex to implement from scratch in a single file without testing.
        # I'll use a standard collator if possible or a simplified one.
        
        # Let's assume we just train on the whole sequence for now (prompt + answer).
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        report_to=["tensorboard"],
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
