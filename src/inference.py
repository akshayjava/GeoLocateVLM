import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from PIL import Image

class GeoLocator:
    def __init__(self, model_path="models/geolocate_vlm", base_model_id="google/paligemma-3b-pt-224"):
        print(f"Loading base model {base_model_id}...")
        
        # If model_path doesn't exist locally, assume it's a HF repo or we want base model
        import os
        if not os.path.exists(model_path) and model_path != base_model_id:
            print(f"Model path {model_path} not found. Using base model {base_model_id} for zero-shot.")
            model_path = base_model_id
            load_adapters = False
        else:
            load_adapters = True

        self.processor = PaliGemmaProcessor.from_pretrained(model_path)
        
        # Load base model
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # Load adapters if applicable
        if load_adapters and model_path != base_model_id:
            print(f"Loading adapters from {model_path}...")
            try:
                self.model = PeftModel.from_pretrained(self.model, model_path)
            except Exception as e:
                print(f"Failed to load adapters: {e}. Continuing with base model.")
        
        self.model.eval()
        
    def predict(self, image_path, prompt="Where was this photo taken?"):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
            
        result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Result usually contains the prompt + answer.
        # We might need to strip the prompt.
        return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        locator = GeoLocator()
        print(locator.predict(sys.argv[1]))
    else:
        print("Usage: python src/inference.py <image_path>")
