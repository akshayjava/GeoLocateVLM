import gradio as gr
from src.inference import GeoLocator

# Initialize model globally to avoid reloading
try:
    locator = GeoLocator()
except Exception as e:
    print(f"Could not load model: {e}")
    locator = None

def predict_location(image):
    if locator is None:
        return "Model not loaded. Please train the model first."
    
    # Save temp image
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    result = locator.predict(temp_path)
    return result

if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict_location,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="On-Device Geolocation VLM",
        description="Upload an image to estimate its location."
    )
    iface.launch()
