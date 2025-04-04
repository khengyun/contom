import gradio as gr
import base64
import requests
from PIL import Image
import io

def predict_api(image: Image.Image):
    """
    - Converts the input PIL image to base64.
    - Sends the base64 image to the FastAPI endpoint.
    - Returns the segmentation mask and the class ratios for "thit" and "ruot".
    """
    # Convert image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    api_url = "http://api:8000/predict"
    payload = {"image_base64": img_str}
    response = requests.post(api_url, json=payload)
    
    if response.status_code != 200:
        return None, f"Error: API returned status code {response.status_code}"
    
    data = response.json()
    mask_base64 = data.get("mask")
    ratio_thit = data.get("ratio_thit")
    ratio_ruot = data.get("ratio_ruot")
    
    # Decode mask from base64
    mask_data = base64.b64decode(mask_base64)
    mask_image = Image.open(io.BytesIO(mask_data))
    
    # Format ratio information with class names
    ratios_info = f"thit: {ratio_thit:.4f}\nruot: {ratio_ruot:.4f}"
    
    return mask_image, ratios_info

iface = gr.Interface(
    fn=predict_api,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[gr.Image(type="pil", label="Predicted Mask"),
             gr.Textbox(label="Class Ratios")],
    title="Segmentation API Test",
    description="Upload an image to test the segmentation API. The response includes a predicted mask and class ratios for 'thit' and 'ruot'."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
