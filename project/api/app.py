from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import onnxruntime
import numpy as np
import cv2
import base64
import io
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI()

# Define the request model
class ImageRequest(BaseModel):
    image_base64: str

# Load the ONNX model
session = onnxruntime.InferenceSession("unet_segmentation.onnx")

# Define image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_base64: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data.")
    
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension
    return image.numpy()

def postprocess_output(model_output: np.ndarray):
    """
    Processes the raw model output to:
    - Compute the predicted mask using argmax.
    - Calculate ratios for "thit" (class 1) and "ruot" (class 2).
    - Convert the mask to a PNG image encoded in base64.
    """
    # model_output shape assumed to be (1, num_classes, H, W)
    output = model_output[0]  # shape: (num_classes, H, W)
    pred_mask = np.argmax(output, axis=0)  # shape: (H, W)
    
    # Count pixels for each class (ignoring background which is assumed to be 0)
    count_thit = np.sum(pred_mask == 1)
    count_ruot = np.sum(pred_mask == 2)
    total = count_thit + count_ruot
    if total > 0:
        ratio_thit = float(count_thit) / total
        ratio_ruot = float(count_ruot) / total
    else:
        ratio_thit = 0.0
        ratio_ruot = 0.0

    # Convert predicted mask to a visible format (background:0, thit:127, ruot:255)
    mask_vis = np.zeros_like(pred_mask, dtype=np.uint8)
    mask_vis[pred_mask == 1] = 127  # "thit"
    mask_vis[pred_mask == 2] = 255  # "ruot"
    
    pil_mask = Image.fromarray(mask_vis)
    buffered = io.BytesIO()
    pil_mask.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return mask_base64, ratio_thit, ratio_ruot

@app.post("/predict")
async def predict(request: ImageRequest):
    image_base64 = request.image_base64
    input_tensor = preprocess_image(image_base64)
    
    input_name = session.get_inputs()[0].name
    inputs = {input_name: input_tensor}
    
    outputs = session.run(None, inputs)
    mask_base64, ratio_thit, ratio_ruot = postprocess_output(outputs[0])
    
    return {
        "mask": mask_base64,
        "ratio_thit": ratio_thit,
        "ratio_ruot": ratio_ruot
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
