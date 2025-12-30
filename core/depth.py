import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np

class DepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Large-hf", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Depth Model: {model_id} on {self.device}...")
        
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        print("Model loaded.")

    def estimate(self, image: Image.Image) -> np.ndarray:
        """
        Estimates depth map from a PIL Image.
        Returns a numpy array of shape (H, W) with relative depth values.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        return prediction.squeeze().cpu().numpy()
