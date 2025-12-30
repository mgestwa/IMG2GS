import io
import os
import shutil
import uuid
import cv2
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np

from core.depth import DepthEstimator
from core.geometry import project_to_3d, save_ply

# Configuration
FILES_DIR = "generated_files"
os.makedirs(FILES_DIR, exist_ok=True)

# Global state for the model
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    print("Initializing Depth Estimator...")
    model_state["estimator"] = DepthEstimator()
    print("Depth Estimator ready.")
    yield
    # Cleanup
    model_state.clear()
    # Optional: cleanup FILES_DIR?

app = FastAPI(title="img2gs-local API", lifespan=lifespan)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount generated files
app.mount("/files", StaticFiles(directory=FILES_DIR), name="files")

def cleanup_file(path: str):
    """Background task to remove file after some time (if needed)."""
    # For now, we prefer keeping them for the session so the viewer can re-fetch if needed.
    # We could implement a periodic cleanup or delete after X minutes.
    pass

@app.post("/process", summary="Convert Image to 3D Gaussian Splat PLY")
async def process_image(file: UploadFile = File(...)):
    """
    Uploads an image, generates a depth map, projects to 3D, and returns the URL to the .ply file.
    """
    # 1. Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Estimate Depth
    estimator = model_state.get("estimator")
    if not estimator:
        raise RuntimeError("Model not initialized.")
    
    depth = estimator.estimate(image)
    
    # QUALITY UPGRADE: Upsample to create a DENSE point cloud.
    # By doubling the resolution, we create 4x more points, shrinking the gaps between them.
    # This works like "Super Resolution" for the 3D geometry.
    SCALE_FACTOR = 2
    
    # 1. Upscale Image (RGB)
    original_w, original_h = image.size
    new_w, new_h = original_w * SCALE_FACTOR, original_h * SCALE_FACTOR
    image_dense = image.resize((new_w, new_h), Image.BICUBIC)
    
    # 2. Upscale Depth
    # Depth is a float array. We can use PIL 'F' mode or CV2. PIL 'F' is convenient here.
    depth_pil = Image.fromarray(depth, mode='F')
    depth_dense_pil = depth_pil.resize((new_w, new_h), Image.BICUBIC)
    depth_dense = np.array(depth_dense_pil)
    
    # QUALITY UPGRADE: Denoising
    # Bilateral filter smoothes flat areas while keeping edges sharp.
    # We need to ensure it's float32 for cv2.
    depth_dense = depth_dense.astype(np.float32)
    # d=5 (diameter), sigmaColor=2.0 (depth range tolerance), sigmaSpace=7.0 (pixel range)
    depth_dense = cv2.bilateralFilter(depth_dense, d=5, sigmaColor=2.0, sigmaSpace=7.0)
    
    # 3. Project to 3D using the high-res data
    xyz, rgb = project_to_3d(image_dense, depth_dense)
    
    # 4. Save to filesystem
    filename = f"{uuid.uuid4()}.ply"
    file_path = os.path.join(FILES_DIR, filename)
    
    save_ply(xyz, rgb, file_path)
    
    # 5. Return URL
    return JSONResponse({
        "status": "success",
        "url": f"/files/{filename}",
        "filename": filename
    })

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

from fastapi.responses import FileResponse
