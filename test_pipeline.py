import os
import numpy as np
from PIL import Image, ImageDraw
from core.depth import DepthEstimator
from core.geometry import project_to_3d, save_ply

def create_dummy_image(path):
    img = Image.new('RGB', (512, 512), color='blue')
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 400, 400], fill='red')
    d.ellipse([200, 200, 300, 300], fill='yellow')
    img.save(path)
    return img

def main():
    image_path = "test_image.jpg"
    output_path = "output.ply"

    print("Step 1: Creating dummy image...")
    image = create_dummy_image(image_path)
    
    print("Step 2: Initializing Depth Estimator...")
    estimator = DepthEstimator()
    
    print("Step 3: Estimating depth...")
    depth = estimator.estimate(image)
    print(f"Depth map shape: {depth.shape}, Range: [{depth.min()}, {depth.max()}]")
    
    print("Step 4: Projecting to 3D...")
    xyz, rgb = project_to_3d(image, depth)
    print(f"Point cloud shape: {xyz.shape}")
    
    print("Step 5: Saving .ply...")
    save_ply(xyz, rgb, output_path)
    
    print(f"Success! Output saved to {os.path.abspath(output_path)}")
    
    # Cleanup dummy image
    if os.path.exists(image_path):
        os.remove(image_path)

if __name__ == "__main__":
    main()
