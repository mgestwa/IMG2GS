import os
import subprocess
import sys
from PIL import Image, ImageDraw
import numpy as np
from core.depth import DepthEstimator
from core.geometry import project_to_3d, save_ply

def create_dummy_data():
    img_path = "verify_image.jpg"
    ply_path = "verify_initial.ply"
    
    # 1. Create Image
    print("[Verify] Creating dummy image...")
    img = Image.new('RGB', (512, 512), color='blue')
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 400, 400], fill='red')
    d.ellipse([200, 200, 300, 300], fill='yellow')
    img.save(img_path)
    
    # 2. Create PLY (using existing pipeline logic)
    print("[Verify] Generating initial PLY...")
    # Mock depth (flat plane for simplicity or random)
    depth = np.ones((512, 512), dtype=np.float32) * 5.0 # Depth = 5.0 meters
    
    # Project
    xyz, rgb = project_to_3d(img, depth)
    save_ply(xyz, rgb, ply_path)
    
    return img_path, ply_path

def main():
    img_path, ply_path = create_dummy_data()
    output_path = "verify_output.ply"
    
    # 3. Run Optimization Script
    print("[Verify] Running optimization...")
    cmd = [
        sys.executable, "run_optimization.py",
        "--ply", ply_path,
        "--image", img_path,
        "--output", output_path,
        "--iters", "50"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print(f"[Verify] SUCCESS! Output saved to {output_path}")
    else:
        print("[Verify] FAILED!")

if __name__ == "__main__":
    main()
