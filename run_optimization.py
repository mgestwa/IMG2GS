import argparse
import torch
import numpy as np
from PIL import Image
from plyfile import PlyData
import os # Added missing import
from core.optimization import GaussianOptimizer

def load_ply(path: str):
    """Loads points and colors from a PLY file."""
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    # Check for direct RGB colors
    if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1)
        colors = colors.astype(np.float32) / 255.0
    # Check for SH (3DGS format)
    elif 'f_dc_0' in vertex and 'f_dc_1' in vertex and 'f_dc_2' in vertex:
        f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
        SH_C0 = 0.28209479177387814
        colors = f_dc * SH_C0 + 0.5
        colors = np.clip(colors, 0.0, 1.0)
    else:
        # Fallback
        print("Warning: No color data found, using white.")
        colors = np.ones_like(points)
        
    return points, colors

def get_default_camera(w, h, fov_degrees=60.0):
    """
    Returns default View Matrix and K for an image centered at origin.
    The 'project_to_3d' function usually assumes camera is at origin looking down -Z or +Z.
    We need to match the coordinate system used in Phase 2.
    Depth Estimators usually give depth relative to the camera.
    So camera is at (0,0,0).
    """
    
    # 1. Intrinsics (K)
    fov_rad = np.deg2rad(fov_degrees)
    focal = 0.5 * w / np.tan(0.5 * fov_rad)
    
    K = torch.tensor([
        [focal, 0, w/2],
        [0, focal, h/2],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # 2. View Matrix (World -> Camera)
    # If points are already in camera space (which they are from project_to_3d),
    # then World Space == Camera Space.
    # So View Matrix is Identity.
    view_mat = torch.eye(4, dtype=torch.float32)
    
    return view_mat, K

def main():
    parser = argparse.ArgumentParser(description="Run 3DGS Optimization (Phase 3)")
    parser.add_argument("--ply", required=True, help="Path to initial .ply file")
    parser.add_argument("--image", required=True, help="Path to reference image")
    parser.add_argument("--output", default="optimized.ply", help="Output .ply file")
    parser.add_argument("--iters", type=int, default=200, help="Number of iterations")
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading PLY: {args.ply}")
    points, colors = load_ply(args.ply)
    
    print(f"Loading Image: {args.image}")
    pil_image = Image.open(args.image).convert("RGB")
    W, H = pil_image.size
    gt_image = torch.tensor(np.array(pil_image) / 255.0, dtype=torch.float32, device="cuda")
    
    # 2. Setup Camera
    print(f"Setting up camera (FOV=60, {W}x{H})")
    view_mat, K = get_default_camera(W, H)
    view_mat = view_mat.to("cuda")
    K = K.to("cuda")
    
    # 3. Initialize Optimizer
    optimizer = GaussianOptimizer(points, colors, device="cuda")
    
    # 4. Optimization Loop
    print(f"Starting optimization for {args.iters} iterations...")
    for i in range(args.iters):
        loss, _ = optimizer.optimize_step(gt_image, view_mat, K)
        
        if i % 20 == 0:
            print(f"Step {i:04d} | Loss: {loss:.6f}")
            
    # 5. Save result
    optimizer.save_ply(args.output)
    print("Done!")

if __name__ == "__main__":
    main()
