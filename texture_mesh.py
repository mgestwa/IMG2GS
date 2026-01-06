import argparse
import sys
import torch
import numpy as np
import os
from PIL import Image

from core.mesh_ingest import load_mesh, sample_mesh
from core.optimization import GaussianOptimizer
from core.geometry import get_intrinsics

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return img

def main():
    parser = argparse.ArgumentParser(description="Texture Project: Colorize Mesh using 3DGS Optimization")
    parser.add_argument("mesh_path", type=str, help="Input Mesh (.obj)")
    parser.add_argument("image_path", type=str, help="Input Image (.jpg/png) to project")
    parser.add_argument("--output", type=str, default="textured_mesh.ply", help="Output PLY path")
    parser.add_argument("--iters", type=int, default=200, help="Optimization iterations")
    parser.add_argument("--points", type=int, default=500_000, help="Number of points to sample")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    print(f"Loading Mesh: {args.mesh_path}")
    mesh = load_mesh(args.mesh_path)
    
    print(f"Loading Image: {args.image_path}")
    pil_image = load_image(args.image_path)
    W, H = pil_image.size
    
    # Prepare GT Image Tensor
    gt_image_tensor = torch.tensor(np.array(pil_image) / 255.0, dtype=torch.float32, device=device) # [H, W, 3]
    
    # 2. Sample Mesh (Geometry)
    print("Sampling surface...")
    xyz, _, scale_radius = sample_mesh(mesh, num_points=args.points)
    
    # Initialize with GRAY color to ensure we are actually learning
    # (Starting with white is also fine, but gray shows contrast better if learning fails)
    gray_colors = np.ones_like(xyz) * 0.5 
    
    # 3. Normalize Mesh (Center & Scale)
    # Move centroid to 0,0,0
    center = xyz.mean(axis=0)
    xyz_centered = xyz - center
    
    # Scale to fit in unit sphere (radius 1)
    # We use max distance from center as radius
    radius = np.max(np.linalg.norm(xyz_centered, axis=1))
    scale_factor = 1.0 / radius
    xyz_normalized = xyz_centered * scale_factor
    
    print(f"[Auto-Scale] Center: {center}, Radius before: {radius:.2f}, Scaling by: {scale_factor:.4f}")
    
    # Update Scale Radius for splats (proportional to new scale)
    scale_radius_normalized = scale_radius * scale_factor

    # Initializes Optimizer with NORMALIZED coordinates
    optimizer = GaussianOptimizer(xyz_normalized, gray_colors, device=device)
    
    # 4. Setup Geometry (Solid Look) & Freeze
    # Log scale of the normalized splat radius
    log_scale = np.log(scale_radius_normalized)
    
    with torch.no_grad():
        optimizer.scales.data.fill_(log_scale)
        optimizer.opacities.data.fill_(10.0) # Solid
        
    # SWITCH TO COLOR ONLY MODE
    optimizer.switch_to_color_optimization(lr=0.02) # Slightly higher LR for colors
    
    # 5. Camera Setup (Auto-Fit)
    # If object is radius 1, placed at 0,0,0.
    # We want to view it from a distance.
    # Distance = 2.5 * Radius usually gives good view.
    camera_dist = 2.5
    
    print(f"[Camera] Auto-positioning at Z={camera_dist}")
    
    viewmat = torch.eye(4, device=device)
    # World-to-Camera Translation.
    # Camera at (0, 0, dist) looking at (0, 0, 0).
    # This means world coordinates need to be shifted by -dist in Z relative to camera.
    viewmat[2, 3] = camera_dist 
    
    # Intrinsics
    K_np = get_intrinsics(W, H)
    K = torch.tensor(K_np, dtype=torch.float32, device=device)
    
    # 6. Optimization Loop
    print(f"Starting Texture Optimization ({args.iters} iters)...")
    for i in range(args.iters):
        loss, _ = optimizer.optimize_step(gt_image_tensor, viewmat, K)
        
        if i % 20 == 0:
            print(f"Iter {i:03d}: Loss {loss:.4f}")
            
    # 7. Save
    optimizer.save_ply(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
