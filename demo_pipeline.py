import argparse
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image
import uuid

# Add current directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.depth import DepthEstimator
from core.geometry import project_to_3d, save_ply
from core.optimization import GaussianOptimizer


def run_demo(image_path, output_dir="demo_output", iterations=100):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Demo] Processing: {image_path}")
    
    # ---------------------------------------------------------
    # Phase 2: Image -> Point Cloud
    # ---------------------------------------------------------
    print("\n[Phase 2] Generating Initial Point Cloud...")
    
    # 1. Load Image
    img_pil = Image.open(image_path).convert("RGB")
    W, H = img_pil.size
    print(f"  Image Size: {W}x{H}")
    
    # 2. Depth Estimation
    print("  Estimating Depth...")
    estimator = DepthEstimator()
    depth = estimator.estimate(img_pil)
    
    # Upscaling (Quality Upgrade logic from main.py)
    SCALE_FACTOR = 2
    new_w, new_h = W * SCALE_FACTOR, H * SCALE_FACTOR
    img_dense = img_pil.resize((new_w, new_h), Image.BICUBIC)
    depth_pil = Image.fromarray(depth, mode='F')
    depth_dense_pil = depth_pil.resize((new_w, new_h), Image.BICUBIC)
    depth_dense = np.array(depth_dense_pil).astype(np.float32)
    # Denoise
    depth_dense = cv2.bilateralFilter(depth_dense, d=5, sigmaColor=2.0, sigmaSpace=7.0)
    
    # 3. Project to 3D
    print("  Projecting to 3D...")
    xyz, rgb = project_to_3d(img_dense, depth_dense)
    
    initial_ply_path = os.path.join(output_dir, "initial.ply")
    save_ply(xyz, rgb, initial_ply_path)
    print(f"  Saved initial PLY: {initial_ply_path}")
    
    # ---------------------------------------------------------
    # Phase 3: 3DGS Optimization
    # ---------------------------------------------------------
    print("\n[Phase 3] Running 3DGS Optimization...")
    
    # Load Reference Image (ground truth for loss)
    # We use the Upscaled image as GT since our points are upscaled
    gt_image_np = np.array(img_dense)
    if len(gt_image_np.shape) == 2:
        print("DEBUG: Image is 2D (Grayscale?), converting to 3D by stacking.")
        gt_image_np = np.stack([gt_image_np]*3, axis=-1)
    elif gt_image_np.shape[2] == 4:
         print("DEBUG: Image is RGBA, dropping Alpha.")
         gt_image_np = gt_image_np[:, :, :3]
         
    gt_image_np = gt_image_np / 255.0
    gt_image = torch.tensor(gt_image_np, dtype=torch.float32, device='cuda')
    
    # Initialize Optimizer
    # We pass the points and colors directly from memory
    optimizer = GaussianOptimizer(xyz, rgb, device='cuda')
    
    # Camera Setup (Simple Pinhole for now, matching the projection logic)
    # FOV of 60 degrees is assumed in project_to_3d usually, we need to match it.
    fov_rad = 60.0 * np.pi / 180.0
    # Focal length f assuming FOV 60 horizontally
    # tan(theta/2) = (W/2) / f  => f = (W/2) / tan(30)
    fx = (new_w / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx # Assuming square pixels
    cx = new_w / 2.0
    cy = new_h / 2.0
    
    # Intrinsics Matrix K (3x3)
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device='cuda', dtype=torch.float32)
    
    # View Matrix (Identity, as points are projected from this camera view)
    viewmat = torch.eye(4, device='cuda', dtype=torch.float32)
    
    # Training Loop
    print(f"  Optimizing for {iterations} iterations...")
    import traceback
    try:
        for i in range(iterations):
            loss, _ = optimizer.optimize_step(gt_image, viewmat, K)
            
            if i % 10 == 0:
                print(f"  Iter {i:03d}: Loss {loss:.4f}")
                
        # Save Final Result
        final_ply_path = os.path.join(output_dir, "optimized_gs.ply")
        optimizer.save_ply(final_ply_path)
        print(f"  Saved Optimized PLY: {final_ply_path}")
        
        # Render Final Image for Verification
        print("  Rendering result...")
        with torch.no_grad():
            render_colors = optimizer.render(viewmat, K, new_w, new_h)
            # Convert to numpy image
            render_img = (render_colors.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            render_path = os.path.join(output_dir, "render_result.png")
            Image.fromarray(render_img).save(render_path)
            print(f"  Saved Render to: {render_path}")
            
    except Exception as e:
        print("ERROR IN PHASE 3:")
        traceback.print_exc()
        with open(os.path.join(output_dir, "error_log.txt"), "w") as f:
            f.write(traceback.format_exc())
            
    return os.path.join(output_dir, "optimized_gs.ply"), os.path.join(output_dir, "render_result.png")

    return final_ply_path, render_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End Image to 3DGS Pipeline")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--iters", type=int, default=100, help="Number of optimization iterations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        sys.exit(1)
        
    run_demo(args.image, iterations=args.iters)
