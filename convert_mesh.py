import argparse
import sys
import torch
import numpy as np
import os
from core.mesh_ingest import load_mesh, sample_mesh
from core.optimization import GaussianOptimizer

def main():
    parser = argparse.ArgumentParser(description="Convert Mesh (OBJ/GLTF/IFC) to 3D Gaussian Splatting PLY")
    parser.add_argument("input_path", type=str, help="Path to input mesh file (.obj, .gltf, .ply, .stl)")
    parser.add_argument("--output", type=str, default=None, help="Output PLY path (default: input_dir/mesh_gs.ply)")
    parser.add_argument("--points", type=int, default=1_000_000, help="Number of points to sample (default: 1,000,000)")
    parser.add_argument("--solid", action="store_true", help="Force solid opacity (no transparency)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Mesh
    try:
        mesh = load_mesh(args.input_path)
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        sys.exit(1)
        
    # 2. Sample Points
    print("Sampling surface...")
    xyz, rgb, scale_radius = sample_mesh(mesh, args.points)
    
    # 3. Initialize Optimizer (as a container for 3DGS params)
    print("Initializing Gaussian Container...")
    # GaussianOptimizer expects [N, 3] tensors on device
    # But it also creates its own params. 
    # We will let it init, then override.
    optimizer = GaussianOptimizer(xyz, rgb, device=device)
    
    # 4. Override Parameters for "Solid" Mesh look
    # Scale:
    # Optimizer defaults to -5.0. We want log(scale_radius).
    log_scale = np.log(scale_radius)
    print(f"Setting Splat Radius: {scale_radius:.6f} (Log: {log_scale:.2f})")
    
    with torch.no_grad():
        # Override Scales
        optimizer.scales.data.fill_(log_scale)
        
        # Override Opacity
        # 3DGS usually optimizes logits. 
        # sigmoid(10) ~ 1.0 (Solid)
        # sigmoid(-10) ~ 0.0 (Transparent)
        # If we want solid look, set high logit.
        if args.solid:
            print("Forcing 100% Opacity...")
            optimizer.opacities.data.fill_(10.0) 
        else:
            # Default init is 0.0 (sigmoid(0) = 0.5 opacity)
            # Maybe for mesh we want it quite solid by default too?
            # Let's default to something high like 3.0 (sigmoid(3) ~ 0.95)
            optimizer.opacities.data.fill_(5.0)

    # 5. Save
    if args.output:
        out_path = args.output
    else:
        # Default: same dir as input, named mesh_gs.ply
        base_dir = os.path.dirname(args.input_path)
        out_path = os.path.join(base_dir, "mesh_gs.ply")
        
    optimizer.save_ply(out_path)
    print("Conversion Complete!")

if __name__ == "__main__":
    main()
