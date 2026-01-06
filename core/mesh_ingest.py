import trimesh
import numpy as np
import os

def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Loads a mesh file (OBJ, GLTF, PLY, etc.) using trimesh.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")
    
    # trimesh.load can return a Scene or a Trimesh or a list
    mesh = trimesh.load(path, force='mesh')
    
    # If it's a Scene, we dump all geometries into a single mesh
    if isinstance(mesh, trimesh.Scene):
        print(f"[MeshIngest] Input is a Scene with {len(mesh.geometry)} geometries. Merging...")
        mesh = trimesh.util.concatenate(
            tuple(trimesh.util.concatenate(g) for g in mesh.geometry.values())
        )
        
    print(f"[MeshIngest] Loaded mesh. Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    return mesh

def sample_mesh(mesh: trimesh.Trimesh, num_points: int = 1_000_000):
    """
    Samples points uniformly from the mesh surface.
    Returns:
        xyz: [N, 3] positions
        rgb: [N, 3] colors (0-1)
        scale: float (estimated optimal scale for splats)
    """
    print(f"[MeshIngest] Sampling {num_points} points from surface...")
    
    # Sample points and get face indices to look up colors
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    # --- Colors ---
    # Default color (White/Light Grey)
    colors = np.ones((num_points, 3), dtype=np.float32) * 0.8
    
    # Try to extract texture/color information
    if hasattr(mesh.visual, 'kind'):
        if mesh.visual.kind == 'texture' and mesh.visual.uv is not None:
            # Texture lookup is complex in raw trimesh without rendering.
            # Simplified: Use simple vertex color interpolation if available, or fallback.
            # For now, we will stick to default white for uncolored meshes, 
            # unless vertex colors exist.
            pass
        elif mesh.visual.kind == 'vertex':
            # Interpolate vertex colors
            # face_indices tells us which face each point is on.
            # We can get the 3 vertices of that face and interpolate.
            # For massive speed, we just take the color of the first vertex of the face.
            faces = mesh.faces[face_indices]
            vertex_colors = mesh.visual.vertex_colors # RGBA usually
            
            # Take color of the first vertex of the face (Nearest Neighbor approx for now)
            # Improving this to barycentric takes more code, maybe later if requested.
            c = vertex_colors[faces[:, 0]] 
            colors = c[:, :3] / 255.0 # Drop Alpha, normalize
            
    # --- Scale Estimation ---
    # We want splats to slightly overlap.
    # Estimate average distance between points.
    # Surface Area / N ~= Area per point.
    # Radius ~ sqrt(Area per point)
    area = mesh.area
    area_per_point = area / num_points
    # Radius of a circle with that area: A = pi*r^2 => r = sqrt(A/pi)
    estimated_radius = np.sqrt(area_per_point / np.pi)
    
    # We apply a slight overlap factor
    scale = estimated_radius * 1.5 
    
    print(f"[MeshIngest] Sampling done. Est. Radius: {scale:.6f}, Area: {area:.2f}")
    
    return points.astype(np.float32), colors.astype(np.float32), float(scale)
