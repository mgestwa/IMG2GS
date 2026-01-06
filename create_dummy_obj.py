import trimesh
import numpy as np

def create_cube():
    # Create a simple box 
    mesh = trimesh.creation.box(extents=(2, 2, 2))
    
    # Assign some random vertex colors for fun
    mesh.visual.vertex_colors = np.random.uniform(0, 255, (len(mesh.vertices), 4)).astype(np.uint8)
    
    path = "dummy_cube.obj"
    mesh.export(path)
    print(f"Created {path} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

if __name__ == "__main__":
    create_cube()
