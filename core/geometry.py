import numpy as np
import plyfile
from PIL import Image

def get_intrinsics(width, height, fov=55.0):
    """
    Computes camera intrinsics for a given image size and field of view.
    """
    f = 0.5 * width / np.tan(0.5 * np.radians(fov))
    cx = width / 2
    cy = height / 2
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

def project_to_3d(image: Image.Image, depth: np.ndarray, fov=55.0):
    """
    Project RGBD image to a 3D point cloud.
    """
    width, height = image.size
    
    # Create coordinate grid
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    # Get intrinsics
    K = get_intrinsics(width, height, fov)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Depth Anything V2 outputs relative disparity (higher = closer).
    # We need to convert this to metric-like depth (Z).
    # Z ~ 1 / Disparity
    
    # Normalize disparity to 0..1 range first for consistency
    depth_min = depth.min()
    depth_max = depth.max()
    disparity = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    
    # Invert to get Depth (Z)
    # Add epsilon to avoid division by zero
    # Scale factor (e.g. 10) determines how "deep" the scene looks relative to width
    z_3d = 1.0 / (disparity + 0.05)
    
    # Optional: Normalize Z to a specific range (e.g., mean distance = 5)
    # But usually 1/d is sufficient structure. Use a scalar to keep coordinates manageable.
    z_3d = z_3d * 5.0
    
    # Backprojection
    # X = (u - cx) * Z / fx
    x_3d = (xx - cx) * z_3d / fx
    y_3d = (yy - cy) * z_3d / fy
    
    # Stack to (H*W, 3)
    xyz = np.stack((x_3d, y_3d, z_3d), axis=-1).reshape(-1, 3)
    
    # Colors (H*W, 3), normalized to [0, 1]
    rgb = np.array(image).reshape(-1, 3) / 255.0
    
    return xyz, rgb

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels (features_dc, features_rest, opacity, scaling, rotation) will be handled in saving
    return l

def save_ply(xyz, rgb, output_path):
    """
    Saves the data as a .ply file compatible with 3D Gaussian Splatting viewers.
    Typically requires:
    - x, y, z
    - f_dc_0, f_dc_1, f_dc_2 (SH coefficients for DC term - derived from RGB)
    - opacity
    - scale_0, scale_1, scale_2
    - rot_0, rot_1, rot_2, rot_3
    """
    
    num_points = xyz.shape[0]
    
    # -- 1. Positions --
    # x, y, z
    
    # -- 2. Normals (optional, usually ignored by GS viewers but good for standard tools) --
    normals = np.zeros_like(xyz)

    # -- 3. SH Coefficients (f_dc) --
    # RGB [0, 1] -> SH DC term
    # C0 = 0.28209479177387814
    # color = C0 * SH
    # SH = (color - 0.5) / C0 ... approximate conversion used in standard GS is:
    # SH_0 = (RGB - 0.5) / 0.28209... actually standard conversion is straightforward:
    # color = 0.5 + C0 * SH  => SH = (color - 0.5) / C0
    
    C0 = 0.28209479177387814
    f_dc = (rgb - 0.5) / C0
    
    # -- 4. Opacity --
    # Inverse sigmoid is usually applied in GS training code, but ply file stores raw value?
    # Usually standard .ply for GS stores raw values that go through sigmoid activation.
    # For "opacity=1", we need a high value. sigmoid(x) ~= 1 -> x is large (e.g., 100).
    # Wait, original .ply from 3DGS stores opacity after inverse sigmoid? No, usually before.
    # Let's assume standard GS viewer expects "logit" of opacity.
    # However, many simple viewers might expect 0..1. 
    # Let's stick effectively to "almost opaque" -> large logit.
    opacities = np.ones((num_points, 1)) * 100.0 

    # -- 5. Scales --
    # Logs of scales. Small splats.
    scales = np.ones((num_points, 3)) * -4.0 # exp(-4) is small

    # -- 6. Rotations --
    # Quaternion (w, x, y, z) = (1, 0, 0, 0)
    rots = np.zeros((num_points, 4))
    rots[:, 0] = 1.0
    
    # Construct structured array for PlyFile
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]

    elements = np.empty(num_points, dtype=dtype_full)
    
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opacities[:, 0]
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rots[:, 0]
    elements['rot_1'] = rots[:, 1]
    elements['rot_2'] = rots[:, 2]
    elements['rot_3'] = rots[:, 3]

    el = plyfile.PlyElement.describe(elements, 'vertex')
    plyfile.PlyData([el]).write(output_path)
