import open3d as o3d
import numpy as np

def scene_selection(path):
    
    pcd_original = o3d.io.read_point_cloud('rgbd-scenes-v2/pcdscenes/01.pcd')
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.005)
    
    # pcd_downsampled.paint_uniform_color([1,0,0])
    
    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
    
    
    return pcd_downsampled