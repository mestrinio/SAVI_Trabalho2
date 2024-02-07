#!/usr/bin/env python3

import open3d as o3d
import os



def ply_to_pcd(input, output):
    ply_cloud = o3d.io.read_point_cloud(input) #ply
    o3d.io.write_point_cloud(output, ply_cloud)#pcd



if __name__ == "__main__":

    ply_dir = 'rgbd-scenes-v2/pc/'
    pcd_dir = 'rgbd-scenes-v2/pcdscenes/'

    # Create if not existent
    os.makedirs(pcd_dir, exist_ok=True)

    # for ply files
    for file_name in os.listdir(ply_dir):
        
        if file_name.endswith(".ply"):
            input = os.path.join(ply_dir, file_name)
            output = os.path.join(pcd_dir, file_name.replace(".ply", ".pcd"))
            ply_to_pcd(input, output)
