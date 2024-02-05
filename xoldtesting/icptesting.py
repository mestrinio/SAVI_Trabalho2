#!/usr/bin/env python3

from copy import deepcopy
import math
import os
import open3d as o3d
import numpy as np
#from matplotlib import cm
from more_itertools import locate
from open3d.visualization import gui
from open3d.visualization import rendering
#import call_model
#from scene_png.objects import getpngfromscene
import argparse
from scene_selection import scene_selection
import re
import glob

file_path = 'data/objects_pcd/rgbd-dataset/'
filenames = glob.glob(file_path + '***/**/*.pcd')
pattern = '([a-z_]+)(?=_\d)'

object_paths = 'objects_pcd/objects_to_icp/'
from time import sleep
#def separate_objects():
    
    
    
def main():
    
    pcd_separate_object = o3d.io.read_point_cloud('objects_pcd/objects_to_icp/object_pcd_000.pcd')
    
    #--------------------------------------
    # ICP for object classification
    # --------------------------------------
    pcd_dataset = o3d.io.read_point_cloud('data/objects_pcd/rgbd-dataset/bowl/bowl_1/bowl_1_1_1.pcd')
    pcd_dataset_ds = pcd_dataset.voxel_down_sample(voxel_size=0.005)
    pcd_dataset_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    sleep(5)
    pcd_dataset_ds.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
    
    
    Tinit = np.eye(4, dtype=float)  # null transformation
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd_dataset_ds, pcd_separate_object, 0.9, Tinit,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    #print('object idx ' + str(idx))
    print('reg_p2p = ' + str(reg_p2p))
    print("Transformation is:")
    print(reg_p2p.transformation)
    pcd_separate_object.extend({'transformation': reg_p2p.transformation, 'rmse': reg_p2p.inlier_rmse})
    rmse = pcd_separate_object['rmse']
    print('RMSEEEEEE',rmse)
    o3d.draw_registration_result(pcd_separate_object, pcd_dataset_ds,
                             np.linalg.inv(pcd_separate_object['transformation']))
    
    #for idx, pcd_separate_object in enumerate(pcd_separate_objects):
    #    Tinit = np.eye(4, dtype=float)  # null transformation
    #    reg_p2p = o3d.pipelines.registration.registration_icp(pcd_dataset_ds, pcd_separate_object, 0.9, Tinit,
    #                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #                                                          o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    #    print('object idx ' + str(idx))
    #    print('reg_p2p = ' + str(reg_p2p))
    #    print("Transformation is:")
    #    print(reg_p2p.transformation)
    #    objects_data.append({'transformation': reg_p2p.transformation, 'rmse': reg_p2p.inlier_rmse})
    #    # draw_registration_result(pcd_separate_object, pcd_dataset_ds, np.linalg.inv(reg_p2p.transformation))
    # Select which of the objects in the table is a cereal box by getting the minimum rmse
    #min_rmse = None
    #min_rmse_idx = None
    #for idx, object_data in enumerate(objects_data):
    #    if min_rmse is None:  # first object, use as minimum
    #        min_rmse = object_data['rmse']
    #        min_rmse_idx = idx
    #    if object_data['rmse'] < min_rmse:
    #        min_rmse = object_data['rmse']
    #        min_rmse_idx = idx
    #print('Object idx ' + str(min_rmse_idx) + ' is the cereal box')
    #draw_registration_result(pcd_separate_objects[min_rmse_idx], pcd_dataset_ds,
    #                         np.linalg.inv(objects_data[min_rmse_idx]['transformation']))
    #print(objects_data)
    
    
    '''
    for filename in filenames:
        match = re.search(pattern, filename)
        label = match.group(1)
        
        file** acerto
    
    
        #--------------------------------------
        # ICP for object classification
        # --------------------------------------
        pcd_dataset = o3d.io.read_point_cloud('../data/cereal_box_2_2_40.pcd')
        pcd_dataset_ds = pcd_dataset.voxel_down_sample(voxel_size=0.005)
        pcd_dataset_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_dataset_ds.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

        objects_data = []
        for idx, pcd_separate_object in enumerate(pcd_separate_objects):

            Tinit = np.eye(4, dtype=float)  # null transformation
            reg_p2p = o3d.pipelines.registration.registration_icp(pcd_dataset_ds, pcd_separate_object, 0.9, Tinit,
                                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                  o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

            print('object idx ' + str(idx))
            print('reg_p2p = ' + str(reg_p2p))

            print("Transformation is:")
            print(reg_p2p.transformation)

            objects_data.append({'transformation': reg_p2p.transformation, 'rmse': reg_p2p.inlier_rmse})
            # draw_registration_result(pcd_separate_object, pcd_dataset_ds, np.linalg.inv(reg_p2p.transformation))

        # Select which of the objects in the table is a cereal box by getting the minimum rmse
        min_rmse = None
        min_rmse_idx = None

        for idx, object_data in enumerate(objects_data):

            if min_rmse is None:  # first object, use as minimum
                min_rmse = object_data['rmse']
                min_rmse_idx = idx

            if object_data['rmse'] < min_rmse:
                min_rmse = object_data['rmse']
                min_rmse_idx = idx

        print('Object idx ' + str(min_rmse_idx) + ' is the cereal box')
        draw_registration_result(pcd_separate_objects[min_rmse_idx], pcd_dataset_ds,
                                 np.linalg.inv(objects_data[min_rmse_idx]['transformation']))

        print(objects_data)
'''

    
if __name__ == "__main__":
    #separate_objects()
    main()


    
    
    
    
    
    
    