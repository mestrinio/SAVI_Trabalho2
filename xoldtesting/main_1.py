#!/usr/bin/env python3

# -----------------------------------------------------------------
# Project: PSR 2022-2023
# Author: Miguel Riem Oliveira
# Inspired in:
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
# -----------------------------------------------------------------

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




view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [2.6540005122611348, 2.3321821423160629, 0.85104994623420782],
                "boundingbox_min": [-2.5261458770339673, -2.1656718060235378, -0.55877501755379944],
                "field_of_view": 60.0,
                "front": [0.75672239933786944, 0.34169632162348007, 0.55732830013316348],
                "lookat": [0.046395260625899069, 0.011783639768603466, -0.10144691776517496],
                "up": [-0.50476400916821107, -0.2363660920597864, 0.83026764695055955],
                "zoom": 0.30119999999999997
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


def draw_registration_result(source, target, transformation):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def main():
    # -----------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------
    labels = ['apple', 'ball', 'banana', 'bell pepper', 'binder', 'bowl', 'calculator',
               'camera', 'cap', 'cell phone', 'cereal box', 'coffee mug', 'comb', 'dry battery',
                'flashlight', 'food bag', 'food box', 'food can', 'food cup', 'food jar',
                'garlic', 'glue stick', 'greens', 'hand towel', 'instant noodles', 'keyboard',
                'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook',
                'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato',
                'rubber eraser', 'scissors', 'shampoo', 'soda can', 'sponge', 'stapler', 
                'tomato', 'toothbrush', 'toothpaste', 'water bottle']
    
    
    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    pcd_original = o3d.io.read_point_cloud('rgbd-scenes-v2/pcdscenes/01.pcd')
    
    # Downsample using voxel grid ------------------------------------
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.003)
    # pcd_downsampled.paint_uniform_color([1,0,0])

    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
    
    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    initial_scene = []
    initial_scene.append(frame_world)
    
    

    
    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    # Downsample using voxel grid ------------------------------------
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.005)
    # pcd_downsampled.paint_uniform_color([1,0,0])

    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    # Create transformation T1 only with rotation
    T1 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T1[3, 3] = 1

    # Add null rotation
    R = pcd_downsampled.get_rotation_matrix_from_xyz((110*math.pi/180, 0, 40*math.pi/180))
    T1[0:3, 0:3] = R
    # T[0:3, 0] = [1, 0, 0]  # add n vector
    # T[0:3, 1] = [0, 1, 0]  # add s vector
    # T[0:3, 2] = [0, 0, 1]  # add a vector

    # Add a translation
    T1[0:3, 3] = [0, 0, 0]
    print('T1=\n' + str(T1))

    # Create transformation T2 only with translation
    T2 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T2[3, 3] = 1

    # Add null rotation
    T2[0:3, 0] = [1, 0, 0]  # add n vector
    T2[0:3, 1] = [0, 1, 0]  # add s vector
    T2[0:3, 2] = [0, 0, 1]  # add a vector

    # Add a translation
    T2[0:3, 3] = [0.8, 1, -0.35]
    R = pcd_downsampled.get_rotation_matrix_from_xyz((2*math.pi/180, 0*math.pi/180, 0*math.pi/180))
    T2[0:3, 0:3] = R
    print('T2=\n' + str(T2))

    T = np.dot(T1, T2)
    print('T=\n' + str(T))

    # Create table ref system and apply transformation to it
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))

    frame_table = frame_table.transform(T)

    pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T))

    # Ex3 - Create crop the points in the table

    # Create a vector3d with the points in the boundingbox
    np_vertices = np.ndarray((8, 3), dtype=float)

    sx = sy = 0.6
    sz_top = 0.4
    sz_bottom = -0.1
    np_vertices[0, 0:3] = [sx, sy, sz_top]
    np_vertices[1, 0:3] = [sx, -sy, sz_top]
    np_vertices[2, 0:3] = [-sx, -sy, sz_top]
    np_vertices[3, 0:3] = [-sx, sy, sz_top]
    np_vertices[4, 0:3] = [sx, sy, sz_bottom]
    np_vertices[5, 0:3] = [sx, -sy, sz_bottom]
    np_vertices[6, 0:3] = [-sx, -sy, sz_bottom]
    np_vertices[7, 0:3] = [-sx, sy, sz_bottom]

    print('np_vertices =\n' + str(np_vertices))

    vertices = o3d.utility.Vector3dVector(np_vertices)

    # Create a bounding box
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)
    print(bbox)

    # Crop the original point cloud using the bounding box
    pcd_cropped = pcd_downsampled.crop(bbox)

    # --------------------------------------
    # Plane segmentation
    # --------------------------------------
    plane_model, inlier_idxs = pcd_cropped.segment_plane(distance_threshold=0.02,
                                                         ransac_n=3, num_iterations=100)

    a, b, c, d = plane_model
    pcd_table = pcd_cropped.select_by_index(inlier_idxs, invert=False)
    pcd_table.paint_uniform_color([1, 0, 0])

    pcd_objects = pcd_cropped.select_by_index(inlier_idxs, invert=True)

    # --------------------------------------
    # Clustering
    # --------------------------------------

    labels = pcd_objects.cluster_dbscan(eps=0.02, min_points=50, print_progress=True)

    print("Max label:", max(labels))

    group_idxs = list(set(labels))
    # group_idxs.remove(-1)  # remove last group because its the group on the unassigned points
    num_groups = len(group_idxs)
    #colormap = cm.Pastel1(range(0, num_groups))

    pcd_separate_objects = []
    for group_idx in group_idxs:  # Cycle all groups, i.e.,

        group_points_idxs = list(locate(labels, lambda x: x == group_idx))

        pcd_separate_object = pcd_objects.select_by_index(group_points_idxs, invert=False)

        #color = colormap[group_idx, 0:3]
        #pcd_separate_object.paint_uniform_color(color)
        pcd_separate_objects.append(pcd_separate_object)

#    # --------------------------------------
#    # ICP for object classification
#    # --------------------------------------
#    pcd_cereal_box = o3d.io.read_point_cloud('../data/cereal_box_2_2_40.pcd')
#    pcd_cereal_box_ds = pcd_cereal_box.voxel_down_sample(voxel_size=0.005)
#    pcd_cereal_box_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#    pcd_cereal_box_ds.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
#
#    objects_data = []
#    for idx, pcd_separate_object in enumerate(pcd_separate_objects):
#
#        Tinit = np.eye(4, dtype=float)  # null transformation
#        reg_p2p = o3d.pipelines.registration.registration_icp(pcd_cereal_box_ds, pcd_separate_object, 0.9, Tinit,
#                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
#
#        print('object idx ' + str(idx))
#        print('reg_p2p = ' + str(reg_p2p))
#
#        print("Transformation is:")
#        print(reg_p2p.transformation)
#
#        objects_data.append({'transformation': reg_p2p.transformation, 'rmse': reg_p2p.inlier_rmse})
#        # draw_registration_result(pcd_separate_object, pcd_cereal_box_ds, np.linalg.inv(reg_p2p.transformation))
#
#    # Select which of the objects in the table is a cereal box by getting the minimum rmse
#    min_rmse = None
#    min_rmse_idx = None
#
#    for idx, object_data in enumerate(objects_data):
#
#        if min_rmse is None:  # first object, use as minimum
#            min_rmse = object_data['rmse']
#            min_rmse_idx = idx
#
#        if object_data['rmse'] < min_rmse:
#            min_rmse = object_data['rmse']
#            min_rmse_idx = idx
#
#    print('Object idx ' + str(min_rmse_idx) + ' is the cereal box')
#    draw_registration_result(pcd_separate_objects[min_rmse_idx], pcd_cereal_box_ds,
#                             np.linalg.inv(objects_data[min_rmse_idx]['transformation']))
#
#    print(objects_data)





    
    # --------------------------------------
    # Visualization ----------------------
    # --------------------------------------
    #pcd_downsampled.paint_uniform_color([0.4, 0.3, 0.3])
    pcd_cropped.paint_uniform_color([0.9, 0.0, 0.0])
    pcd_table.paint_uniform_color([0.0, 0.0, 0.9])

    #pcds_to_draw = [pcd_objects]
    pcds_to_draw = (pcd_separate_objects)

    #pcds_to_draw = [pcd_cereal_box_ds]

    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    initial_scene.append(pcd_downsampled)
    o3d.visualization.draw_geometries(initial_scene,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)
    

    
    good_objects =  []
    show_final = []
    show_final.append(frame_world)
    
    props = {}
    #obb = []
    aabbs = {}
    
    for idx, object_data in enumerate(pcd_separate_objects):
        
        entiti = []
        entiti.append(frame_world)
        entiti.append(object_data)

        maxbound = o3d.geometry.PointCloud.get_max_bound(object_data)
        minbound = o3d.geometry.PointCloud.get_min_bound(object_data)
        print('MAXBOUND',maxbound)
        print('minbound',minbound)
        altura = maxbound [2] - minbound [2]
        comprimento = maxbound [0] - minbound [0]
        largura = maxbound [1] - minbound [1]
        print('ALTURA',altura)
        print('COMP',comprimento)
        print('LARG',largura)
        

        #guardar as point clouds 
        if  largura < 0.50 and comprimento < 0.50:
            if len(object_data.points) > 1500:
                filename = f"objects_pcd/objects_to_icp/object_pcd_{idx:03}.pcd"
                o3d.io.write_point_cloud(filename, object_data) 
                o3d.visualization.draw_geometries(entiti,
                                        zoom=0.3412,
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'], point_show_normal=False)
                
                aabb = object_data.get_axis_aligned_bounding_box()
                aabb.color = (1, 0, 0)
                #aabb = object_data.get_oriented_bounding_box()
                #aabb.color = (1, 0, 0)
                
                aabbs[idx] = aabb
                
                #vis.add_geometry("bounding boxes",aabb)
                
                #label_text = 'objeto'
                #label_text = f"{object_data['label'].capitalize()}\nColor: {object_data['color_name']}\nHeight: {object_data['height']} mm\nWidth: {object_data['width']} mm"
                
                print('AAAAAAAAAAAAAAAAA',maxbound)
                print(aabbs[idx])
                props[idx]={'text_pos':maxbound,'altura':altura,'comprimento':comprimento,'largura':largura,'maxbound':maxbound,'minbound':minbound}
                label_text = props [idx]
                good_objects.append(object_data)
                show_final.append(object_data)
                show_final.append(aabb)
                
                #show_final.append(aabb)

    print(aabbs)
    app = gui.Application.instance
    app.initialize()


    w = app.create_window("Open3D - 3D Text", 1024, 768)

    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling

    widget3d.scene.add_geometry("frame", pcd_downsampled, mat)
    for key,value in aabbs.items():
        widget3d.scene.add_geometry(str(key), value, mat)
        
    for idx,properties in props.items():

        l = widget3d.add_3d_label(properties['text_pos'], "altura:{}\nmaxbound:{}".format(properties['altura'],properties['maxbound']))

        l.color = gui.Color(0,0,0)

        l.scale = 1.5
        

    #for idx in range(0, len(points.points)):
#
    #    l = widget3d.add_3d_label(points.points[idx], "{}".format(idx))
#
    #    l.color = gui.Color(points.colors[idx][0], points.colors[idx][1],
#
    #                        points.colors[idx][2])
#
    #    l.scale = np.random.uniform(0.5, 3.0)

    bbox = widget3d.scene.bounding_box

    widget3d.setup_camera(60.0, bbox, bbox.get_center())

    w.add_child(widget3d)


    app.run()

    
    #getpngfromscene(scene_path='rgbd-scenes-v2/pcdscenes/01.pcd',crops = props)
    show_final.extend(good_objects)
    
    o3d.visualization.draw_geometries(show_final,
                                        zoom=0.3412,
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'], point_show_normal=False)
    
    
    
    
   # app = gui.Application.instance
   # app.initialize()
#
   # 
#
   # w = app.create_window("Open3D - 3D Text", 1024, 768)
#
   # widget3d = gui.SceneWidget()
   # widget3d.scene = rendering.Open3DScene(w.renderer)
#
   # mat = rendering.MaterialRecord()
   # mat.shader = "defaultUnlit"
   # mat.point_size = 5 * w.scaling
#
   # widget3d.scene.add_geometry("Points", points, mat)
#
   # for idx in range(0, len(points.points)):
#
   #     l = widget3d.add_3d_label(points.points[idx], "{}".format(idx))
#
   #     l.color = gui.Color(points.colors[idx][0], points.colors[idx][1],
#
   #                         points.colors[idx][2])
#
   #     l.scale = np.random.uniform(0.5, 3.0)
#
   # bbox = widget3d.scene.bounding_box
#
   # widget3d.setup_camera(60.0, bbox, bbox.get_center())
#
   # w.add_child(widget3d)
#
#
   # app.run()
   # 
    
    
    
    
    
    
    
    
    # Initialize window
    #gui.Application.instance.initialize()
    #window = gui.Application.instance.create_window("Results", 1024, 768)   # 4x3
    #scene = gui.SceneWidget()
    #scene.scene = rendering.Open3DScene(window.renderer)
    #window.add_child(scene)
#
    #for i, obj in enumerate(objects):
#
    #    # Add point cloud
    #    point_cloud = obj['points']
    #    scene.scene.add_geometry(f'object_{i}', point_cloud, rendering.MaterialRecord())
#
    #    # Add bbox
    #    bbox = point_cloud.get_minimal_oriented_bounding_box()
    #    bbox.color = [0, 0, 0]  # colors from obj['color'] are too bright
    #    scene.scene.add_geometry(f'bbox_{i}', bbox, rendering.MaterialRecord())
#
    #    # Add label
    #    label_pos = bbox.get_center()
    #    label_pos[2] += 0.15
    #    label_text = f"{obj['label'].capitalize()}\nColor: {obj['color_name']}\nHeight: {obj['height']} mm\nWidth: {obj['width']} mm"
    #    scene.add_3d_label(label_pos, label_text)
#
    ## Run
    #scene_bounds = o3d.geometry.AxisAlignedBoundingBox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    #scene.setup_camera(60, scene_bounds, [0, 0, 0])
    #gui.Application.instance.run()
    
    #o3d.visualization.draw_geometries(show_final,
    #                                    zoom=0.3412,
    #                                    front=view['trajectory'][0]['front'],
    #                                    lookat=view['trajectory'][0]['lookat'],
    #                                    up=view['trajectory'][0]['up'], point_show_normal=False)
    

    
    #entities = []
    #entities.append(frame_world)
    #entities.append(frame_table)
    #entities.extend(pcds_to_draw)
    #o3d.visualization.draw_geometries(entities,
    #                                  zoom=0.3412,
    #                                  front=view['trajectory'][0]['front'],
    #                                  lookat=view['trajectory'][0]['lookat'],
    #                                  up=view['trajectory'][0]['up'], point_show_normal=False)

    # -----------------------------------------------------------------
    # Termination
    # -----------------------------------------------------------------

    #app = gui.Application.instance
    #app.initialize()
#
    #vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    #vis.show_settings = True
    #vis.add_geometry("Frame", pcd_downsampled)
    #vis.add_geometry("bounding boxes",obb)
    ##for idx in range(0, len(points.points)):
##
    ##    vis.add_3d_label(points.points[idx], "{}".format(idx))
#
    #vis.reset_camera_to_default()
#
    #app.add_window(vis)
    #app.run()
    #
#

if __name__ == "__main__":
    main()
