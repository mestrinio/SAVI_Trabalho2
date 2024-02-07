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
from screenshot import screenshot
from callmodel.call_model import Call_Md_2d


#################### VIEW ########################
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
    
    ########### ARGS ############
    parser = argparse.ArgumentParser(description='Detection Script.')
    parser.add_argument('-s', '--scene_selection', type=str, help='', required=False, 
                        default='rgbd-scenes-v2/pcdscenes/01.pcd')

    args = vars(parser.parse_args()) # creates a dictionary
    print(args)
    scene_path = args['scene_selection']

    
    
    '''######################################################################### SCENE'''
    scene_pcd = scene_selection(scene_path)
    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    
    # initial scene show
    initial_scene = []
    initial_scene.append(frame_world)
    initial_scene.append(scene_pcd)
    

    # VISUALIZATION
    o3d.visualization.draw_geometries(initial_scene,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)
    #screenshot()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    vis.add_geometry(scene_pcd)
    vis.capture_screen_image('cena.jpeg', do_render=False)
    vis.run()
    
    vis.destroy_window()

    
    '''############### TRANSFORMATIONS ################'''
    # Create transformation T1 only with rotation
    T1 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T1[3, 3] = 1

    # Add null rotation
    R = scene_pcd.get_rotation_matrix_from_xyz((110*math.pi/180, 0, 40*math.pi/180))
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
    R = scene_pcd.get_rotation_matrix_from_xyz((2*math.pi/180, 0*math.pi/180, 0*math.pi/180))
    T2[0:3, 0:3] = R
    print('T2=\n' + str(T2))

    T = np.dot(T1, T2)
    print('T=\n' + str(T))

    # Create table ref system and apply transformation to it
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))

    frame_table = frame_table.transform(T)

    scene_pcd = scene_pcd.transform(np.linalg.inv(T))

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
    pcd_cropped = scene_pcd.crop(bbox)


   
    '''###################### Plane segmentation'''
    plane_model, inlier_idxs = pcd_cropped.segment_plane(distance_threshold=0.02,
                                                         ransac_n=3, num_iterations=100)

    a, b, c, d = plane_model
    pcd_table = pcd_cropped.select_by_index(inlier_idxs, invert=False)
    pcd_table.paint_uniform_color([1, 0, 0])

    pcd_objects = pcd_cropped.select_by_index(inlier_idxs, invert=True)
    
    
    '''################################################################################ CLUSTERING'''
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
    
    #pcd_cropped.paint_uniform_color([0.9, 0.0, 0.0])
    #pcd_table.paint_uniform_color([0.0, 0.0, 0.9])
    
    
    
    
    '''------------------------------------------------------CYCLYING THROUGH OBJECTS PHASE HERE----------------------------------------------------------------'''
    ################## VARS INICIALIZATION
    
    good_objects =  [] # REMOVE TABLE OBJS

    props = {} # OBJECT PROPERTIES
    aabbs = {} # BOUNDING BOXES
    
    FINAL_SCENE = []
    FINAL_SCENE.append(frame_world)
    
    
    label_k , label_pred = Call_Md_2d()
    
    i= 0
    '''######################################################## CYCLE THROUGH RIGHT AND WRONG OBJECTS BUT ONLY COUNT GOOD ONES'''
    for idx, object_data in enumerate(pcd_separate_objects):
        
        # Clean scene before showing each object
        object_window = []
        object_window.append(frame_world)
        object_window.append(object_data)

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
        
        
        
        objects_data = []
        # SELECIONAR POINTCLOUDS DE OBJETOS CORRETAS
        if  largura < 0.50 and comprimento < 0.50:
            if len(object_data.points) > 1500:
                
                # SALVAR CADA PCD
                filename = f"objects_pcd/objects_to_icp/object_pcd_{idx:03}.pcd"
                o3d.io.write_point_cloud(filename, object_data) 
                
                # VISUALIZE ONLY GOOD ONES
                o3d.visualization.draw_geometries(object_window,
                                        zoom=0.3412,
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'], point_show_normal=False)
                
                aabb = object_data.get_axis_aligned_bounding_box()
                aabb.color = (1, 0, 0)

                aabbs[idx] = aabb
                
                #vis.add_geometry("bounding boxes",aabb)
                #label_text = 'objeto'
                #label_text = f"{object_data['label'].capitalize()}\nColor: {object_data['color_name']}\nHeight: {object_data['height']} mm\nWidth: {object_data['width']} mm"


                print(aabbs[idx])
                props[idx]={'text_pos':maxbound,'altura':altura,'comprimento':comprimento,'largura':largura,'maxbound':maxbound,'minbound':minbound,'object_name':label_pred[i]}
                
                label_text = props [idx]
                
                pcd_dataset = o3d.io.read_point_cloud('data/objects_pcd/rgbd-dataset/bowl/bowl_1/bowl_1_1_1.pcd')
                pcd_dataset_ds = pcd_dataset.voxel_down_sample(voxel_size=0.005)
                pcd_dataset_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pcd_dataset_ds.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

                Tinit = np.eye(4, dtype=float)  # null transformation
                reg_p2p = o3d.pipelines.registration.registration_icp(pcd_dataset_ds, object_data, 0.9, Tinit,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

                print('object idx ' + str(idx))
                print('reg_p2p = ' + str(reg_p2p))

                print("Transformation is:")
                print(reg_p2p.transformation)

                objects_data.append({'transformation': reg_p2p.transformation, 'rmse': reg_p2p.inlier_rmse})
                #print('Object idx ' + str(min_rmse_idx) + ' is the cereal box')
                draw_registration_result(object_data, pcd_dataset_ds,
                             np.linalg.inv(objects_data[0]['transformation']))

                # SAVE GOOD OBJECTS TO VAR
                good_objects.append(object_data)

                i= i + 1 

                FINAL_SCENE.append(object_data)
                FINAL_SCENE.append(aabb)
        
                
    '''#################################################### INITIALIZE WINDOW GUI'''
    app = gui.Application.instance
    app.initialize()

    w = app.create_window("Open3D - 3D Text", 1024, 768)

    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling
    
    widget3d.scene.add_geometry("frame", scene_pcd, mat)
    
    
    
    
    ############################## CYCLE BOUNDING BOXES
    for key,value in aabbs.items():
        widget3d.scene.add_geometry(str(key), value, mat)
    
    ################################### CYCLE PROPERTIES TO WRITE IN BOUNDING BOXES
    for idx,properties in props.items():

        l = widget3d.add_3d_label(properties['text_pos'], "object name: {}\naltura:{}\nmaxbound:{}".format(properties['object_name'],properties['altura'],properties['maxbound']))

        l.color = gui.Color(0,0,0)

        l.scale = 1.5
    
    
    #################################### Final execution of window GUI
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)
    app.run()
    
    
    ############################################ SHOW FINAL SCENE IN NORMAL WINDOW
    o3d.visualization.draw_geometries(FINAL_SCENE,
                                        zoom=0.3412,
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'], point_show_normal=False)
    
if __name__ == "__main__":
    main()


    
    
    
    
    
    
    