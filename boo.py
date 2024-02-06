#!/usr/bin/env python3

import pyrealsense2 as rs
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
from callmodel.call_model_2 import Call_Md_2d
from rgbd_camera import capture_scene_from_camera
from color_averaging import get_average_color_name
from txtts import text_to_speech,txt_speech
from time import sleep
import multiprocessing as mp


labels = ['apple', 'ball', 'banana', 'bell pepper', 'binder', 'bowl', 'calculator',
               'camera', 'cap', 'cell phone', 'cereal box', 'coffee mug', 'comb', 'dry battery',
                'flashlight', 'food bag', 'food box', 'food can', 'food cup', 'food jar',
                'garlic', 'glue stick', 'greens', 'hand towel', 'instant noodles', 'keyboard',
                'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook',
                'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato',
                'rubber eraser', 'scissors', 'shampoo', 'soda can', 'sponge', 'stapler', 
                'tomato', 'toothbrush', 'toothpaste', 'water bottle']



#################### VIEW ########################
view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.7932640408991203, 2.4171065092593551, 0.70642559403004523 ],
			"boundingbox_min" : [ -2.469507729419512, -2.2752481719085242, -0.98426195988268139 ],
			"field_of_view" : 60.0,
			"front" : [ 0.76730186090479968, 0.40913012498730894, 0.4938222302407016 ],
			"lookat" : [ -0.42008347255107703, -0.20169970201536941, -0.28142890036574197 ],
			"up" : [ -0.46194754979434627, -0.18149047511297473, 0.8681391989089462 ],
			"zoom" : 0.22119999999999995
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

def draw_registration_result(source, target, transformation):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)







def main():
    
    ########### ARGS PARSER ############
    parser = argparse.ArgumentParser(description='Detection Script.')
    parser.add_argument('-s', '--scene_selection', type=str, help='', required=False, 
                        default='rgbd-scenes-v2/pcdscenes/01.pcd')
    parser.add_argument('-c', '--camera', type=str, help='', required=False, 
                        default=0)


    args = vars(parser.parse_args()) # creates a dictionary
    print(args)
    scene_path = args['scene_selection']
    cam=args['camera']
    
    if cam==0:
        #scene_path = "rgbd-scenes-v2/pcdscenes/01.pcd"  
        scene_pcd = scene_selection(scene_path)
        text_to_speech('Scene from directory mode has been selected')
        proc = mp.Process(target=txt_speech)  # instantiating without any argument
        proc.start()
        sleep(2)
        print('Previous saved scene has been selected for detection')
    else:
        try:
            scene_pcd = capture_scene_from_camera()
            print('RGB-D camera mode will now start')
            text_to_speech('RGB-D mode has been selected')
            proc = mp.Process(target=txt_speech)  # instantiating without any argument
            proc.start()
            sleep(2)
        except Exception as e:
            print("Error capturing scene from camera:", e)
            return
    
    
    
    
    ############################################### SCENE'''
    scene_pcd = scene_selection(scene_path)
    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    
    # initial scene show
    initial_scene = []
    initial_scene.append(frame_world)
    initial_scene.append(scene_pcd)
    

    
    #screenshot()
    
    
    
    #vis = o3d.visualization.Visualizer()
    #vis.create_window()
    #render_option = vis.get_render_option()
    #render_option.mesh_show_back_face = True
    #vis.add_geometry(scene_pcd)
    #vis.capture_screen_image('cena.jpeg', do_render=False)
    #vis.run()
    #
    #vis.destroy_window()

    
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
    
    
    text_to_speech('Concluded loading and processing of scene. Displaying initial scene')
    proc = mp.Process(target=txt_speech)  # instantiating without any argument
    proc.start()
    sleep(2)
    
    
    
    # VISUALIZATION
    o3d.visualization.draw_geometries(initial_scene,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)
    
    
    
    '''------------------------------------------------------CYCLYING THROUGH OBJECTS PHASE HERE----------------------------------------------------------------'''
    ################## VARS INICIALIZATION
    
    good_objects =  [] # REMOVE TABLE OBJS

    props = {} # OBJECT PROPERTIES
    aabbs = {} # BOUNDING BOXES
    
    FINAL_SCENE = []
    FINAL_SCENE.append(frame_world)
    
    try:
        label_k , label_pred = Call_Md_2d()
    except:
        label_pred = ['404','404','404','404','404']
        print('MODEL NOT AVAILABLE, running with error...')
    
    
    
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
        color = get_average_color_name(object_data)
        
        
        
        # SELECIONAR POINTCLOUDS DE OBJETOS CORRETAS
        if  largura < 0.50 and comprimento < 0.50:
            if len(object_data.points) > 1500:
                
                # SALVAR CADA PCD
                filename = f"objects_pcd/objects_to_icp/object_pcd_{i:03}.pcd"
                o3d.io.write_point_cloud(filename, object_data) 
                
                
                # VISUALIZE ONLY GOOD ONES
                # o3d.visualization.draw_geometries(object_window,
                #                        zoom=0.3412,
                #                        front=view['trajectory'][0]['front'],
                #                        lookat=view['trajectory'][0]['lookat'],
                #                        up=view['trajectory'][0]['up'], point_show_normal=False)
                
                aabb = object_data.get_axis_aligned_bounding_box()
                aabb.color = (0, 1, 0)

                aabbs[i] = aabb
                
                #vis.add_geometry("bounding boxes",aabb)
                #label_text = 'objeto'
                #label_text = f"{object_data['label'].capitalize()}\nColor: {object_data['color_name']}\nHeight: {object_data['height']} mm\nWidth: {object_data['width']} mm"


                print(aabbs[i])
                centro =np.array([maxbound[0]-(comprimento/2),maxbound[1]-(largura/2),maxbound[2]+0.06])
                centro2 =np.array([maxbound[0],maxbound[1]-(largura/3),0])
                props[i]={'text_pos':centro,'altura':round(altura,2),'comprimento':round(comprimento,2),
                          'largura':round(largura,2),'maxbound':maxbound,'minbound':minbound,'deeplabel':'','centro2':centro2, 'color': color}
                
                #label_text = props [idx]
                

                # SAVE GOOD OBJECTS TO VAR
                good_objects.append(object_data)

                i= i + 1 

                FINAL_SCENE.append(object_data)
                FINAL_SCENE.append(aabb)
    
   
    

    

    
    
    text_to_speech('Separate objects obtained, now running the detections')
    proc = mp.Process(target=txt_speech)  # instantiating without any argument
    proc.start()
    sleep(2)
    
    
    
    '''################################################################################### ICP RUNNING FOR ALL GOOD OBJECTS'''
    
    # --------------------------------------
    # ICP for object classification
    # --------------------------------------
    
    
    #pcd_dataset = o3d.io.read_point_cloud('data/objects_pcd/rgbd-dataset/bowl/bowl_1/bowl_1_1_1.pcd')
    #pcd_dataset = o3d.io.read_point_cloud('objects_pcd/objects_to_png/object_pcd_005.pcd')
    #pcd_dataset_ds = pcd_dataset.voxel_down_sample(voxel_size=0.005)
    #pcd_dataset_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #pcd_dataset_ds.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
    
    objects_data = []
    rmse_temp = []
    lab = ['coffee_mug','cap','bowl_0','soda_can','bowl_1']    
    
    for idx, _ in enumerate(good_objects):
        rmse_temp.append([])

    for label in lab:
        
        pcd_dataset = o3d.io.read_point_cloud('objects_pcd/objects_to_png/'+label+'.pcd')
        pcd_dataset_ds = pcd_dataset.voxel_down_sample(voxel_size=0.005)
        pcd_dataset_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_dataset_ds.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))
        
        for idx, good_object in enumerate(good_objects):
            
            Tinit = np.eye(4, dtype=float)  # null transformation
            reg_p2p = o3d.pipelines.registration.registration_icp(pcd_dataset_ds, good_object, 0.9, Tinit,
                                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))

            print('object idx ' + str(idx))
            print('reg_p2p = ' + str(reg_p2p))

            print("Transformation is:")
            print(reg_p2p.transformation)

            rmse_temp[idx].append(reg_p2p.inlier_rmse)
            objects_data.append({'transformation': reg_p2p.transformation, 'rmse': reg_p2p.inlier_rmse})
            #draw_registration_result(good_object, pcd_dataset_ds, np.linalg.inv(reg_p2p.transformation))

    
    
    # Select which of the objects in the table is a cereal box by getting the minimum rmse
    min_rmse = None
    min_rmse_idx = None
    
    
    obj_w_lab = []
    
    for i, rmses in enumerate(rmse_temp):
        
        lab_idx = rmses.index(min(rmses))
        obj_w_lab.append(lab[lab_idx])
        
    
       

    #for idx, object_data in enumerate(objects_data):
#
    #    if min_rmse is None:  # first object, use as minimum
    #        min_rmse = object_data['rmse']
    #        min_rmse_idx = idx
#
    #    if object_data['rmse'] < min_rmse:
    #        min_rmse = object_data['rmse']
    #        min_rmse_idx = idx
#
    #print('Object idx ' + str(min_rmse_idx) + ' is the')
    #draw_registration_result(pcd_separate_objects[min_rmse_idx], pcd_dataset_ds, np.linalg.inv(objects_data[min_rmse_idx]['transformation']))
    
    
    text_to_speech('All the detections were made, displaying results')
    proc = mp.Process(target=txt_speech)  # instantiating without any argument
    proc.start()
    sleep(2)
    
    
    
    '''#################################################### INITIALIZE WINDOW GUI'''
    app = gui.Application.instance
    app.initialize()

    w = app.create_window("Open3D - 3D Text", 1800, 1000)

    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling
    
    widget3d.scene.add_geometry("frame", scene_pcd, mat)
    widget3d.scene.set_background(color = [0,0,0,0])
    
    
    ############################## CYCLE BOUNDING BOXES
    for key,value in aabbs.items():
        widget3d.scene.add_geometry(str(key), value, mat)
    
    
    scene_description = {}
    ################################### CYCLE PROPERTIES TO WRITE IN BOUNDING BOXES
    for idx,properties in props.items():
        # if idx == 0 or idx ==3:
        #     properties['text_pos'] = properties['maxbound']
        # else:
        #     properties['text_pos'] = properties['minbound']

            
        l = widget3d.add_3d_label(properties['text_pos'], "DeepLabel:{}\nICPLabel:{}".format(label_pred[idx],obj_w_lab[idx]))

        l.color = gui.Color(1,0,0)

        l.scale = 1.1
        
        l = widget3d.add_3d_label(properties['centro2'], "Altura:{}\nComprimento:{}\nLargura:{}\nCor:{}".format(properties['altura'],properties['comprimento'],properties['largura'],properties['color']))

        l.color = gui.Color(1,1,1)

        l.scale = 0.8
        
        
        scene_description[idx] = 'Object{} detected as {} by 2D detection, and as {} by 3D. Height is {}, Length is {}, Width is {}, Color is {}'.format(idx,properties['deeplabel'],obj_w_lab[idx],properties['altura'],properties['comprimento'],properties['largura'],properties['color'])
    
    
    description = ''
    
    for key,value in scene_description.items():
        description = description + value
        
    text_to_speech(description)
    
    
    proc = mp.Process(target=txt_speech)  # instantiating without any argument
    proc.start()
    
    
    #################################### Final execution of window GUI
    bbox_ = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)
    app.run()
    
    
    ############################################ SHOW FINAL SCENE IN NORMAL WINDOW
    #o3d.visualization.draw_geometries(FINAL_SCENE,
    #                                    zoom=0.3412,
    #                                    front=view['trajectory'][0]['front'],
    #                                    lookat=view['trajectory'][0]['lookat'],
    #                                    up=view['trajectory'][0]['up'], point_show_normal=False)
    
    #print('aooooooooooooowwwww potência')
    
    
    
    
    
if __name__ == "__main__":
    main()


    
    
    
    
    
    
    