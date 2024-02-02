import cv2
import os




def getpngfromscene(scene_path,crops):
    scene  = ((os.path.basename(scene_path)).split("."))[0]
    obj_path = "rgbd-scenes-v2/outro/rgbd-scenes-v2/imgs/scene_" + scene + '/00650-color.png'

    image = cv2.imread(obj_path)

    
    