import cv2
import os




def getpngfromscene(scene_path,crops):
    scene  = ((os.path.basename(scene_path)).split("."))[0]
    obj_path = "rgbd-scenes-v2/outro/rgbd-scenes-v2/imgs/scene_" + scene + '/00650-color.png'

    image = cv2.imread(obj_path)

    alturas=[]
    comprimentos=[]
    larguras=[]
    minbounds=[]
    maxbounds=[]
    for i  in crops:
        crop=crops[i]
        
        altura = crop['altura']
        comprimento = crop['comprimento']
        largura = crop['largura']
        maxbound= crop['maxbound']
        minbound = crop['minbound']
        alturas.append(altura)
        comprimentos.append(comprimento)
        larguras.append(largura)
        minbounds.append(minbound)
        maxbounds.append(maxbound)

        
        
    

    




    cv2.imshow('img',image)
    cv2.waitKey(0)
    
    