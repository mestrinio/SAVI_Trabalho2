import cv2

save_paths = ["C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/coffee_mug_0.png",
    "C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/cap_1.png",
    "C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/bowl_2.png",
    "C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/soda_can_3.png",
    "C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/bowl_4.png"]
image_paths = ["C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/print_image_0.png",
    "C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/print_image_1.png",
    "C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/print_image_2.png",
    "C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/print_image_3.png",
    "C:/Users/USER/OneDrive/Documentos/GitHub/SAVI_Trabalho2/objects_pcd/objectspng/print_image_4.png" ]
i=0
for image_path in image_paths:
    img = cv2.imread(image_path)

    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    cv2.imwrite(save_paths[i],img)
    i=i+1