#!/usr/bin/env python3


import glob
import json
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import cv2


def main_():

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    data_path = []
    for i in range(5):
        data_path.append('objects_pcd/objectspng/image_'+ str(i) + '.png')
    
    # image_filenames = glob.glob(data_path + '.png')
    image_filenames = data_path
    # To test the script in good time, select only 1000 of the 25000 images
    import torch

    file = []
    for i,name in enumerate(image_filenames):
        image_path = name
        save_path = "objects_pcd/objectspng/image_n_" + str(i) + ".png"
        # image = Image.open(image_path)
        img = cv2.imread(image_path)
        # Convert the image to RGB (3 channels)
        # if image.mode == 'RGBA':
        #     image = image.convert("RGB")

        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path,img)
        file.append(save_path)
        # image = Image.open(image_path)
        # image.save(save_path)
    
    
    # Now, the input images will have 3 channels as expected by the convolutional layer

    # Use a rule of 70% train, 20% validation, 10% test
    test_file = file #image_filenames
    # train_filenames, remaining_filenames = train_test_split(image_filenames, test_size=0.3)
    # validation_filenames, test_filenames = train_test_split(remaining_filenames, test_size=0.33)

    # print('We have a total of ' + str(len(image_filenames)) + ' images.')
    # print('Used ' + str(len(train_filenames)) + ' train images')
    # print('Used ' + str(len(validation_filenames)) + ' validation images')
    # print('Used ' + str(len(test_filenames)) + ' test images')

    d = {     'test_filenames': test_file}

    json_object = json.dumps(d, indent=2)

    # Writing to sample.json
    with open("callmodel/files_from_scene.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main_()
