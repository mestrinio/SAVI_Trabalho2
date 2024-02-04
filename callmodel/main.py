#!/usr/bin/env python3


import glob
import json
from sklearn.model_selection import train_test_split



def main_():

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    data_path = 'objects_pcd/objectspng/'
    image_filenames = glob.glob(data_path + '*.png')
    # To test the script in good time, select only 1000 of the 25000 images

    # Use a rule of 70% train, 20% validation, 10% test
    test_file = image_filenames
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
