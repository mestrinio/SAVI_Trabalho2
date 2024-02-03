import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import cv2
import re


pattern = '([0-9]+)(?=.p)'
# Load PCD file
paths_ =['objects_pcd/objects_to_png/object_pcd_000.pcd','objects_pcd/objects_to_png/object_pcd_001.pcd','objects_pcd/objects_to_png/object_pcd_002.pcd',
        'objects_pcd/objects_to_png/object_pcd_004.pcd','objects_pcd/objects_to_png/object_pcd_005.pcd']
for path_ in paths_:
        match = re.search(pattern, path_)
        number = match.group(1)

for i,path in enumerate(paths_):
    pcd = o3d.io.read_point_cloud(path)

    print("Loaded point cloud:")
    print(pcd)
    # Extract points and colors from the PCD file
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Create an empty image
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # Scale the coordinates to the image size
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    scaled_points = ((points - min_coords) / (max_coords - min_coords) * 127).astype(int)

    # Place the colors in the image
    image[scaled_points[:, 1], scaled_points[:, 0]] = (colors * 255).astype(np.uint8)
    image =np.rot90(image,2)

    # Display the image
    plt.imshow(image)
    plt.show()
    cv2.imshow('ola',image)
    cv2.waitKey()
    cv2.imwrite('objects_pcd/objectspng/'+ str(i) + '.png', image)
# Save the image as a PNG file
    # plt.imsave('objects_pcd/objectspng/output_image'+ str(i) + '_.png', image)

