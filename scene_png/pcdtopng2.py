import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

# Load PCD file
paths =['object_pcd_000.pcd','object_pcd_001.pcd','object_pcd_002.pcd','object_pcd_003.pcd','object_pcd_003.pcd',
        'object_pcd_004.pcd','object_pcd_005.pcd','object_pcd_006.pcd','object_pcd_007.pcd','object_pcd_008.pcd']

# Load PCD file
for path in (paths):
    pcd = o3d.io.read_point_cloud(path)

    # Print basic information about the point cloud
    print("Loaded point cloud:")
    print(pcd)
    # Extract points and colors from the PCD file
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Create an empty image
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    # Scale the coordinates to the image size
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    scaled_points = ((points - min_coords) / (max_coords - min_coords) * 255).astype(int)

    # Place the colors in the image
    image[scaled_points[:, 1], scaled_points[:, 0]] = (colors * 255).astype(np.uint8)

    # Display the image
    plt.imshow(image)
    plt.show()


# Save the image as a PNG file
# plt.imsave('output_image.png', image)

