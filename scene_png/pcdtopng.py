import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

# Load PCD file
pcd = o3d.io.read_point_cloud('your_point_cloud.pcd')

# Convert Point Cloud to Numpy array
points = np.asarray(pcd.points)

# Get the minimum and maximum coordinates to determine the image size
min_coords = np.min(points, axis=0)
max_coords = np.max(points, axis=0)

# Scale the coordinates to the range [0, 255]
scaled_points = ((points - min_coords) / (max_coords - min_coords) * 255).astype(np.uint8)

# Create an empty image with the same size as the scaled points
image = np.zeros((256, 256, 3), dtype=np.uint8)

# Place the scaled points in the image
image[scaled_points[:, 1], scaled_points[:, 0]] = [255, 255, 255]

# Display the image
plt.imshow(image)
plt.show()
