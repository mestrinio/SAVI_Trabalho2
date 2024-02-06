import pcl
import numpy as np

def calculate_mean_color(pcd_file_path):
    # Load PCD file
    cloud = pcl.load(pcd_file_path)

    # Extract RGB values
    rgb_values = pcl.PointCloud()
    pcl.copyPointCloud(cloud, rgb_values)
    rgb_np_array = rgb_values.to_array()

    # Calculate mean color
    mean_color = np.mean(rgb_np_array, axis=0)

    return mean_color

if __name__ == "__main__":
    # Replace 'your_pcd_file.pcd' with the actual path to your PCD file
    pcd_file_path = 'your_pcd_file.pcd'

    mean_color = calculate_mean_color(pcd_file_path)

    print("Mean Color (RGB):", mean_color)
