import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import os

def capture_scene_from_camera():
    """Captures a scene using an RGBD camera and saves it as a PCD file.

    Returns:
        o3d.geometry.PointCloud: The captured point cloud.
    """

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # ... (add your camera configuration settings here)
    config.enable_option(rs.option.emitter_enabled, True)  # Enable IR emitter for better depth in low light
    config.enable_option(rs.option.exposure, 50)  # Adjust exposure for better image quality
    config.enable_option(rs.option.white_balance, 0)  # Set white balance to auto

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # Capture a frame
    frames = pipeline.wait_for_frames()
    # ... (access and process depth and color frames as needed)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=1.0 / 1000.0, depth_trunc=3.0
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )

    # Save the point cloud to a PCD file
    scene_folder = "C:/Users/erfer/OneDrive/Documents/GitHub/SAVI_Trabalho2/rgbd-scenes-v2/scenes"  # Replace with your desired folder name
    os.makedirs(scene_folder, exist_ok=True)
    filename = os.path.join(scene_folder, "captured_scene.pcd")
    o3d.io.write_point_cloud(filename, pcd)

    return pcd

# if __name__ == "__main__":
#     pcd = capture_scene_from_camera()