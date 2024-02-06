import numpy as np
import open3d as o3d

def get_average_color_name(pcd):

    colors = np.asarray(pcd.colors)
    average_color = np.average(colors, axis=0)

    # Scale to 0-255 range for color naming libraries
    average_color = (average_color * 255).astype(int)
    
    # Use external library (e.g., colorsys) for color name
    try:
      import webcolors
      color_name = webcolors.rgb_to_name(average_color)
      
    except (ValueError):
      # Fallback to generic naming based on dominant channel
        color_name = 'Unknown'
        min_diff = 999999

        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r, g, b = webcolors.hex_to_rgb(key)
            rgb_diff = abs(r - average_color[0]) + abs(g - average_color[1]) + abs(b - average_color[2])

            if rgb_diff < min_diff:
                min_diff = rgb_diff
                color_name = name

    return color_name.capitalize()

