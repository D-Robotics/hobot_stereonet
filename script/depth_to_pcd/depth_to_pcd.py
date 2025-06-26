import cv2
import numpy as np
import argparse
import os

def depth_to_pcd(depth_file, calib_params, output_pcd=None):
    """
    Convert depth image to PCD (Point Cloud Data) file
    :param depth_file: Path to 16-bit depth image (PNG format)
    :param calib_params: Camera calibration parameters [fx, fy, cx, cy, baseline]
    :param output_pcd: Output PCD file path (optional)
    """
    # Read depth image (16-bit unsigned)
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Cannot read depth image: {depth_file}")

    # Parse calibration parameters
    fx, fy, cx, cy, baseline = calib_params

    # Generate point cloud
    height, width = depth.shape
    i, j = np.indices((height, width))

    z = depth.flatten() / 1000.0
    x = ((j.flatten() - cx) * z) / fx
    y = ((i.flatten() - cy) * z) / fy
    valid = (z > 0) & np.isfinite(z)
    points = np.stack((x, y, z), axis=1)[valid]

    # Set default output path if not specified
    if output_pcd is None:
        output_pcd = os.path.splitext(depth_file)[0] + ".pcd"

    # Write PCD file (ASCII format)
    with open(output_pcd, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        # Write point coordinates
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} 1.0\n")

    print(f"PCD file saved to: {output_pcd}")

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Convert depth image to PCD file")
    parser.add_argument("depth_file", help="Input depth image (16-bit PNG)")
    parser.add_argument("--calib", required=True,
                        help="Calibration file (format: [fx, fy, cx, cy, baseline] = [...])")
    parser.add_argument("--output", help="Output PCD file path (optional)")
    args = parser.parse_args()

    # Read calibration parameters from file
    with open(args.calib, 'r') as f:
        calib_line = f.readline().strip()
    # Extract parameters from line like: [fx, fy, cx, cy, baseline] = [527.193, 527.193, 659.71, 360.584, 119.893]
    calib_params = list(map(float, calib_line.split('=')[1].strip(' []').split(',')))

    # Perform conversion
    depth_to_pcd(args.depth_file, calib_params, args.output)
