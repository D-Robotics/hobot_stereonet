# StereoNet Python Bindings

Python bindings (pybind11) for the StereoNet depth estimation pipeline. It wraps the
same C++ `StereonetProcess` used by the standalone `infer` executable into a Python
extension module named `dstereonet`, so inference (disparity, depth, point cloud and
visualization) can be run from Python with results identical to the C++ `infer` program.

## Build

-   Dependencies: Python3 + development headers, pybind11, OpenCV (image processing),
    Eigen (matrix operations), DNN (BPU interface). All are resolved at build time
    from `/usr` and at runtime from the board's system library paths
    (`/usr/hobot/lib`, `/usr/lib/aarch64-linux-gnu`).

-   Build: run the build script for the target platform. The scripts compile
    natively on the board (no cross-compile) and produce a shared module
    `build/dstereonet*.so`.

``` bash
cd python_bindings
bash run_build_X5.sh     # or run_build_S100.sh / run_build_S600.sh
```

The build output is a Python extension module `dstereonet` under `build/`.
`infer.py` adds `build/` to `sys.path` and then runs `import dstereonet`.

## Run

``` bash
bash run_py.sh
```

`run_py.sh` is a thin wrapper that calls:

``` bash
python3 infer.py ../config/DStereoV2.4_int16_uncertainty.bin ../standalone/img 0.0 --out_dir ./result
```

### Usage

``` text
python3 infer.py [model_path] [local_img_dir] [uncertainty_th] [--out_dir DIR]
                 [--post_version VER] [--max_memory_count N]
```

| Argument           | Description                      | Default                       |
| ------------------ | -------------------------------- | ----------------------------- |
| model_path         | Path to the stereo model (.bin)  | ./model/DStereoV2.4_int16.bin |
| local_img_dir      | Root directory of input images   | ./img                         |
| uncertainty_th     | Uncertainty threshold for filter | -0.10                         |
| --out_dir          | Result root directory            | ./result                      |
| --post_version     | Post-processing version          | auto                          |
| --max_memory_count | Max memory count for the session | 5                             |

### Input directory format

Same as the C++ `infer` program: the root directory is scanned recursively and each
scene directory is processed. A scene directory must contain a stereo pair plus a
camera intrinsic file. Three layouts are supported:

1. Image pairs named with `left` / `right` in the same directory
   (e.g. `left_xxx.png` + `right_xxx.png`).
2. `left/` and `right/` subdirectories, matched by filename stem.
3. Vertically-stacked single images (top half = left, bottom half = right).

Each scene directory also needs a camera intrinsic file:

-   `camera_intrinsic.txt` - 5 values: `fx fy cx cy baseline` (baseline in meters), or
-   `K.txt` - 10 values: 3x3 K matrix `(fx 0 cx / 0 fy cy / 0 0 1)` followed by baseline.

### Output

Results are written under `<out_dir>/<root_name>/<relative_path>` for each scene:

| Name                    | Description                                        |
| ----------------------- | -------------------------------------------------- |
| left_xxx.png            | Resized left image (model input size)              |
| right_xxx.png           | Resized right image                                |
| disp_xxx.pfm            | Disparity map (pixels)                             |
| uncert_xxx.pfm          | Uncertainty map (if the model outputs uncertainty) |
| epipolar_visual_xxx.png | Epipolar alignment check visualization             |
| visual_disp_xxx.png     | Disparity pseudo-color image                       |
| visual_disp_sf_xxx.png  | Disparity pseudo-color image with speckle filter   |
| depth_xxx.png           | Depth map (uint16, mm), when intrinsic is valid    |
| visual_xxx.png          | Left image overlaid with depth pseudo-color        |
| pointcloud_xxx.pcd      | Colored 3D point cloud                             |
| camera_intrinsic.txt    | Resized intrinsic (fx fy cx cy baseline)           |
| K.txt                   | Resized intrinsic matrix format                    |

## Python API

The extension module `dstereonet` exposes the classes and functions below.

### StereonetProcess

| Member                                                           | Description                                                                                          |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `init(model_path, post_version="auto", max_memory_count=5)`      | Load the model and initialize the BPU session                                                        |
| `get_model_input_size()`                                         | Return `(width, height)` of the model input                                                          |
| `forward_sync(left_nv12, right_nv12, uncertainty_th=-0.10)`      | Run inference; inputs are NV12 uint8 arrays of size `w*h*3/2`; returns `(disp, uncert)` numpy arrays |
| `perspective_disparity_to_depth(disp, cam)`                      | Static; convert disparity to depth (uint16 mm)                                                       |
| `disparity_to_depth(disp, cam)`                                  | Static; dispatch on `cam.rectify_model`                                                              |
| `render_disp_or_depth(input, ...)`                               | Static; render a disparity/depth map to a pseudo-color image                                         |
| `convert_visual_img(rgb, disp, depth, cam, depth_decimal_num=2)` | Static; overlay depth on the left image                                                              |
| `depth_to_pointcloud_rgb(depth, rgb, cam, max_depth=10.0)`       | Static; return a list of `PointXYZRGB`                                                               |
| `dump_pcd_file_rgb(filename, pointcloud, format="binary")`       | Static; write a PCD file                                                                             |

### CameraIntrinsic

Fields: `fx`, `fy`, `cx`, `cy`, `baseline`, `doffs`, `rectify_model`.
Method: `is_valid()`.

### PointXYZ / PointXYZRGB

-   `PointXYZ`: fields `x`, `y`, `z`.
-   `PointXYZRGB`: fields `x`, `y`, `z`, `r`, `g`, `b`.

### check_epipolar_alignment

``` python
visual = dstereonet.check_epipolar_alignment(left_img, right_img, cam)
```

Returns the visualization image (numpy array), or `None` when no valid feature
matches were found.
