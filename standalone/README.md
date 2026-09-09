# StereoInfer

Stereo inference code. It takes stereo left-right images and camera intrinsics as input,
and outputs disparity maps, depth maps, visualization images, point clouds, etc.

## Build

-   Dependencies: OpenCV (image processing), Eigen (matrix operations),
    DNN (BPU interface), NEON (ARM instruction acceleration).
    These libraries are resolved at build time from a sysroot (cross-compile)
    or from `/usr` (native board build), and at runtime from the board's
    system library paths (`/usr/hobot/lib`, `/usr/lib/aarch64-linux-gnu`).
    For the Arm GNU Toolchain (buildroot) build only, the full dependency
    closure (headers + runtime libraries) is bundled into `3rdparty/` and
    committed, so that build compiles and runs independently of the sysroot
    (see below).

-   Build: run the build script for the target platform. The script
    auto-detects the host architecture:

    -   On an x86_64 host it cross-compiles with `aarch64-linux-gnu-gcc/g++`
        against `sysroot_docker/usr_x5` (or `usr_s100`).
    -   On an `aarch64` board it compiles natively against `/usr`.

    S600 only supports native (on-board) compilation.

    `run_build_X5.sh` additionally supports the Arm GNU Toolchain for
    buildroot users, selected explicitly via a command-line argument
    (X5 only): pass `arm` for an interactive prompt (default bin dir
    `/opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin`),
    or pass the bin dir path directly. Auto-detection never selects the Arm
    GNU Toolchain on its own - it always prefers the native compiler or
    `aarch64-linux-gnu-*`.

``` bash
cd standalone
bash run_build_X5.sh     # or run_build_S100.sh / run_build_S600.sh

# X5 with the Arm GNU Toolchain (buildroot):
bash run_build_X5.sh arm                            # interactive prompt for the bin dir
bash run_build_X5.sh /path/to/arm-gnu-toolchain/bin # use the given bin dir directly
```

-   **Self-contained Arm GNU build**: for the Arm GNU Toolchain build only, the
    script populates `standalone/3rdparty/` from the sysroot once, and it is
    committed so the build compiles and runs without the sysroot. It is sorted
    into subdirectories by origin:

    | Directory   | Contents                                                        |
    | ----------- | --------------------------------------------------------------- |
    | `include/`  | Compile-time headers: `eigen3/` (Eigen), `opencv4/` (OpenCV 4.x), `dnn/` (BPU interface) |
    | `hobot/`    | D-Robotics BPU libraries (`libdnn`, `libcnn_intf`, `libhbmem`, `libhbrt_bayes_aarch64`, `libalog`), plus unversioned `.so` link symlinks |
    | `opencv/`   | OpenCV modules (`core`, `imgproc`, `imgcodecs`, `features2d`, `flann`), plus unversioned `.so` link symlinks |
    | `gdal/`     | `libgdal.so.30` (pulled in by OpenCV imgcodecs)                 |
    | `deps/`     | Everything else: gdal's ~119 transitive deps, plus a matching `libstdc++`/`libgcc_s` (which provides `GLIBCXX_3.4.30` that the older Arm GNU toolchain's libstdc++ lacks) |

    Those directories serve both compile (`-I`/`-L`/`-rpath-link`) and run: the
    executables carry a transitive `$ORIGIN/3rdparty/{hobot,opencv,gdal,deps}`
    rpath, so they find the bundled libraries on a buildroot board without any
    `ldconfig` or `LD_LIBRARY_PATH` setup. glibc's own libraries (`libc`, `libm`,
    `ld-linux`, ...) are intentionally not bundled and come from the board's
    system glibc. When the sysroot is present the script re-populates `3rdparty/`
    from it; otherwise the committed `3rdparty/` is used as-is.

- After compilation, a test package `StereoInfer_X5.tar.gz` will be generated in the `build` directory.
The package includes two executables:

| Program   | Description                       |
| --------- | --------------------------------- |
| test_perf | Used for performance benchmarking |
| infer     | Used for offline inference        |

- Copy the test package to the userdata directory on the target board and extract it:

``` bash
cd /userdata/
tar -zxvf StereoInfer_X5.tar.gz
```

## Run

-   After extracting the test package, go into the `StereoInfer` directory.
    Shared libraries are resolved via the board's `ldconfig`
    (`/usr/hobot/lib` and `/usr/lib/aarch64-linux-gnu` are already covered),
    so no `LD_LIBRARY_PATH` or symlink setup is needed. For the buildroot
    package, the bundled libraries are resolved from the adjacent `3rdparty/`
    directory via the embedded rpath instead, which is also automatic.

### 1. Run performance test

``` bash
cd /userdata/StereoInfer
./test_perf ./model/DStereoV2.4_int16_uncertainty.bin 1 30 0.10
```

Parameter Explanation:

![help](docs/help.png)

After execution, you can check the console log output for the program's `fps`, `latency`, `cpu_usage`, `bpu_usage`.
This information is also recorded in `performance_xx.txt` in the current directory.

![console_log_performance](docs/console_log_performance.png)

 | Name      | Description                                                                                                                                                  |
 | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
 | fps       | Frame processed per second                                                                                                                                   |
 | latency   | Including process of 'model inference', 'disparity porcess and convert disparity to depth and point cloud', 'uncertainty filter'                             |
 | cpu_usage | One-core CPU usage for the whole program, including model processing and result saving                                                                       |
 | bpu_usage | One-core BPU usage for the whole system. If other models, such as segmentation or detection, are running at the same time, their usage will also be included |


We have two types of models currently: one includes uncertainty information, while the other does not.
Comparing the results below, we can see that the noise can be filtered out using the uncertainty.

The following files will be generated in the `result` directory:

  | Name                       | without uncertainty Result                                     | with uncertainty Result                                                                                         | Description                                                                                                                                                            |
  | -------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | depth_{timestamp}.png      | ![depth](docs/depth_1765459980707_visual.png)                  | ![depth](docs/depth_1765459116550_visual.png)                                                                   | Depth map aligned with the left image (unit: mm)                                                                                                                       |
  | disparity_{timestamp}.pfm  | ![disparity](docs/disp_1765459980707_visual.png)               | ![disparity](docs/disp_1765459116550_visual.png)                                                                | Disparity map aligned with the left image (unit:pixels)                                                                                                                |
  | visual_{timestamp}.png     | ![visual](docs/visual_1765459980707.jpg)   no black empty hole | ![visual](docs/visual_1765459116550.jpg)   the black empty hole is the bad or edge area filtered by uncertainty | Top: left image; Bottom: depth pseudo-color image. <br> Color gradient red → yellow → green → blue indicates distance from near to far. Numbers show grid point depths |
  | pointcloud_{timestamp}.pcd | ![pcd](docs/pointcloud_1765459980707_visual.png)               | ![pcd](docs/pointcloud_1765459116550_visual.png)                                                                | 3D point cloud generated from the left image                                                                                                                           |

### 2. Run inference

``` bash
./infer ./model/DStereoV2.4_int16_uncertainty.bin ./img 0.10
``` 

#### Input directory format

The `infer` program supports **multi-subdirectory batch inference**.

```text
img/
 ├── scene1/
 │    ├── left_xxx.png
 │    ├── right_xxx.png
 │    ├── camera_intrinsic.txt  (or K.txt)
 │
 ├── scene2/
 │    ├── left_xxx.png
 │    ├── right_xxx.png
 │    ├── camera_intrinsic.txt  (or K.txt)
 │
 └── scene3/
      ├── ...
```

Notes:

* Each subdirectory represents one scene
* Must contain:

  * stereo image pairs
  * camera intrinsic file:

    * `camera_intrinsic.txt` **or**
    * `K.txt`
* The program will automatically search and match stereo pairs


#### Output directory

Results are saved in:

```text
result/
 ├── scene1/
 ├── scene2/
 └── scene3/
```

Each subdirectory corresponds to one input scene.

Output files

The following files will be generated in each result subdirectory:

| Name                 | Description                                           |
| -------------------- | ----------------------------------------------------- |
| depth_xxx.png        | Depth map (unit: mm), aligned with left image         |
| disp_xxx.pfm         | Disparity map (unit: pixels)                          |
| uncert_xxx.pfm       | Uncertainty map (if model supports uncertainty)       |
| visual_xxx.jpg       | Visualization image (depth pseudo-color + left image) |
| pointcloud_xxx.pcd   | Colored 3D point cloud                                |
| camera_intrinsic.txt | Resized camera intrinsic (fx fy cx cy baseline)       |
| K.txt                | Resized intrinsic matrix format                       |

## Visualization Tools

-   Disparity maps, depth maps, and visualization images: It is
    recommended to use
    [cvkit](https://github.com/roboception/cvkit/releases/tag/v2.6.10)
    to open `.pfm` and `.png` files.
-   Point cloud files: It is recommended to use
    [CloudCompare](https://www.cloudcompare.org/) to open `.pcd` files.

