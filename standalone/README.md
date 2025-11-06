# StereoInfer

Stereo inference code. It takes stereo left-right images and camera
intrinsics as input, and outputs disparity maps, depth maps,
visualization images, point clouds, etc.

## Build

-   Dependencies: OpenCV (image processing), Eigen (matrix operations),
    DNN (X5 BPU interface), NEON (ARM instruction acceleration).
    All of these libraries are located in the `3rdparty` directory.

-   Download the compiler

    -   Download link:
        https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads

    -   This example uses `arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz `.
        Please download the corresponding version and extract it.

        ``` bash
        tar -xvf arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
        ```

-   Use cross-compilation for building. Make sure the compiler path in
    `CMakeLists.txt` matches your own installation.

``` cmake
set(CMAKE_C_COMPILER /opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
```

-   Enter the `standalone` directory and run the build script:

``` bash
cd standalone
bash run_build.sh
```

-   After compilation, an algorithm test package `StereoInfer.tar` will
    be generated in the `build` directory.\
    Copy the test package to the `userdata` directory on the X5 board
    and extract it:

``` bash
cd /userdata/
mkdir StereoInfer
tar -xvf StereoInfer.tar -C StereoInfer
```

## Run

-   After extracting the test package, go into the `StereoInfer`
    directory and run the script to create symbolic links:
``` bash
    cd /userdata/StereoInfer
    bash make_ln.sh
```

-   Finally, run the program:

``` bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/userdata/StereoInfer/3rdparty/lib_opencv4.5.4/lib/
./StereoInfer
```

After execution, you can check the console log output for the program's `fps`, `latency`, `cpu_usage`, `bpu_usage`.
This information is also recorded in `performance_xx.txt` in the current directory.  

![console_log_performance](img/console_log_performance.png)

We have two types of models currently: one includes uncertainty information, while the other does not. 
Comparing the results below, we can see that the noise can be filtered out using the uncertainty.

The following files will be generated in the `result` directory:

  |Name                   |          without uncertainty Result    |          with uncertainty Result                 |                      Description |
  |----------|------|------|------|
  |{timestamp}_depth.png  |         ![depth](img/8777028645726_depth.png)    |  None |     Depth map aligned with the left image (unit: mm) |
  |{timestamp}_disparity.pfm |      ![disparity](img/8777028645726_disparity.png) |None |  Disparity map aligned with the left image (unit:pixels) |
  |{timestamp}_visual.jpg    |      ![visual](img/8777028645726_visual.jpg) |    ![visual](img/23859795172752_visual.jpg)    | Top: left image; Bottom: depth pseudo-color image. <br> Color gradient red → yellow → green → blue indicates distance from near to far. Numbers show grid point depths |
  |{timestamp}.pcd  |               ![pcd](img/8777028645726_pcd.png)    |   ![pcd](img/23859795172752_pcd.png)     |    3D point cloud generated from the left image |
-   Disparity maps, depth maps, and visualization images: It is
    recommended to use
    [cvkit](https://github.com/roboception/cvkit/releases/tag/v2.6.10)
    to open `.pfm` and `.png` files.
-   Point cloud files: It is recommended to use
    [CloudCompare](https://www.cloudcompare.org/) to open `.pcd` files.
