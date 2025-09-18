# X5 双目开发交付

# StereoInfer

双目推理代码，输入双目左右图片和相机内参，输出视差图、深度图

## 编译

- 依赖opencv（图像处理）、eigen（矩阵运算）、dnn（X5 BPU接口）、neon（ARM指令加速），这些库都在3rdparty目录下

- 下载编译器
  - 下载地址：https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads
  - 本例使用的是arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz，请下载对应版本并解压
    ```bash
    tar -xvf arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
    ```

- 使用交叉编译进行编译，注意CMakeLists.txt的编译器目录设置为自己对应的目录

```cmake
set(CMAKE_C_COMPILER /root/dockershare/1_RosCode/work_humble_ws_x5/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /root/dockershare/1_RosCode/work_humble_ws_x5/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
```

- 最后执行编译命令

```bash
cd StereoInfer
bash run_build.sh
```

- 编译将生成build目录

## 执行

- 需要将build目录、3rdparty目录、make_ln.sh文件复制到X5板端，例如将这些文件复制到X5目录/userdata/

- 然后在/userdata/目录执行

```
bash make_ln.sh
```

- 最后运行程序

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/userdata/lib_opencv4.5.4/lib/
./stereo_infer
```

- 视差图、深度图打开方式：建议安装[cvkit](https://github.com/roboception/cvkit/releases/tag/v2.6.10)软件打开pfm和png格式图像


# DepthToPointCloud

深度转点云代码，输入深度图和相机内参，输出点云文件

## 编译

- 依赖opencv（图像处理），这些库都在3rdparty目录下

- 配置交叉编译环境，参考上文

- 使用交叉编译进行编译，注意CMakeLists.txt的编译器目录设置为自己对应的目录

```cmake
set(CMAKE_C_COMPILER /opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
```

- 执行编译命令

```bash
cd DepthToPointCloud
bash run_build.sh
```

- 编译将生成build目录

## 执行

- 需要将build目录、3rdparty目录、make_ln.sh文件复制到X5板端，例如将这些文件复制到X5目录/userdata/

- 然后在/userdata/目录执行

```
bash make_ln.sh
```

- 最后运行程序

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/userdata/lib_opencv4.5.4/lib/
./depth_to_pointcloud
```

- 点云文件打开方式：建议安装[CloudCompare](https://www.danielgm.net/cc/)软件打开点云文件
