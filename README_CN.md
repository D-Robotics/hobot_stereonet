# hobot_stereonet

[English](./README.md) | 简体中文

## 功能介绍

hobot_sterenet是地瓜机器人自研的基于深度学习的双目深度算法，算法输入彩色双目图像，输出左视图的深度图，可进一步根据相机内参转为点云。算法兼顾精度和效率，具有较高的使用价值。

算法可搭配多款MIPI和USB双目相机使用，例如230ai、132gs、zed双目相机。

## 准备工作

- RDK X5
- 双目相机

## 支持平台

| 平台   | 运行方式              | 示例功能                                    |
| ------ | --------------------- | ------------------------------------------- |
| RDK X5 | Ubuntu 22.04 (Humble) | 启动双目相机、推理出深度结果，并在Web端显示 |

## 构建hobot_stereonet

### 从TROS.b安装

使用RDK X5的用户可以按照如下手册安装并体验hobot_stereonet的功能：[双目深度算法](https://developer.d-robotics.cc/rdk_doc/Robot_development/boxs/function/hobot_stereonet)

### 从源码构建

- 建议使用交叉编译环境对源码进行编译，交叉编译环境的搭建指南：[5.1.3 源码安装](https://developer.d-robotics.cc/rdk_doc/Robot_development/quick_start/cross_compile)

- 搭建好环境后，执行如下指令编译源码：

```bash
git clone https://github.com/D-Robotics/hobot_stereonet.git
bash ./robot_dev_config/build.sh -p X5 -s hobot_stereonet
```

## 双目模型的版本

目前双目算法已有如下版本可供使用：

| 算法版本 | 算法特性                                           | 对应模型                       |
| -------- | -------------------------------------------------- | ------------------------------ |
| V2.0     | 精度较高、帧率较低，输出15FPS分辨率640*352的深度图 | x5baseplus_alldata_woIsaac.bin |
| V2.1     | 加入置信度，用于过滤视差                           | DStereoV2.1.bin                |
| V2.2     | 精度较低、帧率较高，输出23FPS分辨率640*352的深度图 | DStereoV2.2.bin                |
| V2.3     | 帧率进一步提升，输出27FPS分辨率640*352的深度图     | V22_disp96.bin                 |


## 运行启动文件

#### (1) 搭配RDK X5官方MIPI双目相机启动

- RDK X5官方MIPI双目相机如图所示：

![RDK_Stereo_Cam_230ai](img/RDK_Stereo_Cam_230ai.png)

- 安装方式如图所示，接线请勿接反，会导致左右图对调，双目算法运行错误：

![RDK_X5_230ai](img/RDK_X5_230ai.png)

- 确认相机连接是否正常，通过ssh连接RDK X5，执行以下命令，如果输出如图所示结果，则代表相机连接正常：

```bash
i2cdetect -r -y 4
i2cdetect -r -y 6
```

![i2cdetect_230ai](img/i2cdetect_230ai.png)

- 通过不同的launch文件，启动相应版本的双目算法，通过ssh连接RDK X5，执行以下命令：

- V2.0

```bash
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，其包含了算法和双目相机节点的启动
ros2 launch hobot_stereonet stereonet_model_web_visual_v2.0.launch.py \
mipi_image_width:=640 mipi_image_height:=352 mipi_lpwm_enable:=True mipi_image_framerate:=15.0 \
need_rectify:=False height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0
```

- V2.1

```bash
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，其包含了算法和双目相机节点的启动
ros2 launch hobot_stereonet stereonet_model_web_visual_v2.1.launch.py \
mipi_image_width:=640 mipi_image_height:=352 mipi_lpwm_enable:=True mipi_image_framerate:=25.0 \
need_rectify:=False height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0 uncertainty_th:=0.09
```

- V2.2

```bash
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，其包含了算法和双目相机节点的启动
ros2 launch hobot_stereonet stereonet_model_web_visual_v2.2.launch.py \
mipi_image_width:=640 mipi_image_height:=352 mipi_lpwm_enable:=True mipi_image_framerate:=25.0 \
need_rectify:=False height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0
```

- V2.3

```bash
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，其包含了算法和双目相机节点的启动
ros2 launch hobot_stereonet stereonet_model_web_visual_v2.3.launch.py \
mipi_image_width:=640 mipi_image_height:=352 mipi_lpwm_enable:=True mipi_image_framerate:=30.0 \
need_rectify:=False height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0
```

参数含义如下：

| 名称                 | 参数值      | 说明                                                                       |
| -------------------- | ----------- | -------------------------------------------------------------------------- |
| mipi_image_width     | 设置为640   | MIPI相机的输出分辨率是640*352                                              |
| mipi_image_height    | 设置为352   | MIPI相机的输出分辨率是640*352                                              |
| mipi_lpwm_enable     | 设置为True  | MIPI相机开启硬件同步                                                       |
| mipi_image_framerate | 设置为30.0  | MIPI相机输出帧率为30.0FPS                                                  |
| need_rectify         | 设置为False | 因为官方相机出厂自带标定参数，会自动矫正，不需要加载自定义标定文件进行矫正 |
| height_min           | 设置为-10.0 | 点云最小高度为-10.0m                                                       |
| height_max           | 设置为10.0  | 点云最大高度为10.0m                                                        |
| pc_max_depth         | 设置为5.0   | 是点云最大距离为5.0m                                                       |
| uncertainty_th       | 设置为0.09  | 置信度阈值，仅V2.1版本模型可用                                             |

- 出现如下日志表示双目算法启动成功，`fx/fy/cx/cy/base_line`是相机内参，如果深度图正常，但估计出来的距离有偏差，可能是相机内参存在问题：

![stereonet_run_success_log](img/stereonet_run_success_log.png)

- 通过网页端查看深度图，在浏览器输入 http://ip:8000 (图中RDK X5 ip是192.168.1.100)：

![web_depth_visual](img/web_depth_visual.png)

- 通过rviz2查看点云，需要用户具备一定的ROS2基础，将PC和RDK X5配置到同一个网段，能够相互ping通，订阅双目模型节点发布的相关话题，才可以在rviz2中显示点云，注意rviz2中需要做如下配置：

![stereonet_rviz](img/stereonet_rviz.png)

- 如果用户想保存深度估计结果，可以添加如下参数实现，`save_image_all`打开保存开关，`save_freq`控制保存频率，`save_dir`控制保存的目录（如果目录不存在会自动创建），`save_total`控制保存的总数。程序运行将会保存**相机内参、左右图、视差图、深度图、可视化图**：

```bash
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 这里以V2.0版本的算法为例，其它版本的算法类似加入对应参数即可
ros2 launch hobot_stereonet stereonet_model_web_visual_v2.0.launch.py \
mipi_image_width:=640 mipi_image_height:=352 mipi_lpwm_enable:=True mipi_image_framerate:=15.0 \
need_rectify:=False height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0 \
save_image_all:=True save_freq:=4 save_dir:=./online_result save_total:=10
```

参数含义如下：

| 名称           | 参数值               | 说明                                       |
| -------------- | -------------------- | ------------------------------------------ |
| save_image_all | 设置为True           | 保存图像开关                               |
| save_freq      | 设置为4              | 每隔4帧保存一次，可修改为任意正数          |
| save_dir       | 设置为保存图像的目录 | 可根据需要设置保存位置                     |
| save_total     | 设置为10             | 总共保存10张图像，设置为-1则代表为一直保存 |

![stereonet_save_log](img/stereonet_save_log.png)

![stereonet_save_files](img/stereonet_save_files.png)

#### (2) 本地图片离线回灌

- 如果想利用本地图片评估算法效果，可以使用下列命令指定算法运行模式、图像数据地址以及相机内参，同时要保证图像数据经过去畸变、极线对齐。图片的格式如下图所示，第一张左目图像的命名为left000000.png，第二张左目图像的命名为left000001.png，以此类推。对应的第一张右目图像的命名为right000000.png，第二张右目图像的命名为right000001.png，以此类推。算法按序号遍历图像，直至图像全部计算完毕：

![stereonet_rdk](img/image_format.png)

- 算法离线运行方式如下，通过ssh连接RDK X5，执行以下命令：

- V2.0

```shell
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，注意相机参数的设置，需要手动输入矫正后参数
ros2 launch hobot_stereonet stereonet_model_web_visual.launch.py \
stereonet_model_file_path:=/opt/tros/humble/share/hobot_stereonet/config/x5baseplus_alldata_woIsaac.bin postprocess:=v2 \
use_local_image:=True local_image_path:=./online_result \
need_rectify:=False camera_fx:=216.696533 camera_fy:=216.696533 camera_cx:=335.313477 camera_cy:=182.961578 base_line:=0.070943 \
height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0 \
save_image_all:=True save_dir:=./offline_result image_sleep:=500
```

- V2.1

```shell
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，注意相机参数的设置，需要手动输入矫正后参数
ros2 launch hobot_stereonet stereonet_model_web_visual.launch.py \
stereonet_model_file_path:=/opt/tros/humble/share/hobot_stereonet/config/DStereoV2.1.bin postprocess:=v2.1 \
use_local_image:=True local_image_path:=./online_result \
need_rectify:=False camera_fx:=216.696533 camera_fy:=216.696533 camera_cx:=335.313477 camera_cy:=182.961578 base_line:=0.070943 \
height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0 uncertainty_th:=0.09 \
save_image_all:=True save_dir:=./offline_result image_sleep:=500
```

- V2.2

```shell
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，注意相机参数的设置，需要手动输入矫正后参数
ros2 launch hobot_stereonet stereonet_model_web_visual.launch.py \
stereonet_model_file_path:=/opt/tros/humble/share/hobot_stereonet/config/DStereoV2.2.bin postprocess:=v2.2 \
use_local_image:=True local_image_path:=./online_result \
need_rectify:=False camera_fx:=216.696533 camera_fy:=216.696533 camera_cx:=335.313477 camera_cy:=182.961578 base_line:=0.070943 \
height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0 \
save_image_all:=True save_dir:=./offline_result image_sleep:=500
```

- V2.3

```shell
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，注意相机参数的设置，需要手动输入矫正后参数
ros2 launch hobot_stereonet stereonet_model_web_visual.launch.py \
stereonet_model_file_path:=/opt/tros/humble/share/hobot_stereonet/config/V22_disp96.bin postprocess:=v2.3 \
use_local_image:=True local_image_path:=./online_result \
need_rectify:=False camera_fx:=216.696533 camera_fy:=216.696533 camera_cx:=335.313477 camera_cy:=182.961578 base_line:=0.070943 \
height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0 \
save_image_all:=True save_dir:=./offline_result image_sleep:=500
```

**注意：回灌的图像需要经过极线矫正，并且一定要设置正确的相机参数，否则回灌保存的结果可能是错误的**

![stereonet_offline_log](img/stereonet_offline_log.png)

参数含义如下：

| 名称                      | 参数值                                 | 说明                                                               |
| ------------------------- | -------------------------------------- | ------------------------------------------------------------------ |
| stereonet_model_file_path | 不同版本的双目算法模型文件             | 根据模型版本设置                                                   |
| postprocess               | 不同版本的双目算法模型对应的后处理方法 | 根据模型版本设置                                                   |
| use_local_image           | 设置为True                             | 图片回灌模式开关                                                   |
| local_image_path          | 设置为离线数据目录                     | 回灌图像的地址目录                                                 |
| need_rectify              | 设置为False                            | 回灌图像要求经过极线矫正，不需要开启此开关，但要手动传入矫正后参数 |
| camera_fx                 | 设置为相机矫正后内参fx                 | 相机内参                                                           |
| camera_fy                 | 设置为相机矫正后内参fy                 | 相机内参                                                           |
| camera_cx                 | 设置为相机矫正后内参cx                 | 相机内参                                                           |
| camera_cy                 | 设置为相机矫正后内参cy                 | 相机内参                                                           |
| base_line                 | 设置为相机矫正后基线                   | 基线距离，单位为m                                                  |
| height_min                | 设置为-10.0                            | 点云最小高度为-10.0m                                               |
| height_max                | 设置为10.0                             | 点云最大高度为10.0m                                                |
| pc_max_depth              | 设置为5.0                              | 是点云最大距离为5.0m                                               |
| save_image_all            | 设置为True                             | 保存回灌结果                                                       |
| save_dir                  | 设置为保存图像的目录                   | 可根据需要设置保存位置                                             |
| uncertainty_th            | 设置为0.09                             | 置信度阈值，仅V2.1版本模型可用                                     |

- 算法运行成功后，同样可以通过网页端和rviz显示实时渲染数据，参考上文，离线运行的结果将会保存在`离线数据目录下的result子目录中`，同样会保存**相机内参、左右图、视差图、深度图、可视化图**

#### (3) 搭配ZED双目摄像头启动

- ZED双目摄像头如图所示：

![zed_cam](img/zed_cam.png)

- 将ZED相机通过USB连接RDK X5，然后启动双目算法，通过ssh连接RDK X5，执行以下命令：

- **注意：运行ZED相机RDK X5一定要联网，因为ZED需要联网下载标定文件，可以ping一下任意网站确认板子是否联网**

```shell
ping www.baidu.com
```

- V2.0

```shell
ros2 launch hobot_zed_cam test_stereo_zed_rectify.launch.py \
resolution:=720p dst_width:=640 dst_height:=352 \
stereonet_model_file_path:=/opt/tros/humble/share/hobot_stereonet/config/x5baseplus_alldata_woIsaac.bin postprocess:=v2 \
height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0
```

- V2.1

```shell
ros2 launch hobot_zed_cam test_stereo_zed_rectify.launch.py \
resolution:=720p dst_width:=640 dst_height:=352 \
stereonet_model_file_path:=/opt/tros/humble/share/hobot_stereonet/config/DStereoV2.1.bin postprocess:=v2.1 \
height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0 uncertainty_th:=0.09
```

- V2.2

```shell
ros2 launch hobot_zed_cam test_stereo_zed_rectify.launch.py \
resolution:=720p dst_width:=640 dst_height:=352 \
stereonet_model_file_path:=/opt/tros/humble/share/hobot_stereonet/config/DStereoV2.2.bin postprocess:=v2.2 \
height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0
```

- V2.3

```shell
ros2 launch hobot_zed_cam test_stereo_zed_rectify.launch.py \
resolution:=720p dst_width:=640 dst_height:=352 \
stereonet_model_file_path:=/opt/tros/humble/share/hobot_stereonet/config/V22_disp96.bin postprocess:=v2.3 \
height_min:=-10.0 height_max:=10.0 pc_max_depth:=5.0
```

![stereonet_zed_run_success_log](img/stereonet_zed_run_success_log.png)

联网的情况下程序会自动下载标定文件，如果RDK X5没有联网，可以手动下载标定文件然后上传到RDK X5的`/root/zed/settings/`目录下

- 通过网页端查看深度图，在浏览器输入 http://ip:8000 ，更多**点云可视化**和**保存图像**相关的内容请参考上文设置对应参数

## hobot_stereonet功能包说明

### 订阅话题

| 名称               | 消息类型                     | 说明                                                   |
| ------------------ | ---------------------------- | ------------------------------------------------------ |
| /image_combine_raw | sensor_msgs::msg::Image      | 双目相机节点发布的左右目拼接图像话题，用于模型推理深度 |
| /camera_info_topic | sensor_msgs::msg::CameraInfo | 双目相机节点发布的左右目拼接图像话题，用于模型推理深度 |

### 发布话题

| 名称                                 | 消息类型                      | 说明                                     |
| ------------------------------------ | ----------------------------- | ---------------------------------------- |
| /StereoNetNode/stereonet_depth       | sensor_msgs::msg::Image       | 发布的深度图像，像素值为深度，单位为毫米 |
| /StereoNetNode/stereonet_visual      | sensor_msgs::msg::Image       | 发布的比较直观的可视化渲染图像           |
| /StereoNetNode/stereonet_pointcloud2 | sensor_msgs::msg::PointCloud2 | 发布的点云深度话题                       |

### 其它重要参数

| 名称                   | 参数值                            | 说明                                                                                       |
| ---------------------- | --------------------------------- | ------------------------------------------------------------------------------------------ |
| stereo_image_topic     | 默认 /image_combine_raw           | 订阅双目图像消息的话题名                                                                   |
| camera_info_topic      | 默认 /image_right_raw/camera_info | 订阅相机矫正参数消息的话题名                                                               |
| need_rectify           | 默认 True                         | 是否指定自定义标定文件对图像进行矫正开关                                                   |
| stereo_calib_file_path | 默认 stereo.yaml                  | need_rectify=True的情况下，加载该路径下的标定文件进行标定                                  |
| stereo_combine_mode    | 默认 1                            | 左右目图像往往拼接在一张图上再发布出去，1为上下拼接，0为左右拼接，指示双目算法如何拆分图像 |
| KMean                  | 默认 10                           | 过滤稀疏离群点时每个点的临近点的数目，统计每个点与周围最近10个点的距离                     |
| stdv                   | 默认 0.01                         | 过滤稀疏离群点时判断是否为离群点的阈值，将标准差的倍数设置为0.01                           |
| leaf_size              | 默认 0.05                         | 设置点云的单位密度，表示半径0.05米的三维球内只有一个点                                     |

### 注意事项

1. 模型的输入尺寸为宽：640，高352，相机发布的图像分辨率应为640x352
2. 如果双目相机发布图像的格式为NV12，那么双目图像的拼接方式必须为上下拼接

## 离线工具

### 检查极线对齐精度

极线不对齐会极大影响算法精度。源码目录下的tools文件夹内提供了极线对齐检测工具，可以在PC上检查相机图像的极线对齐精度。工具使用特征提取和匹配算法检查极线是否完全对齐，理想情况下，特征点的y坐标应该完全一致。选取10个最佳的匹配点，检查匹配点的y坐标。


PC上需要预装ROS2以及OpenCV (version: 4)

```shell
# 在PC上进入tools目录，创建build目录并编译
cd tools
mkdir build
cd build
cmake ..
make -j2
# 运行工具，image_path是需要检查的图像的地址
./stereonet_matcher --ros-args -p image_path:=`pwd`/../images/
```

工具运行后会显示出对齐的结果，选取10个特征点对比其在左右目的y坐标，如下图。

![stereonet_rdk](img/matcher.png)

绿色表示y坐标对齐，红色表示没有对齐。匹配的结果同时会保存在本地的"result.jpg"。工具敲入“回车”键读取下一张图，敲入“q”键退出工具。控制台打印的log是具体的匹配结果。

### 深度图/视差图离线渲染工具

- 代码文件路径

```
hobot_stereonet/script/render.py
```

- 运行

1. 安装python依赖包

```shell
pip install -r requirements.txt
```

2. 文件组织格式

![folder_contents.png](img/folder_contents.png)

程序会读取`depth`、`disp`、`left`开头的文件，对应`深度图`、`视差图`、`左图`，其中`深度图`支持`png`格式的`16bit`
图像，`视差图`支持`pfm/tiff`格式的`float32`图像，请固定文件的开头，并保证`深度图`、`视差图`、`左图`的后缀是一致的。

3. 批量渲染

```shell
# 通过img_dir指定文件夹路径，img_type指定渲染深度图还是视差图，save_dir指定保存路径
python render.py --img_dir=./depth_img_dir --img_type=depth --save_dir=./render_depth
```

4. 单张图渲染

```shell
# 通过img_path指定但张图路径，img_type指定渲染深度图还是视差图，save_dir指定保存路径
python render.py --img_path=./depth000001.png --img_type=depth --save_dir=./render_depth
```

5. 参数说明

| 名称                | 默认值                              | 类型   | 说明                                                                              |
| ------------------- | ----------------------------------- | ------ | --------------------------------------------------------------------------------- |
| img_dir             | ''                                  | string | 文件夹目录，按渲染需求包含需要的左图、深度图、视差图                              |
| img_path            | ''                                  | string | 渲染图像的路径，指定单张深度图/视差图的路径                                       |
| img_type            | depth                               | string | 渲染图像的类型，可设置为[depth, disp]，一定要指定渲染的是深度图还是视差图         |
| min_disp            | 2.0                                 | float  | 视差小于该值，不进行渲染，颜色为黑色                                              |
| max_disp            | 192.0                               | float  | 视差大于该值，不进行渲染，颜色为黑色                                              |
| min_depth           | 0.0                                 | float  | 深度小于该值，不进行渲染，颜色为黑色                                              |
| max_depth           | 10000.0 (默认单位是mm，这里表示10m) | float  | 深度大于该值，不进行渲染，颜色为黑色                                              |
| save_dir            | ''                                  | string | 保存结果的目录，会自动创建                                                        |
| save_gif            | False                               | bool   | 是否保存gif图像，设置会True会在结果目录保存gif动图                                |
| need_left_img       | False                               | bool   | 是否在保存结果时附带左图，需要对应的文件夹下有左图，左图会拼接在深度图/视差图上方 |
| need_speckle_filter | True                                | bool   | 是否需要speckle filter，会去除深度图/视差图的散点                                 |

### 结果展示

| 深度渲染结果                                          | 视差渲染结果                                        |
| ----------------------------------------------------- | --------------------------------------------------- |
| ![render_depth000181.png](img/render_depth000181.png) | ![render_disp000181.png](img/render_disp000181.png) |

