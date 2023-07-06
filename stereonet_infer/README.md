Getting Started with Hobot Stereonet Node
=======

# 功能介绍

双目深度估计算法使用地平线开源的`stereonet`算法模型，订阅包含双目图像的话题消息，利用BPU进行算法推理，发布包含双目图像左图和感知结果的话题消息。

# 支持平台

地平线RDK J5

# 开发环境

- 编程语言: C/C++
- 开发平台: J5/X86
- 系统版本：Ubuntu 20.0.4
- 编译工具链:Linux GCC 9.3.0/Linaro GCC 9.3.0

# 参数

| 参数名      | 解释             | 类型   | 支持的配置                 | 是否必须 | 默认值             |
| ------------| -----------------| -------| --------------------------| -------- | -------------------|
| ros_img_topic_name | 发布数据的话题名 | string    | 和订阅话题名一致 | 否       | stereonet_node_output                |
| sub_hbmem_topic_name | 订阅双目图像数据的话题名 | string    | 和发布的双目图像数据话题名一致 | 否       | hbmem_stereo_img |

# 编译

支持在RDK J5 Ubuntu 20.04系统上编译和在PC上使用docker交叉编译。

## RDK J5 Ubuntu 20.04系统上编译

1、编译环境

- 板端已安装RDK J5 Ubuntu 20.04系统。

- 当前编译终端已设置tros.b环境变量：`source /opt/tros/setup.bash`。

- 已安装ROS2软件包构建系统ament_cmake。安装命令：`apt update; apt-get install python3-catkin-pkg; pip3 install empy`

- 已安装ROS2编译工具colcon。安装命令：`pip3 install -U colcon-common-extensions`

2、编译

- 编译命令：`colcon build --packages-select hobot_stereonet`

## docker交叉编译

1、编译环境

- 在docker中编译，并且docker中已经编译好tros.b。docker安装、交叉编译、tros.b编译和部署说明详见[TogetheROS.Bot用户手册](https://developer.horizon.ai/api/v1/fileData/documents_tros/quick_start/cross_compile.html#)。

2、编译

- 编译命令：

  ```shell
  bash robot_dev_config/build.sh -p J5 -s hobot_stereonet
  ```

# 使用介绍

## RDK J5 Ubuntu 20.04系统上运行

```shell
# 配置tros.b环境
source /opt/tros/setup.bash

ros2 launch hobot_stereonet hobot_stereonet_demo.launch.py 
```
