# 功能介绍

双目深度估计算法是使用地平线[OpenExplorer](https://developer.horizon.ai/api/v1/fileData/horizon_j5_open_explorer_cn_doc/hat/source/examples/stereonet.html)在[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)数据集上训练出来的`StereoNet`模型。

算法输入为双目图像数据，分别是左右视图。算法输出为左视图的视差。

此示例使用mipi双目相机作为图像数据输入源，利用BPU进行算法推理，发布包含双目图像左图和感知结果的话题消息，
在PC端rviz2上渲染算法结果。

# 物料清单

双目相机

# 使用方法

## 功能安装

在RDK系统的终端中运行如下指令，即可快速安装：

tros humble 版本
```bash
sudo apt update
sudo apt install -y tros-humble-hobot-stereonet
```

## 启动双目图像发布、算法推理和图像可视化

在RDK系统的终端中运行如下指令启动：

tros humble 版本
```shell
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 启动双目模型launch文件，其包含了算法和双目相机节点的启动
ros2 launch stereonet_model stereonet_model_web_visual.launch.py \
stereo_image_topic:=/image_combine_raw stereo_combine_mode:=1 need_rectify:="True" \
height_min:=0.1 height_max:=1.0 KMean:=10 stdv:=0.01 leaf_size:=0.05

```

另外可以通过 component 的方式启动节点
```shell 
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 终端1 启动双目模型launch文件
ros2 launch stereonet_model stereonet_model_component.launch.py \
stereo_image_topic:=/image_combine_raw stereo_combine_mode:=1 need_rectify:="True" \
height_min:=0.1 height_max:=1.0 KMean:=10 stdv:=0.01 leaf_size:=0.05

# 终端2 启动mipi双目相机launch文件
ros2 launch mipi_cam mipi_cam_dual_channel.launch.py \
mipi_image_width:=1280 mipi_image_height:=640
```

如果想利用本地图片评估算法效果，可以使用下列命令指定算法运行模式、图像数据地址以及相机内参，
同时要保证图像数据经过去畸变、基线对齐。
```shell
# 配置tros.b humble环境
source /opt/tros/humble/setup.bash

# 进入算法数据目录
cd /opt/tros/humble/share/stereonet_model/

# 启动双目模型launch文件
ros2 launch stereonet_model stereonet_model_web_visual.launch.py \
need_rectify:="False" use_local_image:="True" local_image_path:=`pwd`/data/ \
camera_fx:=505.044342 camera_fy:=505.044342 camera_cx:=605.167053 camera_cy:=378.247009 base_line:=0.069046
```
参数含义如下：

| 名称                         | 参数值   | 说明     |
| --------------------------- | ------------------------ | ------------------------------ |
| use_local_image        | 默认 False | 是否启用图片回灌模式                        |
| local_image_path        | - |  回灌图像的地址目录
| camera_fx        | - | 相机内参   
| camera_fy        | - | 相机内参
| camera_cx        | - | 相机内参  
| camera_cy        | - | 相机内参 
| base_line        | - | 基线距离  

图片的格式如下图所示，第一张左目图像的命名为left000000.png，第二张左目图像的命名为left000001.png，以此类推。
对应的第一张右目图像的命名为right000000.png，第二张右目图像的命名为right000001.png，以此类推。
算法按序号遍历图像，直至图像全部计算完毕。
![stereonet_rdk](img/image_format.png)

启动成功后，打开同一网络电脑的rviz2，订阅双目模型节点发布的相关话题，即可看到算法可视化的实时效果：
![stereonet_rdk](img/stereonet_rdk.png)

 也可以在PC上可通过浏览器观察到算法的运行结果，地址为X5的8000端口，比如X5的ip地址为192.168.31.111，
 那么在浏览器输入 192.168.31.111:8000 即可：
![stereonet_rdk](img/web_depth_visual.png)

# 接口说明

## 订阅话题

| 名称         | 消息类型                             | 说明                                     |
| ------------ | ------------------------------------ | ---------------------------------------- |
| /image_combine_raw  | sensor_msgs::msg::Image   | 双目相机节点发布的左右目拼接图像话题，用于模型推理深度             |


## 发布话题

| 名称         | 消息类型                             | 说明                                     |
| ------------ | ------------------------------------ | ---------------------------------------- |
| /StereoNetNode/stereonet_pointcloud2  | sensor_msgs::msg::PointCloud2                | 发布的点云深度话题             |
| /StereoNetNode/stereonet_depth  | sensor_msgs::msg::Image                | 发布的深度图像，像素值为深度，单位为毫米              |
| /StereoNetNode/stereonet_visual  | sensor_msgs::msg::Image                | 发布的比较直观的可视化渲染图像             |

## 参数

| 名称                         | 参数值   | 说明     |
| --------------------------- | ------------------------ | ------------------------------ |
| stereo_image_topic        | 默认 /image_combine_raw | 订阅双目图像消息的话题名                        |
| need_rectify        | 默认 True | 是否对双目数据做基线对齐和去畸变，相机内外参在config/stereo.yaml文件内指定
| stereo_combine_mode        | 默认 1 | 左右目图像往往拼接在一张图上再发布出去，1为上下拼接，0为左右拼接，指示双目算法如何拆分图像   
| height_min        | 默认 -0.2 |  过滤掉相机垂直方向上高度小于height_min的点，单位为米
| height_max        | 默认 999.9 | 过滤掉相机垂直方向上高度大于height_max的点，单位为米   
| KMean        | 默认 10 | 过滤稀疏离群点时每个点的临近点的数目，统计每个点与周围最近10个点的距离 
| stdv        | 默认 0.01 | 过滤稀疏离群点时判断是否为离群点的阈值，将标准差的倍数设置为0.01  
| leaf_size        | 默认 0.05 | 设置点云的单位密度，表示半径0.05米的三维球内只有一个点   

# 算法耗时
当log等级设置为debug时，程序会打印出算法各阶段耗时情况，供用户debug算法性能瓶颈。
```shell
ros2 launch stereonet_model stereonet_model.launch.py \
stereo_image_topic:=/image_combine_raw stereo_combine_mode:=1 need_rectify:="True" log_level:=debug
```
![stereonet_rdk](img/consume.png)
# 注意事项
1. 模型的输入尺寸为宽：1280，高640，相机发布的图像分辨率应为1280x640
2. 如果双目相机发布图像的格式为NV12，那么双目图像的拼接方式必须为上下拼接
