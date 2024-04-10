English| [简体中文](./README_cn.md)

# Function Introduction

The stereo depth estimation algorithm is a `StereoNet` model trained using the Horizon [OpenExplorer](https://developer.horizon.ai/api/v1/fileData/horizon_j5_open_explorer_cn_doc/hat/source/examples/stereonet.html) on the [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) dataset.

The algorithm takes stereo image data as input, consisting of left and right views. The output of the algorithm is the disparity map of the left view.

This example uses the ZED 2i stereo camera as the input source for image data, utilizes BPU for algorithm inference, publishes topic messages containing the left stereo image and perception results, and renders and displays the algorithm results on a PC browser.

# Bill of Materials

ZED 2i stereo camera

# Instructions

## Function Installation

Run the following commands in the terminal of the RDK system for quick installation:

tros foxy:
```bash
sudo apt update
sudo apt install -y tros-hobot-stereonet
sudo apt install -y tros-hobot-stereo-usb-cam
sudo apt install -y tros-hobot-stereonet-render
sudo apt install -y tros-websocket
```

tros humble:
```bash
sudo apt update
sudo apt install -y tros-humble-hobot-stereonet
sudo apt install -y tros-humble-hobot-stereo-usb-cam
sudo apt install -y tros-humble-hobot-stereonet-render
sudo apt install -y tros-humble-websocket
```

## Launch Stereo Image Publishing, Algorithm Inference, and Image Visualization

Run the following commands in the terminal of the RDK system to start:

tros foxy:
```shell
# Configure the tros.b environment
source /opt/tros/setup.bash

# Launch the launch file
ros2 launch hobot_stereonet hobot_stereonet_demo.launch.py 
```

tros humble:
```shell
# Configure the tros.b humble environment
source /opt/tros/humble/setup.bash

# Launch the launch file
ros2 launch hobot_stereonet hobot_stereonet_demo.launch.py 
```

After successful launch, open a browser on the same network computer, visit the IP address of RDK, and you will see the real-time visualization of the algorithm:

![stereonet_rdk](img/stereonet_rdk.png)

The depth estimation visualization using ZED in the same scene is as follows:

![stereonet_zed](img/stereonet_zed.png)

It can be observed that for areas with changes in lighting, the depth estimation accuracy of the deep learning method is higher.

# Interface Description

## Topic

| Name         | Message Type                            | Description                                |
| ------------ | ---------------------------------------- | ------------------------------------------ |
| /image_jpeg  | sensor_msgs/msg/Image                   | Topic for periodically publishing image    |

## Parameters

| Name                         | Parameter Value          | Description                |
| --------------------------- | ------------------------ | -------------------------- |
| sub_hbmem_topic_name        | Default hbmem_stereo_img | Topic name for subscribing to stereo image messages  |

# FAQ
