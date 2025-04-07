#!/bin/bash

mount -o remount,rw /

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/userdata/deps:/userdata/sysroot_docker/usr_x5/lib:/userdata/sysroot_docker/usr_x5/lib/aarch64-linux-gnu/:/userdata/pcl_hid_lib/
source /opt/ros/humble/setup.bash
source /userdata/hobot/install/setup.bash

model_version=v2

for arg in "$@"; do
    if [ "$arg" = "-p" ]; then
        model_version=v3
        break
    fi
done

echo "hobot_steronet model_version is ${model_version}"

#echo 1200000000 >/sys/kernel/debug/clk/bpu_mclk_2x_clk/clk_rate

ros2 launch hobot_stereonet stereonet_model_web_visual_$model_version.launch.py \
need_rectify:="False" save_image_all:=True use_local_image:="True" local_image_path:=./stereonet_images/ \
image_sleep:=80 load_rectify_param:="True"
