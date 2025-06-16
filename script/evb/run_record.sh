#!/bin/bash

mount -o remount,rw /

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/userdata/deps:/userdata/sysroot_docker/usr_x5/lib:/userdata/sysroot_docker/usr_x5/lib/aarch64-linux-gnu/:/userdata/pcl_hid_lib/
source /opt/ros/humble/setup.bash
source /userdata/hobot/install/setup.bash

model_version=v2
save_dir=./stereonet_images

while getopts "d:p" opt; do
  case $opt in
    d)
      save_dir="$OPTARG"
      ;;
    p)
      model_version=v2.3
      ;;
  esac
done

echo "hobot_steronet model_version is ${model_version}"
echo "save_dir is ${save_dir}"
#echo 1200000000 >/sys/kernel/debug/clk/bpu_mclk_2x_clk/clk_rate

ros2 launch hobot_stereonet stereonet_model_web_visual_$model_version.launch.py \
mipi_rotation:=90.0 mipi_lpwm_enable:=True mipi_image_width:=640 mipi_image_height:=352 mipi_image_framerate:=25.0 \
resize_before_rectify:=True save_image_all:=True
