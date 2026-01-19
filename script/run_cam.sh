#!/bin/bash
source /opt/tros/humble/setup.bash
#source /opt/ros/humble/setup.bash
#source /userdata/install/setup.bash
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/userdata/deps

image_width=1280
image_height=1088
framerate=30.0
rotation=90.0
gdc_enable=False
cal_rotation=90.0
lpwm_enable=True
frame_ts_type=realtime
out_format=nv12
channel=2
channel2=0
log_level=ERROR

while [[ $# -gt 0 ]]; do
  case $1 in
    --image_width) image_width=$2; shift 2 ;;
    --image_height) image_height=$2; shift 2 ;;
    --framerate) framerate=$2; shift 2 ;;
    --rotation) rotation=$2; shift 2 ;;
    --gdc_enable) gdc_enable=$2; shift 2 ;;
    --cal_rotation) cal_rotation=$2; shift 2 ;;
    --lpwm_enable) lpwm_enable=$2; shift 2 ;;
    --frame_ts_type) frame_ts_type=$2; shift 2 ;;
    --out_format) out_format=$2; shift 2 ;;
    --channel) channel=$2; shift 2 ;;
    --channel2) channel2=$2; shift 2 ;;
    --log_level) log_level=$2; shift 2 ;;
    *) echo "unknown param: $1"; exit 1 ;;
  esac
done

ros2 run mipi_cam mipi_cam --ros-args \
-p device_mode:=dual -p dual_combine:=1 \
-p image_width:=$image_width -p image_height:=$image_height \
-p framerate:=$framerate -p rotation:=$rotation \
-p gdc_enable:=$gdc_enable -p cal_rotation:=$cal_rotation \
-p lpwm_enable:=$lpwm_enable \
-p frame_ts_type:=$frame_ts_type \
-p out_format:=$out_format \
-p channel:=$channel -p channel2:=$channel2 \
--log-level $log_level