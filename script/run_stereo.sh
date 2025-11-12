#!/bin/bash
source /opt/tros/humble/setup.bash

ros2 pkg prefix mipi_cam
ros2 pkg prefix hobot_stereonet

rm -rfv performance_*.txt

stereonet_version=v2.0
calib_method=none
stereo_calib_file_path=calib.yaml
uncertainty_th=-0.10
render_type=indoor
render_perf=True
speckle_filter_enable=False
max_speckle_size=100
max_disp_diff=1.0
pcl_filter_enable=False
grid_size=0.1
grid_min_point_count=5
save_result_flag=False
save_dir=./stereonet_result
save_freq=1
save_total=-1
mipi_image_width=1920
mipi_image_height=1080
mipi_gdc_enable=True
mipi_rotation=90.0

while [[ $# -gt 0 ]]; do
  case $1 in
    --stereonet_version) stereonet_version=$2; shift 2 ;;
    --calib_method) calib_method=$2; shift 2 ;;
    --stereo_calib_file_path) stereo_calib_file_path=$2; shift 2 ;;
    --uncertainty_th) uncertainty_th=$2; shift 2 ;;
    --render_type) render_type=$2; shift 2 ;;
    --render_perf) render_perf=$2; shift 2 ;;
    --speckle_filter_enable) speckle_filter_enable=$2; shift 2 ;;
    --max_speckle_size) max_speckle_size=$2; shift 2 ;;
    --max_disp_diff) max_disp_diff=$2; shift 2 ;;
    --pcl_filter_enable) pcl_filter_enable=$2; shift 2 ;;
    --grid_size) grid_size=$2; shift 2 ;;
    --grid_min_point_count) grid_min_point_count=$2; shift 2 ;;
    --save_result_flag) save_result_flag=$2; shift 2 ;;
    --save_dir) save_dir=$2; shift 2 ;;
    --save_freq) save_freq=$2; shift 2 ;;
    --save_total) save_total=$2; shift 2 ;;
    --mipi_image_width) mipi_image_width=$2; shift 2 ;;
    --mipi_image_height) mipi_image_height=$2; shift 2 ;;
    --mipi_gdc_enable) mipi_gdc_enable=$2; shift 2 ;;
    --mipi_rotation) mipi_rotation=$2; shift 2 ;;
    *) echo "unknown param: $1"; exit 1 ;;
  esac
done

ros2 launch hobot_stereonet stereonet_model_web_visual_$stereonet_version.launch.py \
mipi_image_width:=$mipi_image_width mipi_image_height:=$mipi_image_height mipi_image_framerate:=30.0 mipi_rotation:=$mipi_rotation \
mipi_gdc_enable:=$mipi_gdc_enable mipi_lpwm_enable:=True mipi_frame_ts_type:=realtime \
calib_method:=$calib_method stereo_calib_file_path:=$stereo_calib_file_path \
uncertainty_th:=$uncertainty_th \
render_type:=$render_type render_perf:=$render_perf \
speckle_filter_enable:=$speckle_filter_enable max_speckle_size:=$max_speckle_size max_disp_diff:=$max_disp_diff \
pointcloud_height_min:=-5.0 pointcloud_height_max:=5.0 pointcloud_depth_max:=5.0 \
pcl_filter_enable:=$pcl_filter_enable grid_size:=$grid_size grid_min_point_count:=$grid_min_point_count \
save_result_flag:=$save_result_flag save_dir:=$save_dir save_freq:=$save_freq save_total:=$save_total