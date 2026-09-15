#!/bin/bash
if [[ -f /opt/tros/humble/setup.bash ]]; then
  source /opt/tros/humble/setup.bash
elif [[ -f /opt/tros/jazzy/setup.bash ]]; then
  source /opt/tros/jazzy/setup.bash
else
  echo "Error: neither Humble nor Jazzy TROS environment was found"
  exit 1
fi

ros2 pkg prefix mipi_cam

image_width=640          # main stream width (full-res combine stream)
image_height=352         # main stream height
framerate=30.0           # main stream framerate
rotation=0.0             # main stream rotation (deg)
gdc_enable=True          # enable GDC distortion correction
cal_rotation=0.0         # calibration rotation (deg)
lpwm_enable=True         # enable low-power mode
frame_ts_type=realtime   # frame timestamp type: sensor | realtime
out_format=nv12          # output pixel format: nv12 | bgr8
channel=2                # camera channel
channel2=0               # camera channel 2
dual_combine=2           # 0: left+right only, 1: add combine, 2: combine only
stream_mode=1            # 1: dual stream = main (GDC-corrected) + sub (pre-GDC portrait); required for the portrait sub stream below
sub_stream_enable=True   # publish portrait /sub_image_combine_raw (544x640) for VIO
sub_image_width=544      # sub stream width
sub_image_height=640     # sub stream height
sub_framerate=30.0       # sub stream framerate
sync_awb=False           # sync AWB (auto white balance) from master to slave camera
sync_ccm=False           # sync CCM (Color Correction Matrix) from master to slave camera: copy master's 3x3 matrix (+offset) that maps sensor raw RGB to standard color space, so left/right images have matched color (helps stereo matching)
sync_ae=False            # sync AE (auto exposure) from master to slave camera
log_level=ERROR          # node log level

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
    --dual_combine) dual_combine=$2; shift 2 ;;
    --stream_mode) stream_mode=$2; shift 2 ;;
    --sub_stream_enable) sub_stream_enable=$2; shift 2 ;;
    --sub_image_width) sub_image_width=$2; shift 2 ;;
    --sub_image_height) sub_image_height=$2; shift 2 ;;
    --sub_framerate) sub_framerate=$2; shift 2 ;;
    --sync_awb) sync_awb=$2; shift 2 ;;
    --sync_ccm) sync_ccm=$2; shift 2 ;;
    --sync_ae) sync_ae=$2; shift 2 ;;
    --log_level) log_level=$2; shift 2 ;;
    *) echo "unknown param: $1"; exit 1 ;;
  esac
done

ros2 run mipi_cam mipi_cam --ros-args \
-p device_mode:=dual -p dual_combine:=$dual_combine -p stream_mode:=$stream_mode \
-p image_width:=$image_width -p image_height:=$image_height \
-p framerate:=$framerate -p rotation:=$rotation \
-p gdc_enable:=$gdc_enable -p cal_rotation:=$cal_rotation \
-p lpwm_enable:=$lpwm_enable \
-p frame_ts_type:=$frame_ts_type \
-p out_format:=$out_format \
-p channel:=$channel -p channel2:=$channel2 \
-p sub_stream_enable:=$sub_stream_enable \
-p sub_image_width:=$sub_image_width -p sub_image_height:=$sub_image_height \
-p sub_framerate:=$sub_framerate \
-p sync_awb:=$sync_awb -p sync_ccm:=$sync_ccm -p sync_ae:=$sync_ae \
--log-level $log_level
