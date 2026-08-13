#!/bin/bash
if [[ -f /opt/tros/humble/setup.bash ]]; then
  source /opt/tros/humble/setup.bash
elif [[ -f /opt/tros/jazzy/setup.bash ]]; then
  source /opt/tros/jazzy/setup.bash
else
  echo "Error: neither Humble nor Jazzy TROS environment was found"
  exit 1
fi

codec_sub_topic=/StereoNetNode/stereonet_visual
codec_in_format=bgr8
codec_pub_topic=/image_jpeg
websocket_image_topic=/image_jpeg
websocket_channel=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --codec_sub_topic) codec_sub_topic=$2; shift 2 ;;
    --codec_in_format) codec_in_format=$2; shift 2 ;;
    --codec_pub_topic) codec_pub_topic=$2; shift 2 ;;
    --websocket_image_topic) websocket_image_topic=$2; shift 2 ;;
    --websocket_channel) websocket_channel=$2; shift 2 ;;
    *) echo "unknown param: $1"; exit 1 ;;
  esac
done

ros2 launch hobot_stereonet codec_web_visual.launch.py \
codec_sub_topic:=$codec_sub_topic codec_in_format:=$codec_in_format codec_pub_topic:=$codec_pub_topic \
websocket_image_topic:=$websocket_image_topic websocket_channel:=$websocket_channel
