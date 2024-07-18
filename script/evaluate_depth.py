import argparse
import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage
import math
import sys
import os

spx = 0
spy = 0
def MouseCb(event, x, y, flags, param):
    global spx, spy
    spx = x
    spy = y


class DepthVisualizer(Node):

    def __init__(self):
        super().__init__('DepthVisualizer')


        self._bridge = cv_bridge.CvBridge()
        self._disp_sub = self.create_subscription(
            Image, '/StereoNetNode/stereonet_depth', self.depth_callback, 10)


        cv2.namedWindow('DepthVisualizer')
        cv2.setMouseCallback('DepthVisualizer', MouseCb)

    def wrap_color_map(self, color_map, depth):
        global spx, spy
        cv2.line(color_map, (spx, 0), (spx, color_map.shape[0] - 1), (255,255,255), 1)
        cv2.line(color_map, (0, spy), (color_map.shape[1] - 1, spy), (255,255,255), 1)
        cv2.putText(color_map, str(depth) + 'mm', (spx, spy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return color_map
        pass

    def depth_callback(self, depth_msg):
        global spx, spy
        print("get depth")
        depth_img = self._bridge.imgmsg_to_cv2(depth_msg)
        depth = depth_img[spy][spx]
        normlized_img = (depth_img - depth_img.min()) / depth_img.max() * 255
        color_map = cv2.applyColorMap(normlized_img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        color_map = self.wrap_color_map(color_map, depth)
        cv2.imshow('DepthVisualizer', color_map)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = DepthVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()