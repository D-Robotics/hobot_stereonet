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
import queue

resize_factor = 2
spx = 670
spy = 332
def MouseCb(event, x, y, flags, param):
    global spx, spy
    spx = x * resize_factor
    spy = y * resize_factor

class DepthVisualizer(Node):

    def __init__(self):
        super().__init__('DepthVisualizer')


        self._bridge = cv_bridge.CvBridge()
        self._disp_sub = self.create_subscription(
           Image, '/StereoNetNode/stereonet_depth', self.depth_callback, 10)

        self._image_sub = self.create_subscription(
            Image, '/StereoNetNode/rectified_image', self.image_callback, 10)


        self._que = queue.Queue(maxsize=100)

        cv2.namedWindow('DepthVisualizer')
        cv2.setMouseCallback('DepthVisualizer', MouseCb)

    def wrap_color_map(self, color_map, depth):
        global spx, spy
        camera_cx = 567.488464
        camera_fx = 489.268860
        camera_cy = 293.104919
        camera_fy = 489.268860
        depth = depth / 1000
        cv2.line(color_map, (spx, 0), (spx, color_map.shape[0] - 1), (255,255,255), 2)
        cv2.line(color_map, (0, spy), (color_map.shape[1] - 1, spy), (255,255,255), 2)
        X = (spx - camera_cx) / camera_fx * depth
        Y = (spy - camera_cy) / camera_fy * depth
        cv2.putText(color_map, str(depth) + 'm', (spx, spy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.putText(color_map, '(' + str(spx) + ',' + str(spy) + ')', (spx, spy + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        return color_map
        pass

    def wrap_rgb_map(self, rgb, depth):
        global spx, spy
        depth = depth / 1000
        cv2.line(rgb, (spx, 0), (spx, rgb.shape[0] - 1), (255,255,255), 2)
        cv2.line(rgb, (0, spy), (rgb.shape[1] - 1, spy), (255,255,255), 2)
        cv2.putText(rgb, str(depth) + 'm', (spx, spy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.putText(rgb, '(' + str(spx) + ',' + str(spy) + ')', (spx, spy + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        return rgb
        pass

    def depth_callback(self, depth_msg):
        global spx, spy
        print("get depth")
        self._que.put(depth_msg)
        cv2.waitKey(1)


    def image_callback(self, img_msg):
        global spx, spy
        depth_msg = self._que.get()
        depth_img = self._bridge.imgmsg_to_cv2(depth_msg)
        if spx >= depth_img.shape[1]:
            spx = depth_img.shape[1] - 1

        if spy >= depth_img.shape[0]:
            spy = depth_img.shape[0] - 1

        depth = depth_img[spy][spx]
        normlized_img = (depth_img - depth_img.min()) / depth_img.max() * 255
        # normlized_img = 0.08*600*3000/depth_img
        color_map = cv2.applyColorMap(normlized_img.astype(np.uint8), cv2.COLORMAP_JET)
        color_map = self.wrap_color_map(color_map, depth)
        img_img = self._bridge.imgmsg_to_cv2(img_msg)
        self._que.task_done()
        rgb_img = self.wrap_rgb_map(img_img, depth)
        render = cv2.vconcat([rgb_img, color_map])
        render = cv2.resize(render, (rgb_img.shape[1] // resize_factor, rgb_img.shape[0] // resize_factor * 2),
                            interpolation=cv2.INTER_LINEAR)
        cv2.imshow('DepthVisualizer', render)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = DepthVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
