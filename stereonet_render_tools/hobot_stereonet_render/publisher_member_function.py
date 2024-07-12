# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
from PIL import Image

import rclpy
from rclpy.node import Node

import sensor_msgs.msg
from std_msgs.msg import Header  
# from hbm_img_msgs.msg.HbmMsg1080P import HbmMsg1080P

import time

class MinimalPublisher(Node):
    scale = 0.00000260443857769133
    f = 527.1931762695312
    B = 119.89382172

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(sensor_msgs.msg.Image, 'image_jpeg', 10)

        print("listen topic stereonet_node_output start....")

        self.subscription_ = self.create_subscription(
            sensor_msgs.msg.Image,
            'stereonet_node_output',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard encoding: {msg.encoding}, stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}, frame_id: {msg.header.frame_id}')

        start_time = time.time()  

        # w = 1280
        # h = 720
        h = msg.height
        w = msg.width
        
        str_msg = msg.data

        infer_data_len = w*h*4
        index_slice = infer_data_len
        index_end = len(str_msg) + 1

        buf_infer = str_msg[0:index_slice]
        buf_jpeg = str_msg[index_slice:index_end]

        buf_infer_len = w*h
        data_infer = np.ndarray(shape=(1, buf_infer_len),
                         dtype=np.uint32, buffer=buf_infer)
        
        header = Header()  
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera'  

        loadData = data_infer.reshape((1, 1, 720, 1280))
        image_pre = loadData * self.scale

        image_pre = image_pre[-1] * 16 * 12
        
        # fname = 'dump_stereonet_render/' + msg.header.frame_id + '.npy'
        # np.save(fname, image_pre)


        Z_image_pre = self.f*self.B/image_pre/1000
        Z_image_pre_color = cv2.applyColorMap(cv2.convertScaleAbs(Z_image_pre.squeeze(0), alpha=9), cv2.COLORMAP_JET)
        
        # fname = 'dump_stereonet_render/' + msg.header.frame_id + '.png'
        # Z_image_pre_color = Image.fromarray(Z_image_pre_color)
        # Z_image_pre_color.save(fname)
        # return

        # disps[-1]是输出的结果，*16*12就可以了
        # f*B/image_pre/1000得到的灰度图
        # convertScaleAbs( )可把任意类型的数据转化为CV_8UC1。
        # cv2.applyColorMap用openCV转换成彩色

        data_jpeg = np.ndarray(shape=(1, len(buf_jpeg)),
                         dtype=np.uint8, buffer=buf_jpeg)
        im = cv2.imdecode(data_jpeg, cv2.IMREAD_ANYCOLOR)

        left_img = Image.fromarray(im.astype('uint8')).convert('RGB')
        b, g, r = left_img.split()
        left_img = Image.merge("RGB", (r, g, b))

        open_cv_image = np.array(left_img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        end_time = time.time()  
        time_cost_ms = (end_time - start_time) * 1000
        self.get_logger().info(f'applyColorMap time cost {time_cost_ms:.3f} ms')

        img_pre = Image.fromarray(Z_image_pre_color.astype('uint8')).convert('RGB')

        size1, size2 = left_img.size, img_pre.size

        # # 'horizontal':
        # joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        # loc1, loc2 = (0, 0), (1280, 0)
        # joint.paste(left_img, loc1)
        # joint.paste(img_pre, loc2)
        # joint.save('horizontal.jpg')

        # 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, 720)
        joint.paste(left_img, loc1)
        joint.paste(img_pre, loc2)

        # dump render img padding with left img
        # fname = 'dump_stereonet_render/' + msg.header.frame_id + '.jpg'
        # joint.save(fname)
        # self.get_logger().info(f'save {fname}')
        # return

        size = joint.size
        w = size[0]
        h = size[1]

        open_cv_image = np.array(joint) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        end_time = time.time()  
        time_cost_ms = (end_time - start_time) * 1000
        self.get_logger().info(f'img joint time cost {time_cost_ms:.3f} ms')

        img_msg = self.cv2_to_imgmsg(open_cv_image, 'jpeg', header, w, h)  

        end_time = time.time()  
        time_cost_ms = (end_time - start_time) * 1000
        self.get_logger().info(f'cv2_to_imgmsg time cost {time_cost_ms:.3f} ms')

        self.publisher_.publish(img_msg)
        self.get_logger().info('Publishing')


    def cv2_to_imgmsg(self, cvim, encoding, header, w, h):
        img_msg = sensor_msgs.msg.Image()
        img_msg.height = h
        img_msg.width = w
        img_msg.header = header
        img_msg.encoding = encoding
        encoder_img = np.array(cv2.imencode('.jpg', cvim)[1])
        img_msg.data = encoder_img.tobytes()  
        img_msg.step = len(img_msg.data)

        return img_msg


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)

    # minimal_subscriber = MinimalSubscriber()
    # rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
