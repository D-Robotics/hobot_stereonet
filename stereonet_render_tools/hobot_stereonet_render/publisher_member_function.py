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

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(sensor_msgs.msg.Image, 'image_jpeg', 10)
        timer_period = 0.03  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)

        print("listen topic stereonet_node_output start....")

        self.subscription_ = self.create_subscription(
            sensor_msgs.msg.Image,
            'stereonet_node_output',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info('I heard encoding: "%s"' % msg.encoding)

        start_time = time.time()  

        # w = 1280
        # h = 720
        h = msg.height
        w = msg.width
        
        str_msg = msg.data
        # print(type(str_msg))

        infer_data_len = w*h*4
        index_slice = infer_data_len
        index_end = len(str_msg) + 1

        # print("index_slice:", index_slice)
        # print("index_end:", index_end)

        buf_infer = str_msg[0:index_slice]
        buf_jpeg = str_msg[index_slice:index_end]

        buf_infer_len = w*h
        data_infer = np.ndarray(shape=(1, buf_infer_len),
                         dtype=np.uint32, buffer=buf_infer)
        
        data_jpeg = np.ndarray(shape=(1, len(buf_jpeg)),
                         dtype=np.uint8, buffer=buf_jpeg)

        # print("data_infer data:")
        # print(type(data_infer))
        # print(data_infer.shape)
        # print(data_infer)

        # print("data_jpeg data:")
        # print(type(data_jpeg))
        # print(data_jpeg.shape)

        im = cv2.imdecode(data_jpeg, cv2.IMREAD_ANYCOLOR)
        # print(type(im))
        # cv2.imwrite("py.jpg", im)

        header = Header()  
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera'  

        left_img = Image.fromarray(im.astype('uint8')).convert('RGB')
        b, g, r = left_img.split()
        left_img = Image.merge("RGB", (r, g, b))
        # print(type(left_img))

        open_cv_image = np.array(left_img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        # img_msg = self.cv2_to_imgmsg(open_cv_image, 'jpeg', header, w, h)  
        # self.publisher_.publish(img_msg)
        # self.get_logger().info('Publishing')


        loadData = data_infer.reshape((1, 1, 720, 1280))
        # print("data_infer:")
        # print(type(loadData))
        # print(loadData.shape)
        # print(loadData)

        
        # data_infer = np.fromfile('output.bin', dtype=np.uint32)
        # loadData = data_infer.reshape((1, 1, 720, 1280))
        # print("np.fromfile output.bin:")
        # print(type(loadData))
        # print(loadData.shape)
        # print(loadData)

        self.get_logger().info('split and reshape')

        image_pre = loadData * 0.00000260443857769133

        f = 527.1931762695312
        B = 119.89382172

        image_pre = image_pre[-1] * 16 * 12
        Z_image_pre = f*B/image_pre/1000

        Z_image_pre_color = cv2.applyColorMap(cv2.convertScaleAbs(Z_image_pre.squeeze(0), alpha=11), cv2.COLORMAP_JET)
        # fn = "dump_render.jpg"
        # cv2.imwrite(fn, Z_image_pre_color)

        # disps[-1]是输出的结果，*16*12就可以了
        # f*B/image_pre/1000得到的灰度图
        # convertScaleAbs( )可把任意类型的数据转化为CV_8UC1。
        # cv2.applyColorMap用openCV转换成彩色

        end_time = time.time()  
        time_cost_ms = (end_time - start_time) * 1000
        self.get_logger().info(f'applyColorMap time cost {time_cost_ms:.3f} ms')


        img_pre = Image.fromarray(Z_image_pre_color.astype('uint8')).convert('RGB')
        b, g, r = img_pre.split()
        img_pre = Image.merge("RGB", (r, g, b))

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
        joint.save('vertical.jpg')
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





    # cvim,w,h
    def render_from_msg(self, msg):
        msg_data = msg.data
        print(type(msg_data))

        # 从array.array创建numpy数组  
        # arr = array.array('B', [0, 1, 2, 3, 4, 5, 6, 7])  
        img_arr = np.frombuffer(msg_data.tobytes(), dtype=np.uint8)  
        
        # 从numpy数组创建PIL图像  
        pil_image = Image.fromarray(img_arr)  
        print(type(pil_image))
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, ::-1].copy() 

        return open_cv_image, 1280, 720


        
        # np.fromfile('/home/kao.zhu/share/nfs/github/tros/j5/tros_ws/output.bin', dtype=np.uint32)
        
        # left_img = Image.open("/mnt/nfs/github/tros/j5/tros_ws/frame_1280_720_left.jpg")    
        left_img = Image.open("/home/kao.zhu/share/nfs/github/tros/j5/tros_ws/frame_1280_720_left.jpg")    

        loadData = loadData.reshape((1, 1, 720, 1280))
        image_pre = loadData * 0.00000260443857769133

        f = 527.1931762695312
        B = 119.89382172

        image_pre = image_pre[-1] * 16 * 12
        Z_image_pre = f*B/image_pre/1000

        Z_image_pre_color = cv2.applyColorMap(cv2.convertScaleAbs(Z_image_pre.squeeze(0), alpha=11), cv2.COLORMAP_JET)
        fn = "dump_render.jpg"
        cv2.imwrite(fn, Z_image_pre_color)

        # disps[-1]是输出的结果，*16*12就可以了
        # f*B/image_pre/1000得到的灰度图
        # convertScaleAbs( )可把任意类型的数据转化为CV_8UC1。
        # cv2.applyColorMap用openCV转换成彩色


        img_pre = Image.fromarray(Z_image_pre_color.astype('uint8')).convert('RGB')
        b, g, r = img_pre.split()
        img_pre = Image.merge("RGB", (r, g, b))


        pil_image = left_img
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

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
        joint.save('vertical.jpg')

        open_cv_image = np.array(joint) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        return open_cv_image, size1[0], size1[1] + size2[1]


    def render(self, file):
        # loadData = np.fromfile('/mnt/nfs/github/tros/j5/tros_ws/output.bin', dtype=np.uint32)
        loadData = np.fromfile('/home/kao.zhu/share/nfs/github/tros/j5/tros_ws/output.bin', dtype=np.uint32)
        
        # left_img = Image.open("/mnt/nfs/github/tros/j5/tros_ws/frame_1280_720_left.jpg")    
        left_img = Image.open("/home/kao.zhu/share/nfs/github/tros/j5/tros_ws/frame_1280_720_left.jpg")    

        loadData = loadData.reshape((1, 1, 720, 1280))
        image_pre = loadData * 0.00000260443857769133

        f = 527.1931762695312
        B = 119.89382172

        image_pre = image_pre[-1] * 16 * 12
        Z_image_pre = f*B/image_pre/1000

        Z_image_pre_color = cv2.applyColorMap(cv2.convertScaleAbs(Z_image_pre.squeeze(0), alpha=11), cv2.COLORMAP_JET)
        fn = "dump_render.jpg"
        cv2.imwrite(fn, Z_image_pre_color)

        # disps[-1]是输出的结果，*16*12就可以了
        # f*B/image_pre/1000得到的灰度图
        # convertScaleAbs( )可把任意类型的数据转化为CV_8UC1。
        # cv2.applyColorMap用openCV转换成彩色


        img_pre = Image.fromarray(Z_image_pre_color.astype('uint8')).convert('RGB')
        b, g, r = img_pre.split()
        img_pre = Image.merge("RGB", (r, g, b))

        pil_image = left_img
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

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
        joint.save('vertical.jpg')

        open_cv_image = np.array(joint) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        return open_cv_image, size1[0], size1[1] + size2[1]

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

    def timer_callback(self):
        # 将OpenCV图像转换为ROS 2图像消息  
        # 设置ROS 2图像消息的header  
        header = Header()  
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera'  

        cvim,w,h = self.render('/home/kao.zhu/share/nfs/github/tros/j5/tros_ws/output.bin')
        img_msg = self.cv2_to_imgmsg(cvim, 'jpeg', header, w, h)  

        self.publisher_.publish(img_msg)
        self.get_logger().info('Publishing')


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
