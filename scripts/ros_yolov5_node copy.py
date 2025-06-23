#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
import os
import time

class YoloV5SimpleAvoid:
    def __init__(self):
        self.bridge = CvBridge()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yolov5_repo = '/home/ydzuo/catkin_ws/src/yolov5'
        weight_path = os.path.join(script_dir, 'best.pt')
        self.model = torch.hub.load(yolov5_repo, 'custom', path=weight_path, source='local')
        self.model.conf = 0.4

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)
        rospy.loginfo('YOLOv5自动避障节点（后退+左转+前进）已启动')

        self.avoiding = False   # 是否正在执行避障
        self.avoid_stage = 0    # 0=无, 1=后退, 2=左转, 3=直行
        self.avoid_time = 0     # 动作开始时间

    def image_callback(self, msg):
        # 动作流程控制
        if self.avoiding:
            now = time.time()
            twist = Twist()
            if self.avoid_stage == 1:
                # 后退
                twist.linear.x = -0.2
                twist.angular.z = 0
                self.cmd_pub.publish(twist)
                if now - self.avoid_time > 1.0:  # 后退1秒
                    self.avoid_stage = 2
                    self.avoid_time = now
            elif self.avoid_stage == 2:
                # 左转60°，角速度0.87rad/s，约0.7秒
                twist.linear.x = 0
                twist.angular.z = 0.87
                self.cmd_pub.publish(twist)
                if now - self.avoid_time > 0.7:  # 左转0.7秒
                    self.avoid_stage = 3
                    self.avoid_time = now
            elif self.avoid_stage == 3:
                # 直行
                twist.linear.x = 0.2
                twist.angular.z = 0
                self.cmd_pub.publish(twist)
                if now - self.avoid_time > 2.0:  # 直行2秒
                    self.avoiding = False
                    self.avoid_stage = 0
                    self.avoid_time = 0
                    rospy.loginfo("避障动作结束，恢复导航。")
            return

        # 正常视觉显示
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_h, img_w = cv_img.shape[:2]

        # 画中央检测区辅助线
        center_x = img_w // 2
        offset = img_w // 8   # 红线偏移量(画面1/8)
        left_red = center_x - offset
        right_red = center_x + offset

        # 绿线：画面中央
        cv2.line(cv_img, (center_x, 0), (center_x, img_h), (0, 255, 0), 2)
        # 红线：中央区域左右界
        cv2.line(cv_img, (left_red, 0), (left_red, img_h), (0, 0, 255), 2)
        cv2.line(cv_img, (right_red, 0), (right_red, img_h), (0, 0, 255), 2)

        # 推理和画框
        results = self.model(cv_img)
        img_with_boxes = np.squeeze(results.render())
        # 再把辅助线画到img_with_boxes
        cv2.line(img_with_boxes, (center_x, 0), (center_x, img_h), (0, 255, 0), 2)
        cv2.line(img_with_boxes, (left_red, 0), (left_red, img_h), (0, 0, 255), 2)
        cv2.line(img_with_boxes, (right_red, 0), (right_red, img_h), (0, 0, 255), 2)
        cv2.imshow("YOLOv5 Detection", img_with_boxes)
        cv2.waitKey(1)

        # 检查can是否在中央区域，只触发一次
        for *box, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            if label == 'can' and conf > 0.4 and left_red <= cx <= right_red:
                rospy.logwarn("检测到can在中央区域，执行后退-左转-前进避障动作！")
                self.avoiding = True
                self.avoid_stage = 1
                self.avoid_time = time.time()
                break

if __name__ == '__main__':
    rospy.init_node('ros_yolov5_node')
    node = YoloV5SimpleAvoid()
    rospy.spin()
