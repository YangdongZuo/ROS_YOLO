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
        rospy.loginfo('YOLOv5自动避障节点（新版中央触发/动作流程）已启动')

        self.avoiding = False
        self.avoid_stage = 0   # 0=无, 1=急停, 2=后退, 3=左转, 4=前进
        self.avoid_time = 0

    def image_callback(self, msg):
        if self.avoiding:
            now = time.time()
            twist = Twist()
            if self.avoid_stage == 1:
                # 急停
                twist.linear.x = 0
                twist.angular.z = 0
                self.cmd_pub.publish(twist)
                if now - self.avoid_time > 0.3:  # 停0.3秒
                    self.avoid_stage = 2
                    self.avoid_time = now
            elif self.avoid_stage == 2:
                # 后退0.5米，假定速度0.2m/s约2.5秒
                twist.linear.x = -0.2
                twist.angular.z = 0
                self.cmd_pub.publish(twist)
                if now - self.avoid_time > 1.0:
                    self.avoid_stage = 3
                    self.avoid_time = now
            elif self.avoid_stage == 3:
                # 左转80°（约1.4rad），0.87rad/s转1.6秒
                twist.linear.x = 0
                twist.angular.z = 0.87
                self.cmd_pub.publish(twist)
                if now - self.avoid_time > 1.4:
                    self.avoid_stage = 4
                    self.avoid_time = now
            elif self.avoid_stage == 4:
                # 前进0.6米，速度0.2m/s约3秒
                twist.linear.x = 0.2
                twist.angular.z = 0
                self.cmd_pub.publish(twist)
                if now - self.avoid_time > 2.0:
                    self.avoiding = False
                    self.avoid_stage = 0
                    self.avoid_time = 0
                    rospy.loginfo("避障动作完成，恢复导航。")
            return

        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_h, img_w = cv_img.shape[:2]

        # 改：中央检测区，红线各在左右1/2位置
        center_x = img_w // 2
        left_red = img_w // 4
        right_red = img_w * 3 // 4

        # 画辅助线
        cv2.line(cv_img, (center_x, 0), (center_x, img_h), (0, 255, 0), 2)  # 绿线中央
        cv2.line(cv_img, (left_red, 0), (left_red, img_h), (0, 0, 255), 2)  # 左红线
        cv2.line(cv_img, (right_red, 0), (right_red, img_h), (0, 0, 255), 2)  # 右红线

        results = self.model(cv_img)
        img_with_boxes = np.squeeze(results.render())
        cv2.line(img_with_boxes, (center_x, 0), (center_x, img_h), (0, 255, 0), 2)
        cv2.line(img_with_boxes, (left_red, 0), (left_red, img_h), (0, 0, 255), 2)
        cv2.line(img_with_boxes, (right_red, 0), (right_red, img_h), (0, 0, 255), 2)
        cv2.imshow("YOLOv5 Detection", img_with_boxes)
        cv2.waitKey(1)

        # 检查can是否在中央检测区，只触发一次
        for *box, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            if label == 'can' and conf > 0.4 and left_red <= cx <= right_red:
                rospy.logwarn("中央检测区检测到can：急停-后退0.5m-左转80°-前进0.6m")
                self.avoiding = True
                self.avoid_stage = 1
                self.avoid_time = time.time()
                break

if __name__ == '__main__':
    rospy.init_node('ros_yolov5_node')
    node = YoloV5SimpleAvoid()
    rospy.spin()
