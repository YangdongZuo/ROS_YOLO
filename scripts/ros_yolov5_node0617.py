#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import cv2
import os
import time
import threading
import numpy as np

class YoloCanHardAvoid:
    def __init__(self):
        self.bridge = CvBridge()
        yolov5_repo = '/home/ydzuo/catkin_ws/src/yolov5'
        weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')
        self.model = torch.hub.load(yolov5_repo, 'custom', path=weight_path, source='local')
        self.model.conf = 0.4

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)
        rospy.loginfo('YOLO检测到can后：急停→后退0.3m→左转90°→前进0.5m，窗口实时弹出，节点已启动')

        self.handled_time = 0
        self.cooldown = 5.0  # 避障动作冷却时间，防止重复触发
        self.avoiding = False  # 当前是否正在执行避障动作

        # ========== 窗口显示线程 ==========
        self.show_img = None
        self.img_lock = threading.Lock()
        self.show_thread = threading.Thread(target=self.imshow_loop, daemon=True)
        self.show_thread.start()

    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_h, img_w = cv_img.shape[:2]
        center_x = img_w // 2
        left_red = img_w // 4
        right_red = img_w * 3 // 4

        results = self.model(cv_img)
        img_with_boxes = np.squeeze(results.render())
        # 辅助线
        cv2.line(img_with_boxes, (center_x, 0), (center_x, img_h), (0,255,0), 2)
        cv2.line(img_with_boxes, (left_red, 0), (left_red, img_h), (0,0,255), 2)
        cv2.line(img_with_boxes, (right_red, 0), (right_red, img_h), (0,0,255), 2)
        with self.img_lock:
            self.show_img = img_with_boxes.copy()

        if self.avoiding:
            return  # 动作执行期间忽略新图片

        now = time.time()
        for *box, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            if label == 'can' and conf > 0.4 and left_red <= cx <= right_red:
                if now - self.handled_time > self.cooldown:
                    rospy.logwarn("检测到can：立即停车-后退-左转90°-前进，压制导航")
                    self.avoiding = True
                    self.handled_time = now
                    threading.Thread(target=self.do_strict_avoid_action, daemon=True).start()
                break

    def do_strict_avoid_action(self, turn_left=True):
        # 1. 急停
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        for _ in range(10):
            self.cmd_pub.publish(twist)
            rospy.sleep(0.03)
        # 2. 后退 0.3m（约0.18m/s * 1.67s）
        twist.linear.x = -0.18
        twist.angular.z = 0
        t0 = time.time()
        while time.time() - t0 < 1.67:
            self.cmd_pub.publish(twist)
            rospy.sleep(0.03)
        twist.linear.x = 0
        for _ in range(10):
            self.cmd_pub.publish(twist)
            rospy.sleep(0.03)
        # 3. 左转 90°（0.9rad/s，1.75s ≈ 90°）
        twist.linear.x = 0
        twist.angular.z = 0.9 if turn_left else -0.9
        t0 = time.time()
        while time.time() - t0 < 1.75:
            self.cmd_pub.publish(twist)
            rospy.sleep(0.03)
        twist.angular.z = 0
        for _ in range(10):
            self.cmd_pub.publish(twist)
            rospy.sleep(0.03)
        # 4. 前进 0.5m（0.22m/s * 2.27s ≈ 0.5m）
        twist.linear.x = 0.22
        twist.angular.z = 0
        t0 = time.time()
        while time.time() - t0 < 2.27:
            self.cmd_pub.publish(twist)
            rospy.sleep(0.03)
        twist.linear.x = 0
        for _ in range(15):
            self.cmd_pub.publish(twist)
            rospy.sleep(0.03)
        rospy.loginfo("避障动作完成，恢复导航。")
        self.avoiding = False

    def imshow_loop(self):
        while not rospy.is_shutdown():
            with self.img_lock:
                img = self.show_img.copy() if self.show_img is not None else None
            if img is not None:
                cv2.imshow("YOLOv5 Detection", img)
                cv2.waitKey(1)
            else:
                time.sleep(0.03)

if __name__ == '__main__':
    rospy.init_node('yolo_can_hardavoid')
    node = YoloCanHardAvoid()
    rospy.spin()
    cv2.destroyAllWindows()
