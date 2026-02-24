#!/usr/bin/env python3
import os, sys

from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import jax

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from octo.model.octo_model import OctoModel

class OctoPublisher(Node):
    def __init__(self):
        super().__init__('octo_publisher')
        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
        self.publisher_ = self.create_publisher(String, '/octo_actions', 10)
        self.image_sub = self.create_subscription(Image, '/robocasa_camera', self.image_callback, 10)
        self.bridge = CvBridge()
        self.latest_image = None
        self.timer = self.create_timer(0.5, self.timer_callback)
    
    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        cv_img = cv2.resize(cv_img, (256, 256))
        self.latest_image = cv_img

    def get_actions(self, img):
        img = img[np.newaxis, np.newaxis, ...]  # (1,1,H,W,C)
        observation = {"image_primary": img, "timestep_pad_mask": np.array([[True]])}
        task = self.model.create_tasks(texts=["pick up the cup"]) 

        action = self.model.sample_actions(
                            observation, 
                            task, 
                            unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"], 
                            rng=jax.random.PRNGKey(0)
                            )
        action = np.array(action)
        first_action = action[0, 0]
        print(first_action) 
        return first_action
    
    def timer_callback(self):
        if self.latest_image is None:
            return
        msg = String()
        actions = self.get_actions(self.latest_image)
        msg.data = str(actions.tolist())
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    octo_publisher = OctoPublisher()
    rclpy.spin(octo_publisher)
    octo_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()