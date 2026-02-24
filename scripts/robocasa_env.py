#!/usr/bin/env python3

import ast
import os
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import imageio
from datetime import datetime

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env


class RobocasaEnvNode(Node):
    def __init__(self):
        super().__init__('robocasa_env_node')

        self.image_pub = self.create_publisher(Image, '/robocasa_camera', 10)
        self.subscription = self.create_subscription(String,'/octo_actions', self.listener_callback, 10)

        self.bridge = CvBridge()
        self.env = self.create_environment()
        self.obs = self.env.reset()

        #keeping the default actions as zero
        self.action_dim = self.env.action_spec[0].shape[0]
        self.default_action = np.zeros(self.action_dim, dtype=np.float32)
        self.default_action[-1] = 1.0

        self.video_writer = None
        self.start_video_writer()

        lang = self.env.get_ep_meta()["lang"]
        self.get_logger().info(f"Instruction: {lang}")
        self.get_logger().info(f"Expected action dimension: {self.action_dim}")
        self.publish_image(self.obs)

    def create_environment(self):
        # env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
        my_env="PrepareCoffee"
        custom_layout_id=11
        custom_style_id=12

        env = create_env(
            env_name=my_env,
            layout_ids=custom_layout_id,
            style_ids=custom_style_id,
            robots="PandaOmron",
            camera_names=[
                "robot0_eye_in_hand",
            ],
            camera_widths=128,
            camera_heights=128,
            render_onscreen=False,
            randomize_cameras=False,
            seed=0,
        )

        print(f'my_env = {env}')
        return env
    
    def start_video_writer(self):
        os.makedirs("rollout_videos", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"rollout_videos/rollout_{timestamp}.mp4"

        self.video_writer = imageio.get_writer(video_path, fps=20)
        self.get_logger().info(f"Recording video to {video_path}")

    def close_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.get_logger().info("Video saved.")
            self.video_writer = None
    
    def publish_image(self, obs):
        img = obs["robot0_eye_in_hand_image"]  # (C,H,W)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ros_img = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        self.image_pub.publish(ros_img)

        if self.video_writer is not None:
            video_img = self.env.sim.render(
                height=512,
                width=768,
                camera_name="robot0_agentview_center"
            )[::-1]
            self.video_writer.append_data(video_img)

    def listener_callback(self, msg):
        """
        Called every time Octo publishes an action
        """
        try:
            action_list = ast.literal_eval(msg.data)
            octo_action = np.array(action_list, dtype=np.float32)

            full_action = self.default_action.copy()
            full_action[:7] = octo_action

        except Exception as e:
            self.get_logger().info(f"Expected action shape: {self.env.action_spec[0].shape}")
            self.get_logger().error(f"Failed to parse action: {e}")
            return

        obs, reward, done, info = self.env.step(full_action)
        self.publish_image(obs)

        self.get_logger().info(f"Step | reward={reward:.3f} | done={done}")

        if done:
            self.get_logger().info("Episode finished, resetting env")
            self.close_video_writer()
            self.obs = self.env.reset()
            self.start_video_writer()
            self.publish_image(self.obs)


def main(args=None):
    rclpy.init(args=args)
    node = RobocasaEnvNode()
    rclpy.spin(node)
    node.close_video_writer()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
