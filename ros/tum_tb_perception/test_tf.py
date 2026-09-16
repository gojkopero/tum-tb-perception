import rclpy
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Point, Quaternion

rclpy.init()
node = Node('test_tf')
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.transform_listener.TransformListener(tf_buffer, node)

import time
time.sleep(1) # wait for tf

pose_msg = PoseStamped()
pose_msg.header.frame_id = 'camera_color_optical_frame'
# we DON'T set stamp!

try:
    res = tf_buffer.transform(pose_msg, 'fr3_link0')
    print("Success!")
except Exception as e:
    print("Exception:", e)
