import rclpy
from rclpy.node import Node
import tf2_ros
import time
import subprocess

# Launch realsense headless temporarily just to get the static TF
proc = subprocess.Popen(["ros2", "launch", "realsense2_camera", "rs_launch.py"])

rclpy.init()
node = Node('tf_getter')
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer, node)

# wait for realsense TF
start = time.time()
while time.time() - start < 10:
    rclpy.spin_once(node, timeout_sec=0.1)
    try:
        trans = tf_buffer.lookup_transform('camera_link', 'camera_color_optical_frame', rclpy.time.Time())
        print(f"XYZ: {trans.transform.translation.x} {trans.transform.translation.y} {trans.transform.translation.z}")
        print(f"Q: {trans.transform.rotation.x} {trans.transform.rotation.y} {trans.transform.rotation.z} {trans.transform.rotation.w}")
        break
    except Exception as e:
        pass

proc.terminate()
rclpy.shutdown()
