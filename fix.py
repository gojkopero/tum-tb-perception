with open('ros/tum_tb_perception/yolo_pnp_pose_estimator_node.py', 'r') as f:
    content = f.read()

content = content.replace(
    'from cv_bridge import CvBridge, CvBridgeError',
    'from cv_bridge import CvBridge, CvBridgeError\nfrom rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy'
)

qos_replace = """        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._img_cb, qos
        )
        self.cam_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._cam_cb, qos
        )
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self._pc_cb, qos
        )"""

content = content.replace(
"""        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._img_cb, 10
        )
        self.cam_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._cam_cb, 10
        )
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self._pc_cb, 10
        )""", qos_replace)

content = content.replace('pt_msg.header.frame_id = camera_frame_id', 'pt_msg.header.frame_id = camera_frame_id\n        pt_msg.header.stamp = rclpy.time.Time().to_msg()')
content = content.replace('pose_msg.header.frame_id = camera_frame_id', 'pose_msg.header.frame_id = camera_frame_id\n        pose_msg.header.stamp = rclpy.time.Time().to_msg()')

with open('ros/tum_tb_perception/yolo_pnp_pose_estimator_node.py', 'w') as f:
    f.write(content)
