#!/usr/bin/env python3

"""
YOLO + PnP pose estimator node for the euROBIN taskboard.

Replaces the legacy pointcloud-based orientation estimation with a pipeline
ported from DesignLearnRG_euROBIN/vision:
  1. YOLOv8 detects taskboard components (ONNX Runtime)
  2. Three keypoints (blue_button, red_button, red_hole) are averaged
     over N frames for stability
  3. OpenCV solvePnP computes the 6-DOF board pose from the 2D–3D
     correspondences using known CAD model coordinates
  4. Secondary detections are projected onto the solved board plane
  5. TF frames are published for all detected components

No pointcloud subscription required – the entire pose is derived from
the RGB image and camera intrinsics.

Subscribes to:
    - /camera/camera/color/image_raw    (sensor_msgs/Image)
    - /camera/camera/color/camera_info  (sensor_msgs/CameraInfo)

Publishes (same topics as existing pose estimator nodes):
    - /tum_tb_perception/object_poses   (ObjectList)
    - /tum_tb_perception/object_markers (MarkerArray)
    - TF: taskboard_frame, blue_button_frame, red_button_frame, ...
"""

import os
import sys
import time
import copy
import collections

import rclpy
import tf2_ros
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from launch_ros.substitutions import FindPackageShare

from geometry_msgs.msg import (
    Point, Pose, Quaternion, Vector3, TransformStamped,
)
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger

# For the ObjectList output (compatibility with existing system):
from tum_tb_perception_msgs.msg import ObjectList, Object

from tum_tb_perception.yolo_detector import YOLODetector, load_class_names_from_yaml
from tum_tb_perception.pnp_pose_estimator import (
    PnPPoseEstimator,
    PNP_KEYPOINT_ORDER,
    PNP_MODEL_POINTS,
    EUROBIN_TO_TUM_LABEL,
)

# Fix occasional Qt visualisation issue on Ubuntu 22.04+:
if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class YoloPnPPoseEstimatorNode(Node):

    def __init__(self):
        super().__init__("yolo_pnp_pose_estimator")

        pkg = FindPackageShare(package="tum_tb_perception").find(
            "tum_tb_perception"
        )

        # ---- Parameters -------------------------------------------------
        self.declare_parameter(
            "model_path",
            os.path.join(pkg, "models/best_taskboard.onnx"),
        )
        self.declare_parameter(
            "class_names_yaml",
            os.path.join(pkg, "config/yolo_classes_taskboard.yaml"),
        )
        self.declare_parameter(
            "model_points_yaml",
            os.path.join(pkg, "config/model_points_taskboard.yaml"),
        )
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter(
            "object_poses_pub_topic", "/tum_tb_perception/object_poses"
        )
        self.declare_parameter(
            "object_marker_pub_topic", "/tum_tb_perception/object_markers"
        )
        self.declare_parameter(
            "detection_image_pub_topic",
            "/tum_tb_perception/yolo_detections",
        )
        self.declare_parameter("taskboard_frame_name", "taskboard_frame")
        self.declare_parameter("desired_reference_frame", "fr3_link0")
        self.declare_parameter("confidence_threshold", 0.2)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("num_samples", 10)
        self.declare_parameter("rate", 10)
        self.declare_parameter("debug", False)
        self.declare_parameter("trigger_service_name", "rerun_pose_estimation")

        # Read parameters:
        self.model_path = self.get_parameter("model_path").value
        self.class_names_yaml = self.get_parameter("class_names_yaml").value
        self.model_points_yaml = self.get_parameter("model_points_yaml").value
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.object_poses_pub_topic = self.get_parameter(
            "object_poses_pub_topic"
        ).value
        self.object_marker_pub_topic = self.get_parameter(
            "object_marker_pub_topic"
        ).value
        self.detection_image_pub_topic = self.get_parameter(
            "detection_image_pub_topic"
        ).value
        self.taskboard_frame_name = self.get_parameter(
            "taskboard_frame_name"
        ).value
        self.desired_reference_frame = self.get_parameter(
            "desired_reference_frame"
        ).value
        self.conf_threshold = self.get_parameter("confidence_threshold").value
        self.iou_threshold = self.get_parameter("iou_threshold").value
        self.num_samples = self.get_parameter("num_samples").value
        self.rate = self.get_parameter("rate").value
        self.debug = self.get_parameter("debug").value
        self.trigger_service_name = self.get_parameter("trigger_service_name").value

        # ---- Subscribers ------------------------------------------------
        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._img_cb, 10
        )
        self.cam_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._cam_cb, 10
        )

        # ---- Publishers -------------------------------------------------
        self.obj_pose_pub = self.create_publisher(
            ObjectList, self.object_poses_pub_topic, 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, self.object_marker_pub_topic, 10
        )
        self.det_img_pub = self.create_publisher(
            Image, self.detection_image_pub_topic, 10
        )

        # ---- Services ---------------------------------------------------
        self.trigger_srv = self.create_service(
            Trigger, self.trigger_service_name, self._trigger_cb
        )

        # ---- Message buffers --------------------------------------------
        self.current_image_msg = None
        self.current_camera_info_msg = None

        # ---- Initialise subsystems --------------------------------------
        self._initialise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self):
        self.bridge = CvBridge()

        # TF:
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.transform_listener.TransformListener(
            self.tf_buffer, self
        )
        self.latest_tf_transforms = []
        self.latest_obj_list = None
        self.latest_marker_arr = None

        # YOLO detector:
        self.get_logger().info(f"Loading YOLO model from {self.model_path}")
        class_names = load_class_names_from_yaml(self.class_names_yaml)
        self.get_logger().info(f"YOLO classes: {class_names}")
        self.yolo = YOLODetector(
            model_path=self.model_path,
            class_names=class_names,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        self.get_logger().info("YOLO model loaded and warmed up.")

        # PnP estimator (will be initialised once camera info is available):
        self.pnp = None

        # Keypoint averaging buffers:
        # Each buffer stores up to num_samples (cx, cy) tuples
        self.keypoint_buffers = {
            key: collections.deque(maxlen=self.num_samples)
            for key in PNP_KEYPOINT_ORDER
        }
        # Buffer for secondary (non-PnP) detections:
        self.secondary_buffers = {}

        # RViz marker bookkeeping:
        self.object_marker_id = 0

        # Colour map for markers (use euROBIN→TUM label mapping):
        self.label_colors = {
            "blue_button":   (43, 160, 43),
            "red_button":    (255, 126, 14),
            "lcd":           (214, 38, 40),
            "slider":        (147, 103, 188),
            "hatch_handle":  (140, 86, 75),
            "red_hole":      (188, 188, 33),
            "taskboard":     (31, 119, 179),
        }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _trigger_cb(self, request, response):
        self.get_logger().info("Trigger received. Resetting pose estimation...")
        self.latest_tf_transforms = []
        self.latest_obj_list = None
        self.latest_marker_arr = None
        for buf in self.keypoint_buffers.values():
            buf.clear()
        for buf in self.secondary_buffers.values():
            buf.clear()
        response.success = True
        response.message = "Pose estimation reset."
        return response

    def _img_cb(self, msg):
        self.current_image_msg = msg

    def _cam_cb(self, msg):
        self.current_camera_info_msg = msg

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _build_camera_matrix(self):
        """Build 3×3 camera intrinsic matrix from CameraInfo."""
        ci = self.current_camera_info_msg
        return np.array(
            [
                [ci.p[0], 0.0, ci.p[2]],
                [0.0, ci.p[5], ci.p[6]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # RViz helpers
    # ------------------------------------------------------------------

    def _point_marker(self, point, frame_id, label="", color=(0, 0, 0)):
        """Create sphere + text markers for a 3D point."""
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = self.object_marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = Pose(
            position=Point(
                x=float(point[0]), y=float(point[1]), z=float(point[2])
            ),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        m.scale.x = m.scale.y = m.scale.z = 0.01
        m.color.r = color[0] / 255.0
        m.color.g = color[1] / 255.0
        m.color.b = color[2] / 255.0
        m.color.a = 1.0
        self.object_marker_id += 1

        t = copy.deepcopy(m)
        t.id = self.object_marker_id
        t.type = Marker.TEXT_VIEW_FACING
        t.text = label
        t.pose.position.x += 0.005 * (len(label) / 2.0)
        t.pose.position.y -= 0.01
        self.object_marker_id += 1

        return m, t

    def _clear_markers(self):
        ma = MarkerArray()
        m = Marker()
        m.id = self.object_marker_id
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self.marker_pub.publish(ma)
        self.object_marker_id = 0

    # ------------------------------------------------------------------
    # Keypoint averaging
    # ------------------------------------------------------------------

    def _update_keypoint_buffers(self, detections):
        """Add detected keypoint centres to the averaging buffers."""
        for det in detections:
            eurobin_name = det["class_name"]
            center = det["center"]  # (cx, cy) in pixels

            if eurobin_name in PNP_KEYPOINT_ORDER:
                self.keypoint_buffers[eurobin_name].append(center)
            else:
                tum_label = EUROBIN_TO_TUM_LABEL.get(eurobin_name, eurobin_name)
                if tum_label not in self.secondary_buffers:
                    self.secondary_buffers[tum_label] = collections.deque(
                        maxlen=self.num_samples
                    )
                self.secondary_buffers[tum_label].append(center)

    def _get_averaged_keypoints(self):
        """
        Return averaged 2D keypoints if all PnP buffers have
        enough samples.

        Returns
        -------
        keypoints : dict or None
            {label: (cx, cy)} or None if not enough samples.
        """
        for key in PNP_KEYPOINT_ORDER:
            if len(self.keypoint_buffers[key]) < self.num_samples:
                return None

        result = {}
        for key in PNP_KEYPOINT_ORDER:
            pts = np.array(list(self.keypoint_buffers[key]))
            result[key] = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

        return result

    def _get_averaged_secondary(self):
        """Return averaged 2D centres for secondary detections."""
        result = {}
        for label, buf in self.secondary_buffers.items():
            if len(buf) >= 3:  # need at least a few samples
                pts = np.array(list(buf))
                result[label] = (
                    float(pts[:, 0].mean()),
                    float(pts[:, 1].mean()),
                )
        return result

    # ------------------------------------------------------------------
    # TF transform helpers
    # ------------------------------------------------------------------

    def _transform_point_to_base(self, point_camera, camera_frame_id):
        """
        Transform a 3D point from camera frame to the desired
        reference frame using the TF tree.

        Returns (x, y, z) in reference frame or None on failure.
        """
        import tf2_geometry_msgs

        pt_msg = tf2_geometry_msgs.PointStamped(
            point=Point(
                x=float(point_camera[0]),
                y=float(point_camera[1]),
                z=float(point_camera[2]),
            )
        )
        pt_msg.header.frame_id = camera_frame_id
        pt_msg.header.stamp = rclpy.time.Time().to_msg()

        try:
            pt_transformed = self.tf_buffer.transform(
                pt_msg,
                self.desired_reference_frame
            )
            return np.array(
                [
                    pt_transformed.point.x,
                    pt_transformed.point.y,
                    pt_transformed.point.z,
                ]
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().error(f"TF transform error: {e}")
            return None

    def _transform_pose_to_base(self, rvec, tvec, camera_frame_id):
        """
        Transform the board pose (rvec, tvec in camera frame) to a
        PoseStamped in the desired reference frame.

        Returns (position_xyz, quaternion_xyzw) or (None, None).
        """
        import tf2_geometry_msgs
        import cv2 as cv

        R_board_in_cam, _ = cv.Rodrigues(rvec)
        t = tvec.flatten()

        # Build quaternion from rotation matrix:
        from tf_transformations import quaternion_from_matrix

        T = np.eye(4)
        T[:3, :3] = R_board_in_cam
        T[:3, 3] = t
        q = quaternion_from_matrix(T)  # (x, y, z, w)
        q = q / np.linalg.norm(q)

        pose_msg = tf2_geometry_msgs.PoseStamped(
            pose=Pose(
                position=Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                orientation=Quaternion(
                    x=float(q[0]),
                    y=float(q[1]),
                    z=float(q[2]),
                    w=float(q[3]),
                ),
            )
        )
        pose_msg.header.frame_id = camera_frame_id
        pose_msg.header.stamp = rclpy.time.Time().to_msg()

        try:
            pose_base = self.tf_buffer.transform(
                pose_msg,
                self.desired_reference_frame
            )
            pos = pose_base.pose.position
            ori = pose_base.pose.orientation
            return (
                np.array([pos.x, pos.y, pos.z]),
                np.array([ori.x, ori.y, ori.z, ori.w]),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().error(f"TF pose transform error: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_node(self):
        try:
            while rclpy.ok():
                rclpy.spin_once(self)

                # Continuously re-publish latest TFs and markers, then skip processing:
                if self.latest_tf_transforms:
                    now = self.get_clock().now().to_msg()
                    for tf_msg in self.latest_tf_transforms:
                        tf_msg.header.stamp = now
                    self.tf_broadcaster.sendTransform(
                        self.latest_tf_transforms
                    )
                    
                    if self.latest_obj_list:
                        for obj in self.latest_obj_list.objects:
                            obj.header.stamp = now
                        self.obj_pose_pub.publish(self.latest_obj_list)
                        
                    if self.latest_marker_arr:
                        for m in self.latest_marker_arr.markers:
                            m.header.stamp = now
                        self.marker_pub.publish(self.latest_marker_arr)

                    time.sleep(1.0 / self.rate)
                    continue

                if (
                    self.current_image_msg is None
                    or self.current_camera_info_msg is None
                ):
                    time.sleep(1.0 / self.rate)
                    continue

                # Initialise PnP estimator on first camera info:
                if self.pnp is None:
                    cam_matrix = self._build_camera_matrix()
                    self.pnp = PnPPoseEstimator(cam_matrix)
                    self.get_logger().info(
                        "PnP estimator initialised with camera intrinsics."
                    )

                # ---- YOLO detection ------------------------------------
                try:
                    img_bgr = self.bridge.imgmsg_to_cv2(
                        self.current_image_msg, "bgr8"
                    )
                except CvBridgeError as e:
                    self.get_logger().error(f"CvBridge error: {e}")
                    time.sleep(1.0 / self.rate)
                    continue

                t0 = time.time()
                detections = self.yolo.detect(img_bgr)
                yolo_time = time.time() - t0

                if self.debug:
                    self.get_logger().info(
                        f"YOLO: {len(detections)} detections in "
                        f"{yolo_time:.3f}s"
                    )
                    for d in detections:
                        self.get_logger().info(
                            f"  {d['class_name']}: conf={d['confidence']:.2f} "
                            f"center={d['center']}"
                        )

                # Publish annotated detection image for debugging:
                self._publish_detection_image(img_bgr, detections)

                # Update keypoint buffers:
                self._update_keypoint_buffers(detections)

                # Check if we have enough averaged keypoints for PnP:
                avg_keypoints = self._get_averaged_keypoints()
                if avg_keypoints is None:
                    filled = {
                        k: len(v) for k, v in self.keypoint_buffers.items()
                    }
                    self.get_logger().info(
                        f"Collecting keypoints: {filled} / {self.num_samples}"
                    )
                    time.sleep(1.0 / self.rate)
                    continue

                # ---- PnP pose estimation -------------------------------
                t0 = time.time()
                success, rvec, tvec, T_cam_board = self.pnp.estimate_pose(
                    avg_keypoints
                )
                pnp_time = time.time() - t0

                if not success:
                    self.get_logger().error("PnP failed!")
                    time.sleep(1.0 / self.rate)
                    continue

                if self.debug:
                    self.get_logger().info(
                        f"PnP: solved in {pnp_time:.3f}s, "
                        f"tvec={tvec.flatten()}"
                    )

                # ---- Compute component positions in camera frame -------
                camera_frame_id = (
                    self.current_camera_info_msg.header.frame_id
                )

                # Primary components (from model points):
                component_positions_board = {}
                for name, model_pt in PNP_MODEL_POINTS.items():
                    tum_label = EUROBIN_TO_TUM_LABEL.get(name, name)
                    component_positions_board[tum_label] = model_pt.copy()

                # Secondary components (project YOLO detections onto board plane):
                avg_secondary = self._get_averaged_secondary()
                for tum_label, pixel_uv in avg_secondary.items():
                    pt_cam, pt_board = self.pnp.project_point_to_board_plane(
                        pixel_uv, rvec, tvec
                    )
                    if pt_board is not None:
                        component_positions_board[tum_label] = pt_board

                # Taskboard frame is at blue_button (the PnP origin):
                component_positions_board["taskboard"] = np.array([0.0, 0.0, 0.0])

                # ---- Transform to robot base frame ---------------------
                board_pos_base, board_quat_base = self._transform_pose_to_base(
                    rvec, tvec, camera_frame_id
                )

                if board_pos_base is None:
                    self.get_logger().error(
                        "Could not transform board pose to base frame."
                    )
                    time.sleep(1.0 / self.rate)
                    continue
                    
                # Flatten orientation (assume table is perfectly flat):
                import tf_transformations
                roll, pitch, yaw = tf_transformations.euler_from_quaternion(board_quat_base)
                # The board is usually upside down in the robot frame (Z points down into table) 
                # or right-side up. We snap roll/pitch to exactly 180 or 0.
                import math
                if abs(roll) > math.pi / 2:
                    roll = math.pi  # or -math.pi
                else:
                    roll = 0.0
                pitch = 0.0
                flat_quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
                board_quat_base = flat_quat / np.linalg.norm(flat_quat)

                # Compute base positions using flattened orientation (guarantees identical Z height):
                flat_R = tf_transformations.quaternion_matrix(board_quat_base)[:3, :3]
                
                component_positions_base = {}
                for label, pt_board in component_positions_board.items():
                    pt_base = board_pos_base + (flat_R @ pt_board)
                    component_positions_base[label] = pt_base

                # ---- Build TF transforms and publish -------------------
                self._clear_markers()
                new_tfs = []
                marker_arr = MarkerArray()
                obj_list = ObjectList()

                board_quat_msg = Quaternion(
                    x=float(board_quat_base[0]),
                    y=float(board_quat_base[1]),
                    z=float(board_quat_base[2]),
                    w=float(board_quat_base[3]),
                )

                for label, pos_base in component_positions_base.items():
                    # TF:
                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = self.get_clock().now().to_msg()
                    tf_msg.header.frame_id = self.desired_reference_frame
                    tf_msg.child_frame_id = (
                        self.taskboard_frame_name
                        if label == "taskboard"
                        else label + "_frame"
                    )
                    tf_msg.transform.translation = Vector3(
                        x=float(pos_base[0]),
                        y=float(pos_base[1]),
                        z=float(pos_base[2]),
                    )
                    tf_msg.transform.rotation = board_quat_msg
                    new_tfs.append(tf_msg)

                    # Object message:
                    obj_msg = Object(
                        label=label,
                        pose=Pose(
                            position=Point(
                                x=float(pos_base[0]),
                                y=float(pos_base[1]),
                                z=float(pos_base[2]),
                            ),
                            orientation=board_quat_msg,
                        ),
                    )
                    obj_msg.header.stamp = self.get_clock().now().to_msg()
                    obj_msg.header.frame_id = self.desired_reference_frame
                    obj_list.objects.append(obj_msg)

                    # RViz marker:
                    color = self.label_colors.get(label, (128, 128, 128))
                    mm, tm = self._point_marker(
                        pos_base,
                        self.desired_reference_frame,
                        label,
                        color,
                    )
                    marker_arr.markers.append(mm)
                    marker_arr.markers.append(tm)

                # Publish and cache for continuous re-publishing:
                self.latest_tf_transforms = new_tfs
                self.latest_obj_list = obj_list
                self.latest_marker_arr = marker_arr
                
                now = self.get_clock().now().to_msg()
                for tf_m in self.latest_tf_transforms:
                    tf_m.header.stamp = now
                self.tf_broadcaster.sendTransform(self.latest_tf_transforms)

                self.obj_pose_pub.publish(self.latest_obj_list)
                self.marker_pub.publish(self.latest_marker_arr)

                n_frames = len(new_tfs)
                self.get_logger().info(
                    f"Published {n_frames} TF frames "
                    f"(YOLO: {yolo_time:.3f}s, PnP: {pnp_time:.3f}s)"
                )

                # (Buffers act as rolling windows since they are deques with maxlen)

                time.sleep(1.0 / self.rate)

        except KeyboardInterrupt:
            self.get_logger().info("Stopping node")
            raise SystemExit

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def _publish_detection_image(self, img_bgr, detections):
        """Annotate the image with YOLO detections and publish."""
        import cv2

        img_out = img_bgr.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            cx, cy = det["center"]
            label = det["class_name"]
            conf = det["confidence"]

            # Draw bounding box:
            cv2.rectangle(
                img_out, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            # Draw centre:
            cv2.circle(img_out, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            # Label:
            cv2.putText(
                img_out,
                f"{label} {conf:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        try:
            det_msg = self.bridge.cv2_to_imgmsg(img_out, "bgr8")
            det_msg.header = self.current_image_msg.header
            self.det_img_pub.publish(det_msg)
        except CvBridgeError:
            pass


# ======================================================================
# Entry point
# ======================================================================


def main(args=None):
    rclpy.init(args=args)
    node = YoloPnPPoseEstimatorNode()

    # Wait for camera info:
    node.get_logger().info(
        f"Waiting for camera info on {node.camera_info_topic}..."
    )
    try:
        while node.current_camera_info_msg is None:
            rclpy.spin_once(node)
            time.sleep(1.0 / node.rate)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info("Camera info received.")
    node.get_logger().info(
        f"Listening for images on {node.image_topic}"
    )

    try:
        node.run_node()
    except SystemExit:
        rclpy.logging.get_logger("rclpy").info("Stopping node...")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
