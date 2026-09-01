#!/home/brairlab/franka_ws/src/tum-tb-perception/.venv/bin/python3

"""
SAM-based pose estimator node for the taskboard.

Replaces the legacy minimum-bounding-rectangle orientation estimator with a
pipeline that uses:
  1. Existing CNN-based position estimation (unchanged, works well)
  2. MobileSAM segmentation of the taskboard from the RGB image
  3. Mask-edge–based orientation extraction
  4. Procrustes refinement using the known URDF model
  5. 180° disambiguation via component layout

Subscribes to the same topics as the original pose_estimator_node and
publishes on the same output topics, making it a drop-in replacement.
Additionally subscribes to the RGB image topic (required by SAM).
"""

import os
import sys
import time
import copy
import socket
import datetime

import rclpy
import tf2_ros
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import tf2_geometry_msgs

from rclpy.node import Node
from tf_transformations import quaternion_from_matrix
from launch_ros.substitutions import FindPackageShare
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Point, Point32, Pose, Quaternion, Vector3, TransformStamped
from sensor_msgs.msg import Image, PointCloud, PointCloud2, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from tum_tb_perception_msgs.msg import BoundingBoxList, ObjectList, Object

from tum_tb_perception.pose_estimation import TaskboardPoseEstimator
from tum_tb_perception.sam_segmentation import SAMSegmenter
from tum_tb_perception.mask_orientation import extract_orientation_from_mask
from tum_tb_perception.taskboard_model import (
    COMPONENT_POSITIONS_2D, resolve_180_ambiguity,
)
from tum_tb_perception.utils import bbox_list_msg_to_list, obj_list_msg_to_json
from tum_tb_perception.visualization import load_class_color_map
from tum_tb_perception.dataset import load_labels

# Fix occasional Qt visualisation issue on Ubuntu 22.04+:
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH')


class SAMPoseEstimatorNode(Node):

    def __init__(self):
        super().__init__('sam_pose_estimator')

        pkg = FindPackageShare(package='tum_tb_perception').find('tum_tb_perception')

        # ---- Parameters ----
        self.declare_parameter('class_colors_file_path',
                               os.path.join(pkg, 'config/class_colors_taskboard.yaml'))
        self.declare_parameter('output_dir_path', '/tmp')
        self.declare_parameter('taskboard_frame_name', 'taskboard_frame')
        self.declare_parameter('desired_reference_frame', 'base')
        self.declare_parameter('udp_ip', 'localhost')
        self.declare_parameter('udp_output_port', 6000)
        self.declare_parameter('pointcloud_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('detector_result_topic', '/tum_tb_perception/detection_result')
        self.declare_parameter('object_positions_pub_topic', '/tum_tb_perception/object_positions')
        self.declare_parameter('object_poses_pub_topic', '/tum_tb_perception/object_poses')
        self.declare_parameter('object_marker_pub_topic', '/tum_tb_perception/object_markers')
        self.declare_parameter('sam_mask_pub_topic', '/tum_tb_perception/sam_mask')
        self.declare_parameter('cropped_pc_pub_topic', '/tum_tb_perception/cropped_pc')
        self.declare_parameter('labels_file_path', 'config/labels.txt')
        self.declare_parameter('cropped_pc_label', 'taskboard')
        self.declare_parameter('sam_model_type', 'vit_t')
        self.declare_parameter('sam_checkpoint_path',
                               os.path.join(pkg, 'models/mobile_sam.pt'))
        self.declare_parameter('sam_device', 'cpu')
        self.declare_parameter('save_output', False)
        self.declare_parameter('rate', 10)
        self.declare_parameter('debug', False)
        self.declare_parameter('assume_flat_table', True)

        # Read parameters:
        self.class_colors_file_path = self.get_parameter('class_colors_file_path').value
        self.output_dir_path        = self.get_parameter('output_dir_path').value
        self.taskboard_frame_name   = self.get_parameter('taskboard_frame_name').value
        self.desired_reference_frame = self.get_parameter('desired_reference_frame').value
        self.udp_ip                 = self.get_parameter('udp_ip').value
        self.udp_output_port        = self.get_parameter('udp_output_port').value
        self.pointcloud_topic       = self.get_parameter('pointcloud_topic').value
        self.camera_info_topic      = self.get_parameter('camera_info_topic').value
        self.image_topic            = self.get_parameter('image_topic').value
        self.detector_result_topic  = self.get_parameter('detector_result_topic').value
        self.object_positions_pub_topic = self.get_parameter('object_positions_pub_topic').value
        self.object_poses_pub_topic = self.get_parameter('object_poses_pub_topic').value
        self.object_marker_pub_topic = self.get_parameter('object_marker_pub_topic').value
        self.sam_mask_pub_topic     = self.get_parameter('sam_mask_pub_topic').value
        self.cropped_pc_pub_topic   = self.get_parameter('cropped_pc_pub_topic').value
        self.labels_file_path       = self.get_parameter('labels_file_path').value
        self.cropped_pc_label       = self.get_parameter('cropped_pc_label').value
        self.sam_model_type         = self.get_parameter('sam_model_type').value
        self.sam_checkpoint_path    = self.get_parameter('sam_checkpoint_path').value
        self.sam_device             = self.get_parameter('sam_device').value
        self.save_output            = self.get_parameter('save_output').value
        self.rate                   = self.get_parameter('rate').value
        self.debug                  = self.get_parameter('debug').value
        self.assume_flat_table      = self.get_parameter('assume_flat_table').value

        # ---- Subscribers ----
        self.pc2_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self._pc2_cb, 10)
        self.det_sub = self.create_subscription(
            BoundingBoxList, self.detector_result_topic, self._det_cb, 10)
        self.cam_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._cam_cb, 10)
        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._img_cb, 10)

        # ---- Publishers ----
        self.obj_pos_pub  = self.create_publisher(ObjectList, self.object_positions_pub_topic, 10)
        self.obj_pose_pub = self.create_publisher(ObjectList, self.object_poses_pub_topic, 10)
        self.marker_pub   = self.create_publisher(MarkerArray, self.object_marker_pub_topic, 10)
        self.mask_pub     = self.create_publisher(Image, self.sam_mask_pub_topic, 10)
        if self.debug:
            self.crop_pub = self.create_publisher(PointCloud, self.cropped_pc_pub_topic, 10)

        # ---- Message buffers ----
        self.current_pc2_msg       = None
        self.current_detection_msg = None
        self.current_camera_info_msg = None
        self.current_image_msg     = None

        # ---- Initialise subsystems ----
        self._initialise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.transform_listener.TransformListener(
            self.tf_buffer, self)

        self.bridge = CvBridge()

        # UDP output:
        self.get_logger().info(
            f'UDP output: {self.udp_ip}:{self.udp_output_port}')
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Position estimator (reuse existing code – it works well):
        self.class_colors_dict = load_class_color_map(self.class_colors_file_path)
        self.position_estimator = TaskboardPoseEstimator(
            class_colors_dict=self.class_colors_dict)
        self.labels_list = load_labels(self.labels_file_path)

        # SAM segmenter:
        self.get_logger().info('Loading SAM model…')
        self.sam = SAMSegmenter(
            model_type=self.sam_model_type,
            checkpoint_path=self.sam_checkpoint_path,
            device=self.sam_device,
        )

        # Marker bookkeeping:
        self.object_marker_id = 0
        self.latest_tf_transforms = []

        # Output directory:
        if self.save_output:
            subdir = 'sam_pose_output_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir_path = os.path.join(self.output_dir_path, subdir)
            os.makedirs(self.output_dir_path, exist_ok=True)
            self.get_logger().info(f'Saving output to {self.output_dir_path}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _pc2_cb(self, msg):
        self.current_pc2_msg = msg

    def _det_cb(self, msg):
        self.current_detection_msg = msg

    def _cam_cb(self, msg):
        self.current_camera_info_msg = msg

    def _img_cb(self, msg):
        self.current_image_msg = msg

    # ------------------------------------------------------------------
    # RViz helpers (same interface as original node)
    # ------------------------------------------------------------------

    def _point_markers(self, point, frame_id, label='', color=(0., 0., 0.)):
        m = Marker()
        m.header.frame_id = frame_id
        m.id = self.object_marker_id
        m.type = 2  # Sphere
        m.action = 0
        m.pose = Pose(
            position=Point(x=float(point[0]), y=float(point[1]), z=float(point[2])),
            orientation=Quaternion(x=0., y=0., z=0., w=1.))
        m.scale.x = m.scale.y = m.scale.z = 0.01
        m.color.r, m.color.g, m.color.b = color[0]/255., color[1]/255., color[2]/255.
        m.color.a = 1.0
        self.object_marker_id += 1

        t = copy.deepcopy(m)
        t.id = self.object_marker_id
        t.type = 9  # Text
        t.text = label
        t.pose.position.x += 0.005 * (len(label) / 2.)
        t.pose.position.y -= 0.01
        self.object_marker_id += 1
        return m, t

    def _clear_markers(self):
        ma = MarkerArray()
        m = Marker()
        m.id = self.object_marker_id
        m.action = 3  # Delete all
        ma.markers.append(m)
        self.marker_pub.publish(ma)
        self.object_marker_id = 0

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _camera_params(self):
        ci = self.current_camera_info_msg
        return {'f_x': ci.p[0], 'f_y': ci.p[5],
                'c_x': ci.p[2], 'c_y': ci.p[6]}

    # ------------------------------------------------------------------
    # Core orientation pipeline
    # ------------------------------------------------------------------

    def _estimate_orientation(self, image_rgb, bbox_dict_list,
                              tb_points_array, object_positions_dict):
        """
        Full SAM-based orientation estimation pipeline.

        Returns
        -------
        orientation_matrix : ndarray (3, 3) or None
        """
        t0 = time.time()

        # --- 1. Plane fitting (same as original) -------------------------
        # Agglomerative outlier removal is O(N^3), so we must downsample first:
        if len(tb_points_array) > 300:
            indices = np.random.choice(len(tb_points_array), 300, replace=False)
            downsampled_pts = tb_points_array[indices]
        else:
            downsampled_pts = tb_points_array

        tb_pts = self.position_estimator.remove_outliers_agglomerative(
            downsampled_pts, debug=self.debug)

        eigvals, eigvecs = np.linalg.eigh(np.cov(tb_pts.T))
        plane_normal = eigvecs[:, 0].copy()
        if plane_normal[2] < 0:
            plane_normal = -plane_normal
        plane_point = tb_pts.mean(axis=0)

        # --- 2. SAM segmentation -----------------------------------------
        tb_bbox = None
        for bbox in bbox_dict_list:
            if bbox['class'] == 'taskboard':
                tb_bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
                break

        if tb_bbox is None:
            self.get_logger().error('No taskboard bounding box found – '
                                   'cannot run SAM segmentation.')
            return None

        self.get_logger().info('Running SAM segmentation on taskboard bbox…')
        sam_t0 = time.time()
        mask, score = self.sam.segment_from_bbox(image_rgb, tb_bbox)
        self.get_logger().info(
            f'SAM segmentation: score={score:.3f}, '
            f'time={time.time()-sam_t0:.2f}s')

        # Publish mask for debugging:
        try:
            mask_vis = (mask.astype(np.uint8) * 255)
            mask_msg = self.bridge.cv2_to_imgmsg(mask_vis, encoding='mono8')
            mask_msg.header.stamp = self.get_clock().now().to_msg()
            mask_msg.header.frame_id = self.current_camera_info_msg.header.frame_id
            self.mask_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().warn(f'Failed to publish SAM mask: {e}')

        # --- 3. Orientation from mask edges -------------------------------
        cam_params = self._camera_params()
        try:
            if self.debug:
                rotation_matrix, mask_info = extract_orientation_from_mask(
                    mask, cam_params, plane_normal, plane_point, debug=True)
                self.get_logger().info(
                    f'Mask edge lengths: {mask_info["edge_lengths"]}')
            else:
                rotation_matrix = extract_orientation_from_mask(
                    mask, cam_params, plane_normal, plane_point, debug=False)
        except RuntimeError as e:
            self.get_logger().error(
                f'Orientation extraction from mask failed: {e}')
            return None

        # --- 4. Disambiguate 180° using known component layout ------------
        rotation_matrix, flipped = resolve_180_ambiguity(
            rotation_matrix, object_positions_dict, plane_normal, plane_point)
        if flipped:
            self.get_logger().info('Resolved 180° ambiguity (flipped orientation).')

        # --- 5. Apply reorientation for frame convention ------------------
        # Match the convention used by the original node:
        reorientation = np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, -1]], dtype=float)
        orientation_matrix = reorientation @ rotation_matrix.T

        elapsed = time.time() - t0
        self.get_logger().info(
            f'Orientation estimation complete in {elapsed:.2f}s')

        return orientation_matrix

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_node(self):
        try:
            while rclpy.ok():
                rclpy.spin_once(self)

                # Continuously re-publish latest TFs:
                if self.latest_tf_transforms:
                    now = self.get_clock().now().to_msg()
                    for tf_msg in self.latest_tf_transforms:
                        tf_msg.header.stamp = now
                    self.tf_broadcaster.sendTransform(self.latest_tf_transforms)

                if self.current_detection_msg is None:
                    time.sleep(1.0 / self.rate)
                    continue

                self._clear_markers()

                if self.current_pc2_msg is None:
                    self.get_logger().warn(
                        f'No pointcloud on {self.pointcloud_topic}, skipping.')
                    self.current_detection_msg = None
                    continue

                if self.current_image_msg is None:
                    self.get_logger().warn(
                        f'No image on {self.image_topic}, skipping.')
                    self.current_detection_msg = None
                    continue

                # ==============================================================
                # Position Estimation  (reused from existing code)
                # ==============================================================

                self.get_logger().info('Estimating object positions…')
                pos_t0 = time.time()

                pc_points = pc2.read_points_list(
                    self.current_pc2_msg, skip_nans=True,
                    field_names=('x', 'y', 'z'))

                bbox_dict_list = bbox_list_msg_to_list(self.current_detection_msg)

                obj_pos, obj_pts, cropped_pc = \
                    self.position_estimator.estimate_object_positions(
                        bbox_dict_list, pc_points,
                        cropped_pc_label=self.cropped_pc_label,
                        debug=self.debug)

                if 'taskboard' not in obj_pts:
                    self.get_logger().error(
                        'Taskboard not found in positions – skipping.')
                    self.current_detection_msg = None
                    continue

                tb_points = obj_pts['taskboard']

                # Publish position markers:
                obj_list_msg = ObjectList()
                marker_arr = MarkerArray()

                try:
                    for label, pos in obj_pos.items():
                        pt_msg = tf2_geometry_msgs.PointStamped(
                            point=Point(x=float(pos[0]), y=float(pos[1]),
                                        z=float(pos[2])))
                        pt_msg.header.frame_id = \
                            self.current_camera_info_msg.header.frame_id
                        pt_msg = self.tf_buffer.transform(
                            pt_msg, self.desired_reference_frame,
                            rclpy.duration.Duration(seconds=1.0))

                        obj_msg = Object(
                            label=label,
                            pose=Pose(position=pt_msg.point,
                                      orientation=Quaternion(
                                          x=0., y=0., z=0., w=1.)))
                        obj_msg.header = pt_msg.header
                        obj_list_msg.objects.append(obj_msg)

                        p = [pt_msg.point.x, pt_msg.point.y, pt_msg.point.z]
                        mm, tm = self._point_markers(
                            p, self.desired_reference_frame, label,
                            self.class_colors_dict[label])
                        marker_arr.markers.append(mm)
                        marker_arr.markers.append(tm)

                except (tf2_ros.LookupException,
                        tf2_ros.ConnectivityException) as e:
                    self.get_logger().error(f'TF error (positions): {e}')
                    self.current_detection_msg = None
                    continue

                self.obj_pos_pub.publish(obj_list_msg)
                self.marker_pub.publish(marker_arr)

                self.get_logger().info(
                    f'Positions estimated in {time.time()-pos_t0:.2f}s')

                # ==============================================================
                # SAM-based Orientation Estimation
                # ==============================================================

                # Convert image message to numpy:
                try:
                    image_bgr = self.bridge.imgmsg_to_cv2(
                        self.current_image_msg, 'bgr8')
                    import cv2
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                except CvBridgeError as e:
                    self.get_logger().error(f'CvBridge error: {e}')
                    self.current_detection_msg = None
                    continue

                orientation_matrix = self._estimate_orientation(
                    image_rgb, bbox_dict_list, tb_points, obj_pos)

                orientation_quaternion = None
                if orientation_matrix is not None:
                    # Build 4×4 homogeneous matrix for quaternion conversion:
                    tf_mat = np.eye(4)
                    tf_mat[:3, :3] = orientation_matrix
                    orientation_quaternion = np.array(
                        quaternion_from_matrix(tf_mat))
                    orientation_quaternion /= np.linalg.norm(
                        orientation_quaternion)
                else:
                    self.get_logger().error(
                        'Orientation estimation failed. '
                        'Will not publish taskboard frame.')

                self.current_detection_msg = None

                # ==============================================================
                # Publish Poses & TF
                # ==============================================================

                if obj_list_msg is not None and orientation_quaternion is not None:
                    tb_quat = Quaternion(
                        x=float(orientation_quaternion[0]),
                        y=float(orientation_quaternion[1]),
                        z=float(orientation_quaternion[2]),
                        w=float(orientation_quaternion[3]))

                    # Transform TB pose to desired frame:
                    tb_pos = obj_pos['taskboard']
                    pose_msg = tf2_geometry_msgs.PoseStamped(
                        pose=Pose(
                            position=Point(x=float(tb_pos[0]),
                                           y=float(tb_pos[1]),
                                           z=float(tb_pos[2])),
                            orientation=tb_quat))
                    pose_msg.header.frame_id = \
                        self.current_camera_info_msg.header.frame_id
                    try:
                        pose_msg = self.tf_buffer.transform(
                            pose_msg, self.desired_reference_frame)
                    except (tf2_ros.LookupException,
                            tf2_ros.ConnectivityException) as e:
                        self.get_logger().error(
                            f'TF error (orientation): {e}')
                        continue

                    # Optionally flatten the orientation so XY plane aligns exactly with reference frame
                    if getattr(self, 'assume_flat_table', True):
                        import tf_transformations
                        import math
                        q = [pose_msg.pose.orientation.x,
                             pose_msg.pose.orientation.y,
                             pose_msg.pose.orientation.z,
                             pose_msg.pose.orientation.w]
                        roll, pitch, yaw = tf_transformations.euler_from_quaternion(q)
                        
                        # Snap roll and pitch to 0 or PI to flatten against the reference XY plane
                        def snap_angle(angle):
                            # Normalize angle to -pi..pi
                            angle = (angle + math.pi) % (2 * math.pi) - math.pi
                            return 0.0 if abs(angle) < math.pi / 2.0 else math.pi
                            
                        q_flat = tf_transformations.quaternion_from_euler(snap_angle(roll), snap_angle(pitch), yaw)
                        pose_msg.pose.orientation = Quaternion(
                            x=float(q_flat[0]), y=float(q_flat[1]), z=float(q_flat[2]), w=float(q_flat[3]))

                    tb_quat = pose_msg.pose.orientation

                    new_tfs = []

                    # Taskboard frame:
                    tf_msg = TransformStamped()
                    tf_msg.header = pose_msg.header
                    tf_msg.child_frame_id = self.taskboard_frame_name
                    tf_msg.transform.translation = Vector3(
                        x=pose_msg.pose.position.x,
                        y=pose_msg.pose.position.y,
                        z=pose_msg.pose.position.z)
                    tf_msg.transform.rotation = tb_quat
                    new_tfs.append(tf_msg)

                    # Per-object frames:
                    updated_obj_list = ObjectList()
                    self.object_marker_id = 0
                    for obj_msg in obj_list_msg.objects:
                        obj_msg.pose.orientation = tb_quat
                        updated_obj_list.objects.append(obj_msg)

                        tf_msg = TransformStamped()
                        tf_msg.header.stamp = self.get_clock().now().to_msg()
                        tf_msg.header.frame_id = self.desired_reference_frame
                        tf_msg.child_frame_id = obj_msg.label + '_frame'
                        tf_msg.transform.translation = Vector3(
                            x=float(obj_msg.pose.position.x),
                            y=float(obj_msg.pose.position.y),
                            z=float(obj_msg.pose.position.z))
                        tf_msg.transform.rotation = tb_quat
                        new_tfs.append(tf_msg)

                    self.latest_tf_transforms = new_tfs
                    now = self.get_clock().now().to_msg()
                    for tf_m in self.latest_tf_transforms:
                        tf_m.header.stamp = now
                    self.tf_broadcaster.sendTransform(self.latest_tf_transforms)
                    self.obj_pose_pub.publish(updated_obj_list)

                    # UDP output:
                    udp_msg = obj_list_msg_to_json(
                        updated_obj_list, orientation_success=True)
                    self.udp_socket.sendto(
                        udp_msg.encode(),
                        (self.udp_ip, self.udp_output_port))

                    self.get_logger().info(
                        f'Published {self.taskboard_frame_name} and '
                        f'{len(new_tfs)-1} object frames.')

                time.sleep(1.0 / self.rate)

        except KeyboardInterrupt:
            self.get_logger().info('Stopping node')
            raise SystemExit


# ======================================================================
# Entry point
# ======================================================================

def main(args=None):
    rclpy.init(args=args)
    node = SAMPoseEstimatorNode()

    # Wait for first camera info:
    node.get_logger().info(
        f'Waiting for camera info on {node.camera_info_topic}…')
    try:
        while node.current_camera_info_msg is None:
            rclpy.spin_once(node)
            time.sleep(1.0 / node.rate)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info('Camera info received.')
    node.position_estimator.load_camera_params(node._camera_params())

    node.get_logger().info(
        f'Listening for detections on {node.detector_result_topic}')

    try:
        node.run_node()
    except SystemExit:
        rclpy.logging.get_logger('rclpy').info('Stopping node…')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
