#!/usr/bin/env python3

"""
Launch file for the YOLO + PnP pose estimator node.

This node uses the euROBIN vision pipeline approach:
  1. YOLOv8 ONNX detection for keypoint extraction
  2. OpenCV solvePnP for 6-DOF board pose estimation
  3. No pointcloud or separate CNN detector required
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare(package="tum_tb_perception")

    args = [
        DeclareLaunchArgument(
            "model_path",
            default_value=PathJoinSubstitution(
                [pkg, "models", "best_taskboard.onnx"]
            ),
            description="Path to the YOLOv8 ONNX model file",
        ),
        DeclareLaunchArgument(
            "class_names_yaml",
            default_value=PathJoinSubstitution(
                [pkg, "config", "yolo_classes_taskboard.yaml"]
            ),
            description="Path to YOLO class names YAML file",
        ),
        DeclareLaunchArgument(
            "model_points_yaml",
            default_value=PathJoinSubstitution(
                [pkg, "config", "model_points_taskboard.yaml"]
            ),
            description="Path to PnP 3D model points YAML file",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/camera/color/image_raw",
            description="RGB Image input topic",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/color/camera_info",
            description="CameraInfo input topic",
        ),
        DeclareLaunchArgument(
            "pointcloud_topic",
            default_value="/camera/camera/depth/color/points",
            description="PointCloud2 input topic (for exact Z heights)",
        ),
        DeclareLaunchArgument(
            "object_poses_pub_topic",
            default_value="/tum_tb_perception/object_poses",
            description="ObjectList poses output topic",
        ),
        DeclareLaunchArgument(
            "object_marker_pub_topic",
            default_value="/tum_tb_perception/object_markers",
            description="RViz MarkerArray output topic",
        ),
        DeclareLaunchArgument(
            "detection_image_pub_topic",
            default_value="/tum_tb_perception/yolo_detections",
            description="Annotated detection image output topic",
        ),
        DeclareLaunchArgument(
            "taskboard_frame_name",
            default_value="taskboard_frame",
            description="Name of the published taskboard TF frame",
        ),
        DeclareLaunchArgument(
            "desired_reference_frame",
            default_value="fr3_link0",
            description="Target TF frame to express poses in",
        ),
        DeclareLaunchArgument(
            "confidence_threshold",
            default_value="0.2",
            description="YOLO detection confidence threshold",
        ),
        DeclareLaunchArgument(
            "iou_threshold",
            default_value="0.5",
            description="YOLO NMS IoU threshold",
        ),
        DeclareLaunchArgument(
            "num_samples",
            default_value="10",
            description="Number of frames to average keypoints over",
        ),
        DeclareLaunchArgument(
            "rate",
            default_value="10",
            description="Node loop rate in Hz",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="False",
            description="Enable debug logging",
        ),
    ]

    node = Node(
        package="tum_tb_perception",
        namespace="tum_tb_perception",
        executable="yolo_pnp_pose_estimator_node.py",
        name="yolo_pnp_pose_estimator",
        parameters=[
            {"model_path": LaunchConfiguration("model_path")},
            {"class_names_yaml": LaunchConfiguration("class_names_yaml")},
            {"model_points_yaml": LaunchConfiguration("model_points_yaml")},
            {"image_topic": LaunchConfiguration("image_topic")},
            {"camera_info_topic": LaunchConfiguration("camera_info_topic")},
            {"pointcloud_topic": LaunchConfiguration("pointcloud_topic")},
            {
                "object_poses_pub_topic": LaunchConfiguration(
                    "object_poses_pub_topic"
                )
            },
            {
                "object_marker_pub_topic": LaunchConfiguration(
                    "object_marker_pub_topic"
                )
            },
            {
                "detection_image_pub_topic": LaunchConfiguration(
                    "detection_image_pub_topic"
                )
            },
            {
                "taskboard_frame_name": LaunchConfiguration(
                    "taskboard_frame_name"
                )
            },
            {
                "desired_reference_frame": LaunchConfiguration(
                    "desired_reference_frame"
                )
            },
            {
                "confidence_threshold": LaunchConfiguration(
                    "confidence_threshold"
                )
            },
            {"iou_threshold": LaunchConfiguration("iou_threshold")},
            {"num_samples": LaunchConfiguration("num_samples")},
            {"rate": LaunchConfiguration("rate")},
            {"debug": LaunchConfiguration("debug")},
        ],
    )

    return LaunchDescription(args + [node])
