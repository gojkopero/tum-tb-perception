#!/usr/bin/env python3

"""
Launch file for the SAM-based pose estimator node.
Drop-in replacement for pose_estimator.launch.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare(package='tum_tb_perception')

    args = [
        DeclareLaunchArgument('class_colors_file_path',
            default_value=PathJoinSubstitution([pkg, 'config', 'class_colors_taskboard.yaml']),
            description='Path to class-colour YAML file'),
        DeclareLaunchArgument('output_dir_path', default_value='/tmp',
            description='Directory for debug output'),
        DeclareLaunchArgument('taskboard_frame_name', default_value='taskboard_frame',
            description='Name of the published taskboard TF frame'),
        DeclareLaunchArgument('desired_reference_frame', default_value='base',
            description='Target TF frame to express poses in'),
        DeclareLaunchArgument('udp_ip', default_value='localhost',
            description='UDP output IP address'),
        DeclareLaunchArgument('udp_output_port', default_value='6000',
            description='UDP output port'),
        DeclareLaunchArgument('pointcloud_topic',
            default_value='/camera/camera/depth/color/points',
            description='PointCloud2 input topic'),
        DeclareLaunchArgument('camera_info_topic',
            default_value='/camera/camera/color/camera_info',
            description='CameraInfo input topic'),
        DeclareLaunchArgument('image_topic',
            default_value='/camera/camera/color/image_raw',
            description='RGB Image input topic (required by SAM)'),
        DeclareLaunchArgument('detector_result_topic',
            default_value='/tum_tb_perception/detection_result',
            description='CNN detection BoundingBoxList input topic'),
        DeclareLaunchArgument('object_positions_pub_topic',
            default_value='/tum_tb_perception/object_positions',
            description='ObjectList positions output topic'),
        DeclareLaunchArgument('object_poses_pub_topic',
            default_value='/tum_tb_perception/object_poses',
            description='ObjectList poses output topic'),
        DeclareLaunchArgument('object_marker_pub_topic',
            default_value='/tum_tb_perception/object_markers',
            description='RViz MarkerArray output topic'),
        DeclareLaunchArgument('sam_mask_pub_topic',
            default_value='/tum_tb_perception/sam_mask',
            description='Debug: SAM mask image output topic'),
        DeclareLaunchArgument('cropped_pc_pub_topic',
            default_value='/tum_tb_perception/cropped_pc',
            description='Debug: cropped PointCloud output topic'),
        DeclareLaunchArgument('labels_file_path',
            default_value=PathJoinSubstitution([pkg, 'config', 'labels.txt']),
            description='Path to class labels file'),
        DeclareLaunchArgument('cropped_pc_label', default_value='taskboard',
            description='Label for which to publish cropped pointcloud'),
        DeclareLaunchArgument('sam_model_type', default_value='vit_t',
            description='SAM model type (vit_t=MobileSAM, vit_b=SAM-B, vit_h=SAM-H)'),
        DeclareLaunchArgument('sam_checkpoint_path',
            default_value=PathJoinSubstitution([pkg, 'models', 'mobile_sam.pt']),
            description='Path to SAM/MobileSAM checkpoint file'),
        DeclareLaunchArgument('sam_device', default_value='cpu',
            description='Torch device for SAM inference (cpu or cuda)'),
        DeclareLaunchArgument('save_output', default_value='False',
            description='Save debug output to disk'),
        DeclareLaunchArgument('rate', default_value='10',
            description='Node loop rate in Hz'),
        DeclareLaunchArgument('debug', default_value='False',
            description='Enable debug logging and visualisations'),
        DeclareLaunchArgument('assume_flat_table', default_value='True',
            description='Force taskboard XY plane to be parallel with the reference frame'),
    ]

    node = Node(
        package='tum_tb_perception',
        namespace='tum_tb_perception',
        executable='sam_pose_estimator_node.py',
        name='sam_pose_estimator',
        parameters=[
            {'class_colors_file_path':    LaunchConfiguration('class_colors_file_path')},
            {'output_dir_path':           LaunchConfiguration('output_dir_path')},
            {'taskboard_frame_name':      LaunchConfiguration('taskboard_frame_name')},
            {'desired_reference_frame':   LaunchConfiguration('desired_reference_frame')},
            {'udp_ip':                    LaunchConfiguration('udp_ip')},
            {'udp_output_port':           LaunchConfiguration('udp_output_port')},
            {'pointcloud_topic':          LaunchConfiguration('pointcloud_topic')},
            {'camera_info_topic':         LaunchConfiguration('camera_info_topic')},
            {'image_topic':               LaunchConfiguration('image_topic')},
            {'detector_result_topic':     LaunchConfiguration('detector_result_topic')},
            {'object_positions_pub_topic': LaunchConfiguration('object_positions_pub_topic')},
            {'object_poses_pub_topic':    LaunchConfiguration('object_poses_pub_topic')},
            {'object_marker_pub_topic':   LaunchConfiguration('object_marker_pub_topic')},
            {'sam_mask_pub_topic':        LaunchConfiguration('sam_mask_pub_topic')},
            {'cropped_pc_pub_topic':      LaunchConfiguration('cropped_pc_pub_topic')},
            {'labels_file_path':          LaunchConfiguration('labels_file_path')},
            {'cropped_pc_label':          LaunchConfiguration('cropped_pc_label')},
            {'sam_model_type':            LaunchConfiguration('sam_model_type')},
            {'sam_checkpoint_path':       LaunchConfiguration('sam_checkpoint_path')},
            {'sam_device':                LaunchConfiguration('sam_device')},
            {'save_output':               LaunchConfiguration('save_output')},
            {'rate':                      LaunchConfiguration('rate')},
            {'debug':                     LaunchConfiguration('debug')},
            {'assume_flat_table':         LaunchConfiguration('assume_flat_table')},
        ],
    )

    return LaunchDescription(args + [node])
