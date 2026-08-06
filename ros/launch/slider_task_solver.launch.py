#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    template_source_image_dir_path_launch_arg = DeclareLaunchArgument(
        'template_source_image_dir_path',
        default_value=PathJoinSubstitution([
            FindPackageShare(package='tum_tb_perception'),
            'models', 'slider_solver_templates_images'
        ]),
        description='Path to directory containing slider marker template images.'
    )
    output_dir_path_launch_arg = DeclareLaunchArgument(
        'output_dir_path',
        default_value='/tmp',
        description='Path to output directory for saving results.'
    )
    task_stage_launch_arg = DeclareLaunchArgument(
        'task_stage',
        default_value='1',
        description='Slider task stage: 1 = red to white center, 2 = red to green.'
    )
    detection_score_threshold_launch_arg = DeclareLaunchArgument(
        'detection_score_threshold',
        default_value='0.7',
        description='Template matching score threshold.'
    )
    run_on_ros_trigger_launch_arg = DeclareLaunchArgument(
        'run_on_ros_trigger',
        default_value='True',
        description='If True, run solver only on ROS trigger messages.'
    )
    run_on_udp_trigger_launch_arg = DeclareLaunchArgument(
        'run_on_udp_trigger',
        default_value='False',
        description='If True, run solver only on UDP trigger messages.'
    )
    udp_ip_launch_arg = DeclareLaunchArgument(
        'udp_ip',
        default_value='localhost',
        description='UDP IP address for trigger/output messages.'
    )
    udp_trigger_port_launch_arg = DeclareLaunchArgument(
        'udp_trigger_port',
        default_value='7000',
        description='UDP port for receiving trigger messages.'
    )
    udp_output_port_launch_arg = DeclareLaunchArgument(
        'udp_output_port',
        default_value='8000',
        description='UDP port for sending output messages.'
    )
    image_topic_launch_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/camera/color/image_raw',
        description='ROS topic for input camera images.'
    )
    trigger_topic_launch_arg = DeclareLaunchArgument(
        'trigger_topic',
        default_value='/tum_tb_perception/slider_solver_trigger',
        description='ROS topic for trigger messages.'
    )
    image_pub_topic_launch_arg = DeclareLaunchArgument(
        'image_pub_topic',
        default_value='/tum_tb_perception/slider_solver_images',
        description='ROS topic for publishing annotated output images.'
    )
    input_image_pub_topic_launch_arg = DeclareLaunchArgument(
        'input_image_pub_topic',
        default_value='/tum_tb_perception/slider_solver_input_images',
        description='ROS topic for publishing input images.'
    )
    slider_distance_pub_topic_launch_arg = DeclareLaunchArgument(
        'slider_distance_pub_topic',
        default_value='/tum_tb_perception/slider_solver_result',
        description='ROS topic for publishing estimated slider motion distance.'
    )
    publish_visual_output_launch_arg = DeclareLaunchArgument(
        'publish_visual_output',
        default_value='True',
        description='If True, publish annotated images for visualization.'
    )
    save_output_launch_arg = DeclareLaunchArgument(
        'save_output',
        default_value='False',
        description='If True, save input and output images to disk.'
    )
    rate_launch_arg = DeclareLaunchArgument(
        'rate',
        default_value='10',
        description='Processing rate (Hz).'
    )
    debug_launch_arg = DeclareLaunchArgument(
        'debug',
        default_value='False',
        description='If True, print additional debug messages.'
    )

    slider_task_solver_node = Node(
        package='tum_tb_perception',
        namespace='tum_tb_perception',
        executable='slider_task_solver_node.py',
        name='slider_task_solver_node',
        parameters=[
            {'template_source_image_dir_path': LaunchConfiguration('template_source_image_dir_path')},
            {'output_dir_path': LaunchConfiguration('output_dir_path')},
            {'task_stage': LaunchConfiguration('task_stage')},
            {'detection_score_threshold': LaunchConfiguration('detection_score_threshold')},
            {'run_on_ros_trigger': LaunchConfiguration('run_on_ros_trigger')},
            {'run_on_udp_trigger': LaunchConfiguration('run_on_udp_trigger')},
            {'udp_ip': LaunchConfiguration('udp_ip')},
            {'udp_trigger_port': LaunchConfiguration('udp_trigger_port')},
            {'udp_output_port': LaunchConfiguration('udp_output_port')},
            {'image_topic': LaunchConfiguration('image_topic')},
            {'trigger_topic': LaunchConfiguration('trigger_topic')},
            {'image_pub_topic': LaunchConfiguration('image_pub_topic')},
            {'input_image_pub_topic': LaunchConfiguration('input_image_pub_topic')},
            {'slider_distance_pub_topic': LaunchConfiguration('slider_distance_pub_topic')},
            {'publish_visual_output': LaunchConfiguration('publish_visual_output')},
            {'save_output': LaunchConfiguration('save_output')},
            {'rate': LaunchConfiguration('rate')},
            {'debug': LaunchConfiguration('debug')},
        ],
    )

    return LaunchDescription([
        template_source_image_dir_path_launch_arg,
        output_dir_path_launch_arg,
        task_stage_launch_arg,
        detection_score_threshold_launch_arg,
        run_on_ros_trigger_launch_arg,
        run_on_udp_trigger_launch_arg,
        udp_ip_launch_arg,
        udp_trigger_port_launch_arg,
        udp_output_port_launch_arg,
        image_topic_launch_arg,
        trigger_topic_launch_arg,
        image_pub_topic_launch_arg,
        input_image_pub_topic_launch_arg,
        slider_distance_pub_topic_launch_arg,
        publish_visual_output_launch_arg,
        save_output_launch_arg,
        rate_launch_arg,
        debug_launch_arg,
        slider_task_solver_node,
    ])
