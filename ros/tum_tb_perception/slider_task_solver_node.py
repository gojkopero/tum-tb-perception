#!/usr/bin/env python3
"""
Estimates the distance by which the slider must be moved to solve
the taskboard slider task by determining the positions of triangle markers
on the LCD.
Runs continuously, processing every incoming image.

Optionally runs the solver only after a ROS/UDP trigger message.
Optionally saves all input and output (annotated with results) images.

Ported from the ROS1 (noetic) version to ROS2 (Jazzy).
"""

import os
import sys
import json
import time
import socket
import signal
import datetime

import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from launch_ros.substitutions import FindPackageShare

from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Image

# --- Constants ---------------------------------------------------------------

cv2_text_label_font_ = cv2.FONT_HERSHEY_SIMPLEX
cv2_text_label_font_scale_ = 0.35
template_matching_method_ = cv2.TM_CCOEFF_NORMED
text_label_colors_ = dict(zip(
    ['red', 'white_center', 'green', 'lcd'],
    [(0, 0, 255), (255, 255, 255), (0, 255, 0), (255, 0, 0)]
))

# Task stage options:
#   1: moving red triangle to center white triangle
#   2: moving red triangle to green triangle
task_stage_options_ = [1, 2]

udp_buffer_size_ = 1024

# --- Helper Functions (pure OpenCV, no ROS dependency) -----------------------

def load_template_images(image_dir_path,
                         object_ids=['red', 'white_center', 'green', 'lcd'],
                         logger=None, debug=False):
    """
    Loads all images that can be used as templates for the given object_ids
    from the image_dir_path directory.

    Note: expects all images within image_dir_path whose filename contains the
    substring "template" to be candidates for extracting the templates.

    Parameters
    ----------
    image_dir_path: str
        Path to directory containing candidate template images.
    object_ids: list
        Template object ID strings (included in filenames).
    logger: rclpy logger or None
        Logger for debug messages.
    debug: bool
        Whether to print some debugging messages.

    Returns
    -------
    template_images_dict: dict
        Mapping between object IDs and lists of ndarrays containing loaded images.
    """
    template_images_dict = {}
    image_filename_list = [filename for filename in os.listdir(image_dir_path)
                           if 'template' in filename]

    for object_id in object_ids:
        if object_id not in template_images_dict.keys():
            template_images_dict[object_id] = []
        for image_filename in image_filename_list:
            if object_id in image_filename:
                if debug and logger:
                    logger.info(f'[DEBUG] Using image for {object_id}: '
                                f'{image_filename}')
                image_path = os.path.join(image_dir_path, image_filename)
                image_array = cv2.imread(image_path)  # Note: in BGR format
                template_images_dict[object_id].append(image_array)
        if not template_images_dict[object_id]:
            raise AssertionError(f'Unable to extract a single template'
                                 f' for object {object_id}')

    return template_images_dict


def lcd_marker_to_slider_distance(estimated_pixel_distance):
    """
    Transforms the image-detected distance between two LCD markers to an
    estimate of the distance required to move the slider to align the markers.

    Parameters
    ----------
    estimated_pixel_distance: float
        Estimated distance between source and target LCD markers (pixels)

    Returns
    -------
    estimated_slider_distance: float
        Estimated distance to move the slider to align the LCD markers
    """
    multiplicative_constant = 0.14
    estimated_slider_distance = estimated_pixel_distance * multiplicative_constant
    return estimated_slider_distance


def check_task_stage_value(task_stage, logger=None):
    """
    Checks the validity of the given task_stage value.

    Parameters
    ----------
    task_stage: int
        Represents the current slider task stage:
          - 1: red marker to the center white marker
          - 2: red marker to the green marker
    logger: rclpy logger or None

    Returns
    -------
    valid: bool
        Whether the task stage value is valid
    """
    if task_stage in task_stage_options_:
        if logger:
            if task_stage == 1:
                logger.info('Running for task stage 1: '
                            'moving the red marker to the center white marker.')
            elif task_stage == 2:
                logger.info('Running for task stage 2: '
                            'moving the red marker to the green marker.')
    else:
        if logger:
            logger.warn(f'Invalid value for task stage parameter! '
                        f'Must be one of {task_stage_options_}')
        return False

    return True


def set_task_stage(task_stage, template_images_dict):
    """
    Sets parameters according to slider task stage:
      - Set initial and goal triangles' IDs
      - Ignore specific templates

    Parameters
    ----------
    task_stage: int
        Represents the current slider task stage.
    template_images_dict: dict
        Mapping between object IDs and lists of ndarrays containing template images.

    Returns
    -------
    template_images_dict: dict
        Mapping between object IDs and lists of ndarrays containing template images.
    initial_point_id: str
        ID of "initial" marker: ['red', 'white_center', 'green']
    goal_point_id: str
        ID of "goal" marker: ['red', 'white_center', 'green']
    """
    if task_stage == 1:
        initial_point_id, goal_point_id = 'red', 'white_center'
        template_images_dict.pop('green', None)
    elif task_stage == 2:
        initial_point_id, goal_point_id = 'red', 'green'
        template_images_dict.pop('white_center', None)

    return template_images_dict, initial_point_id, goal_point_id


# --- ROS2 Node ---------------------------------------------------------------

class SliderTaskSolverNode(Node):

    def __init__(self):
        super().__init__('slider_task_solver_node')

        self.pkg_share_path = FindPackageShare(
            package='tum_tb_perception'
        ).find('tum_tb_perception')

        # Declare parameters:
        self.declare_parameter(
            'template_source_image_dir_path',
            os.path.join(self.pkg_share_path,
                         'models/slider_solver_templates_images'))
        self.declare_parameter('output_dir_path', '/tmp')
        self.declare_parameter('task_stage', 1)
        self.declare_parameter('detection_score_threshold', 0.7)
        self.declare_parameter('run_on_ros_trigger', True)
        self.declare_parameter('run_on_udp_trigger', False)
        self.declare_parameter('udp_ip', 'localhost')
        self.declare_parameter('udp_trigger_port', 7000)
        self.declare_parameter('udp_output_port', 8000)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('trigger_topic',
                               '/tum_tb_perception/slider_solver_trigger')
        self.declare_parameter('image_pub_topic',
                               '/tum_tb_perception/slider_solver_images')
        self.declare_parameter('input_image_pub_topic',
                               '/tum_tb_perception/slider_solver_input_images')
        self.declare_parameter('slider_distance_pub_topic',
                               '/tum_tb_perception/slider_solver_result')
        self.declare_parameter('publish_visual_output', True)
        self.declare_parameter('save_output', False)
        self.declare_parameter('rate', 10)
        self.declare_parameter('debug', False)

        # Read parameters:
        self.template_source_image_dir_path = self.get_parameter(
            'template_source_image_dir_path').value
        self.output_dir_path = self.get_parameter('output_dir_path').value
        self.task_stage = self.get_parameter('task_stage').value
        self.detection_score_threshold = self.get_parameter(
            'detection_score_threshold').value
        self.run_on_ros_trigger = self.get_parameter('run_on_ros_trigger').value
        self.run_on_udp_trigger = self.get_parameter('run_on_udp_trigger').value
        self.udp_ip = self.get_parameter('udp_ip').value
        self.udp_trigger_port = self.get_parameter('udp_trigger_port').value
        self.udp_output_port = self.get_parameter('udp_output_port').value
        self.image_topic = self.get_parameter('image_topic').value
        self.trigger_service = self.get_parameter('trigger_service').value
        self.image_pub_topic = self.get_parameter('image_pub_topic').value
        self.input_image_pub_topic = self.get_parameter(
            'input_image_pub_topic').value
        self.slider_distance_pub_topic = self.get_parameter(
            'slider_distance_pub_topic').value
        self.publish_visual_output = self.get_parameter(
            'publish_visual_output').value
        self.save_output = self.get_parameter('save_output').value
        self.rate = self.get_parameter('rate').value
        self.debug = self.get_parameter('debug').value

        # Initialize subscribers:
        self.image_subscription = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)

        if self.run_on_ros_trigger:
            self.trigger_subscription = self.create_subscription(
                Bool, self.trigger_topic, self.trigger_callback, 10)

        # Initialize publishers:
        self.slider_distance_publisher = self.create_publisher(
            Float32, self.slider_distance_pub_topic, 10)

        if self.publish_visual_output:
            self.slider_solver_image_publisher = self.create_publisher(
                Image, self.image_pub_topic, 10)
            self.input_image_publisher = self.create_publisher(
                Image, self.input_image_pub_topic, 10)

        # Initialize state:
        self.current_image_msg = None
        self.ros_triggered = False
        self.bridge = CvBridge()

        # Initialize data:
        self.initialize()

    def initialize(self):
        """Initialize solver: validate settings, load templates, set up output."""
        # Verify trigger setting (ROS OR UDP):
        if self.run_on_ros_trigger and self.run_on_udp_trigger:
            self.get_logger().error(
                'Node supports trigger messages from either ROS or UDP, '
                'but run_on_ros_trigger and run_on_udp_trigger were both '
                'set to true!')
            self.get_logger().error('Terminating.')
            raise SystemExit

        # Validate task stage:
        if not check_task_stage_value(self.task_stage, self.get_logger()):
            self.get_logger().error(
                f'Invalid initial value for task stage parameter! '
                f'Must be one of {task_stage_options_}')

        # Set up output data directory:
        if self.save_output:
            output_sub_dir_path = ('slider_solver_output_' +
                                   datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.output_dir_path = os.path.join(self.output_dir_path,
                                                output_sub_dir_path)
            self.get_logger().info(
                f'Saving output data in {self.output_dir_path}')
            if not os.path.isdir(self.output_dir_path):
                self.get_logger().info('Output directory does not exist! '
                                       'Creating now...')
                os.makedirs(self.output_dir_path)

        # Load marker template images:
        self.get_logger().info('Loading marker templates...')
        marker_loading_start_time = time.time()

        self.orig_template_images_dict = load_template_images(
            self.template_source_image_dir_path,
            object_ids=['red', 'white_center', 'green'],
            logger=self.get_logger(),
            debug=self.debug
        )

        self.get_logger().info('Found markers:')
        for template_id, result_list in self.orig_template_images_dict.items():
            self.get_logger().info(f'  - {template_id}: {len(result_list)}')

        # Set parameters according to slider task stage:
        self.template_images_dict = dict(self.orig_template_images_dict)
        self.template_images_dict, self.initial_point_id, self.goal_point_id = \
            set_task_stage(self.task_stage, self.template_images_dict)

        elapsed_time = time.time() - marker_loading_start_time
        self.get_logger().info(f'Template loading finished in {elapsed_time:.3f}s')

        # UDP output socket:
        self.get_logger().info(
            'Initializing UDP socket with address family AF_INET and type SOCK_DGRAM')
        self.get_logger().info(
            f'Will send output messages over IP {self.udp_ip} '
            f'and port {self.udp_output_port}.')
        self.udp_output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def image_callback(self, msg):
        self.current_image_msg = msg

    def trigger_callback(self, msg):
        self.get_logger().info('Received trigger ROS message')
        self.ros_triggered = msg.data

    def run_node(self):
        """Main processing loop."""
        self.get_logger().info(f'Subscribing to topic: {self.image_topic}')
        self.get_logger().info('Waiting for reception of first message...')

        try:
            while self.current_image_msg is None:
                rclpy.spin_once(self)
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.get_logger().info('Terminating...')
            return

        self.get_logger().info('Received first image message')

        if self.run_on_ros_trigger:
            self.get_logger().info(
                f'Will estimate solution distance on the latest image '
                f'message at every trigger ROS message on topic '
                f'{self.trigger_topic}...')
        elif self.run_on_udp_trigger:
            self.get_logger().info(
                f'Will estimate solution distance on the latest image '
                f'message at every trigger UDP message over IP {self.udp_ip} '
                f'and port {self.udp_trigger_port}...')

            udp_trigger_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_trigger_socket.settimeout(None)
            udp_trigger_socket.bind((self.udp_ip, self.udp_trigger_port))
        else:
            self.get_logger().info(
                'Will estimate solution distance for every incoming '
                'image message...')

        try:
            while rclpy.ok():
                if self.run_on_ros_trigger:
                    signal.signal(signal.SIGINT, signal.SIG_DFL)
                    while not self.ros_triggered:
                        rclpy.spin_once(self)
                        time.sleep(float(1. / self.rate))
                    self.ros_triggered = False
                elif self.run_on_udp_trigger:
                    signal.signal(signal.SIGINT, signal.SIG_DFL)
                    udp_msg, udp_addr = udp_trigger_socket.recvfrom(
                        udp_buffer_size_)

                    self.get_logger().info('Received trigger UDP message')
                    udp_msg_data = json.loads(udp_msg.decode())

                    try:
                        if (type(udp_msg_data['trigger']).__name__ != 'str' or
                                udp_msg_data['trigger'] != "True"):
                            self.get_logger().warn(
                                'Received invalid value in UDP message dict '
                                f'for key trigger: {udp_msg_data["trigger"]}. '
                                'Will only trigger on "True". Ignoring...')
                            continue
                    except Exception:
                        self.get_logger().warn(
                            'Could not access trigger information in UDP '
                            'message! Please check message format! Ignoring...')
                        continue

                    # Optionally update task_stage from UDP message:
                    if 'task_stage' in udp_msg_data.keys():
                        if type(udp_msg_data['task_stage']).__name__ != 'str':
                            self.get_logger().warn(
                                'Received invalid value in UDP message dict '
                                'for task_stage. Must be str. Ignoring...')
                            continue
                        try:
                            input_task_stage_value = int(
                                udp_msg_data['task_stage'])
                        except ValueError:
                            self.get_logger().warn(
                                f'Could not cast given task_stage to int: '
                                f'{udp_msg_data["task_stage"]}! Ignoring...')
                            continue

                        if not check_task_stage_value(input_task_stage_value,
                                                     self.get_logger()):
                            self.get_logger().warn(
                                f'Ignoring input value and using current '
                                f'default: {self.task_stage}')
                        else:
                            self.task_stage = input_task_stage_value

                    self.template_images_dict = dict(
                        self.orig_template_images_dict)
                    (self.template_images_dict, self.initial_point_id,
                     self.goal_point_id) = set_task_stage(
                        self.task_stage, self.template_images_dict)
                else:
                    rclpy.spin_once(self)
                    time.sleep(float(1. / self.rate))

                if self.run_on_ros_trigger or self.run_on_udp_trigger:
                    self.get_logger().info('Estimating slider distance...')
                start_time = time.time()

                # Convert image message to OpenCV format:
                try:
                    image_cv = self.bridge.imgmsg_to_cv2(
                        self.current_image_msg, "bgr8")
                except CvBridgeError as e:
                    self.get_logger().warn(
                        'Failed to convert image message to opencv format!')
                    self.get_logger().warn(f'Error: {e}')
                    continue

                # Detect templates in input image:
                vis_image_array = image_cv.copy()
                template_positions_dict = {}
                detection_scores_dict = {}

                for template_id, temp_image_array_list in \
                        self.template_images_dict.items():
                    for temp_image_array in temp_image_array_list:
                        # Apply template matching:
                        res = cv2.matchTemplate(
                            image_cv, temp_image_array,
                            template_matching_method_)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                        if self.debug:
                            self.get_logger().info(
                                f'[DEBUG] Checking for template: {template_id}')
                            self.get_logger().info(
                                f'[DEBUG] Current score: {max_val}')

                        if max_val < self.detection_score_threshold:
                            if self.debug:
                                self.get_logger().info(
                                    f'[DEBUG] Could not find a detection with '
                                    f'a score higher than threshold '
                                    f'{self.detection_score_threshold}.')
                                self.get_logger().info(
                                    '[DEBUG] Trying another template source...')
                        else:
                            if self.debug:
                                self.get_logger().info(
                                    '[DEBUG] Suitable detection found!')
                            detection_scores_dict[template_id] = max_val

                            w = temp_image_array.shape[1]
                            h = temp_image_array.shape[0]
                            top_left = max_loc
                            bottom_right = (top_left[0] + w, top_left[1] + h)
                            centroid = (int(top_left[0] + (w / 2.)),
                                        int(top_left[1] + (h / 2.)))
                            template_positions_dict[template_id] = centroid

                            # Annotate image with BB and centroid point:
                            cv2.rectangle(
                                vis_image_array, top_left, bottom_right,
                                color=text_label_colors_[template_id],
                                thickness=2)
                            cv2.circle(
                                vis_image_array, centroid, 1,
                                color=(0., 0., 0.), thickness=1)

                            # Annotate image with faded text labels:
                            overlay = np.copy(vis_image_array)
                            overlay = cv2.rectangle(
                                overlay,
                                (top_left[0], top_left[1] - 15),
                                (top_left[0] + w, top_left[1]),
                                text_label_colors_[template_id], -1)
                            overlay = cv2.putText(
                                overlay, template_id,
                                (top_left[0], top_left[1] - 5),
                                cv2_text_label_font_,
                                cv2_text_label_font_scale_,
                                (0., 0., 0.), 1)
                            alpha = 0.5
                            cv2.addWeighted(
                                overlay, alpha, vis_image_array,
                                1 - alpha, 0, vis_image_array)
                            break
                    else:
                        self.get_logger().warn(
                            f'"{template_id}" could not be detected!')

                if self.debug:
                    self.get_logger().info(
                        f'[DEBUG] Template positions: {template_positions_dict}')
                    self.get_logger().info(
                        f'[DEBUG] Detection scores: {detection_scores_dict}')

                # Estimate distance between positions in image space:
                estimated_pixel_distance = None
                if (self.initial_point_id in template_positions_dict.keys() and
                        detection_scores_dict[self.initial_point_id] >=
                        self.detection_score_threshold):
                    if (self.goal_point_id in template_positions_dict.keys() and
                            detection_scores_dict[self.goal_point_id] >=
                            self.detection_score_threshold):
                        point_1 = template_positions_dict[self.initial_point_id]
                        point_2 = template_positions_dict[self.goal_point_id]
                        x_distance = point_2[0] - point_1[0]
                        estimated_pixel_distance = x_distance

                        # Draw arrow indicating direction of estimated motion:
                        arrow_y_position = int(
                            (point_1[1] + point_2[1]) / 2.)
                        text_label_position = (
                            int((point_1[0] + point_2[0]) / 2.) - 15,
                            arrow_y_position - 45)
                        cv2.arrowedLine(
                            vis_image_array,
                            (point_1[0], arrow_y_position),
                            (point_2[0], arrow_y_position),
                            color=(0, 0, 0), thickness=2, tipLength=0.2)
                        vis_image_array = cv2.putText(
                            vis_image_array,
                            'Dist.: ' + str(estimated_pixel_distance),
                            text_label_position,
                            cv2_text_label_font_,
                            cv2_text_label_font_scale_ * 2,
                            (0., 0., 0.), 1)
                    else:
                        self.get_logger().warn(
                            f'Goal point template ({self.goal_point_id}) '
                            f'was not reliably detected in image!')
                else:
                    self.get_logger().warn(
                        f'Initial point template ({self.initial_point_id}) '
                        f'was not reliably detected in image!')

                if estimated_pixel_distance is None:
                    self.get_logger().error(
                        'Could not estimate slider motion distance!')
                else:
                    estimated_slider_distance = lcd_marker_to_slider_distance(
                        estimated_pixel_distance)

                    # Publish result as Float32:
                    slider_solution_msg = Float32()
                    slider_solution_msg.data = float(estimated_slider_distance)
                    self.slider_distance_publisher.publish(slider_solution_msg)

                    if self.debug:
                        self.get_logger().info(
                            f'Estimated distance between templates '
                            f'{self.initial_point_id} and '
                            f'{self.goal_point_id} in image space: '
                            f'{estimated_pixel_distance}')
                        self.get_logger().info(
                            f'Estimated distance to move the slider to '
                            f'solve the task: {estimated_slider_distance}.')

                    # Send result over UDP socket:
                    self.get_logger().info(
                        'Sending estimation result over UDP...')
                    output_dict = {
                        'slider_motion_distance': str(estimated_slider_distance)
                    }
                    udp_message = json.dumps(output_dict, indent=4)
                    if self.debug:
                        self.get_logger().info(
                            f'[DEBUG] UDP message:\n{udp_message}')
                    udp_message = udp_message.encode()
                    self.udp_output_socket.sendto(
                        udp_message, (self.udp_ip, self.udp_output_port))

                # Publish visual output:
                if self.publish_visual_output:
                    debug_image_msg = self.bridge.cv2_to_imgmsg(
                        vis_image_array, encoding="bgr8")
                    debug_image_msg.header.stamp = \
                        self.get_clock().now().to_msg()
                    debug_image_msg.header.frame_id = \
                        self.current_image_msg.header.frame_id
                    self.slider_solver_image_publisher.publish(debug_image_msg)

                    input_image_msg = self.bridge.cv2_to_imgmsg(
                        image_cv, encoding="bgr8")
                    input_image_msg.header.stamp = \
                        self.get_clock().now().to_msg()
                    input_image_msg.header.frame_id = \
                        self.current_image_msg.header.frame_id
                    self.input_image_publisher.publish(input_image_msg)

                elapsed_time = time.time() - start_time
                self.get_logger().info(
                    f'Finished in {elapsed_time:.3f}s')

                # Optionally save results:
                if self.save_output:
                    detection_str = datetime.datetime.now().strftime(
                        "%Y%m%d_%H%M%S")
                    cv2.imwrite(
                        os.path.join(
                            self.output_dir_path,
                            f'slider_solver_output_image_{detection_str}.png'),
                        vis_image_array)
                    cv2.imwrite(
                        os.path.join(
                            self.output_dir_path,
                            f'slider_solver_input_image_{detection_str}.png'),
                        image_cv)

        except KeyboardInterrupt:
            self.get_logger().info('Stopping node')
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = SliderTaskSolverNode()

    try:
        node.run_node()
    except SystemExit:
        rclpy.logging.get_logger('rclpy').info(
            'Stopping slider_task_solver_node...')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
