#!/usr/bin/env python3

"""
PnP-based taskboard pose estimator.

Uses OpenCV solvePnP to estimate the 6-DOF pose of the taskboard from
2D keypoint detections and known 3D model points (derived from the
DesignLearnRG_euROBIN CAD model).

Board frame convention (origin at blue_button):
    Z |    / Y
      |   /
      |  /
      | /
      |/
      o--------------
      ^              X
    Blue-Button
"""

import numpy as np
import cv2
import yaml


# Class-name mapping from euROBIN YOLO labels → existing tum-tb-perception labels:
EUROBIN_TO_TUM_LABEL = {
    "blue_button":  "blue_button",
    "door_handle":  "hatch_handle",
    "led_screen":   "lcd",
    "red_button":   "red_button",
    "red_hole":     "red_hole",
    "slider":       "slider",
}

# 3D model points in the board-local frame (blue_button = origin, Z = 0).
# These come from DesignLearnRG_euROBIN/vision/resource/model_data.yaml.
PNP_MODEL_POINTS = {
    "blue_button": np.array([0.0,   0.0,   0.0]),
    "red_button":  np.array([0.0,   0.014, 0.0]),
    "red_hole":    np.array([0.058, 0.025, 0.0]),
}

# Ordered list of the three keypoint labels used for PnP:
PNP_KEYPOINT_ORDER = ["blue_button", "red_button", "red_hole"]


class PnPPoseEstimator:
    """
    Estimates the 6-DOF pose of the taskboard from 2D keypoint
    detections and known 3D CAD model points via OpenCV solvePnP.

    Parameters
    ----------
    camera_matrix : ndarray (3, 3)
        Camera intrinsic matrix.
    dist_coeffs : ndarray or None
        Distortion coefficients (default: zero distortion).
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs=None):
        self.camera_matrix = camera_matrix.astype(np.float64)
        self.dist_coeffs = (
            dist_coeffs if dist_coeffs is not None
            else np.zeros(4, dtype=np.float64)
        )

        # Build the ordered 3D model point array:
        self.model_points_3d = np.array(
            [PNP_MODEL_POINTS[k] for k in PNP_KEYPOINT_ORDER],
            dtype=np.float64,
        )

    def estimate_pose(self, keypoints_2d: dict):
        """
        Estimate the board pose from detected 2D keypoints.

        Parameters
        ----------
        keypoints_2d : dict
            Mapping from euROBIN class name → (cx, cy) pixel centre.
            Must contain all three PNP_KEYPOINT_ORDER entries.

        Returns
        -------
        success : bool
        rvec : ndarray (3, 1) – rotation vector (Rodrigues)
        tvec : ndarray (3, 1) – translation vector
        transform_4x4 : ndarray (4, 4) – camera ← board homogeneous transform
        """
        # Check that all three keypoints are present:
        for key in PNP_KEYPOINT_ORDER:
            if key not in keypoints_2d:
                return False, None, None, None

        image_points = np.array(
            [keypoints_2d[k] for k in PNP_KEYPOINT_ORDER],
            dtype=np.float64,
        )

        # solvePnP with SOLVEPNP_SQPNP (works for ≥ 3 points):
        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not success:
            return False, None, None, None

        # Build 4×4 homogeneous transform (camera ← board):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        return True, rvec, tvec, T

    def project_point_to_board_plane(
        self, pixel_uv, rvec, tvec,
    ):
        """
        Project a 2D pixel onto the board's Z = 0 plane using the
        solved board pose, returning the 3D position in camera frame.

        Parameters
        ----------
        pixel_uv : tuple (u, v)
            Pixel coordinates.
        rvec, tvec : ndarray
            Board pose from solvePnP.

        Returns
        -------
        point_camera : ndarray (3,) – 3D position in camera frame.
        point_board : ndarray (3,) – 3D position in board frame.
        """
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        # Camera intrinsics:
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        u, v = pixel_uv

        # Ray direction in camera frame:
        ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

        # Board plane in camera frame: n · (p - t) = 0
        # where n is the board Z-axis in camera frame (3rd column of R)
        n = R[:, 2]  # board Z in camera frame
        p0 = t       # board origin in camera frame

        # Ray–plane intersection:
        denom = np.dot(n, ray_cam)
        if abs(denom) < 1e-10:
            return None, None

        s = np.dot(n, p0) / denom
        point_camera = s * ray_cam

        # Transform to board frame:
        point_board = R.T @ (point_camera - t)

        return point_camera, point_board

    def board_point_to_camera(self, board_point, rvec, tvec):
        """
        Transform a 3D point from board frame to camera frame.

        Parameters
        ----------
        board_point : ndarray (3,)
        rvec, tvec : ndarray

        Returns
        -------
        point_camera : ndarray (3,)
        """
        R, _ = cv2.Rodrigues(rvec)
        return (R @ board_point) + tvec.flatten()


def load_model_points(yaml_path: str) -> dict:
    """Load PnP model points from a YAML config file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    pnp = data.get("PnPPoints", {})
    result = {}
    for name, coords in pnp.items():
        result[name] = np.array(coords, dtype=np.float64)
    return result
