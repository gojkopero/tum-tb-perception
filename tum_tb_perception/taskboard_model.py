#!/usr/bin/env python3

"""
Taskboard physical model derived from the URDF at:
https://github.com/peterso/robotlearningblock/blob/main/idf/taskboard/web_interfaces/task_board.urdf

Provides known component positions in the board-local frame and
a Procrustes (SVD) solver for aligning detected positions to the model.
"""

import numpy as np


# 2D positions (X, Y) of detectable components in the board-local frame (metres).
# Origin is at the centre of the board (task_board_main link).
# X-axis: along the wider dimension (positive toward buttons).
# Y-axis: along the narrower dimension (positive toward cable-wrap posts).
# Z-axis: surface normal pointing up/outward.
COMPONENT_POSITIONS_2D = {
    'red_button':            np.array([0.0962,  0.0309]),
    'blue_button':           np.array([0.0962,  0.0439]),
    'lcd':                   np.array([0.0962, -0.0231]),
    'slider':                np.array([0.0604, -0.0345]),
    'hatch_handle':          np.array([-0.0473, 0.0331]),   # knob when lid closed
    'multimeter_probe':      np.array([-0.1018, -0.0070]),
    'multimeter_socket':     np.array([-0.1018, -0.0430]),
    'multimeter_connector':  np.array([0.0384,  0.0314]),   # avg of red & black plugs
    'wire_connector':        np.array([0.0384,  0.0189]),
    'cable_holder':          np.array([0.0073,  0.0934]),   # avg of left & right posts
}

# Approximate board dimensions (metres) inferred from component positions.
BOARD_WIDTH_X  = 0.220   # along X-axis (wider)
BOARD_HEIGHT_Y = 0.160   # along Y-axis (narrower)
BOARD_ASPECT_RATIO = BOARD_WIDTH_X / BOARD_HEIGHT_Y  # ~1.375


def procrustes_2d(model_pts, detected_pts):
    """
    Kabsch algorithm in 2-D: find optimal rotation R and translation t
    such that  detected ≈ R @ model + t  (least-squares sense).

    Parameters
    ----------
    model_pts : ndarray (N, 2)
        Known positions in the board-local frame.
    detected_pts : ndarray (N, 2)
        Corresponding positions in the detection frame (projected onto plane).

    Returns
    -------
    R : ndarray (2, 2)
        Optimal rotation matrix.
    t : ndarray (2,)
        Optimal translation vector.
    residual : float
        RMS residual after alignment.
    """
    assert model_pts.shape == detected_pts.shape and model_pts.shape[1] == 2

    src_c = model_pts.mean(axis=0)
    tgt_c = detected_pts.mean(axis=0)

    src = model_pts - src_c
    tgt = detected_pts - tgt_c

    H = src.T @ tgt
    U, _, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, d]) @ U.T

    t = tgt_c - R @ src_c

    aligned = (R @ model_pts.T).T + t
    residual = np.sqrt(np.mean(np.sum((aligned - detected_pts) ** 2, axis=1)))

    return R, t, residual


def resolve_180_ambiguity(rotation_3x3, object_positions_dict, plane_normal, plane_point):
    """
    Check whether the estimated orientation is flipped 180° by testing
    whether known 'positive-X' components (buttons) actually end up on
    the positive-X side of the estimated frame.

    Parameters
    ----------
    rotation_3x3 : ndarray (3, 3)
        Estimated rotation matrix (columns = board X, Y, Z in camera frame).
    object_positions_dict : dict
        label -> 3D position (ndarray) in camera frame.
    plane_normal : ndarray (3,)
    plane_point : ndarray (3,)

    Returns
    -------
    rotation_3x3 : ndarray (3, 3)
        Possibly flipped rotation matrix.
    flipped : bool
    """
    # Components known to be at positive X in model frame:
    positive_x_labels = ['red_button', 'blue_button', 'lcd']
    # Components known to be at negative X in model frame:
    negative_x_labels = ['multimeter_probe', 'multimeter_socket']

    x_axis = rotation_3x3[:, 0]
    board_centre = plane_point  # approximate

    score = 0.0
    for label in positive_x_labels:
        if label in object_positions_dict:
            vec = object_positions_dict[label] - board_centre
            score += np.dot(vec, x_axis)

    for label in negative_x_labels:
        if label in object_positions_dict:
            vec = object_positions_dict[label] - board_centre
            score -= np.dot(vec, x_axis)

    if score < 0:
        # Flip: rotate 180° about Z-axis (negate X and Y columns)
        rotation_3x3 = rotation_3x3.copy()
        rotation_3x3[:, 0] = -rotation_3x3[:, 0]
        rotation_3x3[:, 1] = -rotation_3x3[:, 1]
        return rotation_3x3, True

    return rotation_3x3, False
