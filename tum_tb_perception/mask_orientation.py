#!/usr/bin/env python3

"""
Extracts the 3-D orientation of the taskboard from a 2-D binary mask
produced by SAM.

Pipeline:
  1. Contour detection on the mask → largest contour
  2. cv2.minAreaRect → oriented bounding rectangle in pixel space
  3. Back-project the 4 rectangle corners onto the best-fit 3-D plane
  4. Compute 3-D edge vectors → board X and Y axes
  5. Orthogonalise with the depth-derived plane normal (Z axis)
  6. Return a proper 3×3 rotation matrix
"""

import cv2
import numpy as np

from tum_tb_perception.taskboard_model import BOARD_ASPECT_RATIO


def _backproject_pixel_to_plane(u, v, fx, fy, cx, cy, plane_normal, plane_point):
    """
    Intersect the ray through pixel (u, v) with the given 3-D plane.

    The camera is at the origin; the ray direction is
        d = ((u-cx)/fx, (v-cy)/fy, 1).
    The plane is defined by  n · (p - p0) = 0.

    Returns the 3-D intersection point (ndarray of shape (3,)).
    """
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    denom = np.dot(plane_normal, d)
    if abs(denom) < 1e-12:
        return None  # ray parallel to plane
    t = np.dot(plane_normal, plane_point) / denom
    if t < 0:
        return None  # plane behind camera
    return t * d


def extract_orientation_from_mask(mask, camera_params, plane_normal, plane_point,
                                  debug=False):
    """
    Estimate the taskboard's 3-D orientation from its SAM segmentation mask.

    Parameters
    ----------
    mask : ndarray (H, W), uint8 or bool
        Binary mask of the taskboard.
    camera_params : dict
        Camera intrinsics with keys 'f_x', 'f_y', 'c_x', 'c_y'.
    plane_normal : ndarray (3,)
        Unit normal of the taskboard's best-fit plane (pointing toward camera).
    plane_point : ndarray (3,)
        A point on the plane (e.g. centroid of taskboard points).
    debug : bool
        If True, return additional diagnostic info.

    Returns
    -------
    rotation_matrix : ndarray (3, 3)
        Columns are the board's X, Y, Z axes expressed in camera frame.
        X = wider edge direction, Y = narrower edge direction, Z = plane normal.
    info : dict (only if debug=True)
        Diagnostic information (contour, rect corners, 3-D corners, etc.).
    """
    fx, fy = camera_params['f_x'], camera_params['f_y']
    cx, cy = camera_params['c_x'], camera_params['c_y']

    # --- Step 1: largest contour ------------------------------------------
    mask_u8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
    if mask_u8.max() == 1:
        mask_u8 = mask_u8 * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('No contours found in the SAM mask.')

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 500:
        raise RuntimeError(f'Largest contour area ({area:.0f} px²) is too small.')

    # --- Step 2: oriented bounding rectangle in pixel space ----------------
    rect = cv2.minAreaRect(largest)  # ((cx,cy), (w,h), angle)
    box_pts_2d = cv2.boxPoints(rect)  # (4, 2) float32 corners

    # --- Step 3: back-project corners onto the 3-D plane ------------------
    corners_3d = []
    for u, v in box_pts_2d:
        pt = _backproject_pixel_to_plane(u, v, fx, fy, cx, cy,
                                         plane_normal, plane_point)
        if pt is None:
            raise RuntimeError(
                f'Could not back-project pixel ({u:.0f}, {v:.0f}) onto the plane.')
        corners_3d.append(pt)
    corners_3d = np.array(corners_3d)  # (4, 3)

    # --- Step 4: edge vectors in 3-D --------------------------------------
    edge_a = corners_3d[1] - corners_3d[0]
    edge_b = corners_3d[2] - corners_3d[1]
    len_a = np.linalg.norm(edge_a)
    len_b = np.linalg.norm(edge_b)

    if len_a < 1e-6 or len_b < 1e-6:
        raise RuntimeError('Degenerate rectangle: edge length ≈ 0.')

    # Assign longer edge → X-axis (board is wider in X):
    if len_a >= len_b:
        x_raw = edge_a / len_a
        y_raw = edge_b / len_b
    else:
        x_raw = edge_b / len_b
        y_raw = edge_a / len_a

    # --- Step 5: orthogonalise with the depth-derived plane normal ---------
    z_axis = plane_normal / np.linalg.norm(plane_normal)

    # Remove any component of x_raw along z, then normalise:
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Y = Z × X  (right-handed)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    if debug:
        info = {
            'contour': largest,
            'rect': rect,
            'box_pts_2d': box_pts_2d,
            'corners_3d': corners_3d,
            'edge_lengths': (len_a, len_b),
        }
        return rotation_matrix, info

    return rotation_matrix
