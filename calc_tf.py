import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. ee_link -> camera_color_optical_frame
T_ee_to_optical = np.eye(4)
T_ee_to_optical[:3, :3] = R.from_quat([0.0077966, 0.0092625, 0.71351237, -0.70053802]).as_matrix()
T_ee_to_optical[:3, 3] = [-0.06137736, 0.02402934, -0.04691784]

# 2. camera_link -> camera_color_optical_frame
T_camera_to_optical = np.eye(4)
T_camera_to_optical[:3, :3] = R.from_quat([0.5048379621529079, -0.49451665157109637, 0.5039116549166059, -0.4966537600053155]).as_matrix()
T_camera_to_optical[:3, 3] = [-0.00018621953495312482, 0.014856887981295586, 0.0001167338268714957]

# 3. ee_link -> camera_link
T_optical_to_camera = np.linalg.inv(T_camera_to_optical)
T_ee_to_camera = np.dot(T_ee_to_optical, T_optical_to_camera)

trans = T_ee_to_camera[:3, 3]
quat = R.from_matrix(T_ee_to_camera[:3, :3]).as_quat()

print(f"args: {trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}")
