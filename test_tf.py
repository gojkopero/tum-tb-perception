import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. User's Hand-Eye Calibration (fr3_hand_tcp -> camera_color_optical_frame directly)
T_ee_to_optical = np.eye(4)
T_ee_to_optical[:3, :3] = R.from_quat([0.0077966, 0.0092625, 0.71351237, -0.70053802]).as_matrix()
T_ee_to_optical[:3, 3] = [-0.06137736, 0.02402934, -0.04691784]

# 2. My Static Transform in Launch File (fr3_hand_tcp -> camera_link)
T_ee_to_camera_link = np.eye(4)
T_ee_to_camera_link[:3, :3] = R.from_quat([0.007725, 0.707307, -0.007172, -0.706828]).as_matrix()
T_ee_to_camera_link[:3, 3] = [-0.061272, 0.009180, -0.046419]

# 3. Realsense internal TF (camera_link -> camera_color_optical_frame)
T_camera_link_to_optical = np.eye(4)
T_camera_link_to_optical[:3, :3] = R.from_quat([0.5048379621529079, -0.49451665157109637, 0.5039116549166059, -0.4966537600053155]).as_matrix()
T_camera_link_to_optical[:3, 3] = [-0.00018621953495312482, 0.014856887981295586, 0.0001167338268714957]

# 4. Chain my transform to get the effective optical frame
T_effective_optical = np.dot(T_ee_to_camera_link, T_camera_link_to_optical)

# 5. Compare T_ee_to_optical with T_effective_optical
diff = np.abs(T_ee_to_optical - T_effective_optical)
print(f"Max difference in matrix: {np.max(diff):.8f}")
