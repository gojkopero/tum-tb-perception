#!/bin/bash
#
# Sets up a Python virtual environment for the SAM pose estimator node.
#
# Uses --system-site-packages so that ROS 2 libraries (rclpy, cv_bridge,
# tf2_ros, sensor_msgs_py, etc.) are inherited from the system, while
# pip-only dependencies (mobile-sam) are installed into the venv.
#
# After running this script:
#   1. Run download_sam_model.sh to download the MobileSAM checkpoint
#   2. Rebuild the package with colcon
#   3. Launch with: ros2 launch tum_tb_perception sam_pose_estimator.launch.py
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS="${SCRIPT_DIR}/requirements_sam.txt"

echo "[INFO] Creating virtual environment at: ${VENV_DIR}"
python3 -m venv --system-site-packages "${VENV_DIR}"

echo "[INFO] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

echo "[INFO] Installing SAM dependencies from ${REQUIREMENTS}..."
pip install -r "${REQUIREMENTS}"

echo ""
echo "[INFO] ✅ Virtual environment ready at: ${VENV_DIR}"
echo "[INFO] Python: $(which python3) ($(python3 --version))"
echo ""
echo "[INFO] Installed SAM packages:"
pip list 2>/dev/null | grep -iE "mobile.sam|segment.anything" || echo "  (none found — check for errors above)"
echo ""
echo "[INFO] Next steps:"
echo "  1. Download the model:  bash ${SCRIPT_DIR}/download_sam_model.sh"
echo "  2. Rebuild the package: cd /home/brairlab/franka_ws && colcon build --packages-select tum_tb_perception"
echo "  3. Source the workspace: source /home/brairlab/franka_ws/install/setup.bash"
echo "  4. Launch the node:     ros2 launch tum_tb_perception sam_pose_estimator.launch.py"
