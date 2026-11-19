# Source: Thyguad/lewm-research@67cf53f26b96033b53b37eb04c89b82c3c88b933
# Original authors: GalilAI-group; MIT. See THIRD_PARTY_NOTICES.md.
import numpy as np

def perturb_camera_angle(xyaxis, deg_dif=[3, 3]):
    """For OGBench Environments: Perturb the camera angle by a small random rotation."""
    xaxis = np.array(xyaxis[:3])
    yaxis = np.array(xyaxis[3:])

    # Compute z-axis
    zaxis = np.cross(xaxis, yaxis)
    zaxis /= np.linalg.norm(zaxis)

    # Small random rotation (e.g. ±3 degrees)
    yaw = np.deg2rad(deg_dif[0])
    pitch = np.deg2rad(deg_dif[1])

    # Build rotation matrices
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])

    # Combine and rotate the basis
    R = R_pitch @ R_yaw
    xaxis_new = R @ xaxis
    yaxis_new = R @ yaxis

    # Flatten back to tuple for MuJoCo
    xyaxes_new = tuple(np.concatenate([xaxis_new, yaxis_new]))

    return xyaxes_new
