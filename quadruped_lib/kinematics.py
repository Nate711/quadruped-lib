"""Inverse kinematics functions for the pybullet pupper model."""

import numpy as np
from quadruped_lib import kinematics_config


def interior_angle(
    side_length_a: float,
    side_length_b: float,
    side_length_c: float,
    clip: bool = True,
) -> float:
    """Calculate the angle between side A and B using law of cosines.
    Args:
        side_length_a:
        side_length_b:
        side_length_c:
        clip:
    Returns:
        Angle between sides A and B
    """
    arccos_argument = (side_length_a ** 2 + side_length_b ** 2 - side_length_c ** 2) / (
        2 * side_length_a * side_length_b
    )
    if clip:
        arccos_argument = np.clip(arccos_argument, -1.0, 1.0)
    return np.arccos(arccos_argument)


def leg_inverse_kinematics_relative_to_hip(
    r_leg_origin_to_foot: np.ndarray,
    leg_index: int,
    config: kinematics_config.KinematicsConfig,
) -> np.ndarray:
    """Find the joint angles for a given hip-relative foot position.

    args:
        r_leg_origin_to_foot: [type]
        leg_index: [type]
        config: [type]

    returns:
        numpy array (3)
            Array of corresponding joint angles.
    """
    (x, y, z) = r_leg_origin_to_foot

    # Distance from the leg origin to the foot, projected into the y-z plane
    r_leg_origin_to_foot_in_yz_plane = (y ** 2 + z ** 2) ** 0.5

    # Distance from the leg's forward/back point of rotation to the foot
    r_hip_to_foot_in_yz_plane = (
        r_leg_origin_to_foot_in_yz_plane ** 2 - config.abduction_offset ** 2
    ) ** 0.5

    # Interior angle of the right triangle formed in the y-z plane by the leg
    # that is coincident to the ab/adduction axis
    # For feet 2 (front left) and 4 (back left), the abduction offset is
    # positive, for the right feet, the abduction offset is negative.
    arccos_argument = (
        config.abduction_offsets()[leg_index] / r_leg_origin_to_foot_in_yz_plane
    )
    arccos_argument = np.clip(arccos_argument, -1.0, 1.0)
    phi = np.arccos(arccos_argument)

    # Angle of the y-z projection of the hip-to-foot vector, relative to the positive y-axis
    hip_foot_angle = np.arctan2(z, y)

    # Ab/adduction angle, relative to the positive y-axis
    abduction_angle = phi + hip_foot_angle

    # theta: Angle between the tilted negative z-axis and the hip-to-foot vector
    theta = np.arctan2(-x, r_hip_to_foot_in_yz_plane)

    # Distance between the hip and foot
    r_hip_to_foot = (r_hip_to_foot_in_yz_plane ** 2 + x ** 2) ** 0.5

    # Angle between the line going from hip to foot and the link L1
    trident = interior_angle(
        config.upper_link_length, r_hip_to_foot, config.lower_link_length
    )

    # Angle of the first link relative to the tilted negative z axis
    hip_angle = theta + trident

    # Angle between the leg links L1 and L2
    beta = interior_angle(
        config.upper_link_length, config.lower_link_length, r_hip_to_foot
    )

    # Angle of the second link relative to the tilted negative z axis
    knee_angle = beta - np.pi

    return np.array([abduction_angle, hip_angle, knee_angle])


def serial_quadruped_inverse_kinematics(
    r_leg_origin_to_foot: np.ndarray, config: kinematics_config.KinematicsConfig
) -> np.ndarray:
    """Find the joint angles for all twelve joints for given foot positions.

    Foot positions are given relative to the center of the body in the body frame.

    Args:
        r_leg_origin_to_foot: numpy array (3,4)
            Matrix of the body-frame foot positions. Each column corresponds to a separate foot.
        config: Config object
            Object of robot configuration parameters.

    Returns:
        numpy array (3,4). Matrix of corresponding joint angles.
    """
    alpha = np.zeros((3, 4))
    for i in range(4):
        body_offset = config.leg_origins()[:, i]
        alpha[:, i] = leg_inverse_kinematics_relative_to_hip(
            r_leg_origin_to_foot[:, i] - body_offset, i, config
        )
    return alpha
