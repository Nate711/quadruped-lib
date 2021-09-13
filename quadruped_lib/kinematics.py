"""Inverse kinematics functions for the pybullet pupper model.

TODO: exceptions/messages for unreachable locations"""

import numpy as np
from quadruped_lib import kinematics_config


def interior_angle(
    side_length_a: float,
    side_length_b: float,
    side_length_c: float,
    noexcept: bool = False,
    eps=1e-7,
) -> float:
    """Calculate the angle between side A and B using law of cosines.
    Args:
        side_length_a:
        side_length_b:
        side_length_c:
        noexcept: Return 0rads if side C is too small to form triangle, pi rads if side C is too big to form triangle.
    Returns:
        Angle between sides A and B
    """
    if not noexcept:
        if side_length_a < 0 or side_length_b < 0 or side_length_c < 0:
            raise ValueError("Side lengths cannot be negative")
        if abs(side_length_a - side_length_b) > side_length_c + eps:
            raise ArithmeticError(
                f"Side c ({side_length_c}) is too short for side lengths a ({side_length_a}) and b({side_length_b})"
            )
        if side_length_a + side_length_b < side_length_c - eps:
            raise ArithmeticError(
                f"Side c ({side_length_c}) is too long for side lengths a ({side_length_a}) and b({side_length_b})"
            )

    arccos_argument = (side_length_a ** 2 + side_length_b ** 2 - side_length_c ** 2) / (
        2 * side_length_a * side_length_b
    )
    arccos_argument = np.clip(arccos_argument, -1.0, 1.0)
    return np.arccos(arccos_argument)


def wrap_angle(theta: float) -> float:
    """Wraps an angle to -pi to pi."""
    while theta < -np.pi:
        theta += 2 * np.pi
    while theta > np.pi:
        theta -= 2 * np.pi
    return theta


def leg_inverse_kinematics_relative_to_hip(
    r_leg_origin_to_foot: np.ndarray,
    leg_index: int,
    config: kinematics_config.KinematicsConfig,
    noexcept: bool = False,
) -> np.ndarray:
    """Find the joint angles for a given hip-relative foot position.

    Will always find the configuration with a forward-bending lower link,
    rather than the configuration that has the same cartesian location
    but a backward-bending lower link.

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
        config.abduction_offset_i(leg_index) / r_leg_origin_to_foot_in_yz_plane
    )
    if noexcept:
        arccos_argument = np.clip(arccos_argument, -1.0, 1.0)
    elif abs(arccos_argument) > 1.0:
        raise ArithmeticError("Desired foot location too close to hip.")
    phi = np.arccos(arccos_argument)

    # Angle of the y-z projection of the hip-to-foot vector, relative to the positive y-axis
    hip_foot_angle = np.arctan2(z, y)

    # Ab/adduction angle, relative to the positive y-axis
    abduction_angle = wrap_angle(phi + hip_foot_angle)

    # theta: Angle between the tilted negative z-axis and the hip-to-foot vector
    theta = np.arctan2(-x, r_hip_to_foot_in_yz_plane)

    # Distance between the hip and foot
    r_hip_to_foot = (r_hip_to_foot_in_yz_plane ** 2 + x ** 2) ** 0.5

    # Angle between the line going from hip to foot and the link L1
    trident = interior_angle(
        config.upper_link_length,
        r_hip_to_foot,
        config.lower_link_length,
        noexcept=noexcept,
    )

    # Angle of the first link relative to the tilted negative z axis
    hip_angle = theta + trident

    # Angle between the leg links L1 and L2
    beta = interior_angle(
        config.upper_link_length,
        config.lower_link_length,
        r_hip_to_foot,
        noexcept=noexcept,
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


def x_rotation_matrix(theta: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def leg_forward_kinematics_relative_to_hip(
    joint_angles: np.ndarray,
    leg_index: int,
    config: kinematics_config.KinematicsConfig,
):
    l1, l2 = config.upper_link_length, config.lower_link_length
    alpha, theta, phi = tuple(joint_angles)

    px = -l1 * np.sin(theta) - l2 * np.sin(theta + phi)
    py = config.abduction_offset_i(leg_index)
    pz = -l1 * np.cos(theta) - l2 * np.cos(theta + phi)

    return x_rotation_matrix(alpha) @ np.array((px, py, pz))
