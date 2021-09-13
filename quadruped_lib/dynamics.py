import numpy as np
from quadruped_lib import kinematics_config


def leg_jacobian(
    joint_angles: np.ndarray,
    kinematics_config: kinematics_config.KinematicsConfig,
    leg_index: int,
):
    """ "
    Jacobian for serial linkage.

    Vertical extended leg config  corresponds to 0 angles
    """
    l1 = kinematics_config.upper_link_length
    l2 = kinematics_config.lower_link_length
    (alpha, theta, phi) = joint_angles

    px = -l1 * np.sin(theta) - l2 * np.sin(theta + phi)
    py = kinematics_config.abduction_offset_i(leg_index)
    pz = -l1 * np.cos(theta) - l2 * np.cos(theta + phi)

    return np.array(
        [
            [0, pz, -l2 * np.cos(theta + phi)],
            [
                -py * np.sin(alpha) - pz * np.cos(alpha),
                np.sin(alpha) * px,
                -l2 * np.sin(alpha) * np.sin(theta + phi),
            ],
            [
                py * np.cos(alpha) - pz * np.sin(alpha),
                -px * np.cos(alpha),
                l2 * np.cos(alpha) * np.sin(theta + phi),
            ],
        ]
    )
