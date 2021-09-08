import numpy as np
from quadruped_lib import kinematics_config
import math

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

    px = -l1 * math.sin(theta) - l2 * math.sin(theta + phi)
    py = kinematics_config.abduction_offset_i(leg_index)
    pz = -l1 * math.cos(theta) - l2 * math.cos(theta + phi)

    return np.array(
        [
            [0, pz, -l2 * math.cos(theta + phi)],
            [
                -py * math.sin(alpha) - pz * math.cos(alpha),
                math.sin(alpha) * px,
                -l2 * math.sin(alpha) * math.sin(theta + phi),
            ],
            [
                py * math.cos(alpha) - pz * math.sin(alpha),
                -px * math.cos(alpha),
                l2 * math.cos(alpha) * math.sin(theta + phi),
            ],
        ]
    )
