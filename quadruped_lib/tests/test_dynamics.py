import unittest
import numpy as np
from quadruped_lib import dynamics
from quadruped_lib import kinematics_config


class TestDynamics(unittest.TestCase):
    def TestStandingJacobian(self):
        config = kinematics_config.KinematicsConfig(
            abduction_offset=0.05,
            upper_link_length=0.1,
            lower_link_length=0.1,
            hip_x_offset=0,
            hip_y_offset=0,
        )
        joint_angles = [0, np.pi / 4, -np.pi / 2]
        jac = dynamics.leg_jacobian(joint_angles, config, leg_index=0)
        expected_jac = np.array(
            [
                [0.0, -np.sqrt(2) * 0.1, -np.sqrt(2) / 2 * 0.1],
                [np.sqrt(2) * 0.1, 0.0, 0.0],
                [-0.05, -0.0, -np.sqrt(2) / 2 * 0.1],
            ]
        )
        self.assertTrue(np.allclose(jac, expected_jac))
