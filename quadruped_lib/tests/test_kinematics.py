import unittest
import numpy as np
from quadruped_lib import kinematics
from quadruped_lib import kinematics_config

class TestKinematics(unittest.TestCase):
    def TestZeroConfiguration(self):
        config = kinematics_config.KinematicsConfig(
            abduction_offset=0.05,
            upper_link_length=0.1,
            lower_link_length=0.1,
            hip_x_offset=0,
            hip_y_offset=0,
        )
        joint_angles = [0, np.pi / 4, -np.pi / 2]
        r = np.array([0, -0.05, -0.2])
        joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(r, 0, config)
        expected_joint_angles = np.array([0, 0, 0])
        self.assertTrue(np.allclose(joint_angles, expected_joint_angles))
