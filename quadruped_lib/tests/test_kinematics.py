import unittest
import numpy as np
from quadruped_lib import kinematics
from quadruped_lib import kinematics_config


class TestKinematics(unittest.TestCase):
    def TestInteriorAngle(self):
        with self.assertRaises(ValueError):
            kinematics.interior_angle(-1, 0, 0)
        with self.assertRaises(ArithmeticError):
            kinematics.interior_angle(1, 1, 5)
        with self.assertRaises(ArithmeticError):
            kinematics.interior_angle(2, 1, 0.5)
        self.assertAlmostEqual(kinematics.interior_angle(1, 1, 2 + 1e-9), np.pi)
        self.assertAlmostEqual(kinematics.interior_angle(2, 1, 1 - 1e-9), 0)
        self.assertAlmostEqual(kinematics.interior_angle(1, 1, 2 ** 0.5), np.pi / 2)
        self.assertAlmostEqual(
            kinematics.interior_angle(0.5, 1, 3 ** 0.5 / 2), np.pi / 3
        )

    def TestZeroConfiguration(self):
        for l1 in [0.2, 0.5, 1.0]:
            config = kinematics_config.KinematicsConfig(
                abduction_offset=0.05,
                upper_link_length=l1,
                lower_link_length=0.1,
                hip_x_offset=0,
                hip_y_offset=0,
            )
            r = np.array([0, -0.05, -0.1 - l1])
            joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(
                r, 0, config
            )
            expected_joint_angles = np.array([0, 0, 0])
            self.assertTrue(np.allclose(joint_angles, expected_joint_angles))

    def TestStandingConfiguration(self):
        config = kinematics_config.KinematicsConfig(
            abduction_offset=0.05,
            upper_link_length=0.1,
            lower_link_length=0.1,
            hip_x_offset=0,
            hip_y_offset=0,
        )
        r = np.array([0, -0.05, -0.2 * 2 ** 0.5 / 2])
        joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(r, 0, config)
        expected_joint_angles = np.array([0, np.pi / 4, -np.pi / 2])
        self.assertTrue(np.allclose(joint_angles, expected_joint_angles))

    def TestArbitraryConfigsInverseKinematics(self):
        config = kinematics_config.KinematicsConfig(
            abduction_offset=0.05,
            upper_link_length=1,
            lower_link_length=1,
            hip_x_offset=0,
            hip_y_offset=0,
        )
        # upper link 45 degs backward, lower link vertical
        r = np.array([-(2 ** 0.5) / 2, -0.05, -(2 ** 0.5) / 2 - 1])
        joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(r, 0, config)
        expected_joint_angles = np.array([0, np.pi / 4, -np.pi / 4])
        self.assertTrue(np.allclose(joint_angles, expected_joint_angles))

        # hip rotated horizontally outwards
        r = np.array([0, -2, 0.05])
        joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(r, 0, config)
        expected_joint_angles = np.array([-np.pi / 2, 0, 0])
        self.assertTrue(np.allclose(joint_angles, expected_joint_angles))

        # leg rotated forwards to horizontal
        r = np.array([2, -0.05, 0])
        joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(r, 0, config)
        expected_joint_angles = np.array([0, -np.pi / 2, 0])
        self.assertTrue(np.allclose(joint_angles, expected_joint_angles))

        # leg rotated backwards to horizontal
        r = np.array([-2, -0.05, 0])
        joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(r, 0, config)
        expected_joint_angles = np.array([0, np.pi / 2, 0])
        self.assertTrue(np.allclose(joint_angles, expected_joint_angles))

    def TestForwardInverseConsistency(self):
        config = kinematics_config.KinematicsConfig(
            abduction_offset=0.05,
            upper_link_length=0.1,
            lower_link_length=0.1,
            hip_x_offset=0,
            hip_y_offset=0,
        )
        for leg_index in range(4):
            for i in range(100):
                cartesian_targets = (
                    np.random.rand() * 0.2 - 0.1,
                    np.random.rand() * 0.1 - 0.05,
                    -np.random.rand() * 0.1 - 0.05,
                )
                joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(
                    cartesian_targets, leg_index=leg_index, config=config
                )
                estimated_cartesian = kinematics.leg_forward_kinematics_relative_to_hip(
                    joint_angles, leg_index=leg_index, config=config, 
                )
                self.assertTrue(np.allclose(cartesian_targets, estimated_cartesian))