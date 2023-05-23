"""
Created by Kyle scherpenzeel
"""

import unittest
import numpy as np

from user_error_calculator import UserErrors


class TestUserErrors(unittest.TestCase):

    def setUp(self):
        self.sats = np.array([[0, 0, 20000], [10000, 0, 20000], [-10000, 0, 20000], [0, 10000, 20000]])
        self.pos_error = 0
        self.position_surface = np.array([0, 0, 0])
        self.allowable = np.ones(6) * 50  # allowable errors for GDOP, PDOP, HDOP, VDOP, TDOP, HHDOP
        self.user_errors = UserErrors(self.sats, self.pos_error, self.position_surface, self.allowable)

    def test_invalid_satellite_input(self):
        # Create an array with only 3 satellites
        sats = np.array([[0, 0, 20000], [10000, 0, 20000], [-10000, 0, 20000]])
        with self.assertRaises(ValueError) as context:
            UserErrors(sats, self.pos_error, self.position_surface, self.allowable)

    def test_invalid_user_position_input(self):
        position_surface=[1,2]
        with self.assertRaises(ValueError) as context:
            UserErrors(self.sats, self.pos_error, position_surface, self.allowable)

    def test_invalid_allowable_error(self):
        allowable = [1, 2,1,5,1]
        with self.assertRaises(ValueError) as context:
              UserErrors(self.sats, self.pos_error, self.position_surface, allowable)
    def test_invalid_pos_error(self):
        pos_error = -1
        with self.assertRaises(ValueError) as context:
              UserErrors(self.sats, pos_error, self.position_surface, self.allowable)

    def test_satellite_error(self):
        error = self.user_errors.satellite_error()
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)

    def test_parameter_covariance_matrix(self):
        self.user_errors.parameter_covariance_matrix()
        self.assertIsInstance(self.user_errors.Q, np.ndarray)
        self.assertIsInstance(self.user_errors.HQ, np.ndarray)

    def test_dop_calculator(self):
        self.user_errors.dop_calculator()
        for key in self.user_errors.DOP:
            self.assertIsInstance(self.user_errors.DOP[key], float)
            self.assertGreater(self.user_errors.DOP[key], 0)

        #Check if GDOP is greater than all other DOPS
        self.assertGreaterEqual(self.user_errors.DOP["GDOP"], self.user_errors.DOP["PDOP"])
        self.assertGreaterEqual(self.user_errors.DOP["GDOP"], self.user_errors.DOP["HDOP"])
        self.assertGreaterEqual(self.user_errors.DOP["GDOP"], self.user_errors.DOP["VDOP"])
        self.assertGreaterEqual(self.user_errors.DOP["GDOP"], self.user_errors.DOP["TDOP"])
        self.assertGreaterEqual(self.user_errors.DOP["GDOP"], self.user_errors.DOP["HHDOP"])

    def test_user_error(self):
        self.user_errors.user_error()
        for err in self.user_errors.Error:
            self.assertIsInstance(err, float)
            self.assertGreater(err, 0)

    def test_allowable_error(self):
        self.user_errors.allowable_error()
        print(self.user_errors.ErrorBudget)
        for err_budget in self.user_errors.ErrorBudget:
            self.assertIsInstance(err_budget, float)
            self.assertGreater(err_budget, 0)

    def test_postive_distances(self):
        self.user_errors.parameter_covariance_matrix()
        self.assertTrue(np.all(self.user_errors.dists > 0))

if __name__ == '__main__':
    unittest.main()
