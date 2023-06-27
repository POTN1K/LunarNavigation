"""
Script to run unit tests, system test for the streets of coverage estimation model

By Ian Maes
"""

import unittest
import numpy as np

# Local Libraries
import sys
sys.path.append('')

from mission_design import incl, n_sats, loop

class MyTestCase(unittest.TestCase):

    def test_cov_angle(self):
        result = incl(1737.4, 1000)[1]
        print(result)
        self.assertAlmostEqual(result, 0.721, 3)

    def test_inclination(self):
        result = incl(1737.4, 1000)[0]
        print(result)
        self.assertAlmostEqual(result, 0.850, 2)

    def test_N_orbits(self):
        result = n_sats(np.pi/4, 20, 1737.4, np.pi / 4, False)[1]
        self.assertAlmostEqual(result, 3, 2)

    def test_N_sats(self):
        result = n_sats(np.pi / 4, 20, 1737.4, np.pi / 4, False)[0]
        self.assertAlmostEqual(result, 60, 2)

    def test_loop_size(self):
        result = np.size(loop(np.array([3500, 4000, 4500, 40000]), np.arange(10, 20, 1), True)[0])
        self.assertEqual(result, 44)

    def test_loop_extreme_min(self):
        result = loop(np.array([0]), np.arange(100, 1000, 100), False)[1][3, 0]
        self.assertEqual(result, 'Fail')

    def test_loop_extreme_max(self):
        result = loop(np.array([100000000]), np.arange(10, 20, 1), False)[1][3, 0]
        self.assertEqual(result, 2)

    def test_loop_val(self):
        result = loop(np.array([3500, 4000, 4500, 40000]), np.arange(10, 20, 1), True)[1][3, 0]
        self.assertEqual(result, 7)

if __name__ == '__main__':
    unittest.main()