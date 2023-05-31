"""Verification and Validation of Earth Constellation File.
By S. Nedelcu"""

# External Libraries
import unittest
import numpy as np

# Local Libraries
import sys
sys.path.append('.')

# Local objects
from mission_design import general_calculations

t, ir, ig, Phi, M, L1, L2, S3, S4, S5, S6, rc_M, rc_L1_R, rc_L2_R, rc_S3_Rr, rc_S4_Rr, rc_S5_Rr, rc_S6_Rr, rc_S3_R,\
    rc_S4_R, rc_S5_R, rc_S6_R, aS3, DS3, aS4, DS4, aS5, DS5, aS6, DS6, aS = general_calculations()


class TestEarth(unittest.TestCase):
    """Class to test the Earth Constellation functions"""

    def test_EO_1_A(self):
        for i in range(len(t)):
            z_M = rc_M[i, 2]
            z_L1 = rc_L1_R[i, 2]
            z_L2 = rc_L2_R[i, 2]
            z_S3 = rc_S3_Rr[i, 2]
            z_S4 = rc_S4_Rr[i, 2]
            z_S5 = rc_S5_Rr[i, 2]
            z_S6 = rc_S6_Rr[i, 2]
            self.assertEqual(z_M, 0)
            self.assertEqual(z_L1, 0)
            self.assertEqual(z_L2, 0)
            self.assertEqual(z_S3, 0)
            self.assertEqual(z_S4, 0)
            self.assertEqual(z_S5, 0)
            self.assertEqual(z_S6, 0)

    def test_EO_1_B(self):
        for i in range(len(t)):
            line_slope1 = (rc_M[i, 1] - rc_L1_R[i, 1]) / (rc_M[i, 0] - rc_L1_R[i, 0])
            line_slope2 = (rc_M[i, 1] - rc_L2_R[i, 1]) / (rc_M[i, 0] - rc_L2_R[i, 0])
            line_slope3 = (rc_L1_R[i, 1] - rc_L2_R[i, 1]) / (rc_L1_R[i, 0] - rc_L2_R[i, 0])
            self.assertTrue(np.abs(np.round(line_slope1, 12)) == np.abs(np.round(line_slope2, 12)) ==
                            np.abs(np.round(line_slope3, 12)))

    def test_EO_2_A(self):
        for i in range(len(t)):
            self.assertEqual(rc_S3_R[i, 0], rc_S3_Rr[i, 0] * np.cos(ir))
            self.assertEqual(rc_S3_R[i, 1], rc_S3_Rr[i, 1])
            self.assertEqual(rc_S3_R[i, 2], rc_S3_Rr[i, 0] * np.sin(ir))


if __name__ == '__main__':
    unittest.main()
