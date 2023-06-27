# External Libraries
import unittest
import numpy as np

# Local Libraries
import sys
sys.path.append('')

# Local obzjects
from mission_design import Gpeak_helical, alpha1_over_2_helical, Lpr, dB, snr, snr_margin, S_lagrange, \
    L_s



class Testlink_budget2(unittest.TestCase):
    def test_Gpeak_helical(self):
        self.assertAlmostEqual(Gpeak_helical(1, 2, 3), 8.9396, 3)

    def test_alpha1_over_2_helical(self):
        self.assertAlmostEqual(alpha1_over_2_helical(1, 2, 3), 60.81635, 3)
        self.assertEqual(alpha1_over_2_helical(0, 1, 2), np.inf)
        self.assertRaises(ZeroDivisionError, alpha1_over_2_helical, 1, 1, 0)

    def test_Lpr(self):
        self.assertEqual(Lpr(1, 2), -3)
        self.assertRaises(ZeroDivisionError, Lpr, 1, 0)

    def test_dB(self):
        self.assertAlmostEqual(dB(2), 3.0103, 3)
        self.assertEqual(dB(0), -np.inf)

    def test_snr(self):
        self.assertEqual(snr(1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 4)

    def test_snr_margin(self):
        self.assertEqual(snr_margin(1, 2), -1)

    def test_S_lagrange(self):
        self.assertAlmostEqual(S_lagrange(1, 2), 1.7321, 3)
        self.assertEqual(S_lagrange(0, 0), 0.0)

    def test_L_s(self):
        self.assertAlmostEqual(L_s(1, 2), 1.58314e-3, 3)
        self.assertRaises(ZeroDivisionError, L_s, 1, 0)


if __name__ == '__main__':
    unittest.main()
