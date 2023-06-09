import unittest
import subprocess

from ttc_linkbudget import decibel, inv_decibel, HelicalAntenna, ParabolicAntenna, LaserAntenna, LinkBudget

class TestTTC_linkbudget(unittest.TestCase):
    def test_decibel(self):
        self.assertAlmostEqual(decibel(2), 3.01, 2)
        self.assertEqual(decibel(0), 0)
    def test_inv_decibel(self):
        self.assertAlmostEqual(inv_decibel(3.01), 2, 1)

class TestHelicalAntenna(unittest.TestCase):
    def test_peak_gain(self):
        self.assertAlmostEqual()


if __name__ == "__main__":
    unittest.main(verbosity=2)