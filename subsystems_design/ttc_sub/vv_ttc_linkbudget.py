"""By J. Geijsberts"""

import unittest
import subprocess
import numpy as np

from ttc_linkbudget import decibel, inv_decibel, HelicalAntenna, ParabolicAntenna, LaserAntenna, LinkBudget

class TestTTC_OutsideFunctions(unittest.TestCase):
    def test_decibel(self):
        self.assertAlmostEqual(decibel(2), 3.01, 2)
        self.assertEqual(decibel(0), 0)
    def test_inv_decibel(self):
        self.assertAlmostEqual(inv_decibel(3.01), 2, 1)

class TestHelicalAntenna(unittest.TestCase):
    a = HelicalAntenna(1,1)
    def test_peak_gain(self):
        self.assertAlmostEqual(self.a.peak_gain(1), 105.7547165, 5)
    def test_half_power_angle(self):
        self.assertAlmostEqual(self.a.half_power_angle(1), 16.55211408, 5)

class TestParabolicAntenna(unittest.TestCase):
    a = ParabolicAntenna(1,0.5)
    def test_peak_gain_transmitter(self):
        self.assertAlmostEqual(self.a.peak_gain_transmitter(1e9), 60.25595861, 5)
    def test_peak_gain_receiver(self):
        self.assertAlmostEqual(self.a.peak_gain_receiver(1e9), 4.934802201e-18, 6)
    def test_half_power_angle(self):
        self.assertEqual(self.a.half_power_angle(1e9), 21)

class TestLaserAntenna(unittest.TestCase):
    a = LaserAntenna(0.8, 1, 1e-6)
    def test_gain(self):
        self.assertAlmostEqual(self.a.gain(1e9), 9.869604401e-18, 5, 'The gain of the laser antenna was not calculated correctly')
    def test_pointing_loss(self):
        self.assertAlmostEqual(self.a.pointing_loss(1e9), 1, 5)

class TestLinkBudgetMethods(unittest.TestCase):
    def setUp(self):
        # Setting up objects for the tests
        self.helical_antenna = HelicalAntenna(0.098, 0.212)
        self.parabolic_antenna = ParabolicAntenna(3, 0.55)
        self.link_budget = LinkBudget(2500e6, 0.5, 0.5, 35, 500, 1, 290, self.parabolic_antenna, self.helical_antenna, 385000000, 1, 1, 10)

    def test_calculateAntennaGain(self):
        # Testing the calculateAntennaGain method
        self.assertAlmostEqual(self.link_budget.calculateAntennaGain(self.helical_antenna, 'transmitter'), 154.32, places=2)
        self.assertAlmostEqual(self.link_budget.calculateAntennaGain(self.parabolic_antenna, 'receiver'), 5776.85, places=2)

    def test_calculateHalfPowerAngle(self):
        # Testing the calculateHalfPowerAngle method
        self.assertAlmostEqual(self.link_budget.calculateHalfPowerAngle(self.helical_antenna), 7.28, places=2)
        self.assertAlmostEqual(self.link_budget.calculateHalfPowerAngle(self.parabolic_antenna), 0.007, places=3)
        
    def test_pointingLoss(self):
        # Testing the pointingLoss method
        self.assertAlmostEqual(self.link_budget.pointingLoss(1, 1), 0.06309573445, places=5)

    def test_spaceLoss(self):
        # Testing the spaceLoss method
        self.assertAlmostEqual(self.link_budget.spaceLoss(1e9, 385000000), 0.04272271194, places=5)

    def test_largestDistance(self):
        # Testing the largestDistance method
        self.assertAlmostEqual(self.link_budget.largestDistance(1737500, 385000000), 386733596.9, places=1)

    def test_calculateSignalToNoiseRatio(self):
        # Testing the calculateSignalToNoiseRatio method
        self.assertAlmostEqual(self.link_budget.calculateSignalToNoiseRatio(power_transmitter=31, loss_factor_transmitter=0.5, loss_factor_receiver=0.5, gain_transmitter=154.32, gain_receiver=5776.85, atmospheric_loss=1, space_loss=6.92e-23, pointing_loss=1, data_rate=500, boltzmann=1.38065E-23, system_temperature=290), -114.53, places=2)

    def test_snr_margin(self):
        # Testing the snr_margin property
        self.assertAlmostEqual(self.link_budget.snr_margin, -124.53, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)