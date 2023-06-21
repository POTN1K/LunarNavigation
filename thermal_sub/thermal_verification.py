import numpy as np
import thermal as th
import unittest


class TestThermal(unittest.TestCase):
    """Class performing unit tests on thermal.py."""

    def test_heat_dissipation(self):
        """unittest to check heat dissipation formula"""
        Q = th.heat_dissipated(1000, 0.7)
        self.assertAlmostEqual(Q, 300.000, places=3)

    def test_visability_factor(self):
        """unittest to check IR correction factor formula"""
        V = th.visibility_factor(1000, 100)
        self.assertAlmostEqual(V, 100.000, places=3)

    def test_albedo_radiation(self):
        """unittest to check albedo factor formula"""
        a = th.albedo_radiation(100, 1)
        self.assertAlmostEqual(a, 100.000, places=3)

    def test_intensities(self):
        """unittest to check intensity formula"""
        I = th.intensities(500, 1000, 2000)
        self.assertAlmostEqual(I, 125.000, places=3)

    def test_equilibrium_temperature(self):
        """unittest to check equilibrium tempereture formula"""
        T = th.equilibrium_temperature(10, 1, 100, 10, 0.1, 0.7, 0.5, 5.67*10**(-8), 100, 0.8)
        self.assertAlmostEqual(T, 159.29030, places=3)

    def test_q_in(self):
        """unittest to check input heat for given temperature formula"""
        q = th.q_in(300, 0.7, 0.5, 5.67*10**(-8), 10, 1, 100, 10, 0.1, 100, 0.8)
        self.assertAlmostEqual(q, 2348.35, places=2)

    def test_phase_change(self):
        """unittest to check mass phase change material formula"""
        m = th.phase_change(500, 1000, 25000, 10, 1000)
        self.assertAlmostEqual(m, 14.2857, places=3)

    def test_Q_PC_day(self):
        """unittest to check heat charging formula"""
        Q_PC = th.Q_PC_day(500, 1000, 10000, 10000, 100000)
        self.assertAlmostEqual(Q_PC, 118.0555, places=3)