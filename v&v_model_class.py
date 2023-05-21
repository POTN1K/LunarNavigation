"""Script to run unit, module, and system tests on the class Model.
By Nikolaus Ricker"""

# External Libraries
import unittest
import numpy as np

# Local objects
from model_class import Satellite, Tower, Lagrange, FixPoint, OrbitPlane, Model
# Constants
r_moon = 1.737e6  # m
miu_moon = 4.9048695e12  # m^3/s^2

# Satellite unittests
class TestSatellite(unittest.TestCase):
    """Class performing unit tests on the satellite object"""

    def test_a(self):
        """Checks semi major axis is larger than pericenter"""
        with self.assertRaises(ValueError):
            s = Satellite(a=1000)
        with self.assertRaises(ValueError):
            s = Satellite(e=0.5)
            print(s.r, s.a)

    def test_e(self):
        """Checks eccentricity is positive"""
        with self.assertRaises(ValueError):
            s = Satellite(e=-0.5)
        
    def test_i(self):
        """Checks inclination is between 0 and 180"""
        with self.assertRaises(ValueError):
            s = Satellite(i=-5)
        with self.assertRaises(ValueError):
            s = Satellite(i=185)

    def test_w(self):
        """Checks argument of periapsis is between 0 and 360"""
        with self.assertRaises(ValueError):
            s = Satellite(w=-5)
        with self.assertRaises(ValueError):
            s = Satellite(w=365)
    
    def test_Omega(self):
        """Checks longitude of ascending node is between 0 and 360"""
        with self.assertRaises(ValueError):
            s = Satellite(Omega=-5)
        with self.assertRaises(ValueError):
            s = Satellite(Omega=365)
    
    def test_nu(self):
        """Checks that true anomaly restarts every 360°"""
        s = Satellite(nu=365)
        self.assertEqual(s.nu, np.deg2rad(5))

    def test_elevation(self):
        """Checks that elevation is positive"""
        with self.assertRaises(ValueError):
            s = Satellite(elevation=-5)

    def test_r_size(self):
        """Test vector r has correct size"""
        s = Satellite()
        self.assertEqual(len(s.r), 3)

    def test_r_conversion(self):
        """Tests conversion from Keplerian to Cartesian is correct"""
        s1 = Satellite()
        r_test = np.array([s1.a,0,0])
        self.assertTrue(np.allclose(s1.r, r_test, atol=1e-2))
        s2 = Satellite(a=2e6, e=0.1)
        r_test = np.array([s2.a*(1-s2.e),0,0])
        self.assertTrue(np.allclose(s2.r, r_test, atol=1e-2))
        s3 = Satellite(a=2e6, w=90)
        r_test = np.array([0,s3.a,0])
        self.assertTrue(np.allclose(s3.r, r_test, atol=1e-2))
        s4 = Satellite(a=2e6, w=45)
        r_test = np.array([np.sqrt(s4.a**2/2),np.sqrt(s4.a**2/2),0])
        self.assertTrue(np.allclose(s4.r, r_test, atol=1e-2))
    
    def test_range(self):
        """Tests that maximum range is calculated correctly"""
        

    def test_range_limit(self):
        """Tests range can only cover half of the Moon at most"""

# Tower unittests

# Lagrange unittests

# FixPoint unittests

# OrbitPlane unittests

# Model unittests

# Model module tests
# Extreme value tests

# Sensitivity Analysis

# Model system tests

if __name__ == '__main__':
    unittest.main()