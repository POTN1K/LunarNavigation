import unittest
import numpy as np
from propagation_calculator import PropagationTime

r_moon = 1.737e6  # m
miu_moon = 4.9048695e12  # m^3/s^2


#Unit test

class MyTestCase(unittest.TestCase):
    def setUp(self):
        """Set up a test case with known parameters and results"""
        self.orbit_parameters = [[20e6, 0, 0, 0, 0, 0], [20e6, 0, 0, 0, 180, 0]]
        self.final_time = 86400
        self.resolution = 900
        self.mass_sat = 250
        self.area_sat = 1
        self.c_radiation = 1
        self.propagation = PropagationTime(self.orbit_parameters, self.final_time,
                                          self.resolution, self.mass_sat,
                                          self.area_sat, self.c_radiation)

    def test_orbital_parameters(self):
        """Test the orbital parameters are being set correctly"""
        expected_orbital_parameters = np.array([[20e6, 0, 0, 0, 0, 0], [20e6, 0, 0, 0, 180, 0]])
        self.assertEqual(np.allclose(self.propagation.orbit_parameters, expected_orbital_parameters, atol=1e-6), True)

    def test_final_time(self):
        """Test the final time is being set correctly"""
        expected_final_time = 86400
        self.assertEqual(self.propagation.final_time, expected_final_time)

    def test_resolution(self):
        """Test the resolution is being set correctly"""
        expected_resolution = 900
        self.assertEqual(self.propagation.resolution, expected_resolution)

    def test_mass_sat(self):
        """Test the satellite mass is being set correctly"""
        expected_mass_sat = 250
        self.assertEqual(self.propagation.mass_sat, expected_mass_sat)

    def test_area_sat(self):
        """Test the satellite area is being set correctly"""
        expected_area_sat = 1
        self.assertEqual(self.propagation.area_sat, expected_area_sat)

    def test_c_radiation(self):
        """Test the radiation coefficient is being set correctly"""
        expected_c_radiation = 1
        self.assertEqual(self.propagation.c_radiation, expected_c_radiation)



    def test_init_with_invalid_orbital_parameters(self):
        """Test initialization with invalid orbital parameters"""
        with self.assertRaises(ValueError):
            PropagationTime([20e6, 0, 0, 0, 0, 0], 86400, 900, 250, 1, 1)

    def test_init_with_invalid_final_time(self):
        """Test initialization with invalid final time"""
        with self.assertRaises(ValueError):
            PropagationTime([[20e6, 0, 0, 0, 0, 0],[25e6, 0, 0, 0, 0, 0]], -100, 900, 250, 1, 1)

    def test_init_with_invalid_resolution(self):
        """Test initialization with invalid resolution"""
        with self.assertRaises(ValueError):
            PropagationTime([[20e6, 0, 0, 0, 0, 0],[25e6, 0, 0, 0, 0, 0]], 86400, 0, 250, 1, 1)

    def test_init_with_invalid_mass_sat(self):
        """Test initialization with invalid satellite mass"""
        with self.assertRaises(ValueError):
            PropagationTime([[20e6, 0, 0, 0, 0, 0], [25e6, 0, 0, 0, 0, 0]], 86400, 900, -1, 1, 1)

    def test_init_with_invalid_area_sat(self):
        """Test initialization with invalid satellite area"""
        with self.assertRaises(ValueError):
            PropagationTime([[20e6, 0, 0, 0, 0, 0], [25e6, 0, 0, 0, 0, 0]], 86400, 900, 250, 0, 1)

    def test_init_with_invalid_c_radiation(self):
        """Test initialization with invalid radiation coefficient"""
        with self.assertRaises(ValueError):
            PropagationTime([[20e6, 0, 0, 0, 0, 0], [25e6, 0, 0, 0, 0, 0]], 86400, 900, 250, 1, -1)


    def test_satellites_count(self):
        """Test that satellites are added correctly"""
        propagation = PropagationTime([[20e6, 0, 0, 0, 0, 0], [20e6, 0, 0, 0, 0, 0]], 86400, 900, 250, 1, 1)
        self.assertEqual(len(propagation.bodies_to_propagate), len(propagation.orbit_parameters))

    def test_bodies_to_propagate_count(self):
        """Test that multiple satellites are correctly counted"""
        propagation = PropagationTime([[20e6, 0, 0, 0, 0, 0], [20e6, 0, 0, 0, 0, 0]], 86400, 900, 250, 1, 1)
        self.assertEqual(len(propagation.central_bodies), 2)

    def test_create_acceleration_models(self):
        """Test that each satellite has exactly 3 acceleration models"""
        for satellite, models in self.propagation.acceleration_models.items():
            self.assertEqual(len(models), 3)



    def test_hohmann_delta_v(self):
        # Check if hohmann is correct by hand

        delta_v = self.propagation.hohmann_delta_v(20000000, 30000000)
        self.assertAlmostEqual(delta_v, 47.25641828)

    def test_inclination_change(self):
        """Test the inclination_change method with known inputs and output"""
        v = 100
        delta_i = 0.2
        expected_delta_v = 19.96668333
        delta_v = self.propagation.inclination_change(v, delta_i)
        self.assertAlmostEqual(delta_v, expected_delta_v, places=1)

    def test_delta_v_to_maintain_orbit(self):
        """Test the delta_v_to_maintain_orbit method with known inputs and output"""
        satellite_name = "LunarSat1"
        start_time = 0
        end_time = 86400
        expected_delta_v = 8.221648755
        delta_v = self.propagation.delta_v_to_maintain_orbit(satellite_name, start_time, end_time)
        print(self.propagation.kepler_elements[0], self.propagation.kepler_elements[-1])
        self.assertAlmostEqual(delta_v, expected_delta_v, places=1)


# Model unittests

# Model module tests
# Extreme value tests

# Sensitivity Analysis

# Model system tests

#Sensitivity Analysis
#system Test
#Extreme value test

if __name__ == '__main__':
    unittest.main()
