import unittest
from navigation_sub.frozen_orbits import FrozenOrbits
import numpy as np
from mission_design import Model, PropagationTime, UserErrors
import matplotlib.pyplot as plt

class TestFrozenOrbits(unittest.TestCase):

    def setUp(self):
        self.frozen_orbits = FrozenOrbits(File=None)
        # self.frozen_orbits.model = Model()
        # self.frozen_orbits.model_adder(np.vstack((self.frozen_orbits.constellation_SP, self.frozen_orbits.constellation_NP, self.frozen_orbits.orbit_Low_I)))
        # self.frozen_orbits.dyn_sim(1000)

    def test_true_anomaly_translation(self):
        satellites = np.array([[8000e3, 0.1, 0.0, 0.0, 0.0, 0.0],
                               [8000e3, 0.1, 0.0, 0.0, 0.0, 120.0],
                               [8000e3, 0.1, 0.0, 0.0, 0.0, 240.0]])
        change = 30.0
        translated_satellites = self.frozen_orbits.true_anomaly_translation(satellites, change)
        expected_satellites = np.array([[8000e3, 0.1, 0.0, 0.0, 0.0, 30.0],
                                        [8000e3, 0.1, 0.0, 0.0, 0.0, 150.0],
                                        [8000e3, 0.1, 0.0, 0.0, 0.0, 270.0]])
        np.testing.assert_almost_equal(translated_satellites, expected_satellites)

    def test_model_adder(self):
        satellites = np.array([[8000e3, 0.1, 0.0, 0.0, 0.0, 0.0],
                               [8000e3, 0.1, 0.0, 0.0, 0.0, 120.0],
                               [8000e3, 0.1, 0.0, 0.0, 0.0, 240.0]])
        self.frozen_orbits.model_adder(satellites)
        num_satellites = len(self.frozen_orbits.model.modules)
        self.assertEqual(num_satellites, 3)

    def test_model_symmetrical_planes(self):
        self.frozen_orbits.model_symmetrical_planes(1)
        self.assertEqual(len(self.frozen_orbits.model.modules), 20)

    def test_DOP_calculator(self):
        instance = FrozenOrbits(None)

        instance.model = Model()
        instance.model_adder(np.vstack((instance.constellation_SP,
                                                      instance.constellation_NP,
                                                      instance.orbit_Low_I)))
        instance.model_symmetrical_planes(10)
        instance.dyn_sim(1000)
        sat_velocities = instance.propagation_time.velocity
        DOP_array = instance.DOP_calculator(sat_velocities)
        self.assertEqual(DOP_array.shape, (10000, 6))

    def test_dyn_sim(self):
        P = 3600.0
        dt = 1.0
        kepler_plot = 0
        satellites = np.array([[8000e3, 0.1, 0.0, 0.0, 0.0, 0.0],
                               [8000e3, 0.1, 0.0, 0.0, 0.0, 120.0],
                               [8000e3, 0.1, 0.0, 0.0, 0.0, 240.0]])
        self.frozen_orbits.model_adder(satellites)
        self.frozen_orbits.dyn_sim(P, dt, kepler_plot)
        num_timesteps = len(self.frozen_orbits.propagation_time.states_array)
        self.assertEqual(num_timesteps, 3601)

    def test_true_anomaly_translation_invalid_change(self):
        satellites = [[8000e3, 0.1, 45, 90, 0, 0]]
        change = '30'
        with self.assertRaises(TypeError):
            self.frozen_orbits.true_anomaly_translation(satellites, change)

    def test_boxplot_no_array(self):
        df = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
              [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
              [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
        with self.assertRaises(TypeError):
            self.frozen_orbits.boxplot(df)

    def test_true_anomaly_translation(self):
        satellites = np.array([
            [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        change = 10.0
        expected_result = np.array([
            [1000.0, 0.0, 0.0, 0.0, 0.0, 10.0],
            [2000.0, 0.0, 0.0, 0.0, 0.0, 10.0],
            [3000.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        ])
        result = self.frozen_orbits.true_anomaly_translation(satellites, change)
        self.assertTrue((result == expected_result).all())

    def test_DOP_TIME(self):
        instance = FrozenOrbits(None)
        instance.model = Model()
        instance.model_adder(np.vstack((instance.constellation_SP,
                                                      instance.constellation_NP,
                                                      instance.orbit_Low_I)))
        instance.model_symmetrical_planes(10)
        instance.dyn_sim(1000)
        instance.DOP_time(instance.propagation_time.kepler_elements, 100)
        self.assertEqual(len(instance.DOP_time_GDOP), 10)
        self.assertEqual(len(instance.DOP_time_PDOP), 10)
        self.assertEqual(len(instance.DOP_time_HDOP), 10)
        self.assertEqual(len(instance.DOP_time_VDOP), 10)
        self.assertEqual(len(instance.DOP_time_TDOP), 10)
        self.assertEqual(len(instance.DOP_time_HHDOP), 10)






if __name__ == '__main__':
    unittest.main()
