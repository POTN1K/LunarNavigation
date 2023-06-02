"""Main File, used to run all simulations.
Maintained by Nikolaus Ricker"""

# External Libraries
import numpy as np
from tudatpy.kernel.astro import element_conversion
import sys

sys.path.append('.')

# Local Libraries
from mission_design import Model, PropagationTime, UserErrors

# cst
miu_moon = 4.9048695e12  # m^3/s^2


# DOP Calculation
# DOP_with_error = []
# error_budget = []
# DOP = []
#
# for i in range(0, 10000):
#     point = i
#     Errors = UserErrors(np.array([sat.r for sat in model.mod_inView_obj[point]]),0, 0,
#                         model.moon[point], [20, 10, 10, 10, 10, 3.5])
#     DOP_with_error.append(Errors.Error)
#     DOP_with_error.append(Errors.DOP_array)
# #     # error_budget.append(Errors.ErrorBudget)
# max_HHDOP = np.max(DOP_with_error, axis=0)[5]
# print(f" The max HHDOP is :{np.round(max_HHDOP,3)}")
# #
# print(f"GDOP, PDOP, HDOP, VDOP, TDOP, HHDOP")
# print(f"DOP WITH ERROR, PERFECT ORBIT, mean:{np.mean(DOP_with_error, axis=0)}, max: {np.max(DOP_with_error, axis=0)}, "
#       f"min: {np.min(DOP_with_error, axis=0)}, ptp: {np.ptp(DOP_with_error, axis=0)},SD: {np.std(DOP_with_error, axis=0)}")

# print(f"ORBIT BUDGET FOR REQUIREMENT, mean:{np.mean(error_budget, axis=0)}, max: {np.max(error_budget, axis=0)}, "
#       f"min: {np.min(error_budget, axis=0)}, ptp: {np.ptp(error_budget, axis=0)},SD: {np.std(error_budget, axis=0)}")



# # # Dynamic Simulation
# satellites = model.getSatellites()
# duration = 86400 * 1
#
# # # PropagationTime(satellites, total_time, time_step, delta_v, n_planes, shift, elevation)
# propagation_time = PropagationTime(satellites, duration, 100, 250, 0, 0)
#
# # print(np.average(np.array(propagation_time.complete_delta_v(0, duration))))
#
# propagation_time.plot_kepler(0)
# propagation_time.plot_time()


class FrozenOrbits:
    """
    Class to check the parameters of frozen orbits,combination and position over time
    """

    def __init__(self):
        self.model = Model()
        self.distances = []
        self.moon_points = []
        self.satellite_indices = []
        self.requirements = [120.4, 10, 10, 10, 120, 3.5]  # GDOP, PDOP, HDOP, VDOP, TDOP, HHDOP
        self.orbit_choices = np.array([[8025.9e3, 0.004, 39.53, 270, 5, 4, 1, 0], [8148.8e3, 0.004, 39.51, 90, 5, 4, 1, 0],
                                       [7298.6e3, 0.001, 39.71, 270, 3, 7, 1, 1], [8669.2e3, 0.024, 39.46, 270, 4, 6, 1, 0],
                                       [8916.6e3, 0.000, 39.41, 90, 4, 6, 1, 1], [8904.4e3, 0.00, 39.41, 90, 4, 6, 1, 1], [7434.8e3, 0.00, 39.67, 270, 3, 7, 1, 1],
                                       [7298.6e3, 0.001, 39.71, 90, 3, 7, 1, 1], [8954.2e3, 0.002, 39.40, 90, 4, 6, 1, 1],
                                       [8536.0e3, 0.025, 39.47, 270, 4, 6, 1, 0], [5701.2e3, 0.002, 40.78, 90, 4, 6, 1, 1],
                                       [8855.4e3, 0.023, 39.43, 270, 4, 6, 1, 0], [8904.4e3, 0, 39.41, 90, 4, 6, 1, 1]])

        """## a: float = r_moon,
                         e: int = 0,
                         i: int = 0,
                         w: int = 0,
                         n_planes: int = 1,
                         n_sat_per_plane: int = 1,
                         dist_type: int = 0,
                         elevation: int = 15) -> Any"""

        self.orbit_ESA_SP = np.array([[9750.73e3, 0.6383, 54.33, 55.18, 277.53, 123.42],
                                      [9750.73e3, 0.6383, 54.33, 55.18, 277.53, 0],
                                      [9750.73e3, 0.6383, 61.96, 121.7, 59.27, 180],
                                      [9750.73e3, 0.6383, 61.96, 121.7, 59.27, 0]])

        self.orbit8sat = np.array([[8049e3, 0.4082, 45, 90, 0, 0],
                                   [8049e3, 0.4082, 45, 270, 0, 0],
                                   [8049e3, 0.4082, 45, 90, 180, 0],
                                   [8049e3, 0.4082, 45, 270, 180, 0]])

        self.constellation_12orbits = np.array([[8049e3, 0.4082, 45, 90, 0, 0],
                                                [8049e3, 0.4082, 45, 270, 0, 0],
                                                [8049e3, 0.4082, 45, 90, 180, 0],
                                                [8049e3, 0.4082, 45, 270, 180, 0]])

        self.constellation_JCT_M2O = np.array([[3737.4030e3, 0.0988, 48.2234, 89.7356, 0.0675, 0],
                                               [13677.7072e3, 0.0820, 40.3348, 86.5479, 0.41, 0]])

        self.constellation_NP = np.array([[6541.4e3, 0.6, 56.2, 270, 0, self.mean_to_true_anomaly(0.6, 0)],
                                     [6541.4e3, 0.6, 56.2, 270, 0, self.mean_to_true_anomaly(0.6, 120)],
                                     [6541.4e3, 0.6, 56.2, 270, 0, self.mean_to_true_anomaly(0.6, 240)]])

        self.constellation_MLO = np.array([[3476e3, 0.038, 15, 90, 0, self.mean_to_true_anomaly(0.038, 0)],
                                      [3476e3, 0.038, 15, 270, 0, self.mean_to_true_anomaly(0.038, 0)],
                                      [5214e3, 0.038, 15, 90, 0, self.mean_to_true_anomaly(0.038, 0)],
                                      [5214e3, 0.038, 15, 270, 0, self.mean_to_true_anomaly(0.038, 0)],
                                      [10000e3, 0.038, 15, 90, 0, self.mean_to_true_anomaly(0.038, 0)],
                                      [10000e3, 0.038, 15, 270, 0, self.mean_to_true_anomaly(0.038, 0)]])

        self.constellation_MLO = np.array([[5214e3, 0.038, 15, 90, 0, 0],
                                           [5214e3, 0.038, 15, 270, 0, 0],
                                           [10000e3, 0.038, 10, 90, 0, 0],
                                           [10000e3, 0.038, 10, 270, 0, 0]])

        self.constellation_MLO_5 = np.array([[5214e3, 0.006, 30, 90, 0, 0],
                                             [5214e3, 0.006, 30, 270, 0, 0],
                                             [10000e3, 0.006, 30, 90, 0, 0],
                                             [10000e3, 0.006, 30, 270, 0, 0]])

    def mean_to_true_anomaly(self, e, M):
        eccentric_anomaly = element_conversion.true_to_eccentric_anomaly(np.deg2rad(M), e)
        mean_anomaly = element_conversion.eccentric_to_mean_anomaly(eccentric_anomaly, e)
        return np.rad2deg(mean_anomaly)

    def true_anomaly_translation(self, satellites, change):
        satellites2 = satellites.copy()
        satellites2[:, 5] += change
        return satellites2

    def model_adder(self, satellites):
        for i in range(0, len(satellites)):
            self.model.addSatellite(satellites[i, 0], satellites[i, 1], satellites[i, 2], satellites[i, 3],
                                    satellites[i, 4], satellites[i, 5]) #, id=i)
        self.model.setCoverage()
        #self.model.plotCoverage()

    def model_symmetrical_planes(self, choice):
        self.model.addSymmetricalPlanes(self.orbit_choices[choice][0], self.orbit_choices[choice][1], self.orbit_choices[choice][2]
                                        , self.orbit_choices[choice][3], self.orbit_choices[choice][4], self.orbit_choices[choice][5], dist_type=self.orbit_choices[choice][6], f =self.orbit_choices[choice][7])
        self.model.setCoverage()
        # self.model.plotCoverage()
    def DOP_calculator(self):
        self.DOP_each_point = []
        self.DOP_each_point_with_error = []
        if np.min(self.model.mod_inView) >= 4:
            for i in range(0, len(self.model.moon)):

                self.distances.append(np.array([sat.r for sat in self.model.mod_inView_obj[i]]))
                self.moon_points.append(self.model.moon[i])
                # self.satellite_indices.append(np.array([sat.id for sat in self.model.mod_inView_obj[i]]))
                Errors = UserErrors(self.distances[-1], 0, 0, self.moon_points[-1], [120.4, 10, 10, 10, 120, 3.5])
                self.DOP_each_point.append(Errors.DOP_array)
                self.DOP_each_point_with_error.append(Errors.DOP_error_array)

            Ephemeris_error = Errors.allowable_error(self.DOP_each_point_with_error)
            print(Ephemeris_error, np.max(self.DOP_each_point, axis=0), np.median(self.DOP_each_point, axis=0))
            return((Ephemeris_error, np.max(self.DOP_each_point, axis=0), np.median(self.DOP_each_point, axis=0)))

    def dyn_sim(self, P, dt=10, kepler_plot=0):
        satellites = self.model.getSatellites()
        duration = 86400 * P/24
        propagation_time = PropagationTime(satellites, duration, dt, 250, 0, 0)
        # print(np.average(np.array(propagation_time.complete_delta_v(0, duration))))
        propagation_time.plot_kepler(kepler_plot)
        propagation_time.plot_time()

    def period_calc(self, satellites):
        P = np.zeros(np.shape(satellites)[0])
        for i in range(np.shape(satellites)[0]):
            P[i] = 2*np.pi * np.sqrt(satellites[i, 0]**3/miu_moon)
        return P/3600

constellations = []
fo = FrozenOrbits()
orbit_choice = 0
fo.model = Model()
fo.model_adder(fo.orbit_ESA_SP)
fo.model_symmetrical_planes(orbit_choice)
fo.DOP_calculator()


for i in range(0, 13):
    fo.model = Model()
    fo.model_symmetrical_planes(i)
    constellations.append(fo.DOP_calculator())

P = fo.period_calc(fo.orbit_choices)[orbit_choice]
print(fo.period_calc(fo.orbit_choices)[orbit_choice])

constellations = np.asarray(constellations)
# print(fo.period_calc(fo.orbit_choices))
print(constellations)

fo.dyn_sim(P)
