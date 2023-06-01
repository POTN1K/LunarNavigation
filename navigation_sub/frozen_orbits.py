"""Frozen Orbits Simulator/Optimisation.
Maintained by Kyle and Serban"""

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
        self.requirements = [20, 10, 10, 10, 10, 3.5]  # GDOP, PDOP, HDOP, VDOP, TDOP, HHDOP

        self.final_orbits = np.array([[8025.9e3, 0.004, 39.53, 270, 0, mean_to_true_anomaly()],
                                      [8049e3, 0.4082, 45, 270, 0, 0],
                                      [8049e3, 0.4082, 45, 90, 180, 0],
                                      [8049e3, 0.4082, 45, 270, 180, 0]])

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

        self.constellation_SP = np.array([[6541.4e3, 0.6, 56.2, 90, 0, 0]])

        self.constellation_NP = np.array([[6541.4e3, 0.6, 56.2, 270, 0, 0]])

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
            self.model.addSatellite(satellites[i][0], satellites[i][1], satellites[i][2], satellites[i][3],
                                    satellites[i][4], satellites[i][5],id=i)
        self.model.setCoverage()
        self.model.plotCoverage()

    def DOP_calculator(self):
        self.DOP_each_point = []
        self.DOP_each_point_with_error = []
        for i in range(0, len(self.model.moon)):
            self.distances.append(np.array([sat.r for sat in self.model.mod_inView_obj[i]]))
            self.moon_points.append(self.model.moon[i])
            self.satellite_indices.append(np.array([sat.id for sat in self.model.mod_inView_obj[i]]))
            Errors = UserErrors(self.distances[-1], 0, 0, self.moon_points[-1], [59, 10, 100, 100, 100, 3.5])
            self.DOP_each_point.append(Errors.DOP_array)
            self.DOP_each_point_with_error.append(Errors.DOP_error_array)

        Ephemeris_error = Errors.allowable_error(np.asarray(self.DOP_each_point_with_error)[:, :-1])
        print(Ephemeris_error, (np.max(self.DOP_each_point, axis=0))[5])

    def dyn_sim(self, satellites):
        self.propagation_time = PropagationTime(satellites, duration, dt, 250, 0, 0)
        #DV = np.round(np.average(np.array(self.propagation_time.complete_delta_v(0, duration))), 3)
        self.propagation_time.plot_kepler(0)
        self.propagation_time.plot_time()

    def period_calc(self, satellites):
        P = np.zeros(np.shape(satellites)[0])
        for i in range(np.shape(satellites)[0]):
            P[i] = 2*np.pi * np.sqrt(satellites[i, 0]**3/miu_moon)
        return P

fo = FrozenOrbits()
satellites = fo.constellation_12orbits

fo.model_adder(satellites)
P = fo.period_calc(satellites)/3600
print(P)
days = 18/24
duration = 86400 * days
dt = 1

fo.dyn_sim(satellites)
fo.DOP_calculator()