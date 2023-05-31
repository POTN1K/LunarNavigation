"""Main File, used to run all simulations.
Maintained by Nikolaus Ricker"""

# External Libraries
import numpy as np
from tudatpy.kernel.astro import element_conversion
import sys

sys.path.append('.')

# Local Libraries
from mission_design import Model, PropagationTime, UserErrors


#
# # -----------------------------------------------------------------------------
# #Create Kepler array
# model = Model()
#
# def M_to_nu(e, M):
#     eccentric_anomaly = element_conversion.true_to_eccentric_anomaly(np.deg2rad(M), e)
#     mean_anomaly = element_conversion.eccentric_to_mean_anomaly(eccentric_anomaly, e)
#     return np.rad2deg(mean_anomaly)
#
# constellation_8orbits = np.array([[8049e3, 0.4082, 45, 90, 0, M_to_nu(0.4082, 0)],
#                                   [8049e3, 0.4082, 45, 90, 0, M_to_nu(0.4082, 180)],
#                                   [8049e3, 0.4082, 45, 270, 0, M_to_nu(0.4082, 0)],
#                                   [8049e3, 0.4082, 45, 270, 0, M_to_nu(0.4082, 180)],
#                                   [8049e3, 0.4082, 45, 90, 180, M_to_nu(0.4082, 132)],
#                                   [8049e3, 0.4082, 45, 90, 180, M_to_nu(0.4082, 228)],
#                                   [8049e3, 0.4082, 45, 270, 180, M_to_nu(0.4082, 132)],
#                                   [8049e3, 0.4082, 45, 270, 180, M_to_nu(0.4082, 228)]])
# constellation_8orbits_2 = constellation_8orbits.copy()
# constellation_8orbits_3= constellation_8orbits.copy()
# constellation_8orbits_2[:, 5] += 45
# constellation_8orbits_3[:, 5] += 135
#
#
# #constellation_SP_orbits = np.array([[]])
# #JCT_M20 = np.array([[]])
# # #MLO = np.array([[]])
#
#
#
#
# satellites_initial = np.vstack((constellation_8orbits , constellation_8orbits_2, constellation_8orbits_3))
# for i in range(0, len(satellites_initial)):
#     model.addSatellite(satellites_initial[i][0],satellites_initial[i][1],satellites_initial[i][2],satellites_initial[i][3],satellites_initial[i][4],satellites_initial[i][5])
# model.setCoverage()
#
# # model.addSatelliteComb(a=8049e3, e=0.4082, i=45, w=[90, 270], Omega=0, nu=[0,150,210])
# # print(model.modules)
#
# # Static Simulation
# # Create model
#
#
# # # Add preliminary orbit
# # model.addSymmetricalPlanes(a=24572000, i=58.69, e=0, w=22.9, n_planes=6, n_sat_per_plane=4 dist_type=1)
# # model.setCoverage()
# # # Plot coverage
# model.plotCoverage()
#
# # # -----------------------------------------------------------------------------
# # Continue?
# flag = input("Press Enter to continue... 'e' to exit.\n")
# if flag != '':
#     exit()
#
# # -----------------------------------------------------------------------------
# # DOP Calculation
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
#
# # print(f"ORBIT BUDGET FOR REQUIREMENT, mean:{np.mean(error_budget, axis=0)}, max: {np.max(error_budget, axis=0)}, "
# #       f"min: {np.min(error_budget, axis=0)}, ptp: {np.ptp(error_budget, axis=0)},SD: {np.std(error_budget, axis=0)}")
#
# # # -----------------------------------------------------------------------------
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
        self.distances = np.array([])
        self.moon_points = np.array([])
        self.satellite_indices = np.array([])
        self.requirements = [20, 10, 10, 10, 10, 3.5]  # GDOP, PDOP, HDOP, VDOP, TDOP, HHDOP
        self.orbit8sat = np.array([[8049e3, 0.4082, 45, 90, 0, 0],
                                   [8049e3, 0.4082, 45, 90, 0, 180],
                                   [8049e3, 0.4082, 45, 270, 0, 0],
                                   [8049e3, 0.4082, 45, 270, 0, 180],
                                   [8049e3, 0.4082, 45, 90, 180, 132],
                                   [8049e3, 0.4082, 45, 90, 180, 228],
                                   [8049e3, 0.4082, 45, 270, 180, 132],
                                   [8049e3, 0.4082, 45, 270, 180, 228]])

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
                                    satellites[i][4], satellites[i][5])
        self.model.setCoverage()

    def points_in_view(self):
        for i in range(0, len(self.model.moon)):
            self.distances = np.append(self.distances, np.array([sat.r for sat in self.model.mod_inView_obj[i]]))
            self.moon_points = np.append(self.moon_points, self.model.moon[i])
            self.satellite_indices = np.append(self.satellite_indices,
                                               np.array([sat.id for sat in self.model.mod_inView_obj[i]]))



FrozenOrbits()
