"""Main File, used to run all simulations.
Maintained by Nikolaus Ricker"""

# External Libraries
import numpy as np
from tudatpy.kernel.astro import element_conversion
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import csv
import pandas as pd
from pylab import cm
sys.path.append('.')
import matplotlib.image as mpimg

# Local Libraries
from mission_design import Model, PropagationTime, UserErrors

# cst
miu_moon = 4.9048695e12  # m^3/s^2
c = 299792458  # m/s

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

    def __init__(self, File):
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

        # self.orbit_choices2 = np.array([[5701.2e3, 0.002, 40.78, 90, 0, 0],
        #                                 [5701.2e3, 0.002, 40.78, 90, 0, 180]])
        #
        # self.Jasper_sat = np.array([[6142.4e3, 0.6, 57.7, 90, 0, 0],
        #                             [6142.4e3, 0.6, 57.7, 90, 0, 0]])

        # self.orbit_ESA_SP = np.array([[9750.73e3, 0.6383, 54.33, 55.18, 277.53, 123.42],
        #                               [9750.73e3, 0.6383, 54.33, 55.18, 277.53, 0],
        #                               [9750.73e3, 0.6383, 61.96, 121.7, 59.27, 180],
        #                               [9750.73e3, 0.6383, 61.96, 121.7, 59.27, 0]])
        #
        # self.orbit8sat = np.array([[8049e3, 0.4082, 45, 90, 0, 0],
        #                            [8049e3, 0.4082, 45, 270, 0, 0],
        #                            [8049e3, 0.4082, 45, 90, 180, 0],
        #                            [8049e3, 0.4082, 45, 270, 180, 0]])
        #
        # self.constellation_12orbits = np.array([[8049e3, 0.4082, 45, 90, 0, 0],
        #                                         [8049e3, 0.4082, 45, 270, 0, 0],
        #                                         [8049e3, 0.4082, 45, 90, 180, 0],
        #                                         [8049e3, 0.4082, 45, 270, 180, 0]])
        #
        # self.constellation_JCT_M2O = np.array([[3737.4030e3, 0.0988, 48.2234, 89.7356, 0.0675, 0],
        #                                        [13677.7072e3, 0.0820, 40.3348, 86.5479, 0.41, 0]])

        self.constellation_NP = np.array([[6541.4e3, 0.6, 56.2, 270, 0, 0],
                                          [6541.4e3, 0.6, 56.2, 270, 0, 120],
                                          [6541.4e3, 0.6, 56.2, 270, 0, 240]])

        self.constellation_SP = np.array([[6541.4e3, 0.6, 56.2, 90, 0, 0],
                                          [6541.4e3, 0.6, 56.2, 90, 0, 120],
                                          [6541.4e3, 0.6, 56.2, 90, 0, 240]])

        # self.constellation_MLO = np.array([[3476e3, 0.038, 15, 90, 0, self.true_to_mean_anomaly(0.038, 0)],
        #                               [3476e3, 0.038, 15, 270, 0, self.true_to_mean_anomaly(0.038, 0)],
        #                               [5214e3, 0.038, 15, 90, 0, self.true_to_mean_anomaly(0.038, 0)],
        #                               [5214e3, 0.038, 15, 270, 0, self.true_to_mean_anomaly(0.038, 0)],
        #                               [10000e3, 0.038, 15, 90, 0, self.true_to_mean_anomaly(0.038, 0)],
        #                               [10000e3, 0.038, 15, 270, 0, self.true_to_mean_anomaly(0.038, 0)]])
        #
        # self.constellation_MLO = np.array([[5214e3, 0.038, 15, 90, 0, 0],
        #                                    [5214e3, 0.038, 15, 270, 0, 0],
        #                                    [10000e3, 0.038, 10, 90, 0, 0],
        #                                    [10000e3, 0.038, 10, 270, 0, 0]])
        #
        # self.constellation_MLO_5 = np.array([[5214e3, 0.006, 30, 90, 0, 0],
        #                                      [5214e3, 0.006, 30, 270, 0, 0],
        #                                      [10000e3, 0.006, 30, 90, 0, 0],
        #                                      [10000e3, 0.006, 30, 270, 0, 0]])

        self.orbit_Low_I = np.array([[10000e3, 0.038, 10, 90, 0, 30],
                                     [10000e3, 0.038, 10, 90, 0, 102],
                                     [10000e3, 0.038, 10, 90, 0, 174],
                                     [10000e3, 0.038, 10, 90, 0, 246],
                                     [10000e3, 0.038, 10, 90, 0, 218]])

        self.DOP = File
        self.data = []

    def true_to_mean_anomaly(self, e, M):
        eccentric_anomaly = element_conversion.true_to_eccentric_anomaly(np.deg2rad(M), e)
        mean_anomaly = element_conversion.eccentric_to_mean_anomaly(eccentric_anomaly, e)
        return np.rad2deg(mean_anomaly)

    def mean_to_true_anomaly(self, e, M):
        true_anomaly = element_conversion.mean_to_true_anomaly(e, np.deg2rad(M))
        return np.rad2deg(true_anomaly)

    def true_anomaly_translation(self, satellites, change):
        satellites2 = satellites.copy()
        satellites2[:, 5] += change
        return satellites2

    def model_adder(self, satellites):
        for i in range(0, len(satellites)):
            self.model.addSatellite(satellites[i, 0], satellites[i, 1], satellites[i, 2], satellites[i, 3],
                                    satellites[i, 4], satellites[i, 5], id=i)
        #self.model.plotCoverage()

    def model_symmetrical_planes(self, choice):
        self.model.addSymmetricalPlanes(self.orbit_choices[choice][0], self.orbit_choices[choice][1], self.orbit_choices[choice][2]
                                        , self.orbit_choices[choice][3], int(self.orbit_choices[choice][4]), int(self.orbit_choices[choice][5]), dist_type=int(self.orbit_choices[choice][6]), f =int(self.orbit_choices[choice][7]), id_start=7)
        self.model.setCoverage()
        # self.model.plotCoverage()

    def DOP_calculator(self, sat_velocities, plotting=False):

        self.DOP_each_point = []
        self.DOP_each_point_with_error = []
        self.velocity_DOP = []
        self.distances = []
        self.satellite_indices = []
        self.maxdist = []
        if np.min(self.model.mod_inView) >= 4:
            for i in range(0, len(self.model.moon)):

                self.distances.append(np.array([sat.r for sat in self.model.mod_inView_obj[i]]))
                self.satellite_indices.append(np.array([sat.id for sat in self.model.mod_inView_obj[i]]))
                self.moon_points.append(self.model.moon[i])
                # self.satellite_indices.append(np.array([sat.id for sat in self.model.mod_inView_obj[i]]))
                Errors = UserErrors(self.distances[-1], sat_velocities, 0, self.moon_points[-1], [120.4, 10, 10, 10, 120, 3.5])
                self.DOP_each_point.append(np.hstack((Errors.DOP_array, Errors.velocity_parameter_cov(self.satellite_indices[-1]))))
                self.DOP_each_point_with_error.append(Errors.DOP_error_array)
            self.DOP_each_point = np.asarray(self.DOP_each_point)
            self.DOP_each_point_with_error = np.asarray(self.DOP_each_point_with_error)
            # self.maxdist.append(np.max(np.concatenate(self.distances)))
            # HHDOP_ephemeris = Errors.allowable_error(self.DOP_each_point)
            if plotting == True:
                self.boxplot(self.DOP_each_point)
            print(np.max(self.DOP_each_point, axis=0))

            Ephemeris_error = Errors.allowable_error(self.DOP_each_point_with_error)
            # print(Ephemeris_error, np.max(self.DOP_each_point, axis=0), np.median(self.DOP_each_point, axis=0))
            return(np.asarray(self.DOP_each_point))
        else:
            print("You suck")
            return np.asarray([["False","False","False","False","False","False"]])

    def boxplot(self, df):
        plt.figure(figsize=(12, 8))
        column_names = ['GDOP', 'PDOP', 'HDOP', 'VDOP', 'TDOP', 'HHDOP']
        sns.boxplot(data=df)
        plt.xticks(range(df.shape[1]), column_names)
        for i in range(df.shape[1]):
            upper_limit = np.percentile(df[:, i], 95)
            plt.plot([i - 0.5, i + 0.5], [upper_limit] * 2, color='red')
        plt.title("Boxplots with 95% lines ")
        plt.show()


    def dyn_sim(self, P, dt=10, kepler_plot=0):
        satellites = self.model.getSatellites()
        duration = P
        self.propagation_time = PropagationTime(satellites, duration, dt, 250)
        # print(np.average(np.array(propagation_time.complete_delta_v(0, duration))))
        # self.propagation_time.plot_kepler(kepler_plot)
        # self.propagation_time.plot_time()

    def ideal_sim(self, P, satellites):
        self.cartesian_states = np.array([(element_conversion.keplerian_to_cartesian_elementwise(
                gravitational_parameter=miu_moon,
                semi_major_axis=satellites[0],
                eccentricity=satellites[1],
                inclination=np.deg2rad(satellites[2]),
                argument_of_periapsis=np.deg2rad(satellites[3]),
                longitude_of_ascending_node=0,
                true_anomaly=0,)[:])])
        self.t = np.linspace(0, int(np.round(P, 0)), int(np.round(P+1, 0)))
        self.M = 360/P * self.t
        self.mu = np.zeros(np.shape(self.M))
        for i in range(len(self.M)):
            self.mu[i] = self.mean_to_true_anomaly(satellites[1], self.M[i])
        for x in range(len(self.mu)):
            self.cartesian_states = np.vstack((self.cartesian_states, element_conversion.keplerian_to_cartesian_elementwise(
                gravitational_parameter=miu_moon,
                semi_major_axis=satellites[0],
                eccentricity=satellites[1],
                inclination=np.deg2rad(satellites[2]),
                argument_of_periapsis=np.deg2rad(satellites[3]),
                longitude_of_ascending_node=0,
                true_anomaly=np.deg2rad(self.mu[x]), )[:]))
        return self.cartesian_states

    def period_calc(self, satellites):
        P = np.zeros(np.shape(satellites)[0])
        for i in range(np.shape(satellites)[0]):
            P[i] = 2*np.pi * np.sqrt(satellites[i, 0]**3/miu_moon)
        return P

    def DOP_time(self, satellites):
        self.DOP_time_GDOP = []
        self.DOP_time_PDOP = []
        self.DOP_time_HDOP = []
        self.DOP_time_VDOP = []
        self.DOP_time_TDOP = []
        self.DOP_time_HHDOP = []
        self.velocity_time_total = []
        self.velocity_time_horizontal = []
        self.velocity_time_vertical = []
        sats = np.arange(0, 35)
        self.combinations = list(itertools.combinations(sats, 2))
        for k in range(0, 40, 5):
            for i in range(0, satellites.shape[0], 1000):
                self.model.resetModel()
                sat_velocities = self.propagation_time.velocity[i]
                for j in range(0, satellites.shape[1]//6):
                    # if j != self.combinations[k][0] and j != self.combinations[k][1]:
                    self.model.addSatellite(satellites[i][j*6], satellites[i][j*6+1], np.rad2deg(satellites[i][j*6+2]), np.rad2deg(satellites[i][j*6+3]),
                                np.rad2deg(satellites[i][j*6+4]), np.rad2deg(satellites[i][j*6+5]), id=j, elevation=k)
                self.model.setCoverage()
                DOPValues = self.DOP_calculator(sat_velocities)
                self.DOP_time_GDOP.append(DOPValues[:, 0])
                self.DOP_time_PDOP.append(DOPValues[:, 1])
                self.DOP_time_HDOP.append(DOPValues[:, 2])
                self.DOP_time_VDOP.append(DOPValues[:, 3])
                self.DOP_time_TDOP.append(DOPValues[:, 4])
                self.DOP_time_HHDOP.append(DOPValues[:, 5])
                self.velocity_time_total.append(DOPValues[:, 6])
                self.velocity_time_horizontal.append(DOPValues[:, 7])
                self.velocity_time_vertical.append(DOPValues[:, 8])
            np.savetxt("model"+str(k)+"VTOT.csv", np.asarray(self.velocity_time_total), delimiter=",")
            np.savetxt("model"+str(k)+"VH.csv",  np.asarray(self.velocity_time_horizontal), delimiter=",")
            np.savetxt("model"+str(k)+"VV.csv",  np.asarray(self.velocity_time_vertical), delimiter=",")
            np.savetxt("model"+str(k)+"GDOP.csv",  np.asarray(self.DOP_time_GDOP), delimiter=",")
            np.savetxt("model"+str(k)+"PDOP.csv",  np.asarray(self.DOP_time_PDOP), delimiter=",")
            np.savetxt("model"+str(k)+"HDOP.csv",  np.asarray(self.DOP_time_HDOP), delimiter=",")
            np.savetxt("model"+str(k)+"VDOP.csv",  np.asarray(self.DOP_time_VDOP), delimiter=",")
            np.savetxt("model"+str(k)+"TDOP.csv",  np.asarray(self.DOP_time_TDOP), delimiter=",")
            np.savetxt("model"+str(k)+"HHDOP.csv",  np.asarray(self.DOP_time_HHDOP), delimiter=",")


constellations = []
fo = FrozenOrbits("model10GDOP.csv")
orbit_choice = 10
fo.model = Model()


# fo.model_adder(fo.Jasper_sat)
fo.model_adder(np.vstack((fo.constellation_SP, fo.constellation_NP, fo.orbit_Low_I)))
fo.model_symmetrical_planes(orbit_choice)


# kepler_plot = [0, 2, 4, 7]
# for i in range(len(kepler_plot)):
#     fo.dyn_sim(86400*30, dt=10, kepler_plot=kepler_plot[i])

# fo.model_adder(fo.orbit_choices2)
# P = fo.period_calc(fo.orbit_choices)[orbit_choice]
#
# with open("statesarray.csv", 'r') as file:
#     reader = csv.reader(file)
#     data = np.array([[float(element) for element in row] for row in reader])
#
# real_cart = data[:38621, 43:46]

# real_kepl = np.zeros((np.shape(real_cart)[0], 6))
# for i in range(5000):
#     real_kepl[i, :] = element_conversion.cartesian_to_keplerian(data[i, 43:49], miu_moon)

# ideal_cart = fo.ideal_sim(P, fo.orbit_choices[orbit_choice])
# ideal_cart = ideal_cart[1:, 0:3]
#
# dist_cart = np.zeros(np.shape(ideal_cart)[0])
# for i in range(np.shape(ideal_cart)[0]):
#     dist_cart[i] = np.sqrt((ideal_cart[i, 0] - real_cart[i, 0]) ** 2 + (ideal_cart[i, 1] - real_cart[i, 1]) ** 2 + (ideal_cart[i, 2] - real_cart[i, 2]) ** 2)
#
# print(np.max(dist_cart), np.argmax(dist_cart))
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(ideal_cart[-2:, 0], ideal_cart[-2:, 1], ideal_cart[-2:, 2], c='g', marker='x')
# ax.scatter3D(real_cart[-2:, 0], real_cart[-2:, 1], real_cart[-2:, 2], c='r', marker='o')
# plt.show()




# fo.model.addSymmetricalPlanes(a=24572000, i=58.69, e=0, w=22.9, n_planes=6, n_sat_per_plane=4, dist_type=1)

# fo.DOP_calculator(True)

# P = fo.period_calc(fo.orbit_choices)[orbit_choice]
# print(fo.period_calc(fo.orbit_choices)[orbit_choice])

# constellations = np.asarray(constellations)

P_max = 2*np.pi * np.sqrt(10000000**3/miu_moon)
fo.dyn_sim(P_max)
fo.propagation_time.plot_time()

# np.savetxt("statesarray_radiation.csv", fo.propagation_time.states_array, delimiter=",")


# with open("statesarray.csv", 'r') as file:
#     reader = csv.reader(file)
#     data = np.array([[float(element) for element in row] for row in reader])
#
# real_cart = data[:38621, 43:46]
#
# with open("statesarray_relativity_radiation.csv", 'r') as file:
#     reader = csv.reader(file)
#     data = np.array([[float(element) for element in row] for row in reader])
#
# assumption_cart = data[:38621, 43:46]
#
# dist1_cart = np.zeros(np.shape(assumption_cart)[0])
# for i in range(np.shape(assumption_cart)[0]):
#     dist1_cart[i] = np.sqrt((assumption_cart[i, 0] - real_cart[i, 0]) ** 2 + (assumption_cart[i, 1] - real_cart[i, 1]) ** 2 + (assumption_cart[i, 2] - real_cart[i, 2]) ** 2)
#
# print(np.where(dist1_cart >= 1)[0][0])
# print(dist1_cart[np.where(dist1_cart >= 1)[0][0]])
# print(np.max(dist1_cart), np.argmax(dist1_cart))
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(assumption_cart[::500, 0], assumption_cart[::500, 1], assumption_cart[::500, 2], c='g', marker='x')
# ax.scatter3D(real_cart[::500, 0], real_cart[::500, 1], real_cart[::500, 2], c='r', marker='o')
# plt.show()

# time = np.arange(0, P_max+1, 1)
# print(np.max(fo.propagation_time.accelsun))
# plt.semilogy(time, fo.propagation_time.accelsun)
# plt.show()

fo.DOP_time(fo.propagation_time.kepler_elements)
# # print(constellations)