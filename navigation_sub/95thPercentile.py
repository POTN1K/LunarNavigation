import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pylab import cm
from mission_design import Model, UserErrors


def Normal_Distrib(self):
    DOPS = ["GDOP", "PDOP", "HDOP", "VDOP", "TDOP", "HHDOP"]
    for i in range(0,2):
        with open(self.DOP, 'r') as file:
            reader = csv.reader(file)
            self.data = np.array([[float(element) for element in row] for row in reader])
            ## time normal distrib

            self.ninetyfifth_percent = np.percentile(self.data, 95, axis=0)
            print(self.ninetyfifth_percent.shape)
        # std_t = np.std(self.data[:, 10])
        # print(mean_t)
        # print(np.max(self.data[:, 10]))

    # x = np.linspace(mean_t - 3 * std_t, mean_t + 3 * std_t, 100)
    # y = (1 / (np.sqrt(2 * np.pi) * std_t)) * np.exp(-0.5 * ((x - mean_t) / std_t) ** 2)
    # plt.plot(x, y, label='Normal Distribution')
    # plt.scatter(self.data[:, 10], np.zeros_like(self.data[:, 10]), alpha=0.5, label=self.DOP + 'in Time')
    # plt.axvline(mean_t - std_t, color='y', linestyle='--', label='1 Sigma')
    # plt.axvline(mean_t + std_t, color='y', linestyle='--')
    # plt.axvline(mean_t - 2 * std_t, color='g', linestyle='--', label='2 Sigma')
    # plt.axvline(mean_t + 2 * std_t, color='g', linestyle='--')
    # plt.axvline(mean_t - 3 * std_t, color='b', linestyle='--', label='3 Sigma')
    # plt.axvline(mean_t + 3 * std_t, color='b', linestyle='--')
    # plt.axvline(120.4, color='r', linestyle='-', label='GDOP Requirement')
    # plt.legend()
    # plt.show()
    # ### Maybe instead of plotting : save the images and compare them later
    ### Make a code w/ if 2sigma > GDOP req then it is bad

# fo = FrozenOrbits("model0")
# Normal_Distrib(fo)
DOPS = ["GDOP", "PDOP", "HDOP", "VDOP", "TDOP", "HHDOP"]
VOP = ["VTOT", "VH", "VV"]
boxplotscancer = np.zeros(10000)
maxdops =[]
for i in range(4, 6):
    filename = "modelreduntot" + DOPS[i] + ".csv"
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = np.array([[float(element) for element in row] for row in reader])
        ninetyfifth_percent = np.percentile(data, 95, axis=0)
    boxplotscancer = np.vstack((boxplotscancer, ninetyfifth_percent))

boxplotscancer = array = np.delete(boxplotscancer, 0, 0).T

# boxplotscancer_V = np.zeros(10000)
# for j in range(0, 1):
#     filename = "modelreduntot" + DOPS[j] + ".csv"
#     with open(filename, 'r') as file:
#         reader = csv.reader(file)
#         data = np.array([[float(element) for element in row] for row in reader])
#         ninetyfifth_percent = np.percentile(data, 95, axis=0)
#     boxplotscancer_V = np.vstack((boxplotscancer_V, ninetyfifth_percent))
#
# boxplotscancer_V = array = np.delete(boxplotscancer_V, 0, 0).T

SNR = 8  # [dB]
BW = 70  # [dB]
CNR = SNR + BW  # [dB]
sigma_V_1 = 0.01*10**(-(CNR-45)/20)  # [m/s]
sigma_V = 0.06  # [m/s]

def allowable_error(DOP_array, DOP_array_V, allowable, allowable_V):
    constraints = np.max(DOP_array, axis=0)
    ephemeris_budget = []
    for i in range(len(constraints)):
        ephemeris_budget.append(np.sqrt((allowable[i] ** 2 / constraints[i] ** 2/4) - UserErrors.satellite_error(0, ORBIT=0) ** 2))
    ephemeris_budget = np.array(ephemeris_budget)

    constraints_V = np.max(DOP_array_V, axis=0)
    print(constraints_V)
    ephemeris_budget_V = []
    for i in range(len(constraints_V)):
        ephemeris_budget_V.append(
            np.sqrt((allowable_V[i] ** 2 / constraints_V[i] ** 2 / 4) - sigma_V ** 2))
    ephemeris_budget_V = np.array(ephemeris_budget_V)

    return ephemeris_budget, ephemeris_budget_V
# print(allowable_error(boxplotscancer, boxplotscancer_V, [120.4, 120, 120, 100, 300, 3.5], [1, 1, 1, 2, 1, 1]))

### Max 95th percentile DOP values
# [6.0822285  5.0641006  4.4073417  3.43252637 3.40788531 3.0694594 ]

### Allowable error for position and velocity for sigma_V from ERDA reader
#(array([ 9.89397936,  0.94944433,  1.10164593,  1.43123656, 17.60414169,
#       16.28726027]), array([0.08220641, 0.09873396, 0.11344685, 0.14566513, 0.14671838,
#       0.16289498]))

### Allowable error for position and velocity for sigma_V from Kyle's source
# (array([ 9.89397936,  0.94944433,  1.10164593,  1.43123656, 17.60414169,
#        16.28726027]), array([0.05619558, 0.07841202, 0.09628207, 0.13273425, 0.13388926,
#        0.15144248]))

### Allowable error for PDOP, + HDOP&VDOP for orbiting and landing and velocity PDOP + Orbiting and landing HDOP & VDOP for sigma_V from Kyle's source
# (array([ 9.89397936,  9.8697038 ,  5.6658798 , 14.56401079, 17.60414169,
#        16.28726027]), array([0.05619558, 0.07841202, 0.09628207, 0.28508512, 0.13388926,
#        0.15144248]))

### Allowable error for PDOP, + HDOP&VDOP for orbiting and landing and velocity PDOP + Orbiting and landing HDOP & VDOP for sigma_V from ERDA reader
# (array([ 9.89397936,  9.8697038 ,  5.6658798 , 14.56401079, 17.60414169,
#        16.28726027]), array([0.08220641, 0.09873396, 0.11344685, 0.29133052, 0.14671838,
#        0.16289498]))

#
# def boxplot(df, df_V):
#     plt.figure(figsize=(12, 8))
#     column_names = DOPS
#     allowable = [120, 100, 100, 100, 120, 3.5]
#     sns.boxplot(data=df)
#     plt.xticks(range(df.shape[1]), column_names)
#     for i in range(df.shape[1]):
#         plt.plot([i - 0.5, i + 0.5], [allowable[i]] * 2, color='red')
#     plt.title("95th percentile points over time")
#     plt.show()
#
#     plt.figure(figsize=(12, 8))
#     column_names_V = DOPS
#     allowable_V = [1, 1, 1, 1, 1, 1]
#     sns.boxplot(data=df_V)
#     plt.xticks(range(df_V.shape[1]), column_names_V)
#     for i in range(df_V.shape[1]):
#         plt.plot([i - 0.5, i + 0.5], [allowable_V[i]] * 2, color='red')
#     plt.title("95th percentile points over time")
#     plt.show()

# boxplot(boxplotscancer, boxplotscancer_V)


r_moon = 1.737e6  # m
miu_moon = 4.9048695e12  # m^3/s^2



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot Moon
# phi = np.linspace(0, 2 * np.pi, 100)
# theta = np.linspace(0, np.pi, 100)
# # Create a meshgrid of phi and theta values
# phi, theta = np.meshgrid(phi, theta)
# # Calculate the x, y, and z coordinates for each point on the sphere
# xM = r_moon * np.sin(theta) * np.cos(phi)
# yM = r_moon * np.sin(theta) * np.sin(phi)
# zM = r_moon * np.cos(theta)
#
# ax.plot_surface(xM, yM, zM, color='grey', alpha=0.2)
#
# boxplotpointsmap = boxplotscancer[:, 5]
#
# # Plot satellites in view
# color_map = cm.ScalarMappable(cmap='PiYG')
# color_map.set_array(boxplotscancer[:, 5])
# #
# ax.scatter(*zip(*Model.createMoon(100)), marker='s', s=1, c=boxplotscancer[:, 5], cmap='PiYG')
# plt.colorbar(color_map)

grid = (boxplotscancer[:, 1]).reshape((100, 100)).T

# Create the plot
plt.imshow(grid, extent=[-180, 180, -90, 90], cmap='PiYG')  # You can choose a different colormap if you prefer
colorbar = plt.colorbar()  # Add a colorbar for reference
colorbar.ax.tick_params(labelsize=20)

colorbar.set_label('HHDOP values', fontsize=25)
plt.xlabel("Longitude [$\degree$]", fontsize=25)
plt.xlim(-180, 180)
plt.ylabel("Latitude [$\degree$]", fontsize=25)
plt.ylim(-90, 90)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], fontsize=20)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], fontsize=20)
plt.show()

# ax.set_title('Satellite coverage')
# ax.set_xlabel('x [$10^7$ m]')
# ax.set_ylabel('y [$10^7$ m]')
# ax.set_zlabel('z [$10^7$ m]')
#
# ax.set_xlim(-r_moon * 3, r_moon * 3)
# ax.set_ylim(-r_moon * 3, r_moon * 3)
# ax.set_zlim(-r_moon * 3, r_moon * 3)
# ax.set_aspect('equal')
# plt.show()