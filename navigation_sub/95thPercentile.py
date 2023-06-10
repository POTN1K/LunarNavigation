import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import cm
from mission_design import Model, UserErrors

def Normal_Distrib(self):
    DOPS = ["GDOP", "PDOP", "HDOP", "VDOP", "TDOP", "HHDOP"]
    for i in range(0,6):
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
for i in range(0, 6):
    filename = "model10NPSPEQ31" + DOPS[i]+ ".csv"
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = np.array([[float(element) for element in row] for row in reader])
        ninetyfifth_percent = np.percentile(data, 95, axis=0)
    boxplotscancer = np.vstack((boxplotscancer, ninetyfifth_percent))

boxplotscancer = array = np.delete(boxplotscancer, 0, 0).T


def allowable_error(DOP_array, allowable):
    constraints = np.max(DOP_array, axis=0)
    ephemeris_budget = []
    for i in range(len(constraints)):
        ephemeris_budget.append(np.sqrt((allowable[i] ** 2 / constraints[i] ** 2/4 ) - UserErrors.satellite_error(0, ORBIT=0) ** 2))
    ephemeris_budget = np.array(ephemeris_budget)
    return ephemeris_budget
print(allowable_error(boxplotscancer,[120.4, 10, 10, 10, 120, 100]))
def boxplot(df):

    plt.figure(figsize=(12, 8))
    column_names = ["GDOP", "PDOP", "HDOP", "VDOP", "TDOP", "HHDOP"]
    allowable = [120, 10, 10, 10, 10, 10]
    sns.boxplot(data=df)
    plt.xticks(range(df.shape[1]), column_names)
    for i in range(df.shape[1]):
        plt.plot([i - 0.5, i + 0.5], [allowable[i]] * 2, color='red')
    plt.title("95th percentile points over time")
    plt.show()

boxplot(boxplotscancer)

r_moon = 1.737e6  # m
miu_moon = 4.9048695e12  # m^3/s^2



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Moon
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
# Create a meshgrid of phi and theta values
phi, theta = np.meshgrid(phi, theta)
# Calculate the x, y, and z coordinates for each point on the sphere
xM = r_moon * np.sin(theta) * np.cos(phi)
yM = r_moon * np.sin(theta) * np.sin(phi)
zM = r_moon * np.cos(theta)

ax.plot_surface(xM, yM, zM, color='grey', alpha=0.2)

boxplotpointsmap = boxplotscancer[:, 1]

# Plot satellites in view
color_map = cm.ScalarMappable(cmap='PiYG')
color_map.set_array(boxplotscancer[:, 0])
#
ax.scatter(*zip(*Model.createMoon(100)), marker='s', s=1, c=boxplotscancer[:, 0], cmap='PiYG')
plt.colorbar(color_map)

# ax.set_title('Satellite coverage')
ax.set_xlabel('x [$10^7$ m]')
ax.set_ylabel('y [$10^7$ m]')
ax.set_zlabel('z [$10^7$ m]')

ax.set_xlim(-r_moon * 3, r_moon * 3)
ax.set_ylim(-r_moon * 3, r_moon * 3)
ax.set_zlim(-r_moon * 3, r_moon * 3)
ax.set_aspect('equal')
plt.show()