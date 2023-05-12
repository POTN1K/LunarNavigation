import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
from tudatpy.kernel.astro import element_conversion

r_moon = 1.737e6  # m

# Define satellite positions
pos = []

w_tot = np.linspace(0, 360, 6+1)[:-1] # Number of satellites per plane 
t_tot = np.linspace(0, 360, 4+1)[:-1] # Number of planes

for w in w_tot:
    for t in t_tot:
        pos.append(element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=4.9048695e12,
            semi_major_axis=24572000,
            eccentricity=0,
            inclination=np.deg2rad(58.69),
            argument_of_periapsis=np.deg2rad(22.9),
            longitude_of_ascending_node=np.deg2rad(w),
            true_anomaly=np.deg2rad(t)
        )[:3])


resolution = 200
# Sphere surface

phi = np.linspace(0, 2 * np.pi, resolution)
theta = np.linspace(0, np.pi, resolution)

# Create a meshgrid of phi and theta values
phi, theta = np.meshgrid(phi, theta)

# Calculate the x, y, and z coordinates for each point on the sphere
xM = r_moon * np.sin(theta) * np.cos(phi)
yM = r_moon * np.sin(theta) * np.sin(phi)
zM = r_moon * np.cos(theta)

# Convert the spherical coordinates to Cartesian coordinates

phi = np.linspace(0, 2 * np.pi, resolution)
theta = np.linspace(0, np.pi, resolution)

sphere_points = []
for i in phi:
    for j in theta:
        x = r_moon * np.sin(j) * np.cos(i)
        y = r_moon * np.sin(j) * np.sin(i)
        z = r_moon * np.cos(j)
        sphere_points.append([x, y, z])


# Define satellite cone


def h(x_r, y_r, z_r, connected_points, alpha=np.pi*100/180, r_moon=1.737e6):
    a = np.sqrt(x_r**2+y_r**2+z_r**2)  # height of the cone
    h_max = 0.5*(2*r_moon*np.cos(alpha) + np.sqrt(2)*np.sqrt(2*a**2-r_moon**2+r_moon**2*np.cos(2*alpha)))
    if h_max>np.sqrt(a**2+r_moon**2):
        h_max = np.sqrt(a**2+r_moon**2)
        
    for i, point in enumerate(sphere_points):
        x, y, z = point[0], point[1], point[2]
        if np.sqrt((x-x_r)**2+(y-y_r)**2+(z-z_r)**2) <= h_max:
            connected_points[i] += 1
    return h_max, connected_points

# Run coverage
cpoints = np.zeros(len(sphere_points))
for i in range(len(pos)):
    h_max, cpoints = h(*pos[i], cpoints)
sphere_points = np.array(sphere_points)


# Plot the points
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(*zip(*sphere_points), s=1)
ax.plot_surface(xM, yM, zM, color='grey', alpha=0.2)
ax.scatter(*zip(*pos), s=10)

color_map = cm.ScalarMappable(cmap='PiYG')
color_map.set_array(cpoints)

ax.scatter(*zip(*sphere_points), marker='s', s=1, c=cpoints, cmap='PiYG')
plt.colorbar(color_map)


plt.show()