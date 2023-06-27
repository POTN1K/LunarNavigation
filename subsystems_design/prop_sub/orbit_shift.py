"""Compute the orbit shift for different time ranges
By C. Spichal"""

import numpy as np
import matplotlib.pyplot as plt

#Test with the OG orbital planes and assume circular

a = 10000e3
mu_moon = 4.9049e+12
T = 2 * np.pi * np.sqrt(a**3 / mu_moon)
delta_max = 180
time_s = np.arange(100000, 5000000, 1000)
time_h = time_s / 3600
time_d = time_h / 24

delta_a = (delta_max / 360 * a * T) / (3 * np.pi * time_s)
delta_V = 2 * (np.sqrt(mu_moon * (2 / a - 1 / (a - delta_a))) - np.sqrt(mu_moon / a))

plt.title("Delta V required for the original orbit constellation satellite positioning")
plt.xlabel("Maneuver time [days]")
plt.ylabel("Delta V [m/s]")
plt.plot(time_d, delta_V)
plt.show()

