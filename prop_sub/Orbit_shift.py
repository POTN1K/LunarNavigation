"Compute the orbit shift for different time ranges"

import numpy as np
import matplotlib.pyplot as plt

#Test with the OG orbital planes and assume circular

a = 10000.2e3
mu_moon = 4.9049e+12
T = 2 * np.pi * np.sqrt(a**3 / mu_moon)
delta_max = 120
time = np.arange(30000, 10000000, 1000)

delta_a = (delta_max / 360 * a * T) / (3 * np.pi * time)
delta_V = 2 * (np.sqrt(mu_moon * (2 / a - 1 / (a - delta_a))) - np.sqrt(mu_moon / a))

plt.title("delta_V required for satellite positioning")
plt.plot(time, delta_V)
plt.show()

