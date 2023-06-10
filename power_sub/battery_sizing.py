import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


e = 0.6
R_m = 1737.4
a = 6541.4
R_s = 696340
AU = 149597871
mu_moon = 4902.800118

alpha = np.pi - np.arctan(AU / (R_m + R_s))
print(alpha)
theta = np.array(np.linspace(0, np.pi, 1000))
y1 = (1 + e * np.cos(theta)) / (np.sin(theta - alpha))
y2 = -a * (1 - e**2) / (R_m * np.sin(alpha))

plt.plot(theta, y1)
plt.axhline(y2)
plt.show()

T = 2 * np.pi * np.sqrt(a**3/mu_moon)
print('eclipse time =', T/3600)
theta1 = fsolve(lambda x: (1 + e * np.cos(x)) / (np.sin(x - alpha))+a * (1 - e**2) / (R_m * np.sin(alpha)), 1)
theta2 = fsolve(lambda x: (1 + e * np.cos(x)) / (np.sin(x - alpha))-a * (1 - e**2) / (R_m * np.sin(alpha)), 2)
print(theta1, theta2)
theta1 = 0.991544
theta2 = 1.91324
print((theta2-theta1)/(2*np.pi))
print('theta1 =', theta1*180/np.pi)
print('theta2 =', theta2*180/np.pi)

eclipse = e * np.sqrt(1 - e**2) / (2 * np.pi) * (np.sin(theta1) / (1 + e * np.cos(theta1)) - np.sin(theta2) / (1 + e * np.cos(theta2))) + 1 / np.pi * (np.arctan((1 - e) * np.tan(theta2/2) / np.sqrt(1 - e**2)) - np.arctan((1 - e) * np.tan(theta1/2) / np.sqrt(1 - e**2)))
print(eclipse)
print(eclipse * T/3600)