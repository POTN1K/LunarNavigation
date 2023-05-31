import numpy as np
r_au = 149597871 #km
r_sun = 696340 #km
r_moon = 1737.4 #km
r_earth = 6371 #km
r_orbit_s = 24000 #km
r_orbit_m = 384400 #km
T_moon = 659 #h



alpha = np.arctan((r_sun - r_moon) / r_au)
x = (r_sun - r_moon) / r_au * (r_orbit_s + r_moon)
theta = np.arctan((r_moon - x) / (r_moon + r_orbit_s))
t_eclipse =  theta / np.pi
print(t_eclipse)
print(x)
print(alpha*180/np.pi)

alpha_e = np.arctan((r_sun - r_earth) / r_au)
print(alpha_e*180/np.pi)
x_e = (r_sun - r_earth) / r_au * (r_earth + r_orbit_m)
theta_e = np.arctan((r_earth - x_e + 2 * (r_orbit_s + r_moon)) / (r_earth + r_orbit_m))
t_eclipse_e = theta_e / np.pi * T_moon
print(t_eclipse_e)