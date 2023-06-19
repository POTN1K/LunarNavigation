import numpy as np
import sa_sizing as sa

# Constants
G = 6.6743015*10**(-11) # N m^2 / (kg^2)
M = 7.34767309 * 10 ** 22 # kg
# mu_moon = 4.90486959 * 10 ** 12
mu_moon = 4902.800118
# mu_earth = 3.9860044188 * 10 ** 14 m^3 / (s^2)
mu_earth = 398600.435507 # km^3 /(s^2)
r = 6541.4 # km
AU = 149600000 # km
r_sat = 25000 # km


T_s = sa.orbital_period(r_orbit_s, mu_moon)
T_m = sa.orbital_period(r_orbit_m, mu_earth)

def eclipse_angle(r_orbit, radius_1, radius_2):
    return np.arctan((radius_1+radius_2)/r_orbit)





tan_alpha = (r_moon + r_sun) / r_au
alpha_em = np.arctan((r_orbit_s * tan_alpha + r_moon) / r_orbit_s)
t_eclipse_moon = alpha_em / np.pi * T_s
print(t_eclipse_moon/3600, 'h')
print(12*365*24/t_eclipse_moon)



T_m = orbital_period(r_orbit_m, mu_earth)

tan_alpha_e = (r_earth + r_sun) / r_au
alpha_es = np.arctan((r_orbit_m * tan_alpha_e + r_earth) / r_orbit_m)
t_eclipse_eath = alpha_es / np.pi * T_m
print(t_eclipse_eath/3600, 'h')

print(int((t_eclipse_moon+t_eclipse_eath)/3600), 'h', ((t_eclipse_moon+t_eclipse_eath)/3600-6)*60, 'min')


alpha = np.arctan((r_sun - r_moon) / r_au)
x = (r_sun - r_moon) / r_au * (r_orbit_s + r_moon)
theta = np.arctan((r_moon - x) / (r_moon + r_orbit_s))
t_eclipse =  theta / np.pi
t_day = 1 - t_eclipse
print(t_eclipse)
print(t_day)
print(x)
print(alpha*180/np.pi)

alpha_e = np.arctan((r_sun - r_earth) / r_au)
print(alpha_e*180/np.pi)
x_e = (r_sun - r_earth) / r_au * (r_earth + r_orbit_m)
theta_e = np.arctan((r_earth - x_e + 2 * (r_orbit_s + r_moon)) / (r_earth + r_orbit_m))
t_eclipse_e = theta_e / np.pi * T_moon
print(t_eclipse_e)