"""
A script used for computations of the Electrical Power System
By Lennart and Mathijs"""

# External Libraries
import numpy as np
from scipy.optimize import fsolve

# Constants
G = 6.6743015*10**(-11) # N m^2 / (kg^2)
M = 7.34767309 * 10 ** 22 # kg
# mu_moon = 4.90486959 * 10 ** 12
mu_moon = 4902.800118
# mu_earth = 3.9860044188 * 10 ** 14 m^3 / (s^2)
mu_earth = 398600.435507 # km^3 /(s^2)
r = (25000 + 1737.4) * 10 ** 3 # km
AU = 149600000 # km
r_sat = 6541.4 # km
sc_moon = 1310 # W / (m^2)


def orbital_period(r_orbit, mu):
    """Function to compute the orbit period of a satellite around the Moon in [s]
    : parameters. orbit height (km) and gravitational constant
    : output. orbital period (s)"""
    return 2 * np.pi * np.sqrt(r_orbit**3/mu)


'''def orbital_velocity(r_orbit, mu):
    """ Function to compute the orbit velocity of a satellite around the Moon in [km/s]
    : parameters. orbit height (km) and gravitational constant
    : output. orbital velocity (km/s)"""
    return np.sqrt(mu/r_orbit)'''


def eclipse_angle(r_orbit, radius_1, radius_2):
    """ Function to compute the eclipse angle between a planet and the sun in [rad]
    : parameters. orbit height (km), radius of the radiating body (km) and radius of the covering body (km)
    : output. eclipse angle (rad)"""
    return np.arctan((radius_1+radius_2)/r_orbit)


'''def eclipse_length(r_orbit_1, r_orbit_2, radius_1, radius_2, mu):
    """ Function to compute the eclipse length [km] that is caused by the Earth
    : parameters. orbit radius of the Earth around the sun (km), of the Moon around the Earth (km)\
    radius of the sun (km), radius of the Earth (km) and gravitational constant of the Earth (km^3//(s^2))
    : outputs. eclipse length (km) and eclipse time (s)"""
    angle = eclipse_angle(r_orbit_1, radius_1, radius_2)
    velocity = orbital_velocity(r_orbit_2, mu)
    height = 2 * r_orbit_2 * np.tan(angle) + 2 * radius_2
    time = height / velocity
    return height, time'''

def eclipse_length(a, orbit_1, radius_1, radius_2, mu):
    """ Function to compute the eclipse time for a circular orbit [s]
    : parameters. semi major axis of orbit (a [km]), radius of Celestial body (radius_1 [km]),
    orbit where the second Celestial body is rotating around (orbit_1 [km]),
    radius of second Celestial body where the semi major axis is applied to (radius_2 [km]),
    gravitational parameter of second Celstial body (mu [km^3/s^2])"""
    alpha = eclipse_angle(orbit_1, radius_1, radius_2)
    gamma = np.arcsin(radius_2 * np.sin(90 + alpha) / a)
    eclipse = (alpha + gamma) / np.pi
    print(alpha + gamma)
    T = orbital_period(a, mu)
    T_eclipse = T * eclipse
    return T_eclipse
print(eclipse_length(6541.4, 149597871, 696340, 1737.4, 4902.800118)/3600)


def eclipse_lenght_elip(e, a, r_m, r_s, AU, mu_m):
    """ Function to compute the eclipse time [s] for an eliptical orbit that is caused by the Moon
     : parameters. eccentricity of satellite orbit (e), semi major axis of satellite (a [km]), radius of the Moon (r_m [km]),
     radius of the Sun (r_s [km]), distance Earth-Sun (AU [km]), gravitational parameter of the Moon (mu_m [km^3/s^2]"""
    alpha = np.pi - np.arctan(AU / (r_m + r_s))
    theta1 = fsolve(lambda x: (1 + e * np.cos(x)) / (np.sin(x - alpha)) + a * (1 - e ** 2) / (r_m * np.sin(alpha)), 1)
    theta2 = fsolve(lambda x: (1 + e * np.cos(x)) / (np.sin(x - alpha)) - a * (1 - e ** 2) / (r_m * np.sin(alpha)), 2)
    eclipse = e * np.sqrt(1 - e ** 2) / (2 * np.pi) * (
                np.sin(theta1) / (1 + e * np.cos(theta1[0])) - np.sin(theta2[0]) / (1 + e * np.cos(theta2[0]))) + 1 / np.pi * (
                          np.arctan((1 - e) * np.tan(theta2[0] / 2) / np.sqrt(1 - e ** 2)) - np.arctan(
                      (1 - e) * np.tan(theta1[0] / 2) / np.sqrt(1 - e ** 2)))
    T = orbital_period(a, mu_m)
    T_eclipse = eclipse * T
    return T_eclipse

print('eclipse', eclipse_lenght_elip(0.6, 6541.4, 1737.4, 696340, 149597871, 4902.800118)/60)


eclipse_moon = eclipse_length(AU, 384400, 696340, 6371, mu_earth)
eclipse_satellite = eclipse_length(AU+384400, r_sat, 696340, 1737.4, mu_moon)
print(eclipse_satellite)

total_eclipse = eclipse_moon[1] + eclipse_satellite[1]
# print(f'Total eclipse time: {round(total_eclipse/3600,2)} [hrs]')


def bol_power(p_eol, degradation, lifetime):
    """ Function to compute the required begin of life power for the satellite [W]
    : parameters. power required at end of life (W), degradation per annum (%/year) and the mission lifetime (year)
    : output. power required at begin of life (W)"""
    return p_eol / ((1-degradation)**lifetime)
# print(bol_power(1700, 0.03, 12))


def sa_size(p_bol, cell_efficiency, sc):
    """ Function to compute the required solar array size to generate enough power for the system [m^2]
    : parameters. power generated at BOL (W), efficiency of the cells (W/m^2) and solar flux (W/m^2)
    : output. size of the solar array (m^2)"""
    p_sa = p_bol / cell_efficiency
    area_sa = p_sa / sc
    return area_sa
# print(sa_size(bol_power(1700, 0.03, 12), 0.32, sc_moon))


def battery_size(T_eclipse, E_specific, E_density, P_req, N_b, n_dis, N_c, V_cdis, V_d, Vhdis, DoD):
    """Function to compute the battery size and volume. T_eclipse = maximum eclipse time [h],
    E_specific = specific energy [Wh/kg] E_density = Energy density [Wh/l],
    P_req = required power from battery during eclipse [W], N_b = number of batteries in parallel,
    n_dis = discharge converter efficiency, N_c = number of series cells per battery,
    V_d = voltage drop in bypass diode in case a cell failed
    V_cdis = voltage per cell, average during discharge, V_hdis = voltage drop in harness from battery to PRU
    Dod = maximum allowable depth of discharge in the worst case eclipse."""
#    cap_battery = (P_req * T_eclipse) / (N_b * n_dis * ((N_c - 1) * V_cdis - V_d - V_hdis) * DoD)
    mass_battery = (T_eclipse * P_req) / (n_dis * E_specific * DoD) #[kg]
    volume_battery = (T_eclipse * P_req) / (n_dis * E_density * DoD) #[l]
    return mass_battery, volume_battery


# a, b = battery_size(6.79, 200, 300, 1500, 0, 0.9, 0, 0, 0, 0, 0.8)
# print(a, b)

