"""
A script used for computations of the Electrical Power System
By Lennart and Mathijs"""

# External Libraries
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants
G = 6.6743015*10**(-11) # N m^2 / (kg^2)
M = 7.34767309 * 10 ** 22 # kg
# mu_moon = 4.90486959 * 10 ** 12
mu_moon = 4902.800118
# mu_earth = 3.9860044188 * 10 ** 14 m^3 / (s^2)
mu_earth = 398600.435507 # km^3 /(s^2)
# r = (25000 + 1737.4) * 10 ** 3 # km
AU = 149600000 # km
r_sat = 6541.4 # km
sc_moon = 1310 # W / (m^2)
margin = 1.2
specific_energy = 168.5 # Wh/kg
energy_density = 300 # Wh/l
eclipse_power = 761 * margin # W
heater_power = 300 # W


def orbital_period(r_orbit, mu):
    """Function to compute the orbit period of a satellite around the Moon in [s]
    : parameters. orbit height (km) and gravitational constant
    : output. orbital period (s)"""
    return 2 * np.pi * np.sqrt(r_orbit**3/mu)


def orbital_velocity(r_orbit, mu):
    """ Function to compute the orbit velocity of a satellite around the Moon in [km/s]
    : parameters. orbit height (km) and gravitational constant
    : output. orbital velocity (km/s)"""
    return np.sqrt(mu/r_orbit)


def eclipse_angle(r_orbit, radius_1, radius_2):
    """ Function to compute the eclipse angle between a planet and the sun in [rad]
    : parameters. orbit height (km), radius of the radiating body (km) and radius of the covering body (km)
    : output. eclipse angle (rad)"""
    return np.arctan((radius_1+radius_2)/r_orbit)


def eclipse_length1(r_orbit_1, r_orbit_2, radius_1, radius_2, mu):
    """ Function to compute the eclipse length [km, s] that is caused by the Earth
    : parameters.
    r_orbit_1 - orbit radius of the Earth around the sun [km],
    r_orbit_2 - of the Moon around the Earth [km],
    radius_1 - radius of the sun [km],
    radius_2 - radius of the Earth [km],
    mu - gravitational constant of the Earth [km^3/(s^2)]
    : outputs.
    height - eclipse length [km],
    time - eclipse time [s]"""
    angle = eclipse_angle(r_orbit_1, radius_1, radius_2)
    velocity = orbital_velocity(r_orbit_2, mu)
    height = 2 * r_orbit_2 * np.tan(angle) + 2 * radius_2
    time = height / velocity
    return height, time


def eclipse_length(a, orbit_1, radius_1, radius_2, mu):
    """ Function to compute the eclipse time for a circular orbit around an occulting body [s]
    : parameters.
    a - semi major axis of orbit around occulting body [km],
    orbit_1 - orbit where the second Celestial body is rotating around the emitting body [km],
    radius_1 - radius of emitting celestial body [km],
    radius_2 - radius of the receiving celestial body (occulting body) [km],
    mu - gravitational parameter of second celestial body (occulting body) [km^3/s^2]
    : outputs.
    eclipse_time - time the body is blocked by the occulting body [s]"""
    alpha = eclipse_angle(orbit_1, radius_1, radius_2)
    gamma = np.arcsin(radius_2 * np.sin(90 + alpha) / a)
    eclipse = (alpha + gamma) / np.pi
    time = orbital_period(a, mu)
    eclipse_time = time * eclipse
    return eclipse_time


def eclipse_length_elip(e, a, r_m, r_s, AU, mu_m):
    """ Function to compute the eclipse time [s] for an elliptical orbit that is caused by the Moon
     : parameters.
     e - eccentricity of satellite orbit [-],
     a - semi major axis of satellite [km],
     r_m - radius of the Moon [km],
     r_s - radius of the Sun [km],
     AU - distance Earth-Sun [km],
     mu_m - gravitational parameter of the Moon [km^3/s^2]"""
    alpha = np.pi - np.arctan(AU / (r_m + r_s))
    theta1 = fsolve(lambda x: (1 + e * np.cos(x)) / (np.sin(x - alpha)) + a * (1 - e ** 2) / (r_m * np.sin(alpha)), 1)
    theta2 = fsolve(lambda x: (1 + e * np.cos(x)) / (np.sin(x - alpha)) - a * (1 - e ** 2) / (r_m * np.sin(alpha)), 2)
    eclipse = e * np.sqrt(1 - e ** 2) / (2 * np.pi) * (
                np.sin(theta1) / (1 + e * np.cos(theta1[0])) - np.sin(theta2[0]) / (1 + e * np.cos(theta2[0]))) + 1 / \
                np.pi * (np.arctan((1 - e) * np.tan(theta2[0] / 2) / np.sqrt(1 - e ** 2)) - np.arctan(
                      (1 - e) * np.tan(theta1[0] / 2) / np.sqrt(1 - e ** 2)))
    time = orbital_period(a, mu_m)
    eclipse_time = eclipse * time
    return eclipse_time


# eclipse_moon = eclipse_length(AU, 384400, 696340, 6371, mu_earth)
# eclipse_satellite = eclipse_length(AU+384400, r_sat, 696340, 1737.4, mu_moon)
# print(eclipse_satellite)

# total_eclipse = eclipse_moon[1] + eclipse_satellite[1]
# print(f'Total eclipse time: {round(total_eclipse/3600,2)} [hrs]')


def bol_power(p_eol, degradation=0.03, lifetime=12):
    """ Function to compute the required begin of life power for the satellite [W]
    : parameters.
    p_eol - power required at end of life [W],
    degradation - degradation per annum [%/year],
    life_time - the mission lifetime [year]
    : output. power required at begin of life (W)"""
    return p_eol / ((1-degradation)**lifetime)


def sa_size(p_bol, sc, cell_efficiency=0.285):
    """ Function to compute the required solar array size to generate enough power for the system [m^2]
    : parameters.
    p_bol - power generated at BOL [W],
    cell_efficiency - efficiency of the cells [W/m^2],
    sc - solar flux around the Moon [W/m^2]
    : output.
    area_sa - size of the solar array [m^2]"""
    p_sa = p_bol / cell_efficiency
    area_sa = p_sa / sc
    return area_sa
# print(sa_size(bol_power(1700, 0.03, 12), 0.32, sc_moon))


def battery_size(eclipse_time, power_required, voltage_d, voltage_hdis, dod, n_c, n_b=1, n_dis=1, voltage_cdis=3.5):
    """Function to compute the battery capacity [Ah], mass [kg] and volume [l].
    : parameters.
    eclipse_time - time duration of the eclipse [hr],
    power_required - power required during eclipse [W],
    voltage_d - voltage drop in bypass diode in case of cell failure [V],
    voltage_hdis = voltage drop in harness from battery to PRU [V],
    dod = maximum allowable depth of discharge in the worst case eclipse [%],
    n_c = number of series cells per battery [#],
    n_b = number of batteries in parallel [#],
    n_dis = discharge converter efficiency [%],
    voltage_cdis = voltage per cell, average during discharge [V]"""
    capacity_cell = power_required * eclipse_time / (n_b * n_dis * ((n_c - 1) * voltage_cdis - voltage_d -\
                                                                       voltage_hdis) * dod)
    capacity_battery = capacity_cell * n_c * voltage_cdis
    mass_battery = (eclipse_time * power_required) / (n_dis * specific_energy * dod) #[kg]
    volume_battery = (eclipse_time * power_required) / (n_dis * energy_density * dod) #[l]
    return capacity_battery, mass_battery, volume_battery, capacity_cell


def charging_power(capacity, charging_time, efficiency=0.9):
    """ A script to compute the power required for charging the batteries during day time [W]
    : parameters.
    capacity - the capacity of the batteries which must be charged [Wh]
    charging_time - the time allowed for charging (period - eclipse time) [h]
    : outputs.
    charging_power - power required for charging of the battery [W]"""
    return capacity / (charging_time * efficiency)


# Code segment


earth_moon = eclipse_length1(AU, 384400, 696340, 6371, mu_earth)[0]/3600 # option Lennart
moon_earth_eclipse = eclipse_length(384400, AU, 696340, 6371, mu_earth)/3600 # Option Mathijs
# capacity, mass, volume, cell_capacity = battery_size(moon_earth_eclipse, 1500, 0.1, 0.1, 0.8)
# print(earth_moon[1]/3600)
# print(capacity, mass, volume, cell_capacity)
eclipse_time = eclipse_length(5701, AU+384400, 696340, 1737.4, mu_moon)/3600
eclipse_time2 = eclipse_length(10000, AU+384400, 696340, 1737.4, mu_moon)/3600

lowest_volume = 99999
for i in range(9, 60):
    if battery_size(eclipse_time, eclipse_power, 0.1, 0.1, 0.4, i)[-1] < 51:
        new_volume = battery_size(eclipse_time, eclipse_power, 0.1, 0.1, 0.4, i)[2]
        if new_volume < lowest_volume:
            lowest_volume = new_volume
            index = i
    else:
        continue


lowest_volume2 = 99999
for i in range(9, 60):
    if battery_size(earth_moon, eclipse_power, 0.1, 0.1, 0.8, i)[-1] < 51:
        # print(battery_size(eclipse_time, eclipse_power, 0.1, 0.1, 0.8, i)[-1])
        new_volume = battery_size(earth_moon, eclipse_power, 0.1, 0.1, 0.8, i)[2]
        if new_volume < lowest_volume2:
            lowest_volume2 = new_volume
            index2 = i
    else:
        continue


battery_1 = battery_size(eclipse_time, eclipse_power, 0.2, 0.1, 0.4, index)
battery_2 = battery_size(moon_earth_eclipse, eclipse_power, 0.2, 0.1, 0.8, index2)
print(f' --- Battery 1: --- \nNumber of cells: {index}, \nBattery capacity: {battery_1[0]} [Wh],\nBattery mass: \
{14.52+index*1.08} [kg], \nBattery volume: {0.553797+index*0.508429} [l]')
print(f' --- Battery 2: --- \nNumber of cells: {index2}, \nBattery capacity: {battery_2[0]} [Wh],\nBattery mass: \
{14.52+index2*1.08} [kg], \nBattery volume: {0.553797+index2*0.508429} [l]')
charging_battery_1 = charging_power(battery_1[0], (orbital_period(5701, mu_moon)/3600 - eclipse_time))
charging_battery_2 = charging_power(battery_2[0], (orbital_period(384400, mu_earth)/3600 - moon_earth_eclipse))
total_charging_power = charging_battery_1 + charging_battery_2
sa_area = sa_size(bol_power(eclipse_power + total_charging_power*margin + heater_power*margin), sc_moon)
print(sa_area/0.9)
print(bol_power(eclipse_power + total_charging_power*margin + heater_power*margin))
# print(eclipse_power + total_charging_power + heater_power)
# for i in range(6,1000):
#     if battery_size(moon_earth_eclipse+eclipse_time, 2266.8, 0.2, 0.1, 0.4, i)[-1] < 51:
#         print(i, battery_size(moon_earth_eclipse+eclipse_time , 2266.8, 0.2, 0.1, 0.4,i))
#         break


for i in range(6,1000):
    if battery_size(eclipse_time+moon_earth_eclipse, eclipse_power, 0.2, 0.1, 0.4, i)[-1] < 51:
        index3 = i
        break

single_battery = battery_size(eclipse_time+moon_earth_eclipse, eclipse_power, 0.2, 0.1, 0.4, index3)
print(f' --- Singular Battery: --- \nNumber of cells: {index3}, \nBattery capacity: {single_battery[0]} [Wh],\nBattery \
mass: {14.52+index3*1.08} [kg], \nBattery volume: {0.553797+index3*0.508429} [l]')
# battery_size(eclipse_time+moon_earth_eclipse, eclipse_power, 0.2, 0.1, 0.4, index)
# print(sa_size(2266.8, sc_moon)/0.85*4)
print(bol_power(1605))

power = np.arange(10,2001, 1)
sa = sa_size(bol_power(power), sc_moon)
plt.plot(power, sa)
plt.show()
