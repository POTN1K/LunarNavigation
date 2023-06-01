"""
A script used for computations of the Electrical Power System
By Lennart and Mathijs"""

# External Libraries
import numpy as np

# Constants
G = 6.6743015*10**(-11) # N m^2 / (kg^2)
M = 7.34767309 * 10 ** 22 # kg
# mu_moon = 4.90486959 * 10 ** 12
mu_moon = 4902.800118
# mu_earth = 3.9860044188 * 10 ** 14 m^3 / (s^2)
mu_earth = 398600.435507 # km^3 /(s^2)
r = (25000 + 1737.4) * 10 ** 3 # km
AU = 149600000 # km
r_sat = 25000 # km
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
    return np.arctan((radius_1-radius_2)/r_orbit)


def eclipse_length(r_orbit_1, r_orbit_2, radius_1, radius_2, mu):
    """ Function to compute the eclipse length [km] that is caused by the Earth
    : parameters. orbit radius of the Earth around the sun (km), of the Moon around the Earth (km)\
    radius of the sun (km), radius of the Earth (km) and gravitational constant of the Earth (km^3//(s^2))
    : outputs. eclipse length (km) and eclipse time (s)"""
    angle = eclipse_angle(r_orbit_1, radius_1, -radius_2)
    velocity = orbital_velocity(r_orbit_2, mu)
    height = 2 * r_orbit_2 * np.tan(angle) + 2 * radius_2
    time = height / velocity
    return height, time


eclipse_moon = eclipse_length(AU, 384400, 696340, 6371, mu_earth)
eclipse_satellite = eclipse_length(AU+384400, r_sat, 696340, 1737.4, mu_moon)

total_eclipse = eclipse_moon[1] + eclipse_satellite[1]
print(f'Total eclipse time: {total_eclipse/3600} [hrs]')