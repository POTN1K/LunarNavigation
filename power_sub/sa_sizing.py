"""
A script used for computations of the Electrical Power System
By Lennart and Mathijs"""

# External Libraries
import numpy as np

# Constants
G = 6.6743015*10**(-11) # N m^2 / (kg^2)
M = 7.34767309 * 10 ** 22 # kg
mu_moon = 4.90486959 * 10 ** 12
mu_earth = 3.9860044188 * 10 ** 14
r = (25000 + 1737.4) * 10 ** 3 # km
AU = 149600000 # km
def orbital_period(r_orbit, mu):
    """Function to compute the orbit period of a satellite around the Moon in [s]
    : parameters. orbit height (m) and gravitational constant
    : output. orbital period (s)"""
    return 2 * np.pi * np.sqrt(r_orbit**3/mu)

def orbital_velocity(r_orbit, mu):
    """ Function to compute the orbit velocity of a satellite around the Moon in [km/s]
    : parameters. orbit height (m) and gravitational constant
    : output. orbital velocity (m/s)"""
    return np.sqrt(mu/r_orbit)

def eclipse_angle(r_orbit, radius_1, radius_2):
    """ Function to compute the eclipse angle between a planet and the sun in [rad]
    : parameters. orbit height (m), radius of the radiating body (m) and radius of the covering body (m)
    : output. eclipse angle (rad)"""
    return np.arctan((radius_1-radius_2)/r_orbit)*180



# print(eclipse_angle(AU, 696340, 1737))