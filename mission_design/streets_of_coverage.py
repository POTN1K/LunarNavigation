"""
Script to estimate the number of satellites needed for a Moon constellation and a Lagrange/Moon constellation
Loop through several options specified by a set of altitudes and number of satellites per orbit
Number of orbital planes especially is not optimised in this estimation, as the same coverage can be achieved using
different amount of orbital planes with the satellites distributed over more orbital planes.

By Ian Maes
"""
import numpy as np

# define radius of the moon
radius_moon = 1737.4
# define a set of altitudes
altitudes = np.array([3500, 4000, 4500, 40000])
# define a set of number of satellites per orbit that needs to be checked
S = np.arange(10, 15, 1)

# calculation of inclination and coverage angle for the different altitudes
def incl(radius, altitude):
    inclination = np.pi / 2 - (
                np.pi - np.arcsin((radius * np.sin(100 * np.pi / 180)) / (radius + altitude)) - 100 * np.pi / 180)
    theta = (np.pi - np.arcsin((radius * np.sin(100 * np.pi / 180)) / (radius + altitude)) - 100 * np.pi / 180)
    return inclination, theta


# calculation of the number of satellites
def n_sats(theta, S, radius, inclination, lagrange):
    # calculation of angle of coverage of each satellite
    between = np.array(np.cos(theta) / np.cos(4 * np.pi / S))

    between[between < 0] = 0
    c_j = np.arccos(between)

    c_j_lag = np.arccos(np.cos(theta) / np.cos(3 * np.pi / S))

    # Calculation of 1/2 width of coverage of each satellite
    c = c_j * radius
    c_lag = c_j_lag * radius
    # Calculation of the circumference of the moon
    circ_moon = np.pi * radius

    # Calculation of the number of orbits at different inclinations that are needed
    n_orbits = np.array(np.ceil(circ_moon / (2 * c / np.sin(inclination))))
    n_orbits_lag = np.ceil(circ_moon / (2 * c_lag / np.sin(inclination)))

    n_orbits[n_orbits < 2.0] = 2

    # Calculation of the number of satellites needed

    n_sats = n_orbits * S
    n_sats_lag = n_orbits_lag * S

    if lagrange:
        n_sats = 1 / 3 * n_sats + 2 / 3 * n_sats_lag

    return n_sats, n_orbits


def loop(altitudes, S, lagrange):
    inclination_angles = []
    theta_angles = []
    j = 0
    while j < len(altitudes):
        inclination_angles.append(incl(radius_moon, altitudes[j])[0])
        theta_angles.append(incl(radius_moon, altitudes[j])[1])
        j += 1

    inclination_angles = np.array(inclination_angles)
    print(inclination_angles * 180 / np.pi)
    inclination_angles = np.full((1, len(inclination_angles)), 90 * np.pi / 180)
    theta_angles = np.array(theta_angles)

    n_sats_array = np.arange(len(altitudes))
    n_orbits_array = np.arange(len(altitudes))

    n = 0
    while n < len(S):
        n_sats_array = np.vstack((n_sats_array, n_sats(theta_angles, S[n], radius_moon, inclination_angles, lagrange)[0]))
        n_orbits_array = np.vstack(
            (n_orbits_array, n_sats(theta_angles, S[n], radius_moon, inclination_angles, lagrange)[1]))
        n += 1
    # Change nan values to 'Fail' string. This means the combination of number of satellites per orbital plane
    # and altitude can not provide 4-fold coverage
    n_sats_array_str = np.where(np.isnan(n_sats_array), 'Fail', n_sats_array.astype(object))
    n_orbits_array_str = np.where(np.isnan(n_orbits_array), 'Fail', n_orbits_array.astype(object))

    print(n_sats_array_str)
    print(n_orbits_array_str)
    return n_sats_array_str, n_orbits_array_str


if __name__ == '__main__':
    loop(altitudes, S, True)
