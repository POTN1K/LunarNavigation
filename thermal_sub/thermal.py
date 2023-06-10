"""
A script for computing parameters of the thermal subsystem
By Lennart and Mathijs"""

# External Libraries
import numpy as np

# Constants
stefan_boltzmann = 5.67e-8
j_s = 1310 # W/(m^2*K)
albedo_moon = 0.07
power_sun = 3.852 * 10 ** 26 # W
AU = 149600000 # km
area = 2.05 * 2.66 # m^2
absorptivity = 0.5 # -


def heat_dissipated(power_req, efficiency):
    """ A script to compute the heat dissipated by a component or subsystem
    : parameters.
    power_req - the power required for the component or system (W)
    efficiency - the power efficiency of a component or system (-)"""
    return power_req * (1-efficiency)


def temperature_change(heat_flow, time, mass, heat_capacity):
    """ A script to compute the temperature change due to the heat flow of the spacecraft (K)
    : parameters.
    heat_flow - heat flow present (W)
    time - time over which temperature changes (s)
    mass - mass of the spacecraft (kg)
    heat_capacity - heat capacity of the spacecraft (J/kg/K)"""
    return heat_flow * time / (mass * heat_capacity)


def visibility_factor(radius, orbit_radius):
    """ A script to compute the visibility of the spacecraft in orbit
    : parameters.
    radius - the radius of the planet about which the spacecraft is orbiting (km)
    orbit_radius - the orbital radius of the spacecraft (km)"""
    return (radius / orbit_radius) ** 2


def albedo_radiation(albedo, sc, visibility):
    """ A script to compute the albedo radiation (W/m^2)
    : parameters.
    albedo - the albedo factor of the Moon (-)
    sc - the solar flux at the furthest distance from the sun (W/m^2)
    visibility - the visibility of the spacecraft with respec to the Sun (-)"""
    return albedo * sc * visibility


def intensities(surface_intensity, radius_emitter, distance):
    """ A script to compute intensity of radiation by a radiating body (W/m^2)
    : parameters.
    surface_intensity - the intensity of the radiation as experienced at the surface (W/m^2)
    radius_emitter - radius of the emitting body (km)
    distance - distance from the centre to the spacecraft (km)"""
    power = surface_intensity * 4 * np.pi * (radius_emitter * 10 ** 3) ** 2
    intensity = power / (4 * np.pi * ((distance * 10 ** 3) ** 2))
    return intensity


infrared_intensity = intensities(1182, 1737, (1737 + 2616))
solar_flux = intensities(63.28 * 10 ** 6, 696340, AU + 384400)
albedo_intensity = albedo_radiation(albedo_moon, solar_flux, visibility_factor(1737, 2616))

# if __name__ == "__main__":
#     print('works')
