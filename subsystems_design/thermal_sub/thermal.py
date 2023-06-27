"""
A script for computing parameters of the thermal subsystem
By L. van der Peet and M. Vereycken"""

# External Libraries
import numpy as np
import matplotlib.pyplot as plt

# Constants
stefan_boltzmann = 5.67e-8
albedo_moon = 0.07
power_sun = 3.852 * 10 ** 26 # W
AU = 149600000 # km


def heat_dissipated(power_req, efficiency):
    """ A script to compute the heat dissipated [W] by a component or subsystem
    : parameters.
    power_req - the power required for the component or system (W)
    efficiency - the power efficiency of a component or system (-)"""
    return power_req * (1-efficiency)


'''def temperature_change(heat_flow, time, mass, heat_capacity):
    """ A script to compute the temperature change due to the heat flow of the spacecraft (K)
    : parameters.
    heat_flow - heat flow present (W)
    time - time over which temperature changes (s)
    mass - mass of the spacecraft (kg)
    heat_capacity - heat capacity of the spacecraft (J/kg/K)"""
    return heat_flow * time / (mass * heat_capacity)'''


def visibility_factor(radius, orbit_radius):
    """ A script to compute the visibility of the spacecraft in orbit
    : parameters.
    radius - the radius of the planet about which the spacecraft is orbiting (km)
    orbit_radius - the orbital radius of the spacecraft (km)"""
    return (radius / orbit_radius) ** 2


def albedo_radiation(albedo, sc):
    """ A script to compute the albedo radiation (W/m^2)
    : parameters.
    albedo - the albedo factor of the Moon (-)
    sc - the solar flux at the furthest distance from the sun (W/m^2)"""
    return albedo * sc


def intensities(surface_intensity, radius_emitter, distance):
    """ A script to compute intensity of radiation by a radiating body (W/m^2)
    : parameters.
    surface_intensity - the intensity of the radiation as experienced at the surface (W/m^2)
    radius_emitter - radius of the emitting body (km)
    distance - distance from the centre to the spacecraft (km)"""
    power = surface_intensity * 4 * np.pi * (radius_emitter * 10 ** 3) ** 2
    intensity = power / (4 * np.pi * ((distance * 10 ** 3) ** 2))
    return intensity


def equilibrium_temperature(area, area_a, J_s, J_m, a_m, a_sc, e_sc, k, P, n):
    """Calculating the equilibrium temperature inside the spacecraft [K]
    area: emissivity area of the spacecraft [m^2], area_a: absorptivity area of the spacecraft [m^2],
    J_s: solar flux constant of the Sun around the Moon [W/m^2], J_m: infrared flux due to Moon temperature [W/m^2],
    a_m: albedo factor of the Moon [], a_sc: absorptivity constant of the SC [], e_sc: emissivity constant of the SC [],
    k: Stefan Boltzmann constant [W/(m^2*K^4)], P: SC Power [W], n: SC efficiency []
    """
    T = (((J_s * area_a * a_sc) + (J_m * area_a * e_sc) + (a_m * a_sc * area_a) + (P * (1 - n)) + 87.45) / (e_sc * k * area))**(1/4)
    return T


def q_in(T, a_sc, e_sc, k, area, area_a, J_s, J_m, a_m, P, n):
    """Calculating the amount of heat that is necessary to keep the spacecraft on a desired temperature [W]
    T: Desired SC temperature [K], area: emissivity area of the spacecraft [m^2], area_a: absorptivity area of the spacecraft [m^2],
    J_s: solar flux constant of the Sun around the Moon [W/m^2], J_m: infrared flux due to Moon temperature [W/m^2],
    a_m: albedo factor of the Moon [], a_sc: absorptivity constant of the SC [], e_sc: emissivity constant of the SC [],
    k: Stefan Boltzmann constant [W/(m^2*K^4)], P: SC Power [W], n: SC efficiency []"""
    q_in = T**4 * e_sc * k * area - J_m * area_a * e_sc + J_s * a_m * a_sc * area_a + J_s * area_a * a_sc - (P * (1 - n))
    return q_in

def phase_change(q_in, q_material, q_phase_material, T_delta, t_eclipse):
    """Calculating the mass of the phase change material [kg]
    q_in: the required energy [W], q_material: material heat capacity in liquid state [J/kg],
    q_phase_material: material fusion heat [J/kg], T_delta: Temperature differend [°C],
    t_eclipse: eclipse time [s]"""
    m_phase = (q_in * t_eclipse) / (q_material * T_delta + q_phase_material)
    return m_phase

def Q_PC_day(q_in, T_Me, T_Ee, T_M, T_E):
    """Calculating the required charging power [W] for phase change material
    q_in: the required energy [W], T_Me: the eclipse time due to the Moon [s],
     T_Ee: the eclipse time due to Earth [s], T_M: orbital period around the Moon [s],
     T_E: orbital period around the Earth [s]"""
    Q_PC_day_m = (q_in * T_Me) / (T_M - T_Me)
    Q_PC_day_e = (q_in * T_Ee) / (T_E - T_Ee - T_E/T_M * T_Me)
    return Q_PC_day_e + Q_PC_day_m

'''V_hp, m_hp = m_heatpipes(8000, 3.18*10**(-3), 5*10**(-4), 0.2+0.4+3.35)
V_com, m_com = m_heatpipes(8000, 0.03, 16.5*10**(-4), 0.1)
solar_flux = intensities(63.28 * 10 ** 6, 696340, AU - (384400+2616))
albedo_intensity_max = albedo_radiation(albedo_moon, 1420, visibility_factor(1737, 2616))
albedo_intensity_min = albedo_radiation(albedo_moon, 1360, visibility_factor(1737, 10466))
infrared_intensity_max = intensities(1182, 1737, 2616)
infrared_intensity_min = intensities(11.757312, 1737, 10466)
print(albedo_intensity_max)
print(albedo_intensity_min)
print(infrared_intensity_max)
print(infrared_intensity_min)
#print('V_hp =', V_hp, 'm_hp =', m_hp)
#print('V_com =', V_com, 'm_com =', m_com)
print('Sun T_max beginning of life hottest case =', equilibrium_temperature(6, 1.5, 1420, infrared_intensity_max, 1420*0.07, 0.12*1.15, 0.18*0.95, 5.67*10**(-8), 2149, 0.7))
print('Sun T_min end of life coldest case =', equilibrium_temperature(6, 1, 1360, infrared_intensity_min, 1360*0.07, 0.12*0.85, 0.18*1.05, 5.67*10**(-8), 1491, 0.95))
print('Eclipise T_min end of life coldest case =', equilibrium_temperature(6, 1, 0, infrared_intensity_min, 0, 0.12*0.85, 0.18*1.05, 5.67*10**(-8), 761.64, 0.95))
print('Q_PC_day =', Q_PC_day(410, 410, 0.956, 4.082, 10.73, 658.85))
print('mass of phase change material =', phase_change(410*1.05, 0, 237000, 0, (4.082+1.271)*3600))
print('T_max end of life =', equilibrium_temperature(6, 2, 1360, 430, 0.07, 0.43*1.15, 0.67*0.95, 5.67*10**(-8), 1980.8, 0.9))
print('T_min', equilibrium_temperature(4.86, 0.81, 0, 430, 0.07, 0.5, 0.15, 5.67*10**(-8), 1123.2, 0.9))

infrared_intensity_max = intensities(1182, 1737, 2616)
infrared_intensity_min = intensities(11.757312, 1737, 10466)
print('equilibrium temperature solar panels during eclipse =', equilibrium_temperature(6.637*2, 6.637, 0, infrared_intensity_min, 0.07, 0.91, 0.8, 5.67*10**(-8), 0, 0.9)-273.15)
print('equilibrium temperature solar panels day eclipse =', equilibrium_temperature(6.637*2, 6.637, 1420, infrared_intensity_max, 0.07, (0.91-0.32*0.97**12), 0.8, 5.67*10**(-8), 0, 0.9)-273.15)
ewa = q_in(5.9+273.15, 0.6, 0.5, 5.67*10**(-8), 4.86, 0.81, infrared_intensity_min, 1000, 0.9)
m_phase = phase_change(ewa, 1000, 200000, 15, 4*3600)
#area_radiator = radiator(324,  0.88, 0.06, 5.67*10**(-8), 273.15+40, 1420, infrared_intensity_max, albedo_intensity, 10)
#a_radiator = check(1618,  0.8, 5.67*10**(-8), 373, 3, solar_flux, infrared_intensity_max, albedo_intensity, 0.2)
"""print('T_eq SA =', equilibrium_temperature(4.86, 1.9, solar_flux, infrared_intensity_max, 0.07, 0.98*(1-0.32), 0.8, 5.67*10**(-8), 0, 0.9))
print('area radiator =', area_radiator)
print(a_radiator)
print(ewa)"""
print('phase change material mass=', m_phase)
"""print('solar flux =', solar_flux)
print('moon infrared intensity =', infrared_intensity_max)
print(f'albedo moon {albedo_intensity}')
print('T_max', equilibrium_temperature(4.86, 0.81, solar_flux, infrared_intensity_max, 0.07, 0.6, 0.5, 5.67*10**(-8), 1500, 0))
print('T_min', equilibrium_temperature(4.86, 0.81, 0, infrared_intensity_min, 0.07, 0.6, 0.5, 5.67*10**(-8), 1800, 0.5))"""'''
if __name__ == "__main__":
    """Printing some graphs for sensitivity analysis and report based on the formulas found above"""
    a_e = np.array(np.linspace(0.1,2,1000))
    e = np.array(np.linspace(0.5,0.25,5))
    T_sc = np.array([])
    for i in range(len(a_e)):
        T = ((1420 + intensities(1182, 1737, 2616) + albedo_radiation(albedo_moon, 1420)) / (5.67*10**(-8) * 6) * 1.5 * a_e[i] + 1664 * 0.05 / (5.67*10**(-8) * 6 * e[::1]))**0.25-273.15
        T_sc = np.append(T_sc, T)
        i = i+1

    plt.plot(a_e, T_sc[0::5], label='emissivity = 0.2')
    plt.plot(a_e, T_sc[1::5], label='emissivity = 0.4')
    plt.plot(a_e, T_sc[2::5], label='emissivity = 0.6')
    plt.plot(a_e, T_sc[3::5], label='emissivity = 0.8')
    plt.plot(a_e, T_sc[4::5], label='emissivity = 1')
    plt.legend()
    plt.xlabel('absorptivity/emissivity')
    plt.ylabel('Temperature [°C]')
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()

    a = np.array(np.linspace(0.0, 0.35, 1000))
    Q_sc = np.array([])
    for i in range(len(a_e)):
        Q1 = 288.15**4 * 5.67*10**(-8) * e[::1] * 6 - (0 + intensities(11.757312, 1737, 10466) + 0) * 1 * a[i] - 1664 * 0.05
        Q2 = 288.15 ** 4 * 5.67 * 10 ** (-8) * e[::1] * 6 - (1420 + intensities(1182, 1737, 2616) + albedo_radiation(albedo_moon, 1420)) * 1.5 * a[i] - 1664 * 0.3
        Q3 = 288.15 ** 4 * 5.67 * 10 ** (-8) * e[::1] * 6 - (1360 + intensities(11.757312, 1737, 10466) + albedo_radiation(albedo_moon, 1360)) * 1 * a[i] - 1664 * 0.05
        Q_sc = np.append(Q_sc, abs(Q1)*2+abs(Q2)+abs(Q3))
        i = i+1


    plt.plot(a, Q_sc[0::5], label='emissivity = 0.05')
    plt.plot(a, Q_sc[1::5], label='emissivity = 0.1')
    plt.plot(a, Q_sc[2::5], label='emissivity = 0.15')
    plt.plot(a, Q_sc[3::5], label='emissivity = 0.20')
    plt.plot(a, Q_sc[4::5], label='emissivity = 0.25')
    plt.legend()
    plt.xlabel('absorptivity')
    plt.ylabel('Q [W]')
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()

