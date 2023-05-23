import numpy as np
from propagation_calculator import PropagationTime
from model_class import Model

model = Model()
model.addSymmetricalPlanes(2.45e7, 0, 58.69, 22.9, 6, 4)
satellites = np.array(model.getSatellites())
def extreme_resolution():
    propagation_time = PropagationTime(resolution=10000000, final_time=100000000)
    propagation_time.plot_kepler(0)

def extreme_small_mass():
    propagation_time = PropagationTime(mass_sat=0.000000001)
    propagation_time.plot_kepler(0)

def extreme_large_mass():
    propagation_time = PropagationTime(mass_sat=1000000000000000000)
    propagation_time.plot_kepler(0)

def extreme_large_final_time():
    propagation_time = PropagationTime(final_time=100000000000000000000)
    propagation_time.plot_kepler(0)

def extreme_small_final_time():
    propagation_time = PropagationTime(final_time=1, resolution=0.1)
    propagation_time.plot_kepler(0)

def extreme_large_a():
    propagation_time = PropagationTime(orbit_parameters=np.array([[20e20, 0, 0, 0, 0, 0], [20e20, 0, 0, 0, 180, 0]]),final_time=86400*1000)
    propagation_time.plot_kepler(0)
    propagation_time.plot_time()

def extreme_small_a():
    propagation_time = PropagationTime(orbit_parameters=np.array([[2e6, 0, 0, 0, 0, 0], [2e6, 0, 0, 0, 180, 0]]))
    propagation_time.plot_kepler(1)
    propagation_time.plot_time()

def extreme_large_i():
    propagation_time = PropagationTime(orbit_parameters=np.array([[20e6, 0, 1000, 0, 0, 0], [20e6, 0, 1000, 0, 180, 0]]))
    propagation_time.plot_kepler(1)
    propagation_time.plot_time()

def extreme_large_e():
    propagation_time = PropagationTime(orbit_parameters=np.array([[20e6, 1000000, 0, 0, 0, 0], [20e6, 100000, 0, 0, 180, 0]]))
    propagation_time.plot_kepler(1)
    propagation_time.plot_time()

def extreme_negative_e():
    propagation_time = PropagationTime(orbit_parameters=np.array([[20e6, -1, 0, 0, 0, 0], [20e6, -1, 0, 0, 180, 0]]))
    propagation_time.plot_kepler(1)
    propagation_time.plot_time()

def extreme_large_argument_of_periapsis():
    propagation_time = PropagationTime(orbit_parameters=np.array([[20e6, 0, 0, 10000, 0, 0], [20e6, 0, 0, 10000, 180, 0]]))
    propagation_time.plot_kepler(1)
    propagation_time.plot_time()

def extreme_negative_argument_of_periapsis():
    propagation_time = PropagationTime(orbit_parameters=np.array([[20e6, 0, 0, -10000, 0, 0], [20e6, 0, 0, -10000, 180, 0]]))
    propagation_time.plot_kepler(1)
    propagation_time.plot_time()


# extreme_resolution() #dropped to 0 for a
# extreme_small_mass() # dropped to 0 for a
# extreme_large_mass() # works surprisingly
# extreme_large_final_time() # runtime error
# extreme_small_final_time() # Plots take a step at least once w.r.t resolution so have to make resolution smaller
# extreme_large_a() #Nothing happens
# extreme_small_a() # dropped to 0 for a and didn't work
# extreme_large_i() # resets it between 0 and 180 and works
# extreme_large_e() # fails completely (doesnt run)
# extreme_negative_e() # fails completely (doesnt run)
# extreme_large_argument_of_periapsis() #resets between 0-360
# extreme_negative_argument_of_periapsis() # resets between 0-360

# Sensitivity:

def sensitivity_mass():
    propagation_time_small = PropagationTime(mass_sat=100, orbit_parameters=satellites)
    propagation_time_large = PropagationTime(mass_sat=10000, orbit_parameters=satellites)
    avg_ranges_s, std_dev_s, max_ranges_s =propagation_time_small.min_max_kepler()
    avg_ranges_l, std_dev_l, max_ranges_l =propagation_time_large.min_max_kepler()
    delta_v_large = np.average(np.array(propagation_time_small.complete_delta_v(0, 86400)))
    delta_v_small = np.average(np.array(propagation_time_large.complete_delta_v(0, 86400)))
    change_range_max = max_ranges_l/max_ranges_s
    change_range_avg = avg_ranges_l / avg_ranges_s
    change_range_sd = std_dev_l/ std_dev_s
    change_delta_v = delta_v_large/delta_v_small
    print(f"max_range :{change_range_max},avg range: {change_range_avg}, sd_range:{change_range_sd}, detla_v change:{change_delta_v}")

def sensitivity_a():
    satellites[:, 0] = satellites[:, 0]/2
    propagation_time_small = PropagationTime(orbit_parameters=satellites)
    satellites[:, 0] = satellites[:, 0]*2
    propagation_time_large = PropagationTime(orbit_parameters=satellites)
    avg_ranges_s, std_dev_s, max_ranges_s =propagation_time_small.min_max_kepler()
    avg_ranges_l, std_dev_l, max_ranges_l =propagation_time_large.min_max_kepler()
    delta_v_large = np.average(np.array(propagation_time_small.complete_delta_v(0, 86400)))
    delta_v_small = np.average(np.array(propagation_time_large.complete_delta_v(0, 86400)))
    change_range_max = max_ranges_l/max_ranges_s
    change_range_avg = avg_ranges_l / avg_ranges_s
    change_range_sd = std_dev_l/ std_dev_s
    change_delta_v = delta_v_large/delta_v_small
    print(f"max_range :{change_range_max},avg range: {change_range_avg}, sd_range:{change_range_sd}, detla_v change:{change_delta_v}")

def sensitivity_resolution():
    propagation_time_small = PropagationTime(orbit_parameters=satellites, resolution=900)
    propagation_time_large = PropagationTime(orbit_parameters=satellites, resolution=1)
    avg_ranges_s, std_dev_s, max_ranges_s =propagation_time_small.min_max_kepler()
    avg_ranges_l, std_dev_l, max_ranges_l =propagation_time_large.min_max_kepler()
    delta_v_large = np.average(np.array(propagation_time_small.complete_delta_v(0, 86400)))
    delta_v_small = np.average(np.array(propagation_time_large.complete_delta_v(0, 86400)))
    change_range_max = max_ranges_l/max_ranges_s
    change_range_avg = avg_ranges_l / avg_ranges_s
    change_range_sd = std_dev_l/ std_dev_s
    change_delta_v = delta_v_large/delta_v_small
    print(f"max_range :{change_range_max},avg range: {change_range_avg}, sd_range:{change_range_sd}, detla_v change:{change_delta_v}")




# sensitivity_mass() #larger mass gives more stable kepler elements
# sensitivity_a()
sensitivity_resolution()








