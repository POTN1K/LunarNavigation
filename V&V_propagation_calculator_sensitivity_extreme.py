import numpy as np
from propagation_calculator import PropagationTime

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








