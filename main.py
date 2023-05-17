"""Main File, used to run all simulations."""

# Local Libraries
from model_class import Model
from earth_constellation import *
from propagation_calculator import PropagationTime

# -----------------------------------------------------------------------------
# Static Simulation
# Create model
model = Model()

# Add Lagrange point
model.addLagrange('L1')
# model.addLagrange('L2')

# Add multiple orbit planes (a, e, i, w, n_planes, n_sat_per_plane, shift, elevation)
# model.addSymmetricalPlanes(2.45e7, 0, 58.69, 22.9, 6, 4)

print(model.modules[0].range)
# Plot coverage
model.plotCoverage()

# Continue?
flag = input("Press Enter to continue... 'e' to exit.\n")
if flag != '':
    exit()
# -----------------------------------------------------------------------------
# Dynamic Simulation

satellites = model.getSatellites()
print(satellites)

# PropagationTime(satellites, total_time, time_step, delta_v, n_planes, shift, elevation)
propagation_time = PropagationTime(satellites, 86400 * 14, 900, 250, 2, 0, 8)
print(np.average(np.array(propagation_time.complete_delta_v(0, 86400*14))))
propagation_time.plot_kepler(1)
propagation_time.plot_time()
