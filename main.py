"""Main File, used to run all simulations."""
import numpy as np

# Local Libraries
from model_class import Model
from earth_constellation import *
from propagation_calculator import PropagationTime
from user_position_calculator import UserErrors

# -----------------------------------------------------------------------------
# Static Simulation
# Create model
model = Model()

# Add Lagrange point
model.addLagrange('L1')
model.addLagrange('L2')

# Add multiple orbit planes (a, e, i, w, n_planes, n_sat_per_plane, shift, elevation)
model.addSymmetricalPlanes(2.45e7, 0, 58.69, 22.9, 5, 4)

# Plot coverage
model.plotCoverage()

# Get satellites in view for a point (Point on the moon surface 0-999)
# r is the cartesian coordinates of a satellite
point = 682
print(f"Surface coordinates: {model.moon[point]}")
sat_position = np.array([sat.r for sat in model.mod_inView_obj[point]])
print(sat_position)


# Continue?
flag = input("Press Enter to continue... 'e' to exit.\n")
if flag != '':
    exit()

#DOP Calculation
DOP_with_error = []
error_budget = []

for i in range(0, 1000):

    point = i
    Errors = UserErrors(np.array([sat.r for sat in model.mod_inView_obj[point]]), 0,
                        model.moon[point], [20, 10, 10, 10, 10, 3.5])
    DOP_with_error.append(Errors.Error)
    error_budget.append(Errors.ErrorBudget)

print(f"GDOP, PDOP, HDOP, VDOP, TDOP, HHDOP")
print(f"DOP WITH ERROR, PERFECT ORBIT, mean:{np.mean(DOP_with_error, axis=0)}, max: {np.max(DOP_with_error, axis=0)}, min: {np.min(DOP_with_error, axis=0)}, ptp: {np.ptp(DOP_with_error, axis=0)},SD: {np.std(DOP_with_error, axis=0)}")
print(f"ORBIT BUDGET FOR REQUIREMENT, mean:{np.mean(error_budget, axis=0)}, max: {np.max(error_budget, axis=0)}, min: {np.min(error_budget, axis=0)}, ptp: {np.ptp(error_budget, axis=0)},SD: {np.std(error_budget, axis=0)}")


# -----------------------------------------------------------------------------
# Dynamic Simulation

satellites = model.getSatellites()


# PropagationTime(satellites, total_time, time_step, delta_v, n_planes, shift, elevation)
propagation_time = PropagationTime(satellites, 86400 * 14, 900, 250, 2, 0)
print(np.average(np.array(propagation_time.complete_delta_v(0, 86400*14))))
propagation_time.plot_kepler(1)
propagation_time.plot_time()

