"""Main File, used to run all simulations.
Maintained by Nikolaus Ricker"""

# External Libraries
import numpy as np

# Local Libraries
from mission_design import Model, PropagationTime, UserErrors

# -----------------------------------------------------------------------------
# Static Simulation
# Create model
model = Model()

# Add preliminary orbit
model.addSymmetricalPlanes(a=24572000, i=58.69, e=0, w=22.9, n_planes=6, n_sat_per_plane=4, dist_type=1)

# # Plot coverage
# model.plotCoverage()
#
# # -----------------------------------------------------------------------------
# # Continue?
# flag = input("Press Enter to continue... 'e' to exit.\n")
# if flag != '':
#     exit()
#
# # -----------------------------------------------------------------------------
# # DOP Calculation
# DOP_with_error = []
# error_budget = []
#
# for i in range(0, 1000):
#     point = i
#     Errors = UserErrors(np.array([sat.r for sat in model.mod_inView_obj[point]]), 0, 0,
#                         model.moon[point], [20, 10, 10, 10, 10, 3.5])
#
#
# print(f"GDOP, PDOP, HDOP, VDOP, TDOP, HHDOP")
# print(f"DOP WITH ERROR, PERFECT ORBIT, mean:{np.mean(DOP_with_error, axis=0)}, max: {np.max(DOP_with_error, axis=0)}, "
#       f"min: {np.min(DOP_with_error, axis=0)}, ptp: {np.ptp(DOP_with_error, axis=0)},SD: {np.std(DOP_with_error, axis=0)}")
# print(f"ORBIT BUDGET FOR REQUIREMENT, mean:{np.mean(error_budget, axis=0)}, max: {np.max(error_budget, axis=0)}, "
#       f"min: {np.min(error_budget, axis=0)}, ptp: {np.ptp(error_budget, axis=0)},SD: {np.std(error_budget, axis=0)}")

# -----------------------------------------------------------------------------
# Dynamic Simulation
satellites = model.getSatellites()
duration = 86400 * 3

# PropagationTime(satellites, total_time, time_step, delta_v, n_planes, shift, elevation)
propagation_time = PropagationTime(satellites, duration, 900, 250, 0, 0)

print(np.average(np.array(propagation_time.complete_delta_v(0, duration))))

propagation_time.plot_kepler(0)
propagation_time.plot_time()
