"""Main File, used to run all simulations."""
import numpy as np

# Local Libraries
from model_class import Model
from propagation_calculator import PropagationTime
from user_error_calculator import UserErrors

# -----------------------------------------------------------------------------
# Static Simulation
# Create model
model = Model()
#
#satperplane = 7
#shiftangle = 30
#model.addOrbitPlane(8049e3, 0.4082, 45, 90, 0, satperplane, shiftangle)
#model.addOrbitPlane(8049e3, 0.4082, 45, 270, 0, satperplane, shiftangle)
#model.addOrbitPlane(8049e3, 0.4082, 45, 90, 180, satperplane, shiftangle)
#model.addOrbitPlane(8049e3, 0.4082, 45, 270, 180, satperplane, shiftangle)


model.addSymmetricalPlanes(a=24572000, i=58.69, e=0, w=22.9, n_planes=6, n_sat_per_plane=4, f=0.2118, dist_type=1)

# # Plot coverage
model.plotCoverage()

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


#Frozen Orbits
#satellites = [[3737822.9, 0.1, np.deg2rad(48.0679), np.deg2rad(0), np.deg2rad(89.9996), np.deg2rad(0.0004)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(90),  np.deg2rad(0), np.deg2rad(0)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(90), np.deg2rad(0),  np.deg2rad(150)],[8049000, 0.4082, np.deg2rad(45), np.deg2rad(90),  np.deg2rad(0), np.deg2rad(210)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(270), np.deg2rad(0), np.deg2rad(0)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(270), np.deg2rad(0), np.deg2rad(150)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(270), np.deg2rad(0), np.deg2rad(210)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(90), np.deg2rad(180), np.deg2rad(107)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(90), np.deg2rad(180), np.deg2rad(180)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(90), np.deg2rad(180), np.deg2rad(253)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(270), np.deg2rad(180), np.deg2rad(107)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(270), np.deg2rad(180), np.deg2rad(180)], [8049000, 0.4082, np.deg2rad(45), np.deg2rad(270), np.deg2rad(180), np.deg2rad(253)]]


# PropagationTime(satellites, total_time, time_step, delta_v, n_planes, shift, elevation)
propagation_time = PropagationTime(satellites, 86400*3 , 900, 250, 0, 0)
print(np.average(np.array(propagation_time.complete_delta_v(0, 86400*3))))
propagation_time.plot_kepler(0)
propagation_time.plot_time()

