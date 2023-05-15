"""Main File, used to run all simulations."""

# Local Libraries
from model_class import Model
from Earth_Const import *
from propagation_calculator import PropagationTime

# -----------------------------------------------------------------------------
# Static Simulation
# Create model
model = Model()

# Add satellite (a, e, i, w, Omega, nu, shift)
model.addSatellite(2.45e7, 0, 58.69, 22.9, 4, 6)

# Add tower (phi, theta, h)
model.addTower(20, 15, 100000)

# Add Lagrange point
model.addLagrange('L1')
model.addLagrange('L2')
    
# Add orbit plane (a, e, i, w, Omega, n_sat, shift)
# model.addOrbitPlane(2.45e7, 0.5, 85, 0, 0, 4)
# model.addOrbitPlane(2.45e7, 0.5, 85, 45, 120, 4)
# model.addOrbitPlane(2.45e7, 0.5, 85, 90, 240, 4)
# model.addOrbitPlane(2.45e7, 0.5, 0, 0, 0, 4)

# Add multiple orbit planes (a, e, i, w, n_planes, n_sat_per_plane, shift, elevation)
model.addSymmetricalPlanes(2.45e7, 0, 58.69, 22.9, 6, 4, 90)

# Add fixed point (r, elevation)
# model.addFixPoint([2.45e7, 3e6, 1e3], 10)


# Add Earth coverage
## Max Distance
# model.addFixPoint(rc_S3_Mn_max, 10)  # Satellite 3 (S3)
# model.addFixPoint(rc_S4_Mn_max, 10)  # S4
# model.addFixPoint(rc_S5_Mn_max, 10)  # S5
# model.addFixPoint(rc_S6_Mn_max, 10)  # S6

## Min dist (probably limiting for coverage)
# model.addFixPoint(rc_S3_Mn_min, 10)
# model.addFixPoint(rc_S4_Mn_min, 10)
# model.addFixPoint(rc_S5_Mn_min, 10)
# model.addFixPoint(rc_S6_Mn_min, 10)


# Get parameters for satellites in the model
model.getParams()

# Plot coverage
model.plotCoverage()

# -----------------------------------------------------------------------------
# Dynamic Simulation

satellites = [[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 0.17453292519943295],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 1.4311699866353502],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 2.6878070480712677],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 3.944444109507185],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 5.201081170943102],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 0.17453292519943295],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 1.4311699866353502],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 2.6878070480712677],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 3.944444109507185],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 5.201081170943102],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 0.17453292519943295],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 1.4311699866353502],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 2.6878070480712677],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 3.944444109507185],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 5.201081170943102]]


propagation_time = PropagationTime(satellites, 86400 * 14, 900, 250, 2, 0, 8)
propagation_time.complete_delta_v(0, 86400*14)
propagation_time.plot_kepler(1)
propagation_time.plot_time()
