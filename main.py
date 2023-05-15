"""Main File, used to run all simulations."""

# Local Libraries
from model_class import Model
from Earth_Const import *

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