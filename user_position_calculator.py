"""
MADE BY KYLE SCHERPENZEEL :(
"""

import numpy as np


sats = np.array([
    [20000, 0, 0],
    [-20000, 0, 0],
    [0, 20000, 0],
    [0, -20000, 0],
    [0, 0, 20000]
])

distances = np.array([19999, 20001, 20000.000025, 20000.000025, 20000.000025])



class UserErrors:
    def __init__(self, sats, pos_error, distances_user_sat):
        self.BROADCAST_EPHEMERIS = pos_error
        self.sats = sats
        self.pos_error = pos_error
        self.distances_user_sat = distances_user_sat
        self.parameter_covariance_matrix()
        self.dop_calculator()




    def satellite_error(self):
        CLOCK_ERROR = 0.4
        RECIEVER_NOISE_AND_RESOLUTION = 0.1
        MULTIPATH = 0.2
        return np.sqrt(CLOCK_ERROR**2 + self.BROADCAST_EPHEMERIS**2 + RECIEVER_NOISE_AND_RESOLUTION**2 + MULTIPATH**2)



    def parameter_covariance_matrix(self):
        rx = np.array([0, 0, 0])
        H = np.ones((len(sats), 4))

        # Compute vectors from receiver to satellites
        vecs = sats - rx

        # Compute distances
        dists = np.linalg.norm(vecs, axis=1)

        # Compute unit vectors
        uvecs = vecs / self.distances_user_sat[:, np.newaxis]

        # Insert unit vectors into geometry matrix
        H[:, :3] = uvecs
        H_inv = np.linalg.pinv(H)

        # Compute covariance matrix
        self.Q = np.dot(H_inv, H_inv.T)

        # Now, we can compute the DOP values
    def dop_calculator(self):
        # GDOP (Geometric DOP) - uses all elements
        GDOP = np.sqrt(np.trace(self.Q))

        # PDOP (Position DOP) - uses the 3D positional elements
        PDOP = np.sqrt(np.trace(self.Q[:3, :3]))

        # HDOP (Horizontal DOP) - uses the horizontal positional elements
        HDOP = np.sqrt(np.trace(self.Q[:2, :2]))

        # VDOP (Vertical DOP) - uses the vertical positional element
        VDOP = np.sqrt(self.Q[2, 2])

        # TDOP (Time DOP) - uses the time element
        TDOP = np.sqrt(self.Q[3, 3])

        self.DOP = {
            "GDOP": GDOP,
            "PDOP": PDOP,
            "HDOP": HDOP,
            "VDOP": VDOP,
            "TDOP": TDOP
        }

Errors = UserErrors(sats,0,distances)
print(Errors.DOP)

