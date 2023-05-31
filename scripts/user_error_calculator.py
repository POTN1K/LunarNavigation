"""
This script works together with the code from Niko to be able to get all the DOP values on the discritized moon. I am
planning to add the same with velocity, but I need to output velocity in the simulation for that to
work. The code also gives a budget for allowable ephemeris error w.r.t the requirements.

MADE BY KYLE SCHERPENZEEL :(
"""

import numpy as np


class UserErrors:
    def __init__(self, sats, sats_velocity, pos_error, position_surface, allowable): #, allowable, parameter#
        self.ErrorBudget = []
        self.ORBIT = pos_error
        self.sats = sats
        self.pos_error = pos_error
        self.sats_velocity = sats_velocity
        # self.point = point
        self.allowable = allowable
        self.position_user = position_surface
        self.parameter_covariance_matrix()
        self.dop_calculator()
        self.user_error()
        # self.allowable_error()


    @property
    def sats(self):
        return self._sats

    @sats.setter
    def sats(self, value):
        if len(value.shape) >= 2 and value.shape[0] >=4:
            self._sats = value
        else:
            raise ValueError("Must be at least 4 satellites")

    @property
    def position_surface(self):
        return self._position_surface

    @position_surface.setter
    def position_surface(self, value):
        if len(value.shape) == (1, 3):
            self._position_surface = value
        else:
            raise ValueError("Must be 3D coordinates")

    @property
    def allowable(self):
        return self._allowable

    @allowable.setter
    def allowable(self, value):
        if np.array(value).shape == (6,):
            self._allowable = value
        else:
            raise ValueError("Must give 6 error requirements")

    @property
    def pos_error(self):
        return self._pos_error

    @pos_error.setter
    def pos_error(self, value):
        if value >= 0:
            self._pos_error = value
        else:
            raise ValueError("Ephemeris error must be equal or greater than 0")

    def satellite_error(self):
        CLOCK_ERROR = 1.1
        RECIEVER_NOISE_AND_RESOLUTION = 0.1
        OTHER = 1
        MULTIPATH = 0.2
        return np.sqrt(CLOCK_ERROR**2 + self.ORBIT**2 + RECIEVER_NOISE_AND_RESOLUTION**2 + MULTIPATH**2 + OTHER**2)



    def parameter_covariance_matrix(self):
        H = np.ones((len(self.sats), 4))

        # Compute vectors from receiver to satellites
        vecs = self.sats - self.position_user

        # Compute distances
        self.dists = np.linalg.norm(vecs, axis=1)

        # Compute unit vectors
        uvecs = vecs / self.dists[:, np.newaxis]

        # Insert unit vectors into geometry matrix
        H[:, :3] = uvecs
        HH = np.vstack((H, [[0, 0, 1, 0]]))
        HH_inv = np.linalg.pinv(HH)
        H_inv = np.linalg.pinv(H)
        # Compute covariance matrix
        self.Q = np.dot(H_inv, H_inv.T)
        self.HQ = np.dot(HH_inv, HH_inv.T)
    def velocity_parameter_cov(self, time_index):
        HV = np.ones((len(self.sats), 4))
        # Compute vectors from receiver to satellites
        vecs = self.sats - self.position_user
        # Compute distances
        self.dists = np.linalg.norm(vecs, axis=1)
        # Compute unit vectors
        uvecs = vecs / self.dists[:, np.newaxis]
        HV[:, :3] = np.dot(uvecs, self.sats_velocity[time_index].reshape((3, (len(self.sats)))))
        HV_inv = np.linalg.pinv(HV)
        self.VQ = np.dot(HV_inv, HV_inv.T)
        print(np.sqrt(np.trace(self.VQ)))





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

        HHDOP = np.sqrt(np.trace(self.HQ[:2, :2]))



        self.DOP = {
            "GDOP": GDOP,
            "PDOP": PDOP,
            "HDOP": HDOP,
            "VDOP": VDOP,
            "TDOP": TDOP,
            "HHDOP": HHDOP
        }
        self.DOP_array = []
        for key in self.DOP:
            self.DOP_array.append(self.DOP[key])
    def user_error(self):
        self.Error = []
        for key in self.DOP:
            self.Error.append(self.DOP[key] * self.satellite_error())

    def allowable_error(self):
        for i in range(len(self.DOP)):
            self.ORBIT = 0
            self.user_error()
            while self.Error[i] <= self.allowable[i]:
                self.ORBIT = self.ORBIT + .05
                self.user_error()
            self.ErrorBudget.append(self.ORBIT)
            self.ORBIT = 0
            self.user_error()

