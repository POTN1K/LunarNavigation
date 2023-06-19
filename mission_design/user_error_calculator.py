"""
This script works together with the code from Niko to be able to get all the DOP values on the discritized moon. I am
planning to add the same with velocity, but I need to output velocity in the simulation for that to
work. The code also gives a budget for allowable ephemeris error w.r.t the requirements.

MADE BY KYLE SCHERPENZEEL :(
"""

import numpy as np


class UserErrors:
    def __init__(self, sats, sats_velocity, pos_error, position_surface, allowable): #, allowable, parameter#
        self.ephemeris_budget = None
        self.DOP_array = None
        self.DOP_error_array = None
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
        # self.allowable_error()


    # @property
    # def sats(self):
    #     return self._sats
    #
    # @sats.setter
    # def sats(self, value):
    #     if len(value.shape) >= 2 and value.shape[0] >=4:
    #         self._sats = value
    #     else:
    #         raise ValueError("Must be at least 4 satellites")

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

    def satellite_error(self, ORBIT):
        CLOCK_ERROR = 0.03
        RECEIVER_NOISE_AND_RESOLUTION = 0.1
        MULTIPATH = 0.2
        DIFF_GROUP_DELAY = 0.15
        return np.sqrt(CLOCK_ERROR**2 + ORBIT**2 + RECEIVER_NOISE_AND_RESOLUTION**2 + MULTIPATH**2 + DIFF_GROUP_DELAY**2)



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
    def velocity_parameter_cov(self, velocity_indeces):
        HV = np.ones((len(self.sats), 4))
        # Compute vectors from receiver to satellites
        vecs = self.sats - self.position_user
        # Compute distances
        self.dists = np.linalg.norm(vecs, axis=1)
        rel_velocities = []
        for i in range(0, len(velocity_indeces)):
            rel_velocities.append([self.sats_velocity[velocity_indeces[i]:velocity_indeces[i]+3]])
        rel_velocities = np.squeeze(np.asarray(rel_velocities))
        # Compute unit vectors
        uvecs = vecs / self.dists[:, np.newaxis]
        HV[:, :3] = uvecs * rel_velocities
        HV_inv = np.linalg.pinv(HV)
        self.VQ = np.dot(HV_inv, HV_inv.T)
        # print(np.sqrt(np.trace(self.VQ)))
        return self.velocity_error_calculator()




    def velocity_error_calculator(self):
        self.velocity3d = np.sqrt(np.trace(self.VQ[:3, :3]))
        self.velocityH =  np.sqrt(np.trace(self.VQ[:2, :2]))
        self.velocityV =  np.sqrt(self.VQ[2, 2])
        return np.asarray([self.velocity3d, self.velocityH, self.velocityV])


        # Now, we can compute the DOP values
    def dop_calculator(self):
        # GDOP (Geometric DOP) - uses all elements
        self.GDOP = np.sqrt(np.trace(self.Q))

        # PDOP (Position DOP) - uses the 3D positional elements
        self.PDOP = np.sqrt(np.trace(self.Q[:3, :3]))

        # HDOP (Horizontal DOP) - uses the horizontal positional elements
        self.HDOP = np.sqrt(np.trace(self.Q[:2, :2]))

        # VDOP (Vertical DOP) - uses the vertical positional element
        self.VDOP = np.sqrt(self.Q[2, 2])

        # TDOP (Time DOP) - uses the time element
        self.TDOP = np.sqrt(self.Q[3, 3])

        self.HHDOP = np.sqrt(np.trace(self.HQ[:2, :2]))



        # self.DOP = {
        #     "GDOP": GDOP,
        #     "PDOP": PDOP,
        #     "HDOP": HDOP,
        #     "VDOP": VDOP,
        #     "TDOP": TDOP,
        #     "HHDOP": HHDOP
        # }

        self.DOP_array = np.array([self.GDOP, self.PDOP, self.HDOP, self.VDOP, self.TDOP, self.HHDOP])
        self.DOP_error_array = self.DOP_array * self.satellite_error(0)
        # for key in self.DOP:
        #     self.DOP_array.append(self.DOP[key])

    def allowable_error(self, DOP_array):
        constraints = np.max(DOP_array, axis=0)
        ephemeris_budget = []
        for i in range(len(constraints)):
            ephemeris_budget.append(np.sqrt((self.allowable[i] ** 2 / constraints[i] ** 2) - self.satellite_error(0) ** 2))
        self.ephemeris_budget = np.array(ephemeris_budget)
        return self.ephemeris_budget
