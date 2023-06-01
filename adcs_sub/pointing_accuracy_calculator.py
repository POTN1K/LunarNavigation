""""jksa;lda
By sdasd"""

#Local libraries
from mission_design import OrbitPlane
import numpy as np

r_moon = 1.737e6  # m



def Insight(self):
        """Calculate the relative distance between the satellites in the orbit plane and the x, y and z components of their relative distance"""
        QR = []
        rel_dist = []
        QP = []
        for i in range(self.n_sat):
            QR.append((self.satellites[i].r - self.satellites[(i + 1) % self.n_sat].r))
            rel_dist.append(np.linalg.norm(self.satellites[i].r - self.satellites[(i + 1) % self.n_sat].r))
            QP.append(- self.satellites[(i + 1) % self.n_sat].r)
            print(QR)
            print(QP)
        d = []

        for i in range(self.n_sat):
            a = np.cross(QP[i], QR[i])
            d.append(np.linalg.norm(a) / rel_dist[i])

        return QR, d, QP

# def SatellitesInsight(self):
#     QR = []
#     QP = []
#     rel_dist = []
#     for i in range(self.n_sat):
#         QR.append((self.satellites[i].r - self.satellites[(i + 1) % self.n_sat].r))
#         rel_dist.append(np.linalg.norm(self.satellites[i].r - self.satellites[(i + 1) % self.n_sat].r))
#         QP.append(- self.satellites[(i + 1) % self.n_sat].r)
#         print(QR)
#         print(QP)
#     d = []
#
#     for i in range(self.n_sat):
#         a = np.cross(QP[i], QR[i])
#         d.append(np.linalg.norm(a) / rel_dist[i])


if __name__=="__main__":

    orb1 = OrbitPlane(a=2e7, e=0.4, i=20, Omega=0, n_sat=4)
    #orb2 = OrbitPlane(a=2e7, e=0.6, i=20, Omega=30, n_sat=4)

    rel1 = Insight()

    #rel2 = orb2.relDistSatellites()

    sats1 = [sat.r for sat in orb1.satellites]

    print(rel1)
    #print(sats1)

