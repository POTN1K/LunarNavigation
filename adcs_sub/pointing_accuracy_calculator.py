""""jksa;lda
By sdasd"""

#Local libraries
from mission_design import OrbitPlane
import numpy as np

r_moon = 1.737e6  # m



def Insight(_orbit):
        """Calculate the relative distance between the satellites in the orbit plane and the x, y and z components of their relative distance. Then, compute the perpendicular distance between the Moon center and relative satellite distance. If it is higher than the Moon radius, then the satellites are in sight, if not, they aren't."""
        QR = []
        rel_dist = []
        QP = []
        for i in range(_orbit.n_sat):
            QR.append((_orbit.satellites[i].r - _orbit.satellites[(i + 1) % _orbit.n_sat].r))
            rel_dist.append(np.linalg.norm(_orbit.satellites[i].r - _orbit.satellites[(i + 1) % _orbit.n_sat].r))
            QP.append(- _orbit.satellites[(i + 1) % _orbit.n_sat].r)
            #print(QR)
        #print(QP)
        d = []

        for i in range(_orbit.n_sat):
            a = np.cross(QP[i], QR[i])
            d.append(np.linalg.norm(a) / rel_dist[i])
            if d[i] > r_moon:
                print('True')
            else:
                print('False')

        return d

def OutOfPlaneDistance(_orbita, _orbitb):

    out_plane_dist = np.zeros((_orbita.n_sat, _orbitb.n_sat))
    print(f'{out_plane_dist=}')
    for i in range(_orbita.n_sat):
        for j in range(_orbitb.n_sat):
            out_plane_dist[i][j] = np.linalg.norm(_orbita.satellites[i].r - _orbitb.satellites[j].r)
    print(f'{out_plane_dist=}')
    return out_plane_dist, np.min(out_plane_dist)


if __name__=="__main__":

    orb1 = OrbitPlane(a=2e7, e=0.4, i=20, Omega=0, n_sat=4)
    orb2 = OrbitPlane(a=2e7, e=0.6, i=40, Omega=30, n_sat=4)

    rel1 = orb1.relDistSatellites()
    print(Insight(orb1))
    print(OutOfPlaneDistance(orb1, orb2)[1], "hello")
    #rel2 = orb2.relDistSatellites()

    sats1 = [sat.r for sat in orb1.satellites]
    #print(sats1)

    #print(rel1)

    #print(sats1)

