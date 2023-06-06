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
    #print(f'{out_plane_dist=}')
    for i in range(_orbita.n_sat):
        for j in range(_orbitb.n_sat):
            out_plane_dist[i][j] = np.linalg.norm(_orbita.satellites[i].r - _orbitb.satellites[j].r)
        print(np.min(out_plane_dist[i]), 'cool')
    print(f'{out_plane_dist=}')
    return out_plane_dist, np.min(out_plane_dist)


if __name__=="__main__":

    orb1 = OrbitPlane(a=5701.2e3, e=0.002, i=40.78, w=90, Omega=0, n_sat=6)
    orb2 = OrbitPlane(a=5701.2e3, e=0.002, i=40.78, w=90, Omega=90, n_sat=6)
    orb3 = OrbitPlane(a=5701.2e3, e=0.002, i=40.78, w=90, Omega=180, n_sat=6)
    orb4 = OrbitPlane(a=5701.2e3, e=0.002, i=40.78, w=90, Omega=270, n_sat=6)
    orb5 = OrbitPlane(a=10000e3, e=0.038, i=10, w=90, Omega=0, n_sat=3, shift=30)
    orb6 = OrbitPlane(a=6541.4, e=0.6, i=56.2, w=90, Omega=0, n_sat=2)
    orb7 = OrbitPlane(a=6541.4, e=0.6, i=56.2, w=270, Omega=0, n_sat=2)



    rel1 = orb1.relDistSatellites()
    rel2 = orb2.relDistSatellites()
    rel3 = orb3.relDistSatellites()
    rel4 = orb4.relDistSatellites()
    rel5 = orb5.relDistSatellites()
    rel6 = orb6.relDistSatellites()
    rel7 = orb7.relDistSatellites()

    print(Insight(orb1), "Insight")
    print(OutOfPlaneDistance(orb1, orb2)[1], "hello")
    print(OutOfPlaneDistance(orb2, orb3)[1], "nice")
    print(OutOfPlaneDistance(orb1, orb3)[1], "sicc")

    #rel2 = orb2.relDistSatellites()

    sats1 = [sat.r for sat in orb1.satellites]
    #print(sats1)

    #print(rel1)

    #print(sats1)

