""""jksa;lda
By sdasd"""

#Local libraries
from mission_design import OrbitPlane
import numpy as np
import csv as csv

r_moon = 1.737e6  # m

with open('Coordinates_in_time.csv', 'r') as file:

    reader = csv.reader(file)
    data = np.array([[float(element) for element in row] for row in reader])
    # print(data)
    data_no_time = data[:, 1:]
    # print(data_no_time)
    positionarray = data_no_time[:, 0:3]
    for i in range(1, len(data_no_time[0]//6)):
        positionarray = np.hstack((positionarray, data_no_time[:, 6*i:6*i+3]))
    positionarray = np.array_split(positionarray, 31)
    # print(positionarray[0], "Position")
    south_pole = positionarray[0:2]
    north_pole = positionarray[2:4]
    low_incl = positionarray[4:7]
    og1 = positionarray[7:13]
    og2 = positionarray[13:19]
    og3 = positionarray[19:25]
    og4 = positionarray[25:31]

    # print(south_pole, 'SOuthh')

def Insight(plane):

        QR = []
        d = []
        """Calculate the relative distance between the satellites in the orbit plane and the x, y and z components of their relative distance. Then, compute the perpendicular distance between the Moon center and relative satellite distance. If it is higher than the Moon radius, then the satellites are in sight, if not, they aren't."""
        for i in range(len(plane)):
            #print(_orbit.satellites[i].r, 'test')

            # QR.append((_orbit.satellites[i].r - _orbit.satellites[(i + 1) % _orbit.n_sat].r))
            QRi = (plane[i] - plane[(i + 1) % len(plane)])
            # print(QRi)
            di = np.sqrt(QRi[0]**2 + QRi[1]**2 + QRi[2]**2)
            QR.append(QRi)
            d.append(di)
        print(np.shape(d))
        return d, len(d)
#
def OutOfPlaneDistance(plane_a, plane_b):

    QP = []
    min = []

    #print(f'{out_plane_dist=}')
    for i in range(len(plane_a)):
        do = np.zeros(len(plane_a[:, 0]), len(plane_a))
        for j in range(plane_b):
            QPi = (plane_a[i] - plane_b[j])
            QP.append(QPi)
            d = np.sqrt(QPi[0] ** 2 + QPi[1] ** 2 + QPi[2] ** 2)
            do[:, i] = d
            for k in range(len(d)):
                min.append(np.min(do[k, :]))
#
#
#
#         #print(np.min(out_plane_dist[i]), 'cool')
#     #print(f'{out_plane_dist=}')
#     return out_plane_dist, np.min(out_plane_dist)


if __name__=="__main__":

    # orb1 = OrbitPlane(a=6541.4e3, e=0.6, i=56.2, w=90, Omega=0, n_sat=2)
    # orb2 = OrbitPlane(a=6541.4e3, e=0.6, i=56.2, w=270, Omega=0, n_sat=2)
    # orb3 = OrbitPlane(a=10000e3, e=0.038, i=10, w=90, Omega=0, n_sat=3, shift=30)
    # orb4 = OrbitPlane(a=5701.2e3, e=0.002, i=40.78, w=90, Omega=0, n_sat=6)
    # orb5 = OrbitPlane(a=5701.2e3, e=0.002, i=40.78, w=90, Omega=90, n_sat=6)
    # orb6 = OrbitPlane(a=5701.2e3, e=0.002, i=40.78, w=90, Omega=180, n_sat=6)
    # orb7 = OrbitPlane(a=5701.2e3, e=0.002, i=40.78, w=90, Omega=270, n_sat=6)




    # rel1 = orb1.relDistSatellites()
    # rel2 = orb2.relDistSatellites()
    # rel3 = orb3.relDistSatellites()
    # rel4 = orb4.relDistSatellites()
    # rel5 = orb5.relDistSatellites()
    # rel6 = orb6.relDistSatellites()
    # rel7 = orb7.relDistSatellites()

    print(Insight(south_pole), "Insight")
    # print(OutOfPlaneDistance(orb1, orb2)[1], "hello")
    # print(OutOfPlaneDistance(orb2, orb3)[1], "nice")
    # print(OutOfPlaneDistance(orb1, orb3)[1], "sicc")

    #rel2 = orb2.relDistSatellites()

    #sats1 = [sat.r for sat in orb1.satellites]
    #print(sats1)

    #print(rel1)

    #print(sats1)

    # def Insight(_orbit):
    #     """Calculate the relative distance between the satellites in the orbit plane and the x, y and z components of their relative distance. Then, compute the perpendicular distance between the Moon center and relative satellite distance. If it is higher than the Moon radius, then the satellites are in sight, if not, they aren't."""
    #     QR = []
    #     rel_dist = []
    #     QP = []
    #     for i in range(_orbit.n_sat):
    #         # print(_orbit.satellites[i].r, 'test')
    #
    #         # QR.append((_orbit.satellites[i].r - _orbit.satellites[(i + 1) % _orbit.n_sat].r))
    #         QR = ((_orbit.satellites[i].r - _orbit.satellites[(i + 1) % _orbit.n_sat].r))
    #         QP = np.sqrt(QR[0] ** 2 + QR[1] ** 2 + QR[2] ** 2)
    #         rel_dist.append(np.linalg.norm(_orbit.satellites[i].r - _orbit.satellites[(i + 1) % _orbit.n_sat].r))
    #         QP.append(- _orbit.satellites[(i + 1) % _orbit.n_sat].r)
    #         # print(QR)
    #     # print(QP)
    #     d = []