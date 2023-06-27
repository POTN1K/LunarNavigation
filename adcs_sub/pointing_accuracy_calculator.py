""""jksa;lda
By sdasd"""

# Local libraries
# from mission_design import OrbitPlane
import numpy as np
import pandas as pd

r_moon = 1.737e6  # m

# Functions
def max_inPlane(plane):
    """Calculate the relative distance between the satellites in the orbit plane and the x, y and z components of their relative distance.
    Then, compute the perpendicular distance between the Moon center and relative satellite distance. If it is higher than the Moon radius, then the satellites are in sight, if not, they aren't."""

    total_d = []
    for i in range(len(plane)):
        _ = (plane[i] - plane[(i + 1) % len(plane)])
        dist = np.linalg.norm(_, axis=1)
        total_d.append(dist)
    return np.max(total_d)


def max_outPlane(plane_a, plane_b):
    """Calculate maximum relative distance between two orbit planes"""
    max_dist = 0
    for sata in plane_a:
        tdist = []
        for satb in plane_b:
            dist = np.linalg.norm(sata - satb, axis=1)
            tdist.append(dist)
        _ = np.max(np.min(tdist, axis=0))
        if _ > max_dist:
            max_dist = _
    return max_dist


def clean_csv():
    df = pd.read_csv('Coordinates_in_time.csv', header=None)
    # Drop time
    df.drop(columns=0, inplace=True)
    # Velocity columns / Drop
    erase_indices = np.array([[i, i + 1, i + 2] for i in range(4, df.shape[1], 6)]).flatten()
    df.drop(columns=erase_indices, inplace=True)
    # Time steps to keep
    keep = np.arange(3, df.shape[0], 10)
    clean_df = df.loc[keep]
    # Separate satellites
    sat = np.array([clean_df[[i, i + 1, i + 2]].to_numpy() for i in range(1, 93 * 2, 6)])
    return sat

def relay_csv():
    df = pd.read_csv('statesarray_Jasper.csv', header= None)
    #Drop time
    df.drop(columns=0, inplace=True)
    #Velocity and replica columns, drop
    erase_indices = np.arange(4, df.shape[1]+1)
    df.drop(columns=erase_indices, inplace=True)
    #Time steps to keep
    keepTime = np.arange(3, df.shape[0], 10)
    relayP = df.loc[keepTime]
    relay = np.array(relayP)
    return relay

    
    
def PAcc(divI, DI, MP, L ):
    DO = MP*DI + L*np.tan(2*divI/MP)
    acc = np.arctan2(DO/2, L) *180/np.pi
    return DO, acc


if __name__ == "__main__":
    # Read csv
    sat = clean_csv()
    relay = relay_csv()

    # Planes
    south_pole = sat[0:2]
    north_pole = sat[2:4]
    low_incl = sat[4:7]
    og1 = sat[7:13]
    og2 = sat[13:19]
    og3 = sat[19:25]
    og4 = sat[25:31]
    
    planes = {"South Pole": south_pole, "North Pole": north_pole, 
              "Low Inclination": low_incl, "OG Orbit 1": og1, 
              "OG Orbit 2": og2, "OG Orbit 3": og3, "OG Orbit 4": og4}





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
    InMax = []
    for key in planes:
        InMax.append(max_inPlane(planes[key]))
    
    OutMax = []
    ii = 0
    for key1 in planes:
        for key2 in planes:
           OutMax.append(max_outPlane(planes[key1], planes[key2]))
        ii = ii + 1
        if ii > 2:
            break
    
    # relayMax = []
    # for keyR in planes:
    #     relayMax.append(max_outPlane(relay, planes[keyR]))
    # print(np.min(relayMax))
                        
    # print(OutMax)
    Max = np.max(InMax + OutMax)
    print(Max)
    
    DO = PAcc(8*10**(-3), 45*10**(-3), 1, Max)
    print(DO)
    # print(OutOfPlaneDistance(orb1, orb2)[1], "hello")
    # print(OutOfPlaneDistance(orb2, orb3)[1], "nice")
    # print(OutOfPlaneDistance(orb1, orb3)[1], "sicc")

    # rel2 = orb2.relDistSatellites()

    # sats1 = [sat.r for sat in orb1.satellites]
    # print(sats1)

    # print(rel1)

    # print(sats1)

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
