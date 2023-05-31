""""jksa;lda
By sdasd"""

#Local libraries
from mission_design import OrbitPlane

def func():
    pass

if __name__=="__main__":

    orb1 = OrbitPlane(a=2e7, i=20, n_sat=4)
    orb2 = OrbitPlane(a=2e7, i=20, Omega=30, n_sat=4)

    rel1 = orb1.relDistSatellites()

    sats1 = [sat.r for sat in orb1.satellites]

    print(rel1)
    print(sats1)

