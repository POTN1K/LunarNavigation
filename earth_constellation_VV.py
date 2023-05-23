# Verif and Validation of the Earth Constellation python file

from earth_constellation import *

def EO_1_A():
    for i in range(len(t)):
        z_M = rc_M[i, 2]
        z_L1 = rc_L1_R[i, 2]
        z_L2 = rc_L2_R[i, 2]
        z_S3 = rc_S3_Rr[i, 2]
        z_S4 = rc_S4_Rr[i, 2]
        z_S5 = rc_S5_Rr[i, 2]
        z_S6 = rc_S6_Rr[i, 2]
        if z_M != z_L1 != z_L2 != z_S3 != z_S4 != z_S5 != z_S6 != 0:
            print("fuck me")
EO_1_A()

def EO_1_B():
    for i in range(len(t)):
        line_slope1 = (rc_M[i, 1] - rc_L1_R[i, 1]) / (rc_M[i, 0] - rc_L1_R[i, 0])
        line_slope2 = (rc_M[i, 1] - rc_L2_R[i, 1]) / (rc_M[i, 0] - rc_L2_R[i, 0])
        line_slope3 = (rc_L1_R[i, 1] - rc_L2_R[i, 1]) / (rc_L1_R[i, 0] - rc_L2_R[i, 0])
        if np.round(line_slope1, 12) != np.round(line_slope2, 12) != np.round(line_slope3, 12):
            print("fuck me, again")
EO_1_B()

def EO_2_A():
    for i in range(len(t)):
        if rc_S3_R[i, 0] != rc_S3_Rr[i, 0]*np.cos(ir):
            print("fuuuuuuuck")
        if rc_S3_R[i, 1] != rc_S3_Rr[i, 1]:
            print("fuuuuuuuck")
        if rc_S3_R[i, 2] != rc_S3_Rr[i, 0]*np.sin(ir):
            print("fuuuuuuuck")
EO_2_A()

