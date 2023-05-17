# This python file calculates the distance between the satellites
# S3, S4, S5 and S6 and the Moon at all times during one period

import numpy as np
import matplotlib.pyplot as plt


### Initialisation

# Parameters
R_SOI = 66100000  # [m] => Radius of the Sphere Of Influence of the Moon
R_M = 384400000  # [m] => Earth-Moon average distance
M_incl = 6.68*np.pi/180  # [rad] Moon obliquity (equator incl wrt its orbit around Earth)
mu_E = 3.986004418 * 10**(14)  # [m^3/s^2] => Earth Gravitational Parameter
P_M = 27.45189357205398 * 24 * 3600  # [s] => Period of a revolution of the Moon around the Earth.
                                     # Real period is 27.322 days but changed that to 27.45189357205398 so that the orbit is circular

# Minimum angle between the projection of the Earth-Moon line on the satellites orbit and the Earth-Satellite line
# for the satellites to be out of the SOI at all times for an inclination angle of 12.7235891 [deg]
alpha0 = 0.1732387  # 0.172796400948 min angle for no penetration of SOI for i = ir_max
                    # 0.17217 [rad] is the min for no penetration at the intersection
# Inclination
ir_max = np.arctan2(R_SOI, R_M)  # [rad] => maximum inclination angle of the red orbit
if np.abs(ir_max*180/np.pi - 9.75694912) > 10**(-9):  # VERIF
    print(ir_max*180/np.pi - 9.75694912)
ir = 12.7235891*np.pi/180  # 60000118*np.pi/180  # [rad] => actual inclination angle of the red orbit : used as an INPUT
ig = - ir  # [rad] => inclination angle of the green orbit

# Time
t0 = 0
tf = P_M
Nb_pts = 20000
dt = P_M/(Nb_pts)
t = np.arange(t0, tf, dt)


### Characteristics of the moon and the six satellites on Lagrange pts (L1, L2) and on inclined orbits (S3, S4, S5 & S6)
M = np.array([R_M, 0, 0])
L1 = np.array([R_M-R_SOI, 0, 0])
L2 = np.array([R_M+R_SOI, 0, 0])
S3 = np.array([R_M, alpha0, ir])
S4 = np.array([R_M, -alpha0, ir])
S5 = np.array([R_M, alpha0, ig])
S6 = np.array([R_M, -alpha0, ig])
Phi = np.pi/2  # [rad] => 3rd spherical coordinate (for Moon ref frame R : orbital plane of the Moon around the Earth is the same as x-y plane
               # for S1, S2, S3, S4 in Rr and Rg respectively)


### Time dependent position of the Moon in cartesian coordinates in R and of S3, S4, S5 and S6 in cartesion coordinates in Rr (for S3 and S4), and Rg (for S5 and S6)

def moon_sat_cart_pos(sat, t):
    rc = np.zeros((len(t), 3))
    R = sat[0]
    Theta = sat[1] + t*np.sqrt(mu_E/R_M**3)
    for i in range(len(t)):
        rc[i, :] = [R * np.cos(Theta[i]) * np.sin(Phi),
                    R * np.sin(Theta[i]) * np.sin(Phi),
                    R * np.cos(Phi)]
        for j in range(np.shape(rc)[1]):
            if np.abs(rc[i, j]) <= 10 ** (-7):
                rc[i, j] = 0
    return rc, Theta

rc_M = moon_sat_cart_pos(M, t)[0]
rc_L1_R = moon_sat_cart_pos(L1, t)[0]
rc_L2_R = moon_sat_cart_pos(L2, t)[0]
rc_S3_Rr = moon_sat_cart_pos(S3, t)[0]
rc_S4_Rr = moon_sat_cart_pos(S4, t)[0]
rc_S5_Rr = moon_sat_cart_pos(S5, t)[0]
rc_S6_Rr = moon_sat_cart_pos(S6, t)[0]

## Transormation of cartesian coordinates of satellites to the R frame from Rr or Rg
def transf(sat, rc):
    i = sat[2]
    TR_Rrg = np.array([[np.cos(i), 0, -np.sin(i)],
                      [    0    , 1,     0     ],
                      [np.sin(i), 0,  np.cos(i)]])
    rc_R = np.zeros((len(t), 3))
    for j in range(len(t)):
        rc_R[j, :] = TR_Rrg.dot(rc[j, :])
    return rc_R

rc_S3_R = transf(S3, rc_S3_Rr)
rc_S4_R = transf(S4, rc_S4_Rr)
rc_S5_R = transf(S5, rc_S5_Rr)
rc_S6_R = transf(S6, rc_S6_Rr)


### Distance requirement

def Dist_req(sat):
    D = rc_M - sat
    for i in range(len(t)):
        D[i, :] = np.sqrt(D[i, 0]**2 + D[i, 1]**2 + D[i, 2]**2)
    D = np.round(D[:, 0]*10**(-3), 3)
    a = []
    for j in range(len(D)):
        if D[j] < R_SOI*10**(-3):
            #print("Too close to the Moon at day :", round(t[j]/(3600*24), 1), "and angle :", round(moon_sat_cart_pos(M, t)[1][j]*180/np.pi, 1))
            a.append(j)
    return D, a


aS3 = Dist_req(rc_S3_R)[1]
DS3 = Dist_req(rc_S3_R)[0]
# print(np.min(DS3), np.argmin(DS3), np.max(DS3), np.argmax(DS3), t[np.argmax(DS3)]/(3600*24), round(moon_sat_cart_pos(S3, t)[1][np.argmax(DS3)]*180/np.pi))

aS4 = Dist_req(rc_S4_R)[1]
DS4 = Dist_req(rc_S4_R)[0]
# print(np.min(DS4), np.argmin(DS4), np.max(DS4), np.argmax(DS4))

aS5 = Dist_req(rc_S5_R)[1]
DS5 = Dist_req(rc_S5_R)[0]
# print(np.min(DS5), np.argmin(DS5), np.max(DS5), np.argmax(DS5))

aS6 = Dist_req(rc_S6_R)[1]
DS6 = Dist_req(rc_S6_R)[0]
# print(np.min(DS6), np.argmin(DS6), np.max(DS6), np.argmax(DS6))

aS = [0]
aS.extend(aS3)
aS.extend(aS4)
aS.extend(aS5)
aS.extend(aS6)
# print(len(aS)-1)


### Transformation from R to M (Niko's Moon coordinates)

def transf_RM(sat, M):
    T_MR = np.array([[      0       , -1,        0       ],
                     [np.cos(M_incl),  0, -np.sin(M_incl)],
                     [np.sin(M_incl),  0,  np.cos(M_incl)]])
    rc_Mn1 = T_MR.dot(sat)
    rc_M_Mn = T_MR.dot(M)
    rc_Mn = rc_Mn1 - rc_M_Mn
    return rc_Mn1, rc_Mn

# S3 and S5 max dist coordinates
rc_S3_Mn_max = transf_RM(rc_S3_R[np.argmax(DS3), :], rc_M[np.argmax(DS3), :])[1]
rc_L1_Mn_S3max = transf_RM(rc_L1_R[np.argmax(DS3), :], rc_M[np.argmax(DS3), :])[1]
rc_L2_Mn_S3max = transf_RM(rc_L2_R[np.argmax(DS3), :], rc_M[np.argmax(DS3), :])[1]
rc_S4_Mn_S3max = transf_RM(rc_S4_R[np.argmax(DS3), :], rc_M[np.argmax(DS3), :])[1]
rc_S5_Mn_S3max = transf_RM(rc_S5_R[np.argmax(DS3), :], rc_M[np.argmax(DS3), :])[1]
rc_S6_Mn_S3max = transf_RM(rc_S6_R[np.argmax(DS3), :], rc_M[np.argmax(DS3), :])[1]

# S4 and S6 max dist coordinates
rc_S4_Mn_max = transf_RM(rc_S4_R[np.argmax(DS4), :], rc_M[np.argmax(DS4), :])[1]
rc_L1_Mn_S4max = transf_RM(rc_L1_R[np.argmax(DS4), :], rc_M[np.argmax(DS4), :])[1]
rc_L2_Mn_S4max = transf_RM(rc_L2_R[np.argmax(DS4), :], rc_M[np.argmax(DS4), :])[1]
rc_S3_Mn_S4max = transf_RM(rc_S3_R[np.argmax(DS4), :], rc_M[np.argmax(DS4), :])[1]
rc_S5_Mn_S4max = transf_RM(rc_S5_R[np.argmax(DS4), :], rc_M[np.argmax(DS4), :])[1]
rc_S6_Mn_S4max = transf_RM(rc_S6_R[np.argmax(DS4), :], rc_M[np.argmax(DS4), :])[1]

# S3 and S5 min dist coordinates
rc_S3_Mn_min = transf_RM(rc_S3_R[np.argmin(DS3), :], rc_M[np.argmin(DS3), :])[1]
rc_L1_Mn_S3min = transf_RM(rc_L1_R[np.argmin(DS3), :], rc_M[np.argmin(DS3), :])[1]
rc_L2_Mn_S3min = transf_RM(rc_L2_R[np.argmin(DS3), :], rc_M[np.argmin(DS3), :])[1]
rc_S4_Mn_S3min = transf_RM(rc_S4_R[np.argmin(DS3), :], rc_M[np.argmin(DS3), :])[1]
rc_S5_Mn_S3min = transf_RM(rc_S5_R[np.argmin(DS3), :], rc_M[np.argmin(DS3), :])[1]
rc_S6_Mn_S3min = transf_RM(rc_S6_R[np.argmin(DS3), :], rc_M[np.argmin(DS3), :])[1]

# S4 and S6 min dist coordinates
rc_S4_Mn_min = transf_RM(rc_S4_R[np.argmin(DS4), :], rc_M[np.argmin(DS4), :])[1]
rc_L1_Mn_S4min = transf_RM(rc_L1_R[np.argmin(DS4), :], rc_M[np.argmin(DS4), :])[1]
rc_L2_Mn_S4min = transf_RM(rc_L2_R[np.argmin(DS4), :], rc_M[np.argmin(DS4), :])[1]
rc_S3_Mn_S4min = transf_RM(rc_S3_R[np.argmin(DS4), :], rc_M[np.argmin(DS4), :])[1]
rc_S5_Mn_S4min = transf_RM(rc_S5_R[np.argmin(DS4), :], rc_M[np.argmin(DS4), :])[1]
rc_S6_Mn_S4min = transf_RM(rc_S6_R[np.argmin(DS4), :], rc_M[np.argmin(DS4), :])[1]

Dist_S3_min = np.sqrt(rc_S3_Mn_min[0] ** 2 + rc_S3_Mn_min[1] ** 2 + rc_S3_Mn_min[2] ** 2)
Dist_S3_max = np.sqrt(rc_S3_Mn_max[0] ** 2 + rc_S3_Mn_max[1] ** 2 + rc_S3_Mn_max[2] ** 2)
Dist_S4_min = np.sqrt(rc_S4_Mn_min[0] ** 2 + rc_S4_Mn_min[1] ** 2 + rc_S4_Mn_min[2] ** 2)
Dist_S4_max = np.sqrt(rc_S4_Mn_max[0] ** 2 + rc_S4_Mn_max[1] ** 2 + rc_S4_Mn_max[2] ** 2)
Dist_S5_min = np.sqrt(rc_S5_Mn_S3min[0] ** 2 + rc_S5_Mn_S3min[1] ** 2 + rc_S5_Mn_S3min[2] ** 2)
Dist_S5_max = np.sqrt(rc_S5_Mn_S3max[0] ** 2 + rc_S5_Mn_S3max[1] ** 2 + rc_S5_Mn_S3max[2] ** 2)
Dist_S6_min = np.sqrt(rc_S6_Mn_S4min[0] ** 2 + rc_S6_Mn_S4min[1] ** 2 + rc_S6_Mn_S4min[2] ** 2)
Dist_S6_max = np.sqrt(rc_S6_Mn_S4max[0] ** 2 + rc_S6_Mn_S4max[1] ** 2 + rc_S6_Mn_S4max[2] ** 2)

# print(rc_S3_Mn_min)
# print(Dist_S3_min)
# print(rc_S3_Mn_max)
# print(Dist_S3_max)
# print(rc_S4_Mn_min)
# print(Dist_S4_min)
# print(rc_S4_Mn_max)
# print(Dist_S4_max)
# print(rc_S5_Mn_min)
# print(Dist_S5_min)
# print(rc_S5_Mn_max)
# print(Dist_S5_max)
# print(rc_S6_Mn_min)
# print(Dist_S6_min)
# print(rc_S6_Mn_max)
# print(Dist_S6_max)

### Plots

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(rc_M[np.argmin(DS3), 0], rc_M[np.argmin(DS3), 1], rc_M[np.argmin(DS3), 2], c='r', marker='^')
# ax.scatter(rc_L1_R[np.argmin(DS3), 0], rc_L1_R[np.argmin(DS3), 1], rc_L1_R[np.argmin(DS3), 2], c='y', marker='o')
# ax.scatter(rc_L2_R[np.argmin(DS3), 0], rc_L2_R[np.argmin(DS3), 1], rc_L2_R[np.argmin(DS3), 2], c='y', marker='o')
# ax.scatter(rc_S3_R[np.argmin(DS3), 0], rc_S3_R[np.argmin(DS3), 1], rc_S3_R[np.argmin(DS3), 2], c='b', marker='o')
# ax.scatter(rc_S4_R[np.argmin(DS3), 0], rc_S4_R[np.argmin(DS3), 1], rc_S4_R[np.argmin(DS3), 2], c='b', marker='o')
# ax.scatter(rc_S5_R[np.argmin(DS3), 0], rc_S5_R[np.argmin(DS3), 1], rc_S5_R[np.argmin(DS3), 2], c='b', marker='o')
# ax.scatter(rc_S6_R[np.argmin(DS3), 0], rc_S6_R[np.argmin(DS3), 1], rc_S6_R[np.argmin(DS3), 2], c='b', marker='o')
#
# ax.scatter(rc_M[np.argmin(DS4), 0], rc_M[np.argmin(DS4), 1], rc_M[np.argmin(DS4), 2], c='r', marker='^')
# ax.scatter(rc_L1_R[np.argmin(DS4), 0], rc_L1_R[np.argmin(DS4), 1], rc_L1_R[np.argmin(DS4), 2], c='y', marker='o')
# ax.scatter(rc_L2_R[np.argmin(DS4), 0], rc_L2_R[np.argmin(DS4), 1], rc_L2_R[np.argmin(DS4), 2], c='y', marker='o')
# ax.scatter(rc_S3_R[np.argmin(DS4), 0], rc_S3_R[np.argmin(DS4), 1], rc_S3_R[np.argmin(DS4), 2], c='y', marker='o')
# ax.scatter(rc_S4_R[np.argmin(DS4), 0], rc_S4_R[np.argmin(DS4), 1], rc_S4_R[np.argmin(DS4), 2], c='y', marker='o')
# ax.scatter(rc_S5_R[np.argmin(DS4), 0], rc_S5_R[np.argmin(DS4), 1], rc_S5_R[np.argmin(DS4), 2], c='y', marker='o')
# ax.scatter(rc_S6_R[np.argmin(DS4), 0], rc_S6_R[np.argmin(DS4), 1], rc_S6_R[np.argmin(DS4), 2], c='y', marker='o')
#
# ax.scatter(rc_M[np.argmax(DS3), 0], rc_M[np.argmax(DS3), 1], rc_M[np.argmax(DS3), 2], c='r', marker='^')
# ax.scatter(rc_L1_R[np.argmax(DS3), 0], rc_L1_R[np.argmax(DS3), 1], rc_L1_R[np.argmax(DS3), 2], c='y', marker='o')
# ax.scatter(rc_L2_R[np.argmax(DS3), 0], rc_L2_R[np.argmax(DS3), 1], rc_L2_R[np.argmax(DS3), 2], c='y', marker='o')
# ax.scatter(rc_S3_R[np.argmax(DS3), 0], rc_S3_R[np.argmax(DS3), 1], rc_S3_R[np.argmax(DS3), 2], c='b', marker='o')
# ax.scatter(rc_S4_R[np.argmax(DS3), 0], rc_S4_R[np.argmax(DS3), 1], rc_S4_R[np.argmax(DS3), 2], c='b', marker='o')
# ax.scatter(rc_S5_R[np.argmax(DS3), 0], rc_S5_R[np.argmax(DS3), 1], rc_S5_R[np.argmax(DS3), 2], c='b', marker='o')
# ax.scatter(rc_S6_R[np.argmax(DS3), 0], rc_S6_R[np.argmax(DS3), 1], rc_S6_R[np.argmax(DS3), 2], c='b', marker='o')
#
# ax.scatter(rc_M[np.argmax(DS4), 0], rc_M[np.argmax(DS4), 1], rc_M[np.argmax(DS4), 2], c='r', marker='^')
# ax.scatter(rc_L1_R[np.argmax(DS4), 0], rc_L1_R[np.argmax(DS4), 1], rc_L1_R[np.argmax(DS4), 2], c='y', marker='o')
# ax.scatter(rc_L2_R[np.argmax(DS4), 0], rc_L2_R[np.argmax(DS4), 1], rc_L2_R[np.argmax(DS4), 2], c='y', marker='o')
# ax.scatter(rc_S3_R[np.argmax(DS4), 0], rc_S3_R[np.argmax(DS4), 1], rc_S3_R[np.argmax(DS4), 2], c='y', marker='o')
# ax.scatter(rc_S4_R[np.argmax(DS4), 0], rc_S4_R[np.argmax(DS4), 1], rc_S4_R[np.argmax(DS4), 2], c='y', marker='o')
# ax.scatter(rc_S5_R[np.argmax(DS4), 0], rc_S5_R[np.argmax(DS4), 1], rc_S5_R[np.argmax(DS4), 2], c='y', marker='o')
# ax.scatter(rc_S6_R[np.argmax(DS4), 0], rc_S6_R[np.argmax(DS4), 1], rc_S6_R[np.argmax(DS4), 2], c='y', marker='o')
#
# for i in range(len(t)):
#    ax.scatter(rc_M[i, 0], rc_M[i, 1], rc_M[i, 2], c='b', marker='o')
#    ax.scatter(rc_L1_R[i, 0], rc_L1_R[i, 1], rc_L1_R[i, 2], c='g', marker='^')
#    ax.scatter(rc_L2_R[i, 0], rc_L2_R[i, 1], rc_L2_R[i, 2], c='g', marker='^')
# for i in range(len(t)):
#    if i in aS :
#        continue
#    else :
#        ax.scatter(rc_S3_R[i, 0], rc_S3_R[i, 1], rc_S3_R[i, 2], c='g', marker='^')
#        ax.scatter(rc_S4_R[i, 0], rc_S4_R[i, 1], rc_S4_R[i, 2], c='g', marker='^')
#        ax.scatter(rc_S5_R[i, 0], rc_S5_R[i, 1], rc_S5_R[i, 2], c='g', marker='^')
#        ax.scatter(rc_S6_R[i, 0], rc_S6_R[i, 1], rc_S6_R[i, 2], c='g', marker='^')
# # for j in range(len(aS3)):
# #    ax.scatter(rc_S3_R[aS3[j], 0], rc_S3_R[aS3[j], 1], rc_S3_R[aS3[j], 2], c='r', marker='^')
# #    ax.scatter(rc_S4_R[aS4[j], 0], rc_S4_R[aS4[j], 1], rc_S4_R[aS4[j], 2], c='r', marker='^')
# #    ax.scatter(rc_S5_R[aS5[j], 0], rc_S5_R[aS5[j], 1], rc_S5_R[aS5[j], 2], c='r', marker='^')
# #    ax.scatter(rc_S6_R[aS6[j], 0], rc_S6_R[aS6[j], 1], rc_S6_R[aS6[j], 2], c='r', marker='^')
# # for k in range(5):
# #    ax.scatter(rc_S3_R[k, 0], rc_S3_R[k, 1], rc_S3_R[k, 2], c='y', marker='^')
# #    ax.scatter(rc_S4_R[k, 0], rc_S4_R[k, 1], rc_S4_R[k, 2], c='y', marker='^')
# #    ax.scatter(rc_S5_R[k, 0], rc_S5_R[k, 1], rc_S5_R[k, 2], c='y', marker='^')
# #    ax.scatter(rc_S6_R[k, 0], rc_S6_R[k, 1], rc_S6_R[k, 2], c='y', marker='^')
# plt.show()
