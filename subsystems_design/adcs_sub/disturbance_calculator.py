# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:42:11 2023

by M. Dinescu
"""

'''
L = 1m
refl = 0.57
acc = 0.45839557342765236
'''

#%%
import numpy as np
''' Parameters '''

dim = [1, 1, 1] #length, width, height
q = 0.57 #reflectivity


''' Functions'''

def As_and_Csp_Rect(dim):
    A_s =[]
    x_sp = []
    y_sp = []
    for i in range(len(dim)):
        xi = -0.5*dim[(i+1)%len(dim)]*dim[(i+2)%len(dim)]/(dim[(i+1)%len(dim)]+dim[(i+2)%len(dim)])
        x_sp.append(xi)
        yi = -xi
        y_sp.append(yi)
        Ai = dim[(i+1)%len(dim)]*dim[(i+2)%len(dim)]
        A_s.append(Ai)
    return np.array(A_s), np.array(x_sp), np.array(y_sp)

#Cg is origin in this system
def SRP(q, U):
    cg_x = np.array([0,0,0])
    cg_y = np.array([0,0,0])
    A_s = U[0]
    x_sp = U[1]
    y_sp = U[2]
    dist_SM = 0.98263  #[AU]
    F_s = 1367.6/ dist_SM**2
    c = 3*10**8
    T = np.zeros((len(A_s), 179))
    for inc in range(1, 180):
        Fx = F_s/c*A_s*(1+q)*np.cos(inc*np.pi/180)
        Fy = F_s/c*A_s*(1+q)*np.sin(inc*np.pi/180)
        T[: , inc-1] = Fy*(np.abs(x_sp- cg_x)) - Fx*(np.abs(y_sp-cg_y))
    # Tm = np.max(np.abs(T))
    inc = np.argmax(np.abs(T))
    Tmax = T[int(inc/179), inc%179]
    return inc,  Tmax

body = {"Rect":[As_and_Csp_Rect(dim)[0], As_and_Csp_Rect(dim)[1], 
                  As_and_Csp_Rect(dim)[2]]}

TD = SRP(q, body["Rect"])[1]
print(TD)

#%%
''' Paramters'''
#From other subsystems
a = 10000 #[km]
mu = 4902.801
I_sp = 266

I_z = 422.06914971714093
I_y = 130.3676592918074
I_x = 418.99796298218376

w = dim[1]
l = dim[0]

#Gimbal Characterisitcs
Gd = 0.338
Gh = 0.338
Gmass = 5.8

#Rotations
theta_acq = 180
theta_sol = 360
theta_G = 360

#Time durations
t_pulse = 5
t_OM = 365*24*60*60
tg = 1.5*60*60 
t_x = 400
t_y = 120
t_z = 400

#Pointing accuracy
PAcc = 0.45839557342765236

''' Functions'''


def Per(a, mu):
    Period = 2*np.pi*np.sqrt(a**3 / mu)
    NOrb = 12*365*24*60*60/Period
    return Period, NOrb

def Gimbal(Gd, Gh, Gmass, I_z, P, tg):
    I_g = 1/4*(Gmass*(Gd/2)**2)+1/12*(5.8*(Gh)**2)
    
    wr = I_g/I_z
    
    
    tgr = P/tg
    return I_g, wr, tgr

def acq(theta, I, t):
    T = 4*theta*np.pi/180*I/t**2
    h = T*t/2
    return T, h


# print(OM)
# print(OM[1]*12/8)
# print(R1, R2, R3)
def maxDesaturationRect(h, t, w, l, I_sp):
    ''' h is the momentum of the wheel
        t is time for momentum dunmping
        w is width of spacecraft
        l is length of spacecraft
        n is number of thrusters in 
          same orientation '''
    if w< l:
        L= l
    elif l<w:
        L=w
    else:
        L=w
    return h/(t*L), h/(L*I_sp*9.81)

#Assume 1 second for PVT services, could be slower if needed


def HowMuchIsYourMom(P, TD, theta_a):    
    # momentum h required for a maximum allowed spacecraft rotation (per orbit) 
    ha = TD/(theta_a*np.pi/180)*P/4
    return ha


P, Orbs = Per(a, mu)

I_g, wr, tgr = Gimbal(Gd, Gmass, I_z, P, tg)


R1 = acq(theta_acq, I_x, t_x)
R2 = acq(theta_acq, I_z, t_z)
R3 = acq(theta_acq, I_y, t_y)
OM = acq(theta_sol, I_z, t_OM)
GM = acq(2*theta_G*wr, I_z, tg)
print(GM)
print(GM[1]*tgr, "Gimbal momentum")

Dumps = HowMuchIsYourMom(P, TD, PAcc)
# Dumps = HowMuchIsYourMom(P, TD, 0.003437823560633145)[1]
FT, Mp = maxDesaturationRect(Dumps, t_pulse, w, l, I_sp)
print(FT, "thrust")
print(Dumps, "here")
print(Mp*Orbs)
print(Mp)



# I_g = 1/4*(5.8*(0.338/2)**2)+1/12*(5.8*0.338**2)

# wr = I_g/422.06914971714093

# P = 2*np.pi*np.sqrt(a**3 / mu)
# tgr = P/tg
# # print(1/wr*360)

# # print(I_g*2*np.pi/P)

# Orbs = 12*365*24*60*60/P
# # print(Orbs)

# def acq(theta, I, t):
#     T = 4*theta*np.pi/180*I/t**2
#     h = T*t/2
#     return T, h

# R1 = acq(180, 418.99796298218376, 400)
# R2 = acq(180, 422.06914971714093, 400)
# R3 = acq(180, 130.3676592918074, 120)
# OM = acq(360, 422.06914971714093, 365*24*60*60)
# GM = acq(2*360*wr, 422.06914971714093, tg)
# print(GM)
# print(GM[1]*tgr, "Gimbal torq")
# # print(OM)
# # print(OM[1]*12/8)
# # print(R1, R2, R3)
# def maxDesaturationRect(h, t, w, l, I_sp):
#     ''' h is the momentum of the wheel
#         t is time for momentum dunmping
#         w is width of spacecraft
#         l is length of spacecraft
#         n is number of thrusters in 
#           same orientation '''
#     if w< l:
#         L= l
#     elif l<w:
#         L=w
#     else:
#         L=w
#     return h/(t*L), h/(L*I_sp*9.81)

# #Assume 1 second for PVT services, could be slower if needed


# def HowMuchIsYourMom(P, TD, theta_a):    
#     # momentum h required for a maximum allowed spacecraft rotation (per orbit) 
#     ha = TD/(theta_a*np.pi/180)*P/4
#     return ha

# Dumps = HowMuchIsYourMom(P, TD, 0.45839557342765236)
# # Dumps = HowMuchIsYourMom(P, TD, 0.003437823560633145)[1]
# FT, Mp = maxDesaturationRect(Dumps, 5, 1, 1, 266)
# print(FT, "thrust")
# print(Dumps, "here")
# print(Mp*Orbs)
# print(Mp)