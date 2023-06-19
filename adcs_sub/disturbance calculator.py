# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:42:11 2023

@author: MD
"""
#%%
import numpy as np

def As_and_Csp_Rect(l, w, h):
    dim = [l, w, h]
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

def As_and_Csp_Cyl(R, h):
    A_s = np.pi*R**2 + h*2*R
    x_sp = 0.5*h/(1+2*h/(np.pi*R))
    y_sp = 0
    return np.array([A_s]), np.array([x_sp]), np.array([y_sp])

#Cg is origin in this system
def SRP(q, U):
    A_s = U[0]
    x_sp = U[1]
    y_sp = U[2]
    dist_SM = 0.98263  #[AU]
    F_s = 1367/ dist_SM**2
    c = 3*10**8
    T = np.zeros((len(A_s), 179))
    for inc in range(1, 180):
        Fx = F_s/c*A_s*(1+q)*np.cos(inc*np.pi/180)
        Fy = F_s/c*A_s*(1+q)*np.sin(inc*np.pi/180)
        T[: , inc-1] = Fy*(np.abs(x_sp)) - Fx*(np.abs(y_sp))
    # Tm = np.max(np.abs(T))
    inc = np.argmax(np.abs(T))
    Tmax = T[int(inc/179), inc%179]
    return inc,  Tmax

body = {"Rect":[As_and_Csp_Rect(0.9, 0.9, 0.9)[0], As_and_Csp_Rect(0.9, 0.9, 0.9)[1], 
                  As_and_Csp_Rect(0.9, 0.9, 0.9)[2]], 
          "Cyl": [As_and_Csp_Cyl(1.5, 3)[0], As_and_Csp_Cyl(1.5, 3)[1], 
                  As_and_Csp_Cyl(1.5, 3)[2]]}

# print("This is the max", SRP(1, body["Cyl"]))
TD = SRP(1, body["Rect"])[1]
print(TD)

#%%

a = 10000 #[km]
mu = 4902.801

P = 2*np.pi*np.sqrt(a**3 / mu)

Orbs = 12*365*24*60*60/P
print(Orbs)
def maxDesaturationRect(h, t, w, l, n, I_sp):
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
    return h/(n*t*L), h/(n*L*I_sp*9.81)

def maxDesaturationCyl(h, t, R, n):
    return h/(n*t*R)

#Assume 1 second for PVT services, could be slower if needed


def HowMuchIsYourMom(P, TD, theta_a):
    #Momentum storage needed to overcome periodic disturbance torque
    hP = np.sqrt(2)/2*TD*P/4
    
    # momentum h required for a maximum allowed spacecraft rotation (per orbit) 
    ha = TD/(theta_a*np.pi/180)*P/4
    
    return hP, ha

def prop(FT, t, I_sp):
    return FT*t/(I_sp*9.81)

def devPs(TD, I_min):
    return 0.5*TD/I_min

Dumps = HowMuchIsYourMom(P, TD, 0.45839557342765236)[1]
FT, Mp = maxDesaturationRect(Dumps, 1, 0.9, 0.9, 1, 266)
print(Dumps, "here")
print(Mp*Orbs)
print(Mp)
