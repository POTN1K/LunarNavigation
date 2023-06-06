# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:42:11 2023

@author: MD
"""
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
def SRP(x_sp, y_sp, q, A_s):
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

# A_s = As_and_Csp_Rect(2, 1.5, 3)[0]

# x_sp = As_and_Csp_Rect(2, 1.5, 3)[1]
# y_sp = As_and_Csp_Rect(2, 1.5, 3)[2]

A_s = As_and_Csp_Cyl(1.5, 3)[0]
print(A_s)

x_sp = As_and_Csp_Cyl(1.5, 3)[1]
y_sp = As_and_Csp_Cyl(1.5, 3)[2]

print("This is the max", SRP(x_sp, y_sp, 1, A_s))

def maxDesaturationRect(h, t, w, l, n):
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
        
    return h/(n*t*L)

def maxDesaturationCyl(h, t, R, n):
    return h/(n*t*R)

#Assume 1 second for PVT services, could be slower if needed
FT = maxDesaturationRect(75*1.04, 1, 1.5, 2, 4)
print(FT)
    