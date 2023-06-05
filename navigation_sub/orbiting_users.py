"""File to locate orbiting users.
Maintained by Serban Nedelcu"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

### Constants
R_M = 1737.4e3  # [m]
A = 8025.9e3  # [m]
R_O = R_M + 200e3  # [m]
dy = 1

### Instersection points w/ Moon (analytical)
z1 = 2*(A - np.sqrt(A**2 - R_M**2))
y1 = np.sqrt(R_M**2 - z1**2)

### Find intersection between sat view and orbit (analytical)
a = 1 + ((z1-A)/y1)**2
b = 2*A*(z1-A)/y1
c = A**2 - R_O**2
d = (b**2) - (4*a*c)
ysol1 = (-b-np.sqrt(d))/(2*a)
ysol2 = (-b+np.sqrt(d))/(2*a)
print(ysol1, ysol2)

### Verif that there is only one intersection point w/ Moon
# length = 100
# a = z1_line[np.where(z1_line==z1)[0][0]-length : np.where(z1_line==z1)[0][0]+length]
# b = z_circle_plus[np.where(np.abs(Circles(R_M)[1]-z1)<10**(-8))[0][0]-length : np.where(np.abs(Circles(R_M)[1]-z1)<10**(-8))[0][0]+length]
# for i in range(len(a)):
#     if np.abs(a[i] - b[i]) < 10**(-5):
#         print(i)
#         print(a[i])
#         print(b[i])

### Define lines
y1_line1 = np.arange(0, R_O+dy, dy)
y1_line = np.sort(np.append(y1_line1, np.array([y1, ysol1, ysol2])))
z1_line = A + (z1-A)/y1 * y1_line
y2_line = np.sort(-y1_line)
z2_line = A - (z1-A)/y1 * y2_line

### Define circles

def Circles(Radius):
    y_1 = np.arange(-Radius, Radius + dy, dy)
    y_2 = np.sort(np.append(y_1, np.array([y1, ysol1, ysol2])))
    y = y = np.concatenate((y_2, y_2[::-1]))
    zp = np.sqrt(Radius**2 - y_2**2)
    zm = -zp
    z = np.concatenate((zp, zm[::-1]))
    return y, z

y_Moon, z_Moon= Circles(R_M)
y_Orbit, z_Orbit = Circles(R_O)



am = np.where(y1_line==ysol2)[0][0]



### Plotting
plt.plot(y1_line, z1_line, "-b", label="sat view")
plt.plot(y2_line, z2_line, "-b")
plt.plot(y_Moon, z_Moon, "-g", label="Moon")
plt.plot(y_Orbit, z_Orbit, "-.r", label="Orbit")
plt.legend()
plt.show()
