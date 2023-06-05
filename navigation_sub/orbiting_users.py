"""File to locate orbiting users.
Maintained by Serban Nedelcu"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

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
y_line = np.concatenate((y2_line, y1_line))
z_line = np.concatenate((z2_line, z1_line))

### Define circles

def Circles(Radius):
    y_1 = np.arange(-Radius, Radius + dy, dy)
    y_2 = np.sort(np.append(y_1, np.array([y1, ysol1, ysol2])))
    y = np.concatenate((y_2, y_2[::-1]))
    zp = np.sqrt(Radius**2 - y_2**2)
    zm = -zp
    z = np.concatenate((zp, zm[::-1]))
    return y, z

y_Moon, z_Moon= Circles(R_M)
y_Orbit, z_Orbit = Circles(R_O)

### Area covered

common_range = np.logical_and(y_Orbit >= ysol1, y_Orbit <= ysol2)
ysol2_indices = np.where(y_Orbit == ysol2)[0]
if len(ysol2_indices) > 0:
    common_range[ysol2_indices[0] + 1:] = True

y_common = y_Orbit[common_range]
z_Orbit_common = z_Orbit[common_range]
z_line_common = np.interp(y_common, y_line, z_line)

right_indices = np.where(y_common >= 0)[0]
y_right = y_common[right_indices]
z_Orbit_right = z_Orbit_common[right_indices]
z_line_right = np.interp(y_right, y_line, z_line)

conditionR = z_Orbit_right > z_line_right
areaR = np.trapz(np.abs(z_Orbit_right[conditionR] - z_line_right[conditionR]), x=y_right[conditionR])

Area = np.pi*(R_O**2 - R_M**2)
print(areaR*2/Area)

### Left side
common_rangeL = np.where(np.logical_and(y_Orbit<=-ysol2, z_Orbit<=0)!=0)[0]
print(len(common_rangeL))
print(len(y_Orbit))
y_commonL = y_Orbit[common_rangeL]
print(len(y_commonL))
z_Orbit_commonL = z_Orbit[common_rangeL]
z_line_commonL = np.interp(y_commonL, y_line, z_line)

left_indices = np.where(y_commonL <= 0)[0]
y_left = y_commonL[left_indices]
z_Orbit_left = z_Orbit_commonL[left_indices]
z_line_left = np.interp(y_left, y_line, z_line)

conditionL = z_Orbit_left > z_line_left
areaL = np.trapz(np.abs(z_Orbit_left[conditionL] - z_line_left[conditionL]), x=y_left[conditionL])

print(areaL*2/Area)

common_rangeL1 = np.where(np.logical_and(y_Orbit<=-ysol1, z_Orbit>=0)!=0)[0]
print(len(common_rangeL1))
print(len(y_Orbit))
y_commonL1 = y_Orbit[common_rangeL1]
print(len(y_commonL1))
z_Orbit_commonL1 = z_Orbit[common_rangeL1]
z_line_commonL1 = np.interp(y_commonL1, y_line, z_line)

left_indices1 = np.where(y_commonL1 <= 0)[0]
y_left1 = y_commonL1[left_indices1]
z_Orbit_left1 = z_Orbit_commonL1[left_indices1]
z_line_left1 = np.interp(y_left1, y_line, z_line)

conditionL1 = z_Orbit_left1 > z_line_left1
areaL1 = np.trapz(np.abs(z_Orbit_left1[conditionL1] - z_line_left1[conditionL1]), x=y_left1[conditionL1])

print((areaL1+areaL)*2/Area)

# common_rangeL1 = np.logical_and(y_Orbit <= -ysol2, y_Orbit >= -R_O)
#
# print(len(y_Orbit))
# print(np.count_nonzero(common_rangeL1))
# y_commonL1 = y_Orbit[common_rangeL1]
# z_Orbit_commonL1 = z_Orbit[common_rangeL1]
# z_line_commonL1 = np.interp(y_commonL1, y_line, z_line)
#
# left_indices1 = np.where(y_commonL1 <= 0)[0]
# y_left1 = y_commonL1[left_indices1]
# z_Orbit_left1 = z_Orbit_commonL1[left_indices1]
# z_line_left1 = np.interp(y_left1, y_line, z_line)
#
# conditionL1 = z_Orbit_left1 > z_line_left1
# areaL1 = np.trapz(np.abs(z_Orbit_left1[conditionL1] - z_line_left1[conditionL1]), x=y_left1[conditionL1])
#
# common_rangeL2 = np.logical_and(y_Orbit <= -ysol1, y_Orbit >= -R_O)
#
# print(len(y_Orbit))
# print(np.count_nonzero(common_rangeL2))
# y_commonL2 = y_Orbit[common_rangeL2]
# z_Orbit_commonL2 = z_Orbit[common_rangeL2]
# z_line_commonL2 = np.interp(y_commonL2, y_line, z_line)
#
# left_indices2 = np.where(y_commonL2 <= 0)[0]
# y_left2 = y_commonL2[left_indices2]
# z_Orbit_left2 = z_Orbit_commonL2[left_indices2]
# z_line_left2 = np.interp(y_left2, y_line, z_line)
#
# conditionL2 = z_Orbit_left2 > z_line_left2
# areaL2 = np.trapz(np.abs(z_Orbit_left2[conditionL2] - z_line_left2[conditionL2]), x=y_left2[conditionL2])
#
# areaL = areaL1 + areaL2
# print(areaL*2/Area)

### Plotting
fig, ax = plt.subplots()
plt.plot(y_line, z_line, "-b", label="sat view")
plt.plot(y_Moon, z_Moon, "-g", label="Moon")
plt.plot(y_Orbit, z_Orbit, "-.r", label="Orbit")
ax.fill_between(y_right, z_line_right, z_Orbit_right, where=conditionR, color='green', alpha=0.5)
ax.fill_between(y_left, z_line_left, z_Orbit_left, where=conditionL, color='green', alpha=0.5)
ax.fill_between(y_left1, z_line_left1, z_Orbit_left1, where=conditionL1, color='blue', alpha=0.5)
plt.legend()
plt.show()
