"""File performing Sensitivity Analysis of structures_sub package.
It is performed on the stage_1structure and detailed_structure scripts.
By N. Ricker"""

# External modules
import numpy as np
import matplotlib.pyplot as plt

# Internal modules
from stage1_structure import Cylinder, RectangularPrism, Structure
from detailed_structure import PropTankSphere, PropTankCylinder, SatelliteStruc


# Propellant Tank Sensitivity Analysis
def sens_thickness_tank1():
    t = []
    for m in range(1, 100):
        t.append(PropTankSphere(m, "Ti-6AL-4V").thickness * 1000)
    plt.plot(t)
    plt.xlabel("Mass [kg]")
    plt.ylabel("Thickness [mm]")
    plt.title("Sensitivity Analysis of Sphere Propellant Tank Thickness")
    plt.show()

def sens_thickness_tank2():
    t = []
    for m in range(1, 100):
        t.append(PropTankCylinder(m, "Ti-6AL-4V", r=0.1).l * 1000)
    plt.plot(t)
    plt.xlabel("Mass [kg]")
    plt.ylabel("Length [mm]")
    plt.title("Sensitivity Analysis of Propellant Tank Length")
    plt.show()

def sens_distanceMMOI():
    ix = []
    iy = []
    iz = []
    for r in range(1, 100):
        s = SatelliteStruc()
        s.add_point_element(1, pos=[r/100, 0, 0])
        ix.append(s.mmoi[0])
        iy.append(s.mmoi[1])
        iz.append(s.mmoi[2])
    plt.plot(ix, label="Ixx")
    plt.plot(iy, label="Iyy")
    plt.plot(iz, label="Izz")
    plt.legend()
    plt.xlabel("Distance from CoM [cm]")
    plt.ylabel("MMoI [kgm^2]")
    plt.title("Sensitivity Analysis of Distance from CoM")
    plt.show()


def sens_solarPanelsMMOI():
    ix = []
    iy = []
    iz = []
    for area in range(1, 100):
        s = SatelliteStruc()
        s.add_panels(4*np.sqrt(area/4), np.sqrt(area/4), 10)
        ix.append(s.mmoi[0])
        iy.append(s.mmoi[1])
        iz.append(s.mmoi[2])
    plt.plot(ix, label="Ixx")
    plt.plot(iy, label="Iyy")
    plt.plot(iz, label="Izz")
    plt.legend()
    plt.xlabel("Distance from CoM [cm]")
    plt.ylabel("MMoI [kgm^2]")
    plt.title("Sensitivity Analysis of Solar Panel Area")
    plt.show()


if __name__ == "__main__":
    sens_thickness_tank1()
    sens_thickness_tank2()
    sens_distanceMMOI()
    sens_solarPanelsMMOI()
