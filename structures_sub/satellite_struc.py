"""Script to create a satellite structure.
By I. Maes, N. Ricker"""

# Global imports
import numpy as np

# Local imports


# Constants
from structures_constants import *


# --------------------------------------------------------------------------- #
class PropellantTank:
    """Class to create a spherical tank for the propellant."""

    def __init__(self, volume, prop_mass, material, pressure=10e6):
        """Initializes the tank.
        :param volume: Volume of the tank. [m^3]
        :param prop_mass: Mass of the propellant. [kg]
        :param material: Material of the tank. [string]
        :param pressure: Pressure of the tank. [Pa]

        """
        self.name = "Propellant Tank"

        self.volume = volume
        self.pressure = pressure
        self.material = material
        self.mass_prop = prop_mass

        self.r = (self.volume*3/(4*np.pi))**(1/3)
        self.thickness = self.pressure*self.r/(2*self.sigma_y/1.1)
        self.mass_struc = self.surface_area() * self.thickness * self.rho

        self.mass = self.mass_struc + self.mass_prop
        self.mmoi = 2 / 3 * self.mass_struc * self.r ** 2 + 2 / 5 * self.mass_prop * self.r ** 2

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self.E = material_properties[material]['E']
        self.rho = material_properties[material]['density']
        self.sigma_y = material_properties[material]['yield_strength']
        self.sigma_u = material_properties[material]['ultimate_strength']
        self.thermal_coeff = material_properties[material]['thermal_coefficient']
        self._material = material

    def surface_area(self):
        """Computes the surface area of the tank.
        :return: Surface area of the tank. [m^2]"""
        return 4*np.pi*self.r**2

    def __repr__(self):
        return f"PropellantTank({self.volume}, {self.material})"


class SatelliteStruc:
    """Class to create a satellite structure. It studies mass, dimensions, volume, stress, and vibrations"""

    def __init__(self, name="Sat_Test"):
        """Initialize the satellite structure.
        :param name: Name of the satellite. [string]
        """
        self.name = name
        self.mass = 0
        self.dim = [0, 0, 0] # [Length, Width, Height]
        self.t = None
        self.support_r1 = None
        self.mmoi = [0, 0, 0] # [Ixx, Iyy, Izz]

        self.material = "Aluminium_7075-T73"

        self.stress = [0, 0, 0, 0] # [tensile_launch, tensile_thermal, compressive_launch, compressive_thermal]
        self.vibration = [0, 0, 0] # [axial, lateral_x, lateral_y]

        self.mass_breakdown = dict()
        self.mmoi_breakdown = dict()

    def update_mass_breakdown(self, key, value):
        if key in self.mass_breakdown:
            self.mass_breakdown[key] += value
        else:
            self.mass_breakdown[key] = value

    def update_mmoi_breakdown(self, key, value):
        if key in self.mmoi_breakdown:
            value = np.array(value)
            self.mmoi_breakdown[key] += value
        else:
            self.mmoi_breakdown[key] = np.array(value)

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        if value < 0:
            raise ValueError("Mass cannot be negative")
        self._mass = value

    @property
    def volume(self):
        return self.dim[0]*self.dim[1]*self.dim[2]

    @property
    def surface_area(self):
        return 2 * (self.dim[0] * self.dim[1] + self.dim[0] * self.dim[2] +
                    self.dim[1] * self.dim[2])

    @property
    def cross_section(self):
        return 2*(self.dim[0] + self.dim[1])*self.t

    @property
    def area_moment_of_inertia(self):
        ix = 2 * (self.dim[1] ** 3 * self.t + self.t ** 3 * self.dim[0]) / 12 + 2 * self.dim[0] * self.t * (self.dim[1] / 2) ** 2
        iy = 2 * (self.dim[0] ** 3 * self.t + self.t ** 3 * self.dim[1]) / 12 + 2 * self.dim[1] * self.t * (self.dim[0] / 2) ** 2
        return [ix, iy]

    @property
    def buckling_limit(self):
        if self.support_r1 is None:
            b = 2*4*(np.pi**2*self.E/(12*(1-0.33**2))*(self.t/self.dim[0])**2 * self.dim[0]*self.t + np.pi**2*self.E/(12*(1-0.33**2))*(self.t/self.dim[1])**2 * self.dim[0]*self.t)/self.cross_section
        else:
            y = 1 - 0.901 * (1 - np.e ** - (1/16 * np.sqrt(self.support_r2/(self.support_r2-self.support_r1))))
            print(self.dim[2]/self.support_r2)
            b = ((2*4*(np.pi**2*self.E/(12*(1-0.33**2))*(self.t/self.dim[0])**2 * self.dim[0]*self.t + np.pi**2*self.E/(12*(1-0.33**2))*(self.t/self.dim[1])**2 * self.dim[0]*self.t)) +0.6 * y * self.support_E * (self.support_r2-self.support_r1)/self.support_r2*self.support_area)/(self.cross_section + self.support_area)
#self.support_E*((9*(self.support_r2-self.support_r1)/self.support_r2)**1.6 + 0.16*((self.support_r2-self.support_r1)/self.dim[2])**1.3)*self.support_area
#0.6 * self.support_E * (self.support_r2-self.support_r1)/self.support_r2*self.support_area
#(self.support_E *self.support_area* 4 * np.pi**2 * self.support_I / (self.dim[2]))
        return b

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self.E = material_properties[material]['E']
        self.rho = material_properties[material]['density']
        self.sigma_y = material_properties[material]['yield_strength']
        self.sigma_u = material_properties[material]['ultimate_strength']
        self.thermal_coeff = material_properties[material]['thermal_coefficient']
        self._material = material

    def add_point_element(self, mass, name='misc', pos=[0, 0, 0]):
        """Adds a point element to the satellite structure.
        :param mass: Mass of the point element. [kg]
        :param name: Name of the point element. [string]
        :param pos: Position of the point element (x,y,z). [m]
        """
        self.mass += mass
        Ixx = mass * (pos[1] ** 2 + pos[2] ** 2)
        Iyy = mass * (pos[0] ** 2 + pos[2] ** 2)
        Izz = mass * (pos[0] ** 2 + pos[1] ** 2)
        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz
        self.update_mass_breakdown(name, mass)
        self.update_mmoi_breakdown(name, [Ixx, Iyy, Izz])

    def add_structure_sub(self, length, width, height, t):
        """Adds the structure shape to the satellite.
        :param length: Length of the structure. [m]
        :param width: Width of the structure. [m]
        :param height: Height of the structure. [m]
        :param t: Thickness of the structure. [m]
        """

        self.dim = [length, width, height]
        self.t = t

        mass_struc = self.rho * t * self.surface_area
        self.mass += mass_struc

        m_panel = mass_struc / 6

        Ixx = 2*(m_panel / 12) * (self.dim[2] ** 2 + self.dim[1] ** 2) + 2*m_panel*(self.dim[1]**2/12 + self.dim[2]**2/4) + 2*m_panel*(self.dim[2]**2/12 + self.dim[1]**2/4)
        Iyy = 2*(m_panel / 12) * (self.dim[0] ** 2 + self.dim[2] ** 2) + 2*m_panel*(self.dim[0]**2/12 + self.dim[2]**2/4) + 2*m_panel*(self.dim[2]**2/12 + self.dim[0]**2/4)
        Izz = 2*(m_panel / 12) * (self.dim[1] ** 2 + self.dim[0] ** 2) + 2*m_panel*(self.dim[0]**2/12 + self.dim[1]**2/4) + 2*m_panel*(self.dim[1]**2/12 + self.dim[0]**2/4)

        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz

        self.update_mass_breakdown('structure', mass_struc)
        self.update_mmoi_breakdown('structure', [Ixx, Iyy, Izz])

    def add_propellant_tank(self, volume, material, pos=[0, 0, 0], pressure=10e6, prop_mass=20):
        """Adds a propellant tank to the satellite.
        :param volume: Volume of the propellant tank. [m^3]
        :param material: Material of the propellant tank. [string]
        :param pos: Position of the propellant tank (x,y,z). [m]
        :param pressure: Pressure of the propellant tank. [Pa]
        :param prop_mass: Mass of the propellant. [kg]
        """
        p = PropellantTank(volume, prop_mass, material, pressure)
        self.mass += p.mass

        Ixx = p.mmoi + p.mass * (pos[1] ** 2 + pos[2] ** 2)
        Iyy = p.mmoi + p.mass * (pos[0] ** 2 + pos[2] ** 2)
        Izz = p.mmoi + p.mass * (pos[0] ** 2 + pos[1] ** 2)
        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz

        self.update_mass_breakdown('propellant_tank', p.mass)
        self.update_mmoi_breakdown('propellant_tank', [Ixx, Iyy, Izz])

    def add_panels(self, a, b, mass, h=0, deployed=True):
        """Adds panels to the structure.
        :param a: length of the panel. [m]
        :param b: width of the panel. [m]
        :param mass: Total mass of the panels to be added. [kg]
        :param h: Height from centroid. [m]
        :param deployed: Whether the panels are deployed or not. [bool]
        """

        self.mass += mass
        xz_dist = self.dim[0] / 2 + a / 2

        if deployed:
            Ixx = mass * (b ** 2 / 12 + h ** 2)
            Iyy = mass * ((b ** 2 + a ** 2) / 12 + xz_dist ** 2 + h**2) # Worst case scenario, where solar panels rotate
            Izz = mass * ((b ** 2 + a ** 2) / 12 + xz_dist ** 2)
        else:
            h += a/2
            Ixx = mass / 12 * (a ** 2 + b ** 2) + mass * h ** 2
            Iyy = mass/12 * a**2 + mass*(xz_dist**2+h**2)
            Izz = mass / 12 * b ** 2 + mass * xz_dist ** 2
        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz
        self.update_mass_breakdown('panels', mass)
        self.update_mmoi_breakdown('panels', [Ixx, Iyy, Izz])

    def add_support(self, r1, r2, material):
        """Adds a support to the satellite.
        :param r: Radius of the support. [m]
        :param material: Material of the support. [string]
        """
        self.support_r1 = r1
        self.support_r2 = r2
        self.support_E = material_properties[material]["E"]
        self.support_rho = material_properties[material]["density"]
        self.support_I = np.pi*r2**4 / 4 - np.pi*r1**4 / 4
        self.support_area = np.pi*self.support_r2**2 - np.pi*self.support_r1**2

        mass = self.dim[2]*self.support_rho*np.pi*r2**2 - self.dim[2]*self.support_rho*np.pi*r1**2
        self.mass += mass

        Ixx = mass * (3*r2**2 + self.dim[2]**2) / 12 - mass * (3*r1**2 + self.dim[2]**2) / 12
        Iyy = mass * (3*r2**2 + self.dim[2]**2) / 12 - mass * (3*r1**2 + self.dim[2]**2) / 12
        Izz = mass * r2**2 / 2 - mass * r1**2 / 2

        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz

        self.mass_breakdown['support'] = mass
        self.mmoi_breakdown['support'] = [Ixx, Iyy, Izz]

    def calculate_stresses(self):
        """Calculates the stresses on the satellite structure. \n
         Stresses - tensile_launch, tensile_thermal, compressive_launch, compressive_thermal
        """
        s1 = self.mass * g * (g_tensile / self.cross_section + g_lateral * self.dim[2] * (self.dim[1] / 2) /self.area_moment_of_inertia[0])
        s2 = self.mass * g * (g_tensile / self.cross_section + g_lateral * self.dim[2] * (self.dim[0] / 2) /self.area_moment_of_inertia[1])
        self.stress[0] = max(s1, s2)
        self.stress[1] = -self.thermal_coeff * (temperatures[0] - 20) * self.E
        s1 = self.mass * g * (g_axial / self.cross_section + g_lateral * self.dim[2] * (self.dim[1] / 2) /self.area_moment_of_inertia[0])
        s2 = self.mass * g * (g_axial / self.cross_section + g_lateral * self.dim[2] * (self.dim[0] / 2) /self.area_moment_of_inertia[1])
        self.stress[2] = max(s1, s2)
        self.stress[3] = self.thermal_coeff * (temperatures[1] - 20) * self.E

    def calculate_vibrations(self):
        """Calculates the vibrations on the satellite structure."""
        self.vibration[0] = 0.25 * np.sqrt(self.E * self.cross_section / (self.mass * self.dim[2]))
        self.vibration[1] = 0.56 * np.sqrt(self.E * self.area_moment_of_inertia[0] / (self.mass * self.dim[2] ** 3))
        self.vibration[2] = 0.56 * np.sqrt(self.E * self.area_moment_of_inertia[1] / (self.mass * self.dim[2] ** 3))

    def compliance(self):
        """Checks if the satellite structure is compliant with the requirements.
        :return: s_compliance, v_axial_compliance, v_lat_compliance, b_compliance
        """
        s_compliance = max(self.stress) < self.sigma_y/1.1
        v_axial_compliance = self.vibration[0] > axial_freq_falcon*1.5
        v_lat_compliance = min(self.vibration[1:]) > lateral_freq_falcon*1.5
        b_compliance = self.stress[2] < self.buckling_limit/1.1
        return s_compliance, v_axial_compliance, v_lat_compliance, b_compliance


if __name__ == "__main__":
    s = SatelliteStruc()
    s.add_point_element(mass=200, name="payload")
    s.add_structure_sub(length=2.5, width=2, height=2, t=1e-3)
    s.add_propellant_tank(volume=0.07, material="Aluminium_7075-T73", prop_mass=100, pressure=2e6)
    s.add_panels(a=4.5, b=1, mass=100)
    s.add_support(r1=.409, r2=0.41,  material="Ti-6AL-4V")
    s.calculate_stresses()
    s.calculate_vibrations()
    print(f"Mass: {s.mass}")
    print(f"Mass Break: {s.mass_breakdown}")
    print(f"MMOI Break: {s.mmoi_breakdown}")
    print(f"Compliance: {s.compliance()}")
    print(f"Buckling limit: {s.buckling_limit}")
    print(f"Stresses: {s.stress}")
    print(f"Vibrations: {s.vibration}")


