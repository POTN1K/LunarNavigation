"""Script to create a satellite structure.
By I. Maes, N. Ricker"""

# Global imports
import numpy as np

# Local imports


# Constants
from structures_constants import *
from tqdm import tqdm


# --------------------------------------------------------------------------- #
class PropTankSphere:
    """Class to create a spherical tank for the propellant."""

    def __init__(self, prop_mass, material, pressure=10e6, volume=None):
        """Initializes the tank.
        :param volume: Volume of the tank. [m^3]
        :param prop_mass: Mass of the propellant. [kg]
        :param material: Material of the tank. [string]
        :param pressure: Pressure of the tank. [Pa]

        """
        self.name = "Propellant Tank"

        self.mass_prop = prop_mass
        self.volume = volume
        self.pressure = pressure
        self.material = material

        self.r = (self.volume * 3 / (4 * np.pi)) ** (1 / 3)
        self.thickness = self.pressure * self.r / (2 * self.sigma_y / 1.1)
        self.mass_struc = self.surface_area() * self.thickness * self.rho

        self.mass = self.mass_struc + self.mass_prop
        self.mmoi = 2 / 3 * self.mass_struc * self.r ** 2 + 2 / 5 * self.mass_prop * self.r ** 2

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, v):
        if v is None:
            self._volume = self.mass_prop / (1.47 * 1000)*1.05 # 5% Margin
        else:
            self._volume = v

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
        return 4 * np.pi * self.r ** 2

    def __repr__(self):
        return f"PropTankSphere({self.volume}, {self.material})"


class PropTankCylinder:
    """Class to create a cylindrical tank for the propellant."""

    def __init__(self, prop_mass, material, pressure=10e6, volume=None, r=0.28):
        self.name = "Propellant Tank"
        self.mass_prop = prop_mass
        self.material = material
        self.pressure = pressure*1.5
        self.volume = volume

        self.r = r

        self.l = (self.volume - 4*np.pi*self.r**3/3) / (np.pi*self.r**2)
        self.thickness = self.pressure * self.r / (self.sigma_y / 1.1)
        self.mass_struc = self.surface_area() * self.thickness * self.rho

        self.mass = self.mass_struc + self.mass_prop
        self.Ixx = 1/2*self.mass*self.r**2
        self.Iyy = 1/12*self.mass*(3*self.r**2+(self.l+2*self.r)**2)
        self.Izz = self.Iyy

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, v):
        if v is None:
            self._volume = self.mass_prop / (1.47 * 1000)*1.1 # 10% Margin
        else:
            self._volume = v

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
        return 2 * np.pi * self.r * self.l + 4 * np.pi * self.r ** 2

class Support:
    """Class to create a support structure."""

    def __init__(self, r1, r2, h, material):
        self.r1 = r1
        self.r2 = r2
        self.h = h
        self.E = material_properties[material]["E"]
        self.rho = material_properties[material]["density"]
        self.I = np.pi * r2 ** 4 / 4 - np.pi * r1 ** 4 / 4
        self.area = np.pi * self.r2 ** 2 - np.pi * self.r1 ** 2
        self.mass = self.rho * self.area * self.h

        self.Ixx = self.mass * (3 * r2 ** 2 + self.h ** 2) / 12 - self.mass * (
                3 * r1 ** 2 + self.h ** 2) / 12
        self.Iyy = self.mass * (3 * r2 ** 2 + self.h ** 2) / 12 - self.mass * (
                3 * r1 ** 2 + self.h ** 2) / 12
        self.Izz = self.mass * r2 ** 2 / 2 - self.mass * r1 ** 2 / 2
        self.volume = self.area * self.h


class SatelliteStruc:
    """Class to create a satellite structure. It studies mass, dimensions, volume, stress, and vibrations"""

    def __init__(self, name="Sat_Test"):
        """Initialize the satellite structure.
        :param name: Name of the satellite. [string]
        """
        self.propellant_tank = None
        self.name = name
        self.mass = 0
        self.dim = [0, 0, 0]  # [Height, Length, Width]
        self.t = None
        self.support = None
        self.mmoi = [0, 0, 0]  # [Ixx, Iyy, Izz]

        self.material = "Aluminium_7075-T73"

        self.stress = [0, 0, 0, 0]  # [tensile_launch, tensile_thermal, compressive_launch, compressive_thermal]
        self.vibration = [0, 0, 0]  # [axial, lateral_x, lateral_y]

        self.mass_breakdown = dict()
        self.mmoi_breakdown = dict()

    def update_mass_breakdown(self, key, value):
        if key in self.mass_breakdown:
            self.mass_breakdown[key] += value
        else:
            self.mass_breakdown[key] = value

    def update_mmoi_breakdown(self, key, value):
        if key in self.mmoi_breakdown:
            value = np.array(value, dtype=float)
            self.mmoi_breakdown[key] += value
        else:
            self.mmoi_breakdown[key] = np.array(value, dtype=float)

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
        return self.dim[0] * self.dim[1] * self.dim[2]

    @property
    def surface_area(self):
        return 2 * (self.dim[0] * self.dim[1] + self.dim[0] * self.dim[2] +
                    self.dim[1] * self.dim[2])

    @property
    def cross_section(self):
        return 2 * (self.dim[2] + self.dim[1]) * self.t

    @property
    def area_moment_of_inertia(self):
        iy = 2 * (self.dim[1] ** 3 * self.t + self.t ** 3 * self.dim[2]) / 12 + 2 * self.dim[2] * self.t * (
                self.dim[1] / 2) ** 2
        iz = 2 * (self.dim[2] ** 3 * self.t + self.t ** 3 * self.dim[1]) / 12 + 2 * self.dim[1] * self.t * (
                self.dim[2] / 2) ** 2
        return [iy, iz]

    @property
    def buckling_limit(self):
        if self.support is None:
            b = 2 * 4 * (np.pi ** 2 * self.E / (12 * (1 - 0.33 ** 2)) * (self.t / self.dim[2]) ** 2 * self.dim[
                2] * self.t + np.pi ** 2 * self.E / (12 * (1 - 0.33 ** 2)) * (self.t / self.dim[1]) ** 2 * self.dim[
                             1] * self.t) / self.cross_section
        else:
            y = 1 - 0.901 * (1 - np.e ** - (1 / 16 * np.sqrt(self.support.r2 / (self.support.r2 - self.support.r1))))
            if self.dim[2] / self.support.r2 <= 5:
                b = ((2 * 4 * (np.pi ** 2 * self.E / (12 * (1 - 0.33 ** 2)) * (self.t / self.dim[0]) ** 2 * self.dim[
                    0] * self.t + np.pi ** 2 * self.E / (12 * (1 - 0.33 ** 2)) * (self.t / self.dim[1]) ** 2 * self.dim[
                                   0] * self.t)) + 0.6 * y * self.support.E * (
                             self.support.r2 - self.support.r1) / self.support.r2 * self.support.area) / (
                            self.cross_section + self.support.area) # OUTDATED
                # self.support_E*((9*(self.support_r2-self.support_r1)/self.support_r2)**1.6 + 0.16*((self.support_r2-self.support_r1)/self.dim[2])**1.3)*self.support_area
                # 0.6 * self.support_E * (self.support_r2-self.support_r1)/self.support_r2*self.support_area
                # (self.support_E *self.support_area* 4 * np.pi**2 * self.support_I / (self.dim[2]))
            else:
                raise ValueError("Support is too short")
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

        self.dim = [height, width, length]
        self.t = t

        mass_struc = self.rho * t * self.surface_area
        self.mass += mass_struc

        m_panel = mass_struc / 6

        Ixx = 2 * (m_panel / 12) * (self.dim[2] ** 2 + self.dim[1] ** 2) + 2 * m_panel * (
                self.dim[1] ** 2 / 12 + self.dim[2] ** 2 / 4) + 2 * m_panel * (
                      self.dim[2] ** 2 / 12 + self.dim[1] ** 2 / 4)
        Iyy = 2 * (m_panel / 12) * (self.dim[0] ** 2 + self.dim[2] ** 2) + 2 * m_panel * (
                self.dim[0] ** 2 / 12 + self.dim[2] ** 2 / 4) + 2 * m_panel * (
                      self.dim[2] ** 2 / 12 + self.dim[0] ** 2 / 4)
        Izz = 2 * (m_panel / 12) * (self.dim[1] ** 2 + self.dim[0] ** 2) + 2 * m_panel * (
                self.dim[0] ** 2 / 12 + self.dim[1] ** 2 / 4) + 2 * m_panel * (
                      self.dim[1] ** 2 / 12 + self.dim[0] ** 2 / 4)

        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz

        self.update_mass_breakdown('skeleton', mass_struc)
        self.update_mmoi_breakdown('skeleton', [Ixx, Iyy, Izz])

    def add_propellant_tank(self, type, material, pos=[0, 0, 0], pressure=10e6, prop_mass=20., volume=None, r=0.2):
        """Adds a propellant tank to the satellite.
        :param type: Type of the propellant tank. (0-Sphere, 1-Cylinder) [int]
        :param material: Material of the propellant tank. [string]
        :param pos: Position of the propellant tank (x,y,z). [m]
        :param pressure: Pressure of the propellant tank. [Pa]
        :param prop_mass: Mass of the propellant. [kg]
        """

        if type == 0:
            self.propellant_tank = PropTankSphere(prop_mass, material, pressure, volume)
            Ixx = self.propellant_tank.mmoi + self.propellant_tank.mass * (pos[1] ** 2 + pos[2] ** 2)
            Iyy = self.propellant_tank.mmoi + self.propellant_tank.mass * (pos[0] ** 2 + pos[2] ** 2)
            Izz = self.propellant_tank.mmoi + self.propellant_tank.mass * (pos[0] ** 2 + pos[1] ** 2)
        elif type == 1:
            self.propellant_tank = PropTankCylinder(prop_mass, material, pressure, volume, r)
            Ixx = self.propellant_tank.Ixx + self.propellant_tank.mass * (pos[1] ** 2 + pos[2] ** 2)
            Iyy = self.propellant_tank.Iyy + self.propellant_tank.mass * (pos[0] ** 2 + pos[2] ** 2)
            Izz = self.propellant_tank.Izz + self.propellant_tank.mass * (pos[0] ** 2 + pos[1] ** 2)
        else:
            raise ValueError('Invalid propellant tank type')

        self.mass += self.propellant_tank.mass

        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz

        self.update_mass_breakdown('Propellant_tank', self.propellant_tank.mass_struc)
        self.update_mass_breakdown('Propellant', self.propellant_tank.mass_prop)
        self.update_mmoi_breakdown('Propellant+Tank', [Ixx, Iyy, Izz])

    def add_panels(self, a, b, mass, h=0, deployed=True):
        """Adds panels to the structure.
        :param a: length of one panel. [m]
        :param b: width of one panel. [m]
        :param mass: Total mass of the panels to be added. [kg]
        :param h: Height from centroid. [m]
        :param deployed: Whether the panels are deployed or not. [bool]
        """

        self.mass += mass
        dist = self.dim[2] / 2 + a / 2 + b/2

        if deployed:
            Iyy = mass * (b ** 2 / 12 + h ** 2)
            Izz = mass * ((b ** 2 + a ** 2) / 12 + dist ** 2 + h ** 2)  # Worst case scenario, where solar panels rotate
            Ixx = mass * ((b ** 2 + a ** 2) / 12 + dist ** 2)
        else:
            h += a / 2
            dist -= a / 2
            Iyy = mass / 12 * (a ** 2 + b ** 2) + mass * h ** 2
            Izz = mass / 12 * a ** 2 + mass * (dist ** 2 + h ** 2)
            Ixx = mass / 12 * b ** 2 + mass * dist ** 2
        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz
        self.update_mass_breakdown('panels', mass)
        self.update_mmoi_breakdown('panels', [Ixx, Iyy, Izz])

    def add_support(self, r, t, material):
        """Adds a support to the satellite.
        :param r: Radius of the support. [m]
        :param t: Thickness of the support. [m]
        :param material: Material of the support. [string]
        """
        r1 = r - 0.5 * t
        r2 = r + 0.5 * t
        self.support = Support(r1, r2, self.dim[2], material)
        self.mass += self.support.mass

        self.mmoi[0] += self.support.Ixx
        self.mmoi[1] += self.support.Iyy
        self.mmoi[2] += self.support.Izz

        self.mass_breakdown['support'] = self.support.mass
        self.mmoi_breakdown['support'] = [self.support.Ixx, self.support.Iyy, self.support.Izz]

    def add_dictionary(self, dictionary):
        """Adds a point mass elements to the satellite from a dictionary.
        :param dictionary: Dictionary containing the point mass elements. [dict]
        """
        for key in dictionary:
            self.add_point_element(dictionary[key]['mass'], dictionary[key]['subsystem'], dictionary[key]['cg'])

    def calculate_stresses(self):
        """Calculates the stresses on the satellite structure. \n
         Stresses - tensile_launch, tensile_thermal, compressive_launch, compressive_thermal
        """
        s1 = self.mass * g * (g_tensile / self.cross_section + g_lateral * self.dim[0] * (self.dim[1] / 2) /
                              self.area_moment_of_inertia[0])
        s2 = self.mass * g * (g_tensile / self.cross_section + g_lateral * self.dim[0] * (self.dim[2] / 2) /
                              self.area_moment_of_inertia[1])
        self.stress[0] = max(s1, s2)
        self.stress[1] = -self.thermal_coeff * (temperatures[0] - 20) * self.E
        s1 = self.mass * g * (g_axial / self.cross_section + g_lateral * self.dim[0] * (self.dim[1] / 2) /
                              self.area_moment_of_inertia[0])
        s2 = self.mass * g * (g_axial / self.cross_section + g_lateral * self.dim[0] * (self.dim[2] / 2) /
                              self.area_moment_of_inertia[1])
        self.stress[2] = max(s1, s2)
        self.stress[3] = self.thermal_coeff * (temperatures[1] - 20) * self.E

    def calculate_vibrations(self):
        """Calculates the vibrations on the satellite structure."""
        self.vibration[0] = 0.25 * np.sqrt(self.E * self.cross_section / (self.mass * self.dim[0]))
        self.vibration[1] = 0.56 * np.sqrt(self.E * self.area_moment_of_inertia[0] / (self.mass * self.dim[0] ** 3))
        self.vibration[2] = 0.56 * np.sqrt(self.E * self.area_moment_of_inertia[1] / (self.mass * self.dim[0] ** 3))

    def compliance(self):
        """Checks if the satellite structure is compliant with the requirements.
        :return: s_compliance, v_axial_compliance, v_lat_compliance, b_compliance
        """
        s_compliance = max(self.stress) < self.sigma_y / 1.1
        v_axial_compliance = self.vibration[0] > axial_freq_falcon * 1.5
        v_lat_compliance = min(self.vibration[1:]) > lateral_freq_falcon * 1.5
        b_compliance = self.stress[2] < self.buckling_limit / 1.1
        return s_compliance, v_axial_compliance, v_lat_compliance, b_compliance


def optimize():
    best = None
    best_mass = 100000
    for l in tqdm(np.arange(0.9, 1, 0.01)):
        for h in np.arange(0.9, 1, 0.05):
            for w in np.arange(0.9, l, 0.02):
                for t_support in np.arange(0.5e-3, 2e-3, 1e-4):
                    for t_struc in np.arange(1e-2, 2e-2, 1e-3):
                        for r in np.arange(h / 5, min(h / 2, w / 2), 2e-3):
                            s = SatelliteStruc.create_standard_satellite(l_struc=l, w_struc=w, h_struc=h,
                                                                         t_struc=t_struc,
                                                                         r_support=r, t_support=t_support,
                                                                         material_support="Ti-6AL-4V")
                            s.calculate_stresses()
                            s.calculate_vibrations()
                            if all(s.compliance()) and s.mass < best_mass:
                                best = s
                                best_mass = s.mass
                                print(f"New best: {best_mass}")
                                print(f"l: {l}, h: {h}, w: {w}, t_support: {t_support}, t_struc: {t_struc}, r: {r}")


if __name__ == "__main__":

    # Create Satellite
    s = SatelliteStruc()

    # Structure
    s.add_structure_sub(length=1, width=1, height=1, t=1.4e-2)

    # Power
    s.add_panels(a=3.699, b=0.925, mass=27.36)

    # Propulsion
    s.add_propellant_tank(type=1, material="Ti-6AL-4V", prop_mass=243.5, pressure=4e6, pos=[0, 0, 0], r=0.27)

    # Add Elements
    s.add_dictionary(components)


    # Calculate
    s.calculate_stresses()
    s.calculate_vibrations()

    # Results
    print(f"Total Mass: {s.mass:.3f}")
    print(f"Dry Mass: {s.mass - s.propellant_tank.mass_prop:.3f}")
    print(f"Propellant Mass: {s.propellant_tank.mass_prop:.3f}")
    print(f"Mass Breakdown: {s.mass_breakdown}")

    print(f"Meets requirements? {all(s.compliance())}")

    print(f"MMOI: {s.mmoi}")

    print(f"Propellant Tank Info: Radius-{s.propellant_tank.r:.3f}, Length-{s.propellant_tank.l:.3f}, "
          f"Volume-{s.propellant_tank.volume:.3f}, Thickness-{s.propellant_tank.thickness:.3f}, Total length-{s.propellant_tank.l+2*s.propellant_tank.r:.3f}")

    print(f"Vibrations: Axial-{s.vibration[0]:.3f}, Lateral 1-{s.vibration[1]:.3f}, Lateral 2-{s.vibration[2]:.3f}")