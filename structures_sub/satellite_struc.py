"""Script to create a satellite structure.
By I. Maes, N. Ricker"""

# Global imports
import numpy as np

# Local imports


# Constants
from structures_constants import *
from tqdm import tqdm


# --------------------------------------------------------------------------- #
class PropellantTank:
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
        return f"PropellantTank({self.volume}, {self.material})"


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

    @staticmethod
    def create_standard_satellite(l_struc, w_struc, h_struc, t_struc, r_support=None, t_support=None, material_support=None):
        """Creates a standard satellite structure.
        :param l_struc: Length of the structure. [m]
        :param w_struc: Width of the structure. [m]
        :param h_struc: Height of the structure. [m]
        :param t_struc: Thickness of the structure. [m]
        :param r_support: Radius of the support. [m]
        :param t_support: Thickness of the support. [m]
        :param material_support: Material of the support. [string]
        :return: SatelliteStruc object.
        """
        s = SatelliteStruc()

        # Structure
        s.add_structure_sub(length=l_struc, width=w_struc, height=h_struc, t=t_struc)

        if r_support is not None and t_support is not None and material_support is not None:
            s.add_support(r=r_support, t=t_support, material=material_support)

        # Power
        s.add_panels(a=2.83, b=0.71, mass=40)
        s.add_point_element(mass=30, name="battery", pos=[0.3, -0.3, 0.3])
        # Propulsion
        s.add_propellant_tank(material="Steel_17-4PH_H1150", prop_mass=212.9, pressure=2e6, pos=[0, -0.3, 0])
        s.add_point_element(mass=0.59, name="Big thruster", pos=[0, -0.5, 0])
        s.add_point_element(mass=1.74, name="Small thrusters")
        s.add_point_element(mass=4.4, name="Valves")
        # ADCS
        s.add_point_element(mass=10, name="cmg", pos=[0, 0, 0.2])
        s.add_point_element(mass=10, name="cmg", pos=[0, 0, -0.2])
        s.add_point_element(mass=1.41, name="starsensor")
        s.add_point_element(mass=0.03, name="sunseùnsor")
        s.add_point_element(mass=13, name="IMU")
        # CDH
        s.add_point_element(mass=10.5, name="computer", pos=[-0.4, 0.4, 0.4])
        # EPS
        s.add_point_element(mass=18.2, name="pcdu", pos=[-0.45, -0.4, 0.4])
        # Navigation
        s.add_point_element(mass=12, name="nsgu", pos=[0, 0.4, 0])
        s.add_point_element(mass=7.6, name="freq", pos=[0.3, 0.4, 0])
        s.add_point_element(mass=5.2, name="Clock Monitor", pos=[-0.4, 0.4, 0])
        s.add_point_element(mass=15.9, name="clock", pos=[0.3, 0.3, -0.3])
        s.add_point_element(mass=15.9, name="clock", pos=[0, 0.3, -0.3])
        s.add_point_element(mass=15.9, name="clock", pos=[-0.3, 0.3, -0.3])
        # TTC
        s.add_point_element(mass=5.8, name="antenna", pos=[0, 0.45, 0])
        s.add_point_element(mass=4, name="laser", pos=[0, 0, 0])
        s.add_point_element(mass=12.6, name="Moon reflector", pos=[0, 0.45, -0.45])

        # Unknown values
        s.add_point_element(mass=100, name="unknown", pos=[0, 0, 0])
        s.add_point_element(mass=100, name='Cables')

        return s

    def __init__(self, name="Sat_Test"):
        """Initialize the satellite structure.
        :param name: Name of the satellite. [string]
        """
        self.name = name
        self.mass = 0
        self.dim = [0, 0, 0]  # [Length, Width, Height]
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
        return self.dim[0] * self.dim[1] * self.dim[2]

    @property
    def surface_area(self):
        return 2 * (self.dim[0] * self.dim[1] + self.dim[0] * self.dim[2] +
                    self.dim[1] * self.dim[2])

    @property
    def cross_section(self):
        return 2 * (self.dim[0] + self.dim[1]) * self.t

    @property
    def area_moment_of_inertia(self):
        ix = 2 * (self.dim[1] ** 3 * self.t + self.t ** 3 * self.dim[0]) / 12 + 2 * self.dim[0] * self.t * (
                self.dim[1] / 2) ** 2
        iy = 2 * (self.dim[0] ** 3 * self.t + self.t ** 3 * self.dim[1]) / 12 + 2 * self.dim[1] * self.t * (
                self.dim[0] / 2) ** 2
        return [ix, iy]

    @property
    def buckling_limit(self):
        if self.support is None:
            b = 2 * 4 * (np.pi ** 2 * self.E / (12 * (1 - 0.33 ** 2)) * (self.t / self.dim[0]) ** 2 * self.dim[
                0] * self.t + np.pi ** 2 * self.E / (12 * (1 - 0.33 ** 2)) * (self.t / self.dim[1]) ** 2 * self.dim[
                             0] * self.t) / self.cross_section
        else:
            y = 1 - 0.901 * (1 - np.e ** - (1 / 16 * np.sqrt(self.support.r2 / (self.support.r2 - self.support.r1))))
            if self.dim[2] / self.support.r2 <= 5:
                b = ((2 * 4 * (np.pi ** 2 * self.E / (12 * (1 - 0.33 ** 2)) * (self.t / self.dim[0]) ** 2 * self.dim[
                    0] * self.t + np.pi ** 2 * self.E / (12 * (1 - 0.33 ** 2)) * (self.t / self.dim[1]) ** 2 * self.dim[
                                   0] * self.t)) + 0.6 * y * self.support.E * (
                             self.support.r2 - self.support.r1) / self.support.r2 * self.support.area) / (
                            self.cross_section + self.support.area)
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

        self.dim = [length, width, height]
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

        self.update_mass_breakdown('structure', mass_struc)
        self.update_mmoi_breakdown('structure', [Ixx, Iyy, Izz])

    def add_propellant_tank(self, material, pos=[0, 0, 0], pressure=10e6, prop_mass=20, volume=None):
        """Adds a propellant tank to the satellite.
        :param volume: Volume of the propellant tank. [m^3]
        :param material: Material of the propellant tank. [string]
        :param pos: Position of the propellant tank (x,y,z). [m]
        :param pressure: Pressure of the propellant tank. [Pa]
        :param prop_mass: Mass of the propellant. [kg]
        """
        self.propellant_tank = PropellantTank(prop_mass, material, pressure, volume)
        self.mass += self.propellant_tank.mass

        Ixx = self.propellant_tank.mmoi + self.propellant_tank.mass * (pos[1] ** 2 + pos[2] ** 2)
        Iyy = self.propellant_tank.mmoi + self.propellant_tank.mass * (pos[0] ** 2 + pos[2] ** 2)
        Izz = self.propellant_tank.mmoi + self.propellant_tank.mass * (pos[0] ** 2 + pos[1] ** 2)
        self.mmoi[0] += Ixx
        self.mmoi[1] += Iyy
        self.mmoi[2] += Izz

        self.update_mass_breakdown('propellant_tank', self.propellant_tank.mass_struc)
        self.update_mass_breakdown('propellant', self.propellant_tank.mass_prop)
        self.update_mmoi_breakdown('propellant+tank', [Ixx, Iyy, Izz])

    def add_panels(self, a, b, mass, h=0, deployed=True):
        """Adds panels to the structure.
        :param a: length of one panel. [m]
        :param b: width of one panel. [m]
        :param mass: Total mass of the panels to be added. [kg]
        :param h: Height from centroid. [m]
        :param deployed: Whether the panels are deployed or not. [bool]
        """

        self.mass += mass
        xz_dist = self.dim[0] / 2 + a / 2

        if deployed:
            Ixx = mass * (b ** 2 / 12 + h ** 2)
            Iyy = mass * ((
                                  b ** 2 + a ** 2) / 12 + xz_dist ** 2 + h ** 2)  # Worst case scenario, where solar panels rotate
            Izz = mass * ((b ** 2 + a ** 2) / 12 + xz_dist ** 2)
        else:
            h += a / 2
            Ixx = mass / 12 * (a ** 2 + b ** 2) + mass * h ** 2
            Iyy = mass / 12 * a ** 2 + mass * (xz_dist ** 2 + h ** 2)
            Izz = mass / 12 * b ** 2 + mass * xz_dist ** 2
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

    def calculate_stresses(self):
        """Calculates the stresses on the satellite structure. \n
         Stresses - tensile_launch, tensile_thermal, compressive_launch, compressive_thermal
        """
        s1 = self.mass * g * (g_tensile / self.cross_section + g_lateral * self.dim[2] * (self.dim[1] / 2) /
                              self.area_moment_of_inertia[0])
        s2 = self.mass * g * (g_tensile / self.cross_section + g_lateral * self.dim[2] * (self.dim[0] / 2) /
                              self.area_moment_of_inertia[1])
        self.stress[0] = max(s1, s2)
        self.stress[1] = -self.thermal_coeff * (temperatures[0] - 20) * self.E
        s1 = self.mass * g * (g_axial / self.cross_section + g_lateral * self.dim[2] * (self.dim[1] / 2) /
                              self.area_moment_of_inertia[0])
        s2 = self.mass * g * (g_axial / self.cross_section + g_lateral * self.dim[2] * (self.dim[0] / 2) /
                              self.area_moment_of_inertia[1])
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
    s = SatelliteStruc.create_standard_satellite(l_struc=0.9, w_struc=0.9, h_struc=0.9, t_struc=1.4e-2)



    # l: 1.0, h: 1.0, w: 0.5, t_support: 0.0015, t_struc: 0.0005, r: 0.2

    # s = SatelliteStruc.create_standard_satellite(l_struc=0.92, w_struc=0.92, h_struc=0.92,
    #                                      t_struc=0.001,
    #                                      r_support=0.19, t_support=0.001,
    #                                      material_support="Ti-6AL-4V")

    s.calculate_stresses()
    s.calculate_vibrations()
    print(f"Mass: {s.mass}")
    print(f"Mass Break: {s.mass_breakdown}")
    # print(f"MMOI Break: {s.mmoi_breakdown}")
    print(f"Compliance: {s.compliance()}")
    # print(f"Buckling limit: {s.buckling_limit}")
    # print(f"Tank radius: {s.propellant_tank.r}")
    # print(f"Stresses: {s.stress}")
    # print(f"Vibrations: {s.vibration}")
    print(s.propellant_tank.r)
