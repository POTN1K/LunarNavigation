"""Script to size the volume of the satellite structure.
By I. Maes, N. Ricker"""

# External imports
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Internal imports

# Constants
from structures_constants import *


# -----------------------------------------------------------------------------

class RectangularPrism:
    """Class to define a 3D rectangular prism."""

    def __init__(self, length=None, width=None, height=None, volume=None):
        """Initializes the rectangular prism.
        Needs at least three values. If four are given, the length, width and height become ratios.
        :param length: Length of the rectangular prism. [m]
        :param width: Width of the rectangular prism. [m]
        :param height: Height of the rectangular prism. [m]
        :param volume: Volume of the rectangular prism. [m^3]
        """
        self.name = "Rectangular prism"
        if volume is None:
            if length is not None and width is not None and height is not None:
                self.length = length
                self.width = width
                self.height = height
            else:
                raise ValueError("Insufficient parameters to calculate the rectangular prism.")
        else:
            if width is not None and height is not None and length is not None:
                scale_factor = (volume / (length * width * height)) ** (1 / 3)
                self.length = length * scale_factor
                self.width = width * scale_factor
                self.height = height * scale_factor
            elif width is not None and height is not None:
                self.length = volume / (width * height)
                self.width = width
                self.height = height
            elif length is not None and height is not None:
                self.length = length
                self.width = volume / (length * height)
                self.height = height
            elif length is not None and width is not None:
                self.length = length
                self.width = width
                self.height = volume / (length * width)
            else:
                raise ValueError("Not enough parameters.")

    def volume(self):
        """Computes the volume of the rectangular prism.
        :return: Volume of the rectangular prism. [m^3]"""
        return self.length * self.width * self.height

    def surface_area(self):
        """Computes the surface area of the rectangular prism.
        :return: Surface area of the rectangular prism. [m^2]"""
        return 2 * (self.length * self.width + self.length * self.height + self.width * self.height)

    def cross_section_area(self, t):
        """Computes the cross-sectional area around z-axis of the rectangular prism."""
        return 2 * (self.length + self.width) * t

    def area_I(self, t):
        """Computes the second moment of area of the cross-section around x/y-axis of the rectangular prism."""
        ix = 2 * (self.length ** 3 * t + t ** 3 * self.width) / 12 + 2 * self.width * t * (self.length / 2) ** 2
        iy = 2 * (self.width ** 3 * t + t ** 3 * self.length) / 12 + 2 * self.length * t * (self.width / 2) ** 2
        return ix, iy

    def MMOI(self, mass):
        """Computation of Mass moment of inertia for the body
        :param: distributed mass [kg]
        :return: Ixx, Iyy, Izz [kg*m^2]"""
        Ixx = (mass / 12) * (self.height ** 2 + self.length ** 2)
        Izz = (mass / 12) * (self.length ** 2 + self.width ** 2)
        Iyy = (mass / 12) * (self.height ** 2 + self.width ** 2)
        return Ixx, Iyy, Izz

    def __str__(self):
        return f"Rectangular Prism: Length - {self.length}[m] x Width - {self.width}[m] x Height - {self.height}[m]"

    def __repr__(self):
        return f"RectangularPrism({self.length}, {self.width}, {self.height})"


class Cylinder:
    """Class to define a 3D cylinder."""

    def __init__(self, radius=None, height=None, volume=None):
        """Initializes the cylinder.
        Needs at least two values. If three are given, the radius and height become ratios.
        :param radius: Radius of the cylinder. [m]
        :param height: height of the cylinder. [m]
        :param volume: Volume of the cylinder. [m^3]
        """
        self.name = "Cylinder"
        if volume is None:
            if radius is not None and height is not None:
                self.radius = radius
                self.height = height
            else:
                raise ValueError("Insufficient parameters to calculate the cylinder.")
        else:
            if radius is not None and height is not None:
                scale_factor = (volume / (np.pi * radius ** 2 * height)) ** (1 / 3)
                self.radius = radius * scale_factor
                self.height = height * scale_factor
            elif radius is not None:
                self.radius = radius
                self.height = volume / (np.pi * radius ** 2)
            elif height is not None:
                self.radius = np.sqrt(volume / (np.pi * height))
                self.height = height
            else:
                raise ValueError("Not enough parameters.")

    def volume(self):
        """Computes the volume of the cylinder.
        :return: Volume of the cylinder. [m^3]"""
        return np.pi * self.radius ** 2 * self.height

    def surface_area(self):
        """Computes the surface area of the cylinder.
        :return: Surface area of the cylinder. [m^2]"""
        return 2 * np.pi * self.radius * (self.radius + self.height)

    def cross_section_area(self, t):
        """Computes the cross-sectional area around z-axis of the rectangular prism."""
        return 2 * np.pi * self.radius * t

    def area_I(self, t):
        """Computes the second moment of area of the cross-section around x/y-axis of the cylinder."""
        i = np.pi / 4 * ((self.radius + t) ** 4 - self.radius ** 4)
        return i, i

    def MMOI(self, mass):
        """Computation of Mass moment of inertia for the body
        :param: distributed mass [kg]
        :return: Ixx, Iyy, Izz [kg*m^2]"""
        Ixx = (mass / 12) * (3 * self.radius ** 2 + self.height ** 2)
        Izz = (mass / 2) * self.radius ** 2
        Iyy = Ixx
        return Ixx, Iyy, Izz

    def __str__(self):
        return f"Cylinder: Radius - {self.radius}[m] x Height - {self.height}[m]"

    def __repr__(self):
        return f"Cylinder({self.radius}, {self.height})"


class Structure:
    """Structure subsystem of the satellite."""

    def __init__(self, shape, material, m0):
        """Initializes the structure.
        :param shape: Shape of the structure. [Object]
        :param material: Material of the structure. [string]
        :param m0: Mass of the structure. [kg]
        """
        self.panels_bool = False
        self._thickness = None
        self.m0 = m0
        self.shape = shape
        self.material = material

        self.compressive_thermal_stress = self.thermal_coeff * (temperatures[1] - 20) * self.E
        self.tensile_thermal_stress = -self.thermal_coeff * (temperatures[0] - 20) * self.E
        self.compressive_stress = None
        self.tensile_stress = None
        self.l_eigeny = None
        self.l_eigenx = None
        self.axial_eigen = None
        self.m_struc = None

    @property
    def compressive_thermal_stress(self):
        return self._compressive_thermal_stress

    @compressive_thermal_stress.setter
    def compressive_thermal_stress(self, compressive_thermal_stress):
        if compressive_thermal_stress < self.yield_strength:
            self._compressive_thermal_stress = compressive_thermal_stress
        else:
            raise ValueError(f"Compressive thermal stress exceeds yield strength of {self.yield_strength}Pa.")

    @property
    def tensile_thermal_stress(self):
        return self._tensile_thermal_stress

    @tensile_thermal_stress.setter
    def tensile_thermal_stress(self, tensile_thermal_stress):
        if tensile_thermal_stress < self.yield_strength:
            self._tensile_thermal_stress = tensile_thermal_stress
        else:
            raise ValueError(f"Tensile thermal stress exceeds yield strength of {self.yield_strength}Pa.")

    @property
    def compressive_stress(self):
        return self._compressive_stress

    @compressive_stress.setter
    def compressive_stress(self, compressive_stress):
        if compressive_stress is None:
            self._compressive_stress = compressive_stress
        elif compressive_stress < self.yield_strength:
            self._compressive_stress = compressive_stress
        else:
            raise ValueError(f"Compressive stress exceeds yield strength of {self.yield_strength}Pa.")

    @property
    def tensile_stress(self):
        return self._tensile_stress

    @tensile_stress.setter
    def tensile_stress(self, tensile_stress):
        if tensile_stress is None:
            self._tensile_stress = tensile_stress
        elif tensile_stress < self.yield_strength:
            self._tensile_stress = tensile_stress
        else:
            raise ValueError(f"Tensile stress exceeds yield strength of {self.yield_strength}Pa.")

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self.E = material_properties[material]['E']
        self.rho = material_properties[material]['density']
        self.yield_strength = material_properties[material]['yield_strength'] * 1.1
        self.ultimate_strength = material_properties[material]['ultimate_strength'] * 1.25
        self.thermal_coeff = material_properties[material]['thermal_coefficient']
        self._material = material

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self.surface_area = shape.surface_area()
        self.Ixx, self.Iyy, self.Izz = shape.MMOI(self.m0)
        self.height = shape.height
        self._shape = shape

    @property
    def thickness(self):
        if self._thickness is None:
            self._thickness = self.thickness_limiting()
            return self._thickness
        else:
            return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        if thickness >= self.thickness_limiting():
            self._thickness = thickness
        else:
            raise ValueError(
                f"Thickness does not meet requirements, must be at least {round(self.thickness_limiting() * 1000, 3)}mm.")

    def thickness_axial_freq(self):
        """Computes the thickness of the structure based on the frequency of the structure.
        :return: Thickness of the structure due to the axial frequency requirement. [m]"""
        if isinstance(self.shape, Cylinder):
            ar = 2 * np.pi * self.shape.radius
        elif isinstance(self.shape, RectangularPrism):
            ar = 2 * self.shape.width + 2 * self.shape.length
        else:
            raise TypeError("Shape not recognized.")
        return mass_init * self.height / (self.E * ar) * (axial_freq_falcon / 0.25) ** 2

    def thickness_lateral_freq(self):
        """Computes the thickness of the structure based on the frequency of the structure.
        :return: Thickness of the structure due to the axial frequency requirement. [m]"""
        I = (lateral_freq_falcon / 0.56) ** 2 * self.m0 * self.height ** 3 / self.E
        if isinstance(self.shape, Cylinder):
            t = -self.shape.radius + ((4 * I + self.shape.radius ** 4 * np.pi) / np.pi) ** (1 / 4)
            return t
        elif isinstance(self.shape, RectangularPrism):
            t1 = fsolve(lambda t: (1 / 6) * self.shape.width * t ** 3 + 2 * self.shape.width * t * (
                    self.shape.length / 2) ** 2 +
                                  (1 / 6) * t * self.shape.length ** 3 - I, 0.000001)[0]
            t2 = fsolve(lambda t: (1 / 6) * self.shape.length * t ** 3 + 2 * self.shape.length * t * (
                    self.shape.width / 2) ** 2 +
                                  (1 / 6) * t * self.shape.width ** 3 - I, 0.000001)[0]
            return max(t1, t2)
        else:
            raise TypeError("Shape not recognized.")

    def thickness_axial_stress(self):
        """Computes the axial stresses in the structure.
        :return: Thickness of the structure due to the axial frequency requirement. [m]"""
        if isinstance(self.shape, Cylinder):
            t = fsolve(
                lambda t: g_axial * g * self.m0 / (2 * np.pi * self.shape.radius * t) + g_lateral * g * self.m0 *
                          self.height * self.shape.radius / (np.pi / 4 * (
                        (self.shape.radius + t) ** 4) - self.shape.radius ** 4) - self.yield_strength,
                0.00000001)[0]
            return t

        elif isinstance(self.shape, RectangularPrism):
            t1 = fsolve(lambda t: g_axial * g * self.m0 / (2 * (self.shape.width + self.shape.length) * t) +
                                  g_lateral * g * self.m0 * self.height * self.shape.length / ((1 / 6) *
                                                                                               self.shape.width * t ** 3 + 2 * self.shape.width * t * (
                                                                                                       self.shape.length / 2) ** 2 +
                                                                                               (
                                                                                                       1 / 6) * t * self.shape.length ** 3) - self.yield_strength,
                        0.000001)[0]
            t2 = fsolve(lambda t: g_axial * g * self.m0 / (2 * (self.shape.width + self.shape.length) * t) +
                                  g_lateral * g * self.m0 * self.height * self.shape.width / ((1 / 6) *
                                                                                              self.shape.length * t ** 3 + 2 * self.shape.length * t * (
                                                                                                      self.shape.width / 2) ** 2 +
                                                                                              (
                                                                                                      1 / 6) * t * self.shape.width ** 3) - self.yield_strength,
                        0.000001)[0]
            return max(t1, t2)
        else:
            raise TypeError("Shape not recognized.")

    def thickness_buckling(self):
        if isinstance(self.shape, Cylinder):
            d = self.shape.radius
        elif isinstance(self.shape, RectangularPrism):
            d = (self.shape.length + self.shape.width) / 4
        else:
            raise TypeError("Shape not recognized.")
        t = fsolve(lambda t: 9 * (t / d) ** 1.6 + 0.16 * (t / self.height) ** 1.3 - (0.6 * 0.33 * t / d) * 1.1, 0.001)
        return t[0]

    def thickness_limiting(self):
        a = self.thickness_axial_freq()
        b = self.thickness_buckling()
        c = self.thickness_lateral_freq()
        d = self.thickness_axial_stress()
        print(a,b,c,d)
        return max(a,b,c,d)

    def compute_characteristics(self):
        """Function to compute characteristics of the structure
        :return: m_struc, axial_eigen, lateral_eigen, tensile_stress, compressive_stress"""
        self.m_struc = self.surface_area * self.thickness * self.rho
        self.axial_eigen = 0.25 * np.sqrt(
            self.E * self.shape.cross_section_area(self.thickness) / (self.m0 * self.height))
        if isinstance(self.shape, Cylinder):
            self.l_eigenx = 0.56 * np.sqrt(self.E * self.shape.area_I(self.thickness)[0] / (self.m0 * self.height ** 3))
            self.l_eigeny = self.l_eigenx
            self.tensile_stress = self.m0 * g * (g_tensile / self.shape.cross_section_area(self.thickness) +
                                                 g_lateral * self.height * self.shape.radius /
                                                 self.shape.area_I(self.thickness)[0])
            self.compressive_stress = self.m0 * g * (g_axial / self.shape.cross_section_area(self.thickness) +
                                                      g_lateral * self.height * self.shape.radius /
                                                      self.shape.area_I(self.thickness)[0])
        elif isinstance(self.shape, RectangularPrism):
            self.l_eigenx = 0.56 * np.sqrt(self.E * self.shape.area_I(self.thickness)[0] / (self.m0 * self.height ** 3))
            self.l_eigeny = 0.56 * np.sqrt(self.E * self.shape.area_I(self.thickness)[1] / (self.m0 * self.height ** 3))
            tstress1 = self.m0 * g * (g_tensile / self.shape.cross_section_area(self.thickness) +
                                      g_lateral * self.height * (self.shape.length / 2) /
                                      self.shape.area_I(self.thickness)[0])
            tstress2 = self.m0 * g * (g_tensile / self.shape.cross_section_area(self.thickness) +
                                      g_lateral * self.height * (self.shape.width / 2) /
                                      self.shape.area_I(self.thickness)[1])
            cstress1 = self.m0 * g * (g_axial / self.shape.cross_section_area(self.thickness) +
                                      g_lateral * self.height * (self.shape.length / 2) /
                                      self.shape.area_I(self.thickness)[0])
            cstress2 = self.m0 * g * (g_axial / self.shape.cross_section_area(self.thickness) +
                                      g_lateral * self.height * (self.shape.width / 2) /
                                      self.shape.area_I(self.thickness)[1])
            self.tensile_stress = max(tstress1, tstress2)
            self.compressive_stress = max(cstress1, cstress2)
        else:
            raise TypeError("Shape not recognized.")

    def add_panels(self, area, mass, deployed=True):
        """Adds panels to the structure.
        :param area: Area of the panels to be added. [m^2]
        :param mass: Mass of the panels to be added. [kg]
        :param deployed: Whether the panels are deployed or not. [bool]
        :return: None"""
        if isinstance(self.shape, Cylinder):
            b = self.shape.radius * 2
            a = area / (2 * b)
            dist = self.shape.radius + a / 2
        elif isinstance(self.shape, RectangularPrism):
            b = self.shape.length
            a = area / (2 * b)
            dist = self.shape.length / 2 + a / 2
        else:
            raise TypeError("Shape not recognized.")
        if deployed:
            self.Izz += mass * ((b ** 2 + a ** 2) / 12 + dist ** 2)
            self.Iyy += mass * ((b ** 2 + a ** 2) / 12 + dist ** 2)
            self.Ixx += mass * b ** 2 / 12
        else:
            self.Izz += mass * dist ** 2
            self.Iyy += mass * dist ** 2
            self.Ixx += mass * dist ** 2
        self.panels_bool = True

    def __str__(self):
        if self.m_struc is None:
            self.compute_characteristics()
        return f"Satellite characteristics:\n" \
               f"{self.shape}\t Thickness - {round(self.thickness * 1000, 3)} [mm]\n" \
               f"MMOI: Ixx - {round(self.Ixx, 5)}\t Iyy - {round(self.Iyy, 5)}\t Izz - {round(self.Izz, 5)} [kgm2]\n" \
               f"Initial mass - {self.m0}\tStructure mass - {round(self.m_struc, 5)} [kg]\n" \
               f"Eigen frequencies: Axial - {round(self.axial_eigen, 5)}\t Lateral - {round(self.l_eigenx, 5)}/{round(self.l_eigeny, 5)} [Hz]\n" \
               f"Stresses: Tensile - {round(self.tensile_stress, 5)}\t Compressive - {round(self.compressive_stress, 5)}\t " \
               f"Thermal - {round(max(self.compressive_thermal_stress, self.tensile_thermal_stress), 5)}[Pa]\n" \
               f"Includes solar panels? {self.panels_bool}"


if __name__ == "__main__":
    mass_no_struc = 1000  # [kg] Mass of the spacecraft without the structure
    mass_struc = 124  # [kg] Mass of the structure
    mass_init = mass_no_struc + mass_struc  # [kg] Mass of the initial sizing of the spacecraft
    vol_init = mass_init * 0.01  # [m^3] Volume of the initial sizing of the spacecraft

    i = 1
    j = 1.3
    rect = RectangularPrism(i, i, j, vol_init)
    s = Structure(rect, "Aluminium_7075-T73", mass_init)
    s.add_panels(8, 100)
    s.compute_characteristics()
    _ = f"{rect.width}x{rect.length}x{rect.height}, {s.m_struc}, {s.tensile_stress}, {s.compressive_stress}, {s.thickness}, {s.Ixx}, {s.Iyy}, {s.Izz}, {s.axial_eigen}, {s.l_eigenx}, {s.l_eigeny}"
    print(_)