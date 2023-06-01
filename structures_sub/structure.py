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

    def MMOI(self, mass):
        """Computation of Mass moment of inertia for the body
        :param: distributed mass [kg]
        :return: Ixx, Iyy, Izz [kg*m^2]"""
        Ixx = (mass / 12) * (self.height ** 2 + self.length ** 2)
        Iyy = (mass / 12) * (self.length ** 2 + self.width ** 2)
        Izz = (mass / 12) * (self.height ** 2 + self.width ** 2)
        return Ixx, Iyy, Izz

    def __str__(self):
        return f"Rectangular Prism: {self.length}[m] x {self.width}[m] x {self.height}[m]"

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

    def MMOI(self, mass):
        """Computation of Mass moment of inertia for the body
        :param: distributed mass [kg]
        :return: Ixx, Iyy, Izz [kg*m^2]"""
        Ixx = (mass / 12) * (3 * self.radius ** 2 + self.height ** 2)
        Iyy = (mass / 2) * self.radius ** 2
        Izz = Ixx
        return Ixx, Iyy, Izz

    def __str__(self):
        return f"Cylinder: {self.radius}[m] x {self.height}[m]"

    def __repr__(self):
        return f"Cylinder({self.radius}, {self.height})"


class Structure:
    """Structure subsystem of the satellite."""

    def __init__(self, shape, material, m0):
        self.m0 = m0
        self.shape = shape
        self.material = material

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self.E = material_properties[material]['E']
        self.rho = material_properties[material]['density']
        self.yield_strength = material_properties[material]['yield_strength']
        self.ultimate_strength = material_properties[material]['ultimate_strength']
        self.thermal_coeff = material_properties[material]['thermal_coefficient']
        self._material = material

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self.volume = shape.volume()
        self.surface_area = shape.surface_area()
        self.Ixx, self.Iyy, self.Izz = shape.MMOI(self.m0)
        self.height = shape.height
        self._shape = shape

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
            t = -self.shape.radius + ((4 * I + self.shape.rafius ** 4 * np.pi) / np.pi) ** (1 / 4)
            return t
        elif isinstance(self.shape, RectangularPrism):
            t1 = fsolve(lambda t: (1 / 6) * self.shape.width * t ** 3 + 2 * self.shape.width * t * (
                    self.shape.length / 2) ** 2 +
                                  (1 / 6) * t * self.shape.length ** 3 - I, 0.000001)
            t2 = fsolve(lambda t: (1 / 6) * self.shape.length * t ** 3 + 2 * self.shape.length * t * (
                    self.shape.width / 2) ** 2 +
                                  (1 / 6) * t * self.shape.width ** 3 - I, 0.000001)
            return max(t1, t2)
        else:
            raise TypeError("Shape not recognized.")

    def thickness_axial_stress(self):
        """Computes the axial stresses in the structure.
        :return: Thickness of the structure due to the axial frequency requirement. [m]"""
        if isinstance(self.shape, Cylinder):
            t = fsolve(
                lambda t: g_axial * g * self.m0 / (2 * np.pi * self.shape.radius * t) + g_lateral * g * self.m0 *
                          self.height * self.radius / (np.pi / 4 * ((self.radius + t) ** 4) - self.radius ** 4),
                0.000001)
            return t

        elif isinstance(self.shape, RectangularPrism):
            t1 = fsolve(lambda t: g_axial * self.m0 / (2 * (
                    self.shape.width + self.shape.length) * t) + g_lateral * g * self.m0 * self.height * self.shape.length / (
                                              (1 / 6) * self.shape.width * t ** 3 + 2 * self.shape.width * t * (
                                              self.shape.length / 2) ** 2 + (1 / 6) * t * self.shape.length ** 3),
                        0.000001)
            t2 = fsolve(lambda t: g_axial * self.m0 / (2 * (
                    self.shape.width + self.shape.length) * t) + g_lateral * g * self.m0 * self.height * self.shape.width / (
                                          (1 / 6) * self.shape.length * t ** 3 + 2 * self.shape.length * t * (
                                          self.shape.width / 2) ** 2 + (1 / 6) * t * self.shape.width ** 3),
                        0.000001)
            return max(t1, t2)
        else:
            raise TypeError("Shape not recognized.")

if __name__ == "__main__":
    mass_init = 500  # 1564.541  # [kg] Mass of the initial sizing of the spacecraft
    vol_init = 15.64154  # [m^3] Volume of the initial sizing of the spacecraft

    shape = RectangularPrism(length=2, width=1.5, height=3)
    struc = Structure(shape, "Aluminium_7075-T73", mass_init)
    print(shape.MMOI(mass_init))


