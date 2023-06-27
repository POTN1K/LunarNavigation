"""File performing Verification and Validation of structures_sub package.
It is performed on the stage_1structure and detailed_structure scripts.
By N. Ricker"""

# External modules
import numpy as np
import unittest

# Internal modules
from stage1_structure import Cylinder, RectangularPrism, Structure
from detailed_structure import PropTankSphere, PropTankCylinder, SatelliteStruc


class TestStage1Cylinder(unittest.TestCase):
    """Class to perform unit tests on stage1_structure.py Cylinder object"""

    def test_volume(self):
        """Test the volume method"""
        cylinder = Cylinder(1, 2)
        self.assertEqual(cylinder.volume(), np.pi * 2)
        cylinder = Cylinder(1, 2, 3)
        self.assertAlmostEqual(cylinder.volume(), 3, 2)

    def test_surface_area(self):
        """Test the surface_area method"""
        cylinder = Cylinder(1, 2)
        self.assertEqual(cylinder.surface_area(), 2 * np.pi + 2 * np.pi * 2)

    def test_cross_section_area(self):
        """Test the cross_section method"""
        cylinder = Cylinder(1, 2)
        self.assertEqual(cylinder.cross_section_area(1e-2), 2 * np.pi * 1e-2)

    def test_area_I(self):
        """Test the areaI method"""
        cylinder = Cylinder(1, 2)
        _ = np.pi / 4 * ((1 + 1e-2) ** 4 - 1 ** 4)
        self.assertEqual(cylinder.area_I(1e-2)[0], _)

    def test_MMOI(self):
        """Test the MMOI method"""
        cylinder = Cylinder(1, 2)
        _ = [100 / 12 * (3 + 4), 100 / 12 * (3 + 4), 100 / 2]
        self.assertEqual(cylinder.MMOI(100)[0], _[0])
        self.assertEqual(cylinder.MMOI(100)[1], _[1])
        self.assertEqual(cylinder.MMOI(100)[2], _[2])

    def test_correctInputs(self):
        """Test the correctInputs method"""
        with self.assertRaises(ValueError):
            cylinder = Cylinder(radius=1)
        with self.assertRaises(ValueError):
            cylinder = Cylinder(height=1)
        with self.assertRaises(ValueError):
            cylinder = Cylinder(volume=1)


class TestStage1RectangularPrism(unittest.TestCase):
    """Class to perform unit tests on stage1_structure.py RectangularPrism object"""

    def test_volume(self):
        """Test the volume method"""
        rectangular_prism = RectangularPrism(1, 2, 3)
        self.assertEqual(rectangular_prism.volume(), 6)

    def test_surface_area(self):
        """Test the surface_area method"""
        rectangular_prism = RectangularPrism(1, 2, 3)
        self.assertEqual(rectangular_prism.surface_area(), 2 * 1 * 2 + 2 * 1 * 3 + 2 * 2 * 3)

    def test_cross_section_area(self):
        """Test the cross_section method"""
        rectangular_prism = RectangularPrism(1, 2, 3)
        self.assertEqual(rectangular_prism.cross_section_area(1e-2), 1e-2 * 2 * (2 + 1))

    def test_area_I(self):
        """Test the areaI method"""
        rectangular_prism = RectangularPrism(1, 2, 3)
        _ = [2 * (1e-2 + 1e-2 ** 3 * 2) / 12 + 2 * 2 * 1e-2 * 0.5 ** 2, 2 * (2 ** 3 * 1e-2 + 1e-2 ** 3) / 12 + 2 * 1e-2]
        self.assertEqual(rectangular_prism.area_I(1e-2)[0], _[0])
        self.assertEqual(rectangular_prism.area_I(1e-2)[1], _[1])

    def test_MMOI(self):
        """Test the MMOI method"""
        rectangular_prism = RectangularPrism(1, 2, 3)
        _ = [1 / 12 * (3 ** 2 + 1 ** 2), 1 / 12 * (3 ** 2 + 2 ** 2), 1 / 12 * (1 ** 2 + 2 ** 2)]
        self.assertEqual(rectangular_prism.MMOI(1)[0], _[0])
        self.assertEqual(rectangular_prism.MMOI(1)[1], _[1])
        self.assertEqual(rectangular_prism.MMOI(1)[2], _[2])

    def test_correctInputs(self):
        """Test the correctInputs method"""
        with self.assertRaises(ValueError):
            RectangularPrism(length=1)
        with self.assertRaises(ValueError):
            RectangularPrism(width=1)
        with self.assertRaises(ValueError):
            RectangularPrism(height=1)
        with self.assertRaises(ValueError):
            RectangularPrism(volume=1)
        with self.assertRaises(ValueError):
            RectangularPrism(length=1, width=1)
        with self.assertRaises(ValueError):
            RectangularPrism(length=1, height=1)
        with self.assertRaises(ValueError):
            RectangularPrism(width=1, height=1)
        with self.assertRaises(ValueError):
            RectangularPrism(volume=1, height=1)
        with self.assertRaises(ValueError):
            RectangularPrism(volume=1, width=1)
        with self.assertRaises(ValueError):
            RectangularPrism(volume=1, length=1)


class TestStage1Structure(unittest.TestCase):
    """Class to perform unit tests on stage1_structure.py Structure object"""

    def test_t_axial_freq(self):
        """Test the t_axial_freq method"""
        s1 = Structure(Cylinder(1, 2), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s1.thickness_axial_freq(), 4.61e-7, 8)
        s2 = Structure(RectangularPrism(1, 2, 3), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s2.thickness_axial_freq(), 7.26e-7, 8)

    def test_t_lateral_freq(self):
        """Test the t_lateral_freq method"""
        s1 = Structure(Cylinder(1, 2), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s1.thickness_lateral_freq(), 1.18e-7, 8)
        s2 = Structure(RectangularPrism(1, 2, 3), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s2.thickness_lateral_freq(), 10.7e-7, 8)

    def test_t_axial_stress(self):
        """Test the t_axial_stress method"""
        s1 = Structure(Cylinder(1, 2), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s1.thickness_axial_stress(), 4.37e-7, 8)
        s2 = Structure(RectangularPrism(1, 2, 3), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s2.thickness_axial_stress(), 29.5e-7, 8)

    def test_thickness_buckling(self):
        """Test the thickness_buckling method"""
        s1 = Structure(Cylinder(1, 2), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s1.thickness_buckling(), 0.00187, 5)
        s2 = Structure(RectangularPrism(1, 2, 3), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s2.thickness_buckling(), 0.00147, 5)

    def test_characteristics(self):
        """Test the characteristics method"""
        s1 = Structure(Cylinder(1, 2), "Aluminium_6061-T6", 10)
        s1.compute_characteristics()
        self.assertAlmostEqual(s1.axial_eigen, 1592.2, 1)
        self.assertAlmostEqual(s1.l_eigenx, 1262.8, 1)
        self.assertAlmostEqual(s1.l_eigeny, 1262.8, 1)
        self.assertAlmostEqual(s1.tensile_stress, 132977.4, 1)
        self.assertAlmostEqual(s1.compressive_stress, 170456.2, 1)

    def test_addPanels(self):
        """Test the addPanels method"""
        s1 = Structure(Cylinder(1, 2), "Aluminium_6061-T6", 10)
        self.assertAlmostEqual(s1.Ixx, 5.83, 2)
        self.assertAlmostEqual(s1.Iyy, 5.83, 2)
        self.assertAlmostEqual(s1.Izz, 5.0, 2)
        s1.add_panels(10, 20)
        self.assertAlmostEqual(s1.Ixx, 12.5, 2)
        self.assertAlmostEqual(s1.Iyy, 124.17, 2)
        self.assertAlmostEqual(s1.Izz, 123.33, 2)


class TestDetailedDesignPropTank(unittest.TestCase):
    """Class to perform unit tests on detailed_design_proptank.py PropTank objects"""

    def test_volume(self):
        """Test the volume creation with margin"""
        s = PropTankSphere(10, "Aluminium_6061-T6")
        self.assertAlmostEqual(s.volume, 0.00748, 4)
        c = PropTankCylinder(10, "Aluminium_6061-T6")
        self.assertAlmostEqual(c.volume, 0.00748, 4)
        s2 = PropTankSphere(10, "Aluminium_6061-T6", volume=0.1)
        self.assertEqual(s2.volume, 0.1)

    def text_sphereMMOI(self):
        """Test the MMOI calculation for a sphere"""
        s = PropTankSphere(10, "Aluminium_6061-T6")
        self.assertAlmostEqual(s.mmoi, 0.0005, 4)

    def text_cylinderMMOI(self):
        """Test the MMOI calculation for a cylinder"""
        c = PropTankCylinder(10, "Aluminium_6061-T6")
        self.assertAlmostEqual(c.Ixx, 0.0004, 7)
        self.assertAlmostEqual(c.Iyy, 0.0004, 7)
        self.assertAlmostEqual(c.Izz, 0.0002, 7)

    def test_sphereThickness(self):
        """Test the thickness calculation for a sphere"""
        s = PropTankSphere(10, "Aluminium_6061-T6")
        self.assertAlmostEqual(s.thickness, 0.00242, 4)

    def test_cylinderThickness(self):
        """Test the thickness calculation for a cylinder"""
        c = PropTankCylinder(10, "Aluminium_6061-T6")
        self.assertAlmostEqual(c.thickness, 0.0167, 4)

    def test_sphereMass(self):
        """Test the mass calculation for a sphere"""
        s = PropTankSphere(10, "Aluminium_6061-T6")
        self.assertAlmostEqual(s.mass, 11.21, 2)

    def test_cylinderMass(self):
        """Test the mass calculation for a cylinder"""
        c = PropTankCylinder(10, "Aluminium_6061-T6")
        self.assertAlmostEqual(c.mass, 27.32, 2)


class TestDetailedStructure(unittest.TestCase):
    """Class to perform unit tests on detailed_structure.py SatelliteStruc objects"""

    def test_initialise(self):
        """Test the initialisation of a SatelliteStruc object"""
        s = SatelliteStruc()
        self.assertEqual(s.mass, 0)
        self.assertEqual(s.volume, 0)
        self.assertEqual(s.surface_area, 0)
        self.assertEqual(s.cross_section, 0)
        self.assertEqual(s.area_moment_of_inertia[0], 0)
        self.assertEqual(s.area_moment_of_inertia[1], 0)
        self.assertEqual(s.mmoi[0], 0)
        self.assertEqual(s.mmoi[1], 0)
        self.assertEqual(s.mmoi[2], 0)
        self.assertEqual(s.t, 0)
        self.assertEqual(s.stress[0], 0)
        self.assertEqual(s.stress[1], 0)
        self.assertEqual(s.stress[2], 0)
        self.assertEqual(s.stress[3], 0)
        self.assertEqual(s.stress[4], 0)
        self.assertEqual(s.vibration[0], 0)
        self.assertEqual(s.vibration[1], 0)
        self.assertEqual(s.vibration[2], 0)

    def test_addMass(self):
        """Test add mass to the system"""
        s = SatelliteStruc()
        s.add_point_element(10, name='1', pos=[0, 0, 0])
        self.assertEqual(s.mass, 10)
        self.assertEqual(len(s.mass_breakdown), 1)
        s.add_point_element(20, name='2', pos=[0, 0, 0])
        self.assertEqual(s.mass, 30)
        self.assertEqual(len(s.mass_breakdown), 2)

    def test_addStructure(self):
        """Test to add a structure to the system"""
        s = SatelliteStruc()
        s.add_structure_sub(1, 1, 1, 1e-2)
        self.assertEqual(s.mass, 168.0)
        self.assertEqual(len(s.mass_breakdown), 1)
        self.assertEqual(s.volume, 1)
        self.assertEqual(s.surface_area, 6)
        self.assertEqual(s.cross_section, 4e-2)
        self.assertAlmostEqual(s.area_moment_of_inertia[0], 0.00667, 4)
        self.assertAlmostEqual(s.area_moment_of_inertia[1], 0.00667, 4)
        self.assertAlmostEqual(s.mmoi[0], 46.667, delta=0.01)
        self.assertAlmostEqual(s.mmoi[1], 46.667, delta=0.01)
        self.assertAlmostEqual(s.mmoi[2], 46.667, delta=0.01)
        self.assertEqual(s.t, 1e-2)

    def test_addOtherElements(self):
        s = SatelliteStruc()
        s.add_structure_sub(1, 1, 1, 1e-2)
        self.assertEqual(len(s.mass_breakdown), 1)
        self.assertEqual(s.mass, 168.0)
        s.add_propellant_tank(1, "Aluminium_6061-T6")
        self.assertEqual(len(s.mass_breakdown), 3)
        self.assertAlmostEqual(s.mass, 198.278, delta=0.01)
        s.add_panels(2, 1, 20)
        self.assertEqual(len(s.mass_breakdown), 4)
        self.assertAlmostEqual(s.mass, 218.278, delta=0.01)

    def test_panelsMMOI(self):
        """Test the MMOI calculation for a panel"""
        s = SatelliteStruc()
        s.add_structure_sub(1, 1, 1, 1e-2)
        s.add_panels(2, 1, 20)
        self.assertAlmostEqual(s.mmoi[0], 135.0, 3)
        self.assertAlmostEqual(s.mmoi[1], 48.33, 2)
        self.assertAlmostEqual(s.mmoi[2], 135.0, 3)

    def test_dictionary(self):
        """Test to add elements through a dictionary"""
        _ = {"Dummy": {
            "subsystem": "Temp1",
            "mass": 1,  # [kg]
            "cg": [0, 0, 0]  # [m]
        },
            "Dummy2": {
                "subsystem": "Temp2",
                "mass": 1,  # [kg]
                "cg": [0, 0, 0]  # [m]
            }}
        s = SatelliteStruc()
        s.add_dictionary(_)
        self.assertEqual(len(s.mass_breakdown), 2)
        self.assertEqual(s.mass, 2)

    def test_MMOI(self):
        """Test the MMOI calculation for the system"""
        s = SatelliteStruc()
        s.add_point_element(10, pos=[1, 0, 0])
        self.assertEqual(s.mmoi[0], 0)
        self.assertEqual(s.mmoi[1], 10)
        self.assertEqual(s.mmoi[2], 10)
        s.add_point_element(10, pos=[0, 1, 0])
        self.assertEqual(s.mmoi[0], 10)
        self.assertEqual(s.mmoi[1], 10)
        self.assertEqual(s.mmoi[2], 20)

    def test_stresses(self):
        """Test stress function"""
        s = SatelliteStruc()
        s.add_structure_sub(1, 1, 1, 1e-2)
        s.calculate_stresses()
        self.assertAlmostEqual(s.stress[0], 1173829.75, 2)
        self.assertAlmostEqual(s.stress[1], 33984000.0, 2)
        self.assertAlmostEqual(s.stress[2], 720779.51, 2)
        self.assertAlmostEqual(s.stress[3], 33984000.0, 2)
        self.assertAlmostEqual(s.stress[4], 1575398.938, 2)

    def test_vibration(self):
        """Test vibration function"""
        s = SatelliteStruc()
        s.add_structure_sub(1, 1, 1, 1e-2)
        s.calculate_vibrations()
        self.assertAlmostEqual(s.vibration[0], 1035.098, 2)
        self.assertAlmostEqual(s.vibration[1], 946.585, 2)
        self.assertAlmostEqual(s.vibration[2], 946.585, 2)

    def test_system(self):
        """Test for compliance of a system"""
        s = SatelliteStruc()
        s.add_structure_sub(1, 1, 1, 1e-4)
        s.add_panels(2, 1, 20)
        s.calculate_stresses()
        s.calculate_vibrations()
        _ = s.compliance()
        self.assertTrue(_[0])
        self.assertTrue(_[1])
        self.assertTrue(_[2])
        self.assertFalse(_[3])
        self.assertTrue(_[4])

if __name__ == '__main__':
    unittest.main()
