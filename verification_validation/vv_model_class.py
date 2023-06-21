"""Script to run unit, module, and system tests on the class Model.
By Nikolaus Ricker"""

# External Libraries
import unittest
import numpy as np
import matplotlib.pyplot as plt

# Local Libraries
import sys
sys.path.append('.')

# Local objects
from mission_design import Satellite, Tower, Lagrange, FixPoint, OrbitPlane, Model

# Constants
r_moon = 1.737e6  # m
miu_moon = 4.9048695e12  # m^3/s^2


# -----------------------------------------------------------------------------
# General Functions unittests
class TestNode(unittest.TestCase):
    """Class performing unit tests on the node object. 
    All nodes (Satellite, Tower, Lagrange, FixPoint) share same functions."""

    def test_r_size(self):
        """Test vector r has correct size"""
        s = Satellite()
        self.assertEqual(len(s.r), 3)
    
    def test_range(self):
        """Tests that maximum range is calculated correctly"""
        s = Satellite(a=2e6, elevation=0)
        self.assertAlmostEqual(s.range, 991378.3, places=1)

    def test_range_limit(self):
        """Tests range can only cover half of the Moon at most"""
        s = Satellite(a=1e17, elevation=0)
        self.assertAlmostEqual(s.range**2, s.a**2+r_moon**2, places=1)

    def test_range_elevation(self):
        """Tests range is reduced by elevation"""
        s = Satellite(a=2e6, elevation=10)
        self.assertAlmostEqual(s.range, 986101.26, places=1)
    
    def test_isInView(self):
        """Tests that isInView returns True if satellite is in view"""
        s1 = Satellite(a=2e6)
        self.assertTrue(s1.isInView([1.9e6,0,0]))
        s2 = Satellite(a=2e6)
        self.assertFalse(s2.isInView([1e6,0,0]))

# Satellite unittests
class TestSatellite(unittest.TestCase):
    """Class performing unit tests on the satellite object"""

    def test_a(self):
        """Checks semi major axis is larger than pericenter"""
        with self.assertRaises(ValueError):
            s = Satellite(a=1000)
        with self.assertRaises(ValueError):
            s = Satellite(e=0.5)
            print(s.r, s.a)

    def test_e(self):
        """Checks eccentricity is positive"""
        with self.assertRaises(ValueError):
            s = Satellite(e=-0.5)
        
    def test_i(self):
        """Checks inclination is between 0 and 180"""
        with self.assertRaises(ValueError):
            s = Satellite(i=-5)
        with self.assertRaises(ValueError):
            s = Satellite(i=185)

    def test_w(self):
        """Checks argument of periapsis is between 0 and 360"""
        with self.assertRaises(ValueError):
            s = Satellite(w=-5)
        with self.assertRaises(ValueError):
            s = Satellite(w=365)
    
    def test_Omega(self):
        """Checks longitude of ascending node is between 0 and 360"""
        with self.assertRaises(ValueError):
            s = Satellite(Omega=-5)
        with self.assertRaises(ValueError):
            s = Satellite(Omega=365)
    
    def test_nu(self):
        """Checks that true anomaly restarts every 360°"""
        s = Satellite(nu=365)
        self.assertEqual(s.nu, np.deg2rad(5))

    def test_elevation(self):
        """Checks that elevation is positive"""
        with self.assertRaises(ValueError):
            s = Satellite(elevation=-5)

    def test_r_conversion(self):
        """Tests conversion from Keplerian to Cartesian is correct"""
        s1 = Satellite()
        r_test = np.array([s1.a,0,0])
        self.assertTrue(np.allclose(s1.r, r_test, atol=1e-2))
        s2 = Satellite(a=2e6, e=0.1)
        r_test = np.array([s2.a*(1-s2.e),0,0])
        self.assertTrue(np.allclose(s2.r, r_test, atol=1e-2))
        s3 = Satellite(a=2e6, w=90)
        r_test = np.array([0,s3.a,0])
        self.assertTrue(np.allclose(s3.r, r_test, atol=1e-2))
        s4 = Satellite(a=2e6, w=45)
        r_test = np.array([np.sqrt(s4.a**2/2),np.sqrt(s4.a**2/2),0])
        self.assertTrue(np.allclose(s4.r, r_test, atol=1e-2))

# Tower unittests
class TestTower(unittest.TestCase):
    """Class performing unit tests on the tower object"""

    def test_phi(self):
        """Checks phi is between -180 and 180"""
        with self.assertRaises(ValueError):
            t = Tower(phi=-185)
        with self.assertRaises(ValueError):
            t = Tower(phi=185)

    def test_theta(self):
        """Checks theta is between -90 and 90"""
        with self.assertRaises(ValueError):
            t = Tower(theta=-95)
        with self.assertRaises(ValueError):
            t = Tower(theta=95)

    def test_h(self):
        """Checks height is positive"""
        with self.assertRaises(ValueError):
            t = Tower(height=-5)

    def test_r_conversion(self):
        """Tests conversion from Polar to Cartesian is correct"""
        t1 = Tower()
        r_test = np.array([r_moon,0,0])
        self.assertTrue(np.allclose(t1.r, r_test, atol=1e-2))
        t2 = Tower(phi=90)
        r_test = np.array([0,r_moon,0])
        self.assertTrue(np.allclose(t2.r, r_test, atol=1e-2))
        t3 = Tower(phi=180)
        r_test = np.array([-r_moon,0,0])
        self.assertTrue(np.allclose(t3.r, r_test, atol=1e-2))
        t4 = Tower(theta=90)
        r_test = np.array([0,0,r_moon])
        self.assertTrue(np.allclose(t4.r, r_test, atol=1e-2))

# Lagrange unittests
class TestLagrange(unittest.TestCase):
    """Class performing unit tests on the Lagrange object"""

    def test_valid_input(self):
        """Checks that valid input is accepted"""
        with self.assertRaises(ValueError):
            l = Lagrange('x')
    
    def test_r(self):
        """Tests correct distance from center of mass"""
        l1 = Lagrange('L1')
        self.assertAlmostEqual(np.linalg.norm(l1.r), 66e6, places=1)
        l2 = Lagrange('L2')
        self.assertAlmostEqual(np.linalg.norm(l2.r), 66e6, places=1)

# FixPoint unittests
class TestFixPoint(unittest.TestCase):
    """Class performing unit tests on the FixPoint object"""

    def test_valid_input(self):
        """Checks that valid input is accepted"""
        with self.assertRaises(ValueError):
            f = FixPoint([1,2])
    
    def test_valid_size(self):
        """Checks that three values are given"""
        with self.assertRaises(ValueError):
            f = FixPoint([1,2])

# OrbitPlane unittests
class TestOrbitPlane(unittest.TestCase):
    """Class performing unit tests on the OrbitPlane object"""

    def test_a(self):
        """Checks semi major axis is larger than pericenter"""
        with self.assertRaises(ValueError):
            s = OrbitPlane(a=1000)
        with self.assertRaises(ValueError):
            s = OrbitPlane(e=0.5)
            print(s.r, s.a)

    def test_e(self):
        """Checks eccentricity is positive"""
        with self.assertRaises(ValueError):
            s = OrbitPlane(e=-0.5)
        
    def test_i(self):
        """Checks inclination is between 0 and 180"""
        with self.assertRaises(ValueError):
            s = OrbitPlane(i=-5)
        with self.assertRaises(ValueError):
            s = OrbitPlane(i=185)

    def test_w(self):
        """Checks argument of periapsis is between 0 and 360"""
        with self.assertRaises(ValueError):
            s = OrbitPlane(w=-5)
        with self.assertRaises(ValueError):
            s = OrbitPlane(w=365)
    
    def test_Omega(self):
        """Checks longitude of ascending node is between 0 and 360"""
        with self.assertRaises(ValueError):
            s = OrbitPlane(Omega=-5)
        with self.assertRaises(ValueError):
            s = OrbitPlane(Omega=365)

    def test_elevation(self):
        """Checks that elevation is positive"""
        with self.assertRaises(ValueError):
            s = OrbitPlane(elevation=-5)

    def test_nsat(self):
        """Checks that number of satellites is positive"""
        with self.assertRaises(ValueError):
            s = OrbitPlane(n_sat=-5)

    def test_creation_satellites(self):
        """Checks that satellites are created correctly"""
        s1 = OrbitPlane()
        self.assertEqual(len(s1.satellites), 1)
        s2 = OrbitPlane(n_sat=5)
        self.assertEqual(len(s2.satellites), 5)
    
    def test_individual_satellites(self):
        """Checks that one satellite is created correctly"""
        s1 = OrbitPlane()
        self.assertEqual(s1.satellites[0].a, s1.a)
        self.assertEqual(s1.satellites[0].e, s1.e)
        self.assertEqual(s1.satellites[0].i, s1.i)
        self.assertEqual(s1.satellites[0].w, s1.w)
        self.assertEqual(s1.satellites[0].Omega, s1.Omega)
        self.assertEqual(s1.satellites[0].elevation, np.deg2rad(s1.elevation))
        
    def test_satellite_distribution(self):
        """Checks that all satellites are distributed correctly"""
        s1 = OrbitPlane(n_sat=5)
        self.assertEqual(s1.satellites[0].nu, 0)
        self.assertEqual(s1.satellites[1].nu, np.deg2rad(72))
        self.assertEqual(s1.satellites[2].nu, np.deg2rad(144))
        self.assertEqual(s1.satellites[3].nu, np.deg2rad(216))
        self.assertEqual(s1.satellites[4].nu, np.deg2rad(288))

    def test_rel_dist_satellites(self):
        """Checks that satellites are distributed correctly relative to each other"""
        s1 = OrbitPlane(a=2e6, e=0, n_sat=6)
        self.assertAlmostEqual(2e6, np.linalg.norm(s1.satellites[0].r-s1.satellites[1].r))

# Model unittests
class TestModel(unittest.TestCase):
    """Class performing unit tests on the Model object"""

    def test_resolution(self):
        """Checks that resolution is positive"""
        with self.assertRaises(ValueError):
            m = Model(resolution=-5)

    def test_addExistingOrbitPlane(self):
        """Checks that an existing OrbitPlane is added correctly"""
        m = Model()
        o = OrbitPlane()
        m.addExistingOrbitPlane(o)
        self.assertEqual(m.n_orbit_planes, 1)
        self.assertEqual(m.orbit_planes[0], o)
        self.assertEqual(m.n_sat, 1)

    def test_addOrbitPlane(self):
        """Checks that a new OrbitPlane is added correctly"""
        m = Model()
        m.addOrbitPlane()
        self.assertEqual(m.n_orbit_planes, 1)
        self.assertEqual(m.n_sat, 1)

    def test_addTower(self):
        """Checks that a new Tower is added correctly"""
        m = Model()
        m.addTower()
        self.assertEqual(m.n_orbit_planes, 0)
        self.assertEqual(m.n_sat, 1)
        
    def test_addSatellite(self):
        """Checks that a new Satellite is added correctly"""
        m = Model()
        m.addSatellite()
        self.assertEqual(m.n_orbit_planes, 0)
        self.assertEqual(m.n_sat, 1)

    def test_addLagrange(self):
        """Checks that a new Lagrange is added correctly"""
        m = Model()
        m.addLagrange('L1')
        self.assertEqual(m.n_sat, 1)
        m.addLagrange('L2')
        self.assertEqual(m.n_sat, 2)
        self.assertEqual(m.n_orbit_planes, 0)
    
    def test_addFixPoint(self):
        """Checks that a new FixPoint is added correctly"""
        m = Model()
        m.addFixPoint(r=[2e7,0,0])
        self.assertEqual(m.n_orbit_planes, 0)
        self.assertEqual(m.n_sat, 1)

    def test_addExistingModule(self):
        """Checks that an existing Module is added correctly"""
        m = Model()
        m.addExistingModule(Satellite())
        self.assertEqual(m.n_sat, 1)
        self.assertTrue(isinstance(m.modules[0], Satellite))

    def test_addSymmetricalPlanes(self):
        """Checks that symmetrical planes are added correctly"""
        m = Model()
        m.addSymmetricalPlanes(a=r_moon+1, n_planes=2)
        self.assertEqual(m.n_orbit_planes, 2)
        self.assertEqual(m.n_sat, 2)

    def test_moonCreation(self):
        """Checks that moon is created correctly"""
        m = Model(resolution=100)
        self.assertAlmostEqual(len(m.moon), 100**2)

    def test_Coverage(self):
        """Checks that coverage is only ran if there are modules"""
        m = Model(resolution=100)
        with self.assertRaises(ValueError):
            m.setCoverage()

    def test_plotCoverage(self):
        """Checks that coverage is plotted only if there are modules"""
        m = Model(resolution=100)
        with self.assertRaises(ValueError):
            m.plotCoverage()

# Model module tests
def plotErrorMesh():
    """Plot error as a function of mesh resolution"""
    coverage = []
    res = [10, 50, 100, 250, 500, 750, 1000, 1500]
    for i in res:
        m = Model(resolution=i)
        m.addSatellite(a=2e6)
        m.setCoverage()
        coverage.append(np.count_nonzero(m.mod_inView)/m.resolution**2)

    coverage = np.array(coverage)
    plt.plot(np.array(res)**2, np.abs(coverage-coverage[-1])/coverage[-1]*100)
    plt.xlabel('Mesh resolution [pixels]')
    plt.ylabel('Discretisation Error [%]')
    plt.show()

def plotChangeInA():
    """Plot change in covered area as a function of a"""
    a = np.linspace(2e6, 1.5e8, 100)
    areaPerPixel = (4*np.pi*(r_moon**2))/100**2
    area = []
    for i in a:
        m = Model(resolution=100)
        m.addSatellite(a=i)
        m.setCoverage()
        area.append(np.count_nonzero(m.mod_inView)/m.resolution**2)

    plt.plot(a, area)
    plt.xlabel('Semi-Major Axis [m]')
    plt.ylabel('Coverage [%]')
    plt.show()

def plotChangeInE():
    """Plot change in covered area as a function of e"""
    e = np.linspace(0, 0.9, 100)
    area = []
    for i in e:
        m = Model(resolution=100)
        m.addSatellite(a=3e6, e=i)
        m.setCoverage()
        area.append(np.count_nonzero(m.mod_inView)/m.resolution**2)

    plt.plot(e, area)
    plt.xlabel('Eccentricity')
    plt.ylabel('Coverage [%]')
    plt.show()

def plotChangeInI():
    """Plot change in covered area as a function of i"""
    i = np.linspace(0, 180, 100)
    area = []
    for j in i:
        m = Model(resolution=100)
        m.addSatellite(a=3e6, i=j)
        m.setCoverage()
        area.append(np.count_nonzero(m.mod_inView)/m.resolution**2)

    plt.plot(i, area)
    plt.xlabel('Inclination [deg]')
    plt.ylabel('Coverage [%]')
    plt.show()

def plotChangeInHeight():
    """Plot change in covered area as a function of height"""
    h = np.linspace(0, 1e3, 100)
    area = []
    for i in h:
        m = Model(resolution=100)
        m.addTower(h=i)
        m.setCoverage()
        area.append(np.count_nonzero(m.mod_inView)/m.resolution**2)

    plt.plot(h, area)
    plt.xlabel('Height [m]')
    plt.ylabel('Coverage [%]')
    plt.show()

class TestExtreme(unittest.TestCase):
    """Class to test extreme value cases"""

    def testLessThanRadius(self):
        """Checks that error is raised if something is positioned inside the moon"""
        m = Model()
        with self.assertRaises(ValueError):
            m.addSatellite(a=r_moon-1)
        with self.assertRaises(ValueError):
            m.addFixPoint(r=[0,0,0])
        with self.assertRaises(ValueError):
            m.addOrbitPlane(a=r_moon-1)

    def testCoverageCloseToMoon(self):
        """Checks system behaviour when coverage is close to moon"""
        m = Model(resolution=100)
        m.addSatellite(a=r_moon+1)
        m.addTower(h=10)
        m.setCoverage()
        self.assertEqual(np.count_nonzero(m.mod_inView), 0)

    def testCoverageFarFromMoon(self):
        """Checks system behaviour when coverage is far from moon"""
        m = Model(resolution=100)
        m.addSatellite(a=1e12)
        m.setCoverage()
        self.assertAlmostEqual(np.count_nonzero(m.mod_inView), m.resolution**2/2, delta=1.1e2)

# Model system tests
def plotFolds():
    """Checks that the system works with folds"""
    m = Model(resolution=100)
    m.addLagrange('L1')
    m.addOrbitPlane(a=3e6, i=45, n_sat=4)
    m.plotCoverage()

# Validation
def plotModelFromSOC1():
    """Plot model from Streets of Coverage"""
    m = Model(resolution=100)
    m.addSymmetricalPlanes(dist_type=1, a=r_moon+25e6, i=13.7, e=0, w=0, n_planes=2, n_sat_per_plane=10)
    m.plotCoverage()

def plotModelFromSOC2():
    """Plot model from Streets of Coverage"""
    m = Model(resolution=100)
    m.addSymmetricalPlanes(dist_type=1, a=r_moon+10e6, i=18.4, e=0, w=0, n_planes=2, n_sat_per_plane=11)
    m.plotCoverage()

def plotModelFromSOC3():
    """Plot model from Streets of Coverage"""
    m = Model(resolution=100)
    m.addSymmetricalPlanes(dist_type=1, a=r_moon+50e5, i=24.7, e=0, w=0, n_planes=3, n_sat_per_plane=12)
    m.plotCoverage()

def plotModelFromSOC4():
    """Plot model from Streets of Coverage"""
    m = Model(resolution=100)
    m.addSymmetricalPlanes(dist_type=1, a=r_moon+25e5, i=33.8, e=0, w=0, n_planes=3, n_sat_per_plane=15)
    m.plotCoverage()

def plotModelFromPaper1():
    """Plot model from Paper"""
    m = Model(resolution=100)
    m.addSymmetricalPlanes(a=24572000, i=58.69, e=0, w=22.9, n_planes=6, n_sat_per_plane=4)
    m.plotCoverage()

def plotModelFromPaper2():
    """Plot model from Paper"""
    m = Model(resolution=100)
    m.addSymmetricalPlanes(a=24589000, i=66.5, e=0.0139, w=132.95, n_planes=3, n_sat_per_plane=8, f=0.4641)
    m.plotCoverage()

if __name__ == '__main__':
    unittest.main()
    # plotErrorMesh()
    # plotChangeInA()
    # plotFolds()
    # plotModelFromSOC1()
    # plotModelFromSOC2()
    # plotModelFromSOC3()
    # plotModelFromSOC4()
    # plotModelFromPaper2()