"""
File with all necessary classes for modelling coverage at a static time.
It is a fixed Moon model, with a rough coverage of the elements orbiting the satellite. \n
The file is divided into five classes. Each class adds to the final class Model, encapsulating the other four.
The classes are:
    - Satellite: Class to create a satellite in a given orbit. \n
    - Tower: Class to create a fixed tower on the moon surface with a given height. \n
    - Lagrange: Class to position a satellite in a Lagrange point. \n
    - FixPoint: Class to position a satellite in a fix point in space. \n
    - OrbitPlane: Class to create a line of satellites on a plane. \n

Class Model creates the environment and calculates the coverage of the satellites. It can add existing elements or create new ones.
It measures coverage, and plots the results.

By Nikolaus Ricker
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
from tudatpy.kernel.astro import element_conversion

# Constants
r_moon = 1.737e6  # m
r_orbit = r_moon + 200e3  # m
miu_moon = 4.9048695e12  # m^3/s^2


class Satellite:
    """Class to define a satellite with its position and cone of view around the Moon."""

    def __init__(self, a=r_moon, e=0, i=0, w=0, Omega=0, nu=0, elevation=10, shift=0, id=0):
        """Initialise the satellite with its Keplerian elements and calculate its position.
        :param a: semi-major axis [m]
        :param e: eccentricity [-]
        :param i: inclination [deg]
        :param w: argument of periapsis [deg]
        :param Omega: longitude of ascending node [deg]
        :param nu: true anomaly [deg]
        :param elevation: (optional) elevation angle [deg]
        :param shift: (optional) shift of the cone of view [deg]
        """
        self.id = id
        self.range = None
        self.e = e
        self.a = a
        self.i = i
        self.w = w
        self.Omega = Omega
        self.shift = shift
        self.nu = nu + shift
        self.r = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=miu_moon,
            semi_major_axis=self.a,
            eccentricity=self.e,
            inclination=self.i,
            argument_of_periapsis=self.w,
            longitude_of_ascending_node=self.Omega,
            true_anomaly=self.nu
        )[:3]
        self.elevation = elevation
        self.range = self.setRange()

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value >= r_moon / (1 - self.e):
            self._a = value
            if self.range is not None:
                self.range = self.setRange()
        else:
            raise ValueError("Pericenter must be larger than Moon's radius.")

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, value):
        if value >= 0 and value < 1:
            self._e = value
            if self.range is not None:
                self.range = self.setRange()
        else:
            raise ValueError("Eccentricity must be between 0 and 1.")

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, value):
        if value >= 0 and value <= 180:
            self._i = np.deg2rad(value)
            if self.range is not None:
                self.range = self.setRange()
        else:
            raise ValueError("Inclination must be between 0 and 180°.")

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        if value >= 0 and value <= 360:
            self._w = np.deg2rad(value)
            if self.range is not None:
                self.range = self.setRange()
        else:
            raise ValueError("Argument of periapsis must be between 0 and 360°.")

    @property
    def Omega(self):
        return self._Omega

    @Omega.setter
    def Omega(self, value):
        if value >= 0 and value <= 360:
            self._Omega = np.deg2rad(value)
            if self.range is not None:
                self.range = self.setRange()
        else:
            raise ValueError("Longitude of ascending node must be between 0 and 360°.")

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        value = value % 360
        self._nu = np.deg2rad(value)
        if self.range is not None:
            self.range = self.setRange()

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        if value >= 0 and value <= 90:
            self._elevation = np.deg2rad(value)
            if self.range is not None:
                self.range = self.setRange()
        else:
            raise ValueError("Elevation must be between 0 and 90°.")

    def setRange(self):
        """Calculate the maximum range achievable by a satellite.
        :param elevation: (optional) elevation angle [deg]
        """
        alpha = np.deg2rad(self.elevation + 90)  # angle between the cone and the horizontal plane
        r_norm = np.linalg.norm(self.r)  # height of the cone
        h_max = 0.5 * (2 * r_moon * np.cos(alpha) + np.sqrt(2) * np.sqrt(
            2 * r_norm ** 2 - r_moon ** 2 + r_moon ** 2 * np.cos(2 * alpha)))
        if h_max > np.sqrt(r_norm ** 2 + r_moon ** 2):
            h_max = np.sqrt(r_norm ** 2 + r_moon ** 2)
        return h_max

    def isInView(self, target):
        """Check if a target is in view of the satellite.
        :param target: target position "Array 3D" [m]
        """
        if self.range is None:
            self.range = self.range()
        if np.linalg.norm(target - self.r) <= self.range:
            return True
        else:
            return False

    def getParams(self):
        """Return the Keplerian elements of the satellite."""
        return self.a, self.e, self.i, self.w, self.Omega, self.nu

    def __repr__(self):
        return f"Satellite(id={self.id}, a={self.a}, e={self.e}, i={self.i}, w={self.w}, Omega={self.Omega}, nu={self.nu})"


class Tower:
    """Class to define a communication tower."""

    def __init__(self, phi=0, theta=0, height=0, elevation=10):
        """Initialise the ground station with its position.
        :param phi: longitude [deg]
        :param theta: latitude [deg]
        :param height: height [m]
        """
        self.phi = phi
        self.theta = theta
        self.height = height
        self.elevation = elevation
        self.r = self.getcoords(self.phi, self.theta, self.height)
        self.range = self.setRange()

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        if value >= -180 and value <= 180:
            self._phi = value
        else:
            raise ValueError("Longitude must be between -180 and 180°.")

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if value >= -90 and value <= 90:
            self._theta = value
        else:
            raise ValueError("Latitude must be between -90 and 90°.")

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value >= 0:
            self._height = value
        else:
            raise ValueError("Height must be positive.")

    def getcoords(self, phi, theta, height):
        """Calculate the position of the tower.
        :param phi: longitude [deg]
        :param theta: latitude [deg]
        :param height: height above sea level [m]
        """
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
        return np.array(
            [(r_moon + height) * np.cos(theta) * np.cos(phi), (r_moon + height) * np.cos(theta) * np.sin(phi),
             (r_moon + height) * np.sin(theta)])

    def setRange(self):
        """Calculate the maximum range achievable by a satellite.
        :param elevation: (optional) elevation angle [deg]
        """
        alpha = np.deg2rad(self.elevation + 90)  # angle between the cone and the horizontal plane
        r_norm = np.linalg.norm(self.r)  # height of the cone
        h_max = 0.5 * (2 * r_moon * np.cos(alpha) + np.sqrt(2) * np.sqrt(
            2 * r_norm ** 2 - r_moon ** 2 + r_moon ** 2 * np.cos(2 * alpha)))
        if h_max > np.sqrt(r_norm ** 2 + r_moon ** 2):
            h_max = np.sqrt(r_norm ** 2 + r_moon ** 2)
        return h_max

    def isInView(self, target):
        """Check if a target is in view of the satellite.
        :param target: target position [m]
        """
        if self.range is None:
            self.range = self.range()
        if np.linalg.norm(target - self.r) <= self.range:
            return True
        else:
            return False


class Lagrange:
    def __init__(self, l_point, elevation=10):
        """Initialise the Lagrange point with its position.
        :param l_point: Lagrange point
        """
        self.l_point = l_point
        self.elevation = elevation

        if self.l_point == "L1":
            self.r = np.array([0, -66e6 * np.cos(np.deg2rad(6.68)), -66e6 * np.sin(np.deg2rad(6.68))])

        if self.l_point == "L2":
            self.r = np.array([0, 66e6 * np.cos(np.deg2rad(6.68)), 66e6 * np.sin(np.deg2rad(6.68))])

        self.range = self.setRange()

    @property
    def l_point(self):
        return self._l_point

    @l_point.setter
    def l_point(self, value):
        if value == "L1" or value == "L2":
            self._l_point = value
        else:
            raise ValueError("Lagrange point must be either 'L1' or 'L2'.")

    def setRange(self):
        """Calculate the maximum range achievable by a satellite.
        """
        alpha = np.deg2rad(self.elevation + 90)  # angle between the cone and the horizontal plane
        r_norm = np.linalg.norm(self.r)  # height of the cone
        h_max = 0.5 * (2 * r_moon * np.cos(alpha) + np.sqrt(2) * np.sqrt(
            2 * r_norm ** 2 - r_moon ** 2 + r_moon ** 2 * np.cos(2 * alpha)))
        if h_max > np.sqrt(r_norm ** 2 + r_moon ** 2):
            h_max = np.sqrt(r_norm ** 2 + r_moon ** 2)
        return h_max

    def isInView(self, target):
        """Check if a target is in view of the satellite.
        :param target: target position [m]
        """
        if self.range is None:
            self.range = self.range()
        if np.linalg.norm(target - self.r) <= self.range:
            return True
        else:
            return False

    def __repr__(self):
        return f"Lagrange point {self.l_point}"


class FixPoint:
    def __init__(self, r, elevation=10):
        """Initialise the fixed point with its position.
        :param r: fixed point position. Must be array (x, y, z) [m]
        """
        self.r = r
        self.elevation = elevation
        self.range = self.setRange()

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        value = np.array(value)
        if len(value) == 3:
            if np.linalg.norm(value) >= r_moon:
                self._r = value
            else:
                raise ValueError("Fixed point must be outside the Moon.")
        else:
            raise ValueError("Position must be a 3D vector.")

    def setRange(self):
        """Calculate the maximum range achievable by a satellite.
        :param elevation: (optional) elevation angle [deg]
        """
        alpha = np.deg2rad(self.elevation + 90)  # angle between the cone and the horizontal plane
        r_norm = np.linalg.norm(self.r)  # height of the cone
        h_max = 0.5 * (2 * r_moon * np.cos(alpha) + np.sqrt(2) * np.sqrt(
            2 * r_norm ** 2 - r_moon ** 2 + r_moon ** 2 * np.cos(2 * alpha)))
        if h_max > np.sqrt(r_norm ** 2 + r_moon ** 2):
            h_max = np.sqrt(r_norm ** 2 + r_moon ** 2)
        return h_max

    def isInView(self, target):
        """Check if a target is in view of the satellite.
        :param target: target position [m]
        """
        if self.range is None:
            self.range = self.range()
        if np.linalg.norm(target - self.r) <= self.range:
            return True
        else:
            return False

    def __repr__(self):
        return f"Fixed point at {self.r}"


class OrbitPlane:
    """Class to define an orbit plane, along with the satellites in it."""

    def __init__(self, a=r_moon, e=0, i=0, w=0, Omega=0, n_sat=1, elevation=10, shift=0, id_start=0):
        """Initialise the orbit plane with its Keplerian elements and calculate the positions of the satellites.
        :param a: semi-major axis [m]
        :param e: eccentricity [-]
        :param i: inclination [deg]
        :param w: argument of periapsis [deg]
        :param Omega: longitude of ascending node [deg]
        :param n_sat: number of satellites in the orbit plane
        :param elevation: (optional) elevation angle [deg]
        """
        self.id_start = id_start
        self.satellites = []
        self.e = e
        self.a = a
        self.i = i
        self.w = w
        self.Omega = Omega
        self.elevation = elevation
        self.shift = shift
        self.n_sat = n_sat
        self.satellites = self.createSatellites()

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value >= r_moon / (1 - self.e):
            self._a = value
            if self.satellites:
                self.satellites = self.createSatellites()
        else:
            raise ValueError("Pericenter must be larger than the Moon's radius.")

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, value):
        if value >= 0 and value < 1:
            self._e = value
            if self.satellites:
                self.satellites = self.createSatellites()
        else:
            raise ValueError("Eccentricity must be between 0 and 1.")

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, value):
        if value >= 0 and value <= 180:
            self._i = value
            if self.satellites:
                self.satellites = self.createSatellites()
        else:
            raise ValueError("Inclination must be between 0 and 180°.")

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        if value >= 0 and value <= 360:
            self._w = value
            if self.satellites:
                self.satellites = self.createSatellites()
        else:
            raise ValueError("Argument of periapsis must be between 0 and 360°.")

    @property
    def Omega(self):
        return self._Omega

    @Omega.setter
    def Omega(self, value):
        if value >= 0 and value <= 360:
            self._Omega = value
            if self.satellites:
                self.satellites = self.createSatellites()
        else:
            raise ValueError("Longitude of ascending node must be between 0 and 360°.")

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        if value >= 0 and value <= 90:
            self._elevation = value
            if self.satellites:
                self.satellites = self.createSatellites()
        else:
            raise ValueError("Elevation must be between 0 and 90°.")

    @property
    def n_sat(self):
        return self._n_sat

    @n_sat.setter
    def n_sat(self, value):
        if value > 0:
            self._n_sat = value
            if self.satellites:
                self.satellites = self.createSatellites()
        else:
            raise ValueError("Number of satellites must be positive.")

    def createSatellites(self):
        """Create the satellites in the orbit plane."""
        satellites = []
        for n in range(self.n_sat):
            satellites.append(
                Satellite(self.a, self.e, self.i, self.w, self.Omega, 360 / self.n_sat * n, self.elevation, self.shift,
                          id=self.id_start + n))
        return satellites

    def relDistSatellites(self):
        """Calculate the relative distance between the satellites in the orbit plane."""
        rel_dist = []
        for i in range(self.n_sat):
            rel_dist.append(np.linalg.norm(self.satellites[i].r - self.satellites[(i + 1) % self.n_sat].r))
        return rel_dist


class Model:
    """Class to define the model of the satellite constellation."""

    def __init__(self, resolution=100):
        """Initialise environment.
        :param resolution: (optional) resolution of the model."""
        self.orbit_planes = []
        self.modules = []
        self.n_sat = 0
        self.n_orbit_planes = 0
        self.resolution = resolution

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        if value > 0:
            self._resolution = value
            self.moon = self.createMoon(value)
            self.mod_inView = np.zeros(len(self.moon))
            self.mod_inView_obj = {i: [] for i in range(len(self.moon))}

            self.orbits = self.createOrbits(value)
            self.mod_inView_orbits = np.zeros(len(self.orbits))
            self.mod_inView_obj_orbits = {i: [] for i in range(len(self.orbits))}
        else:
            raise ValueError("Resolution must be positive.")

    def resetModel(self):
        self.orbit_planes = []
        self.modules = []
        self.n_sat = 0
        self.n_orbit_planes = 0
        self.mod_inView = np.zeros(len(self.moon))
        self.mod_inView_obj = {i: [] for i in range(len(self.moon))}

        self.mod_inView_orbits = np.zeros(len(self.orbits))
        self.mod_inView_obj_orbits = {i: [] for i in range(len(self.orbits))}

    def addExistingOrbitPlane(self, orbit):
        """Add an existing orbit plane to the model.
        :param orbit: orbit plane (Object)
        """
        self.orbit_planes.append(orbit)
        self.modules.append(orbit.satellites)
        self.n_sat += orbit.n_sat
        self.n_orbit_planes += 1

    def addOrbitPlane(self, a=r_moon, e=0, i=0, w=0, Omega=0, n_sat=1, shift=0, elevation=10, id_start=0):
        """Add an orbit plane to the model.
        :param a: semi-major axis [m]
        :param e: eccentricity [-]
        :param i: inclination [deg]
        :param w: argument of periapsis [deg]
        :param Omega: longitude of ascending node [deg]
        :param n_sat: number of satellites in the orbit plane
        :param elevation: (optional) elevation angle [deg]
        """
        self.id_start = id_start
        n_orbit = OrbitPlane(a, e, i, w, Omega, n_sat, elevation, shift)
        self.orbit_planes.append(n_orbit)
        for s in n_orbit.satellites:
            self.modules.append(s)
        self.n_sat += n_sat
        self.n_orbit_planes += 1

    def addExistingModule(self, module):
        """Add an existing satellite or Tower to the model.
        :param module: satellite/tower (Object)
        """
        self.modules.append(module)
        self.n_sat += 1

    def addSatellite(self, a=r_moon, e=0, i=0, w=0, Omega=0, nu=0, shift=0, elevation=10, id=0):
        """Add a satellite to the model.
        :param a: semi-major axis [m]
        :param e: eccentricity [-]
        :param i: inclination [deg]
        :param w: argument of periapsis [deg]
        :param Omega: longitude of ascending node [deg]
        :param nu: true anomaly [deg]
        :param elevation: (optional) elevation angle [deg]
        :param shift: (optional) shift of the satellite in the orbit plane [deg]
        """
        n_sat = Satellite(a, e, i, w, Omega, nu, elevation, shift)
        self.modules.append(n_sat)
        self.n_sat += 1

    def addTower(self, phi=0, theta=0, h=0):
        n_tower = Tower(phi, theta, h)
        self.modules.append(n_tower)
        self.n_sat += 1

    def addLagrange(self, l_point='L1'):
        n_point = Lagrange(l_point)
        self.modules.append(n_point)
        self.n_sat += 1

    def addFixPoint(self, r, elevation=10):
        n_point = FixPoint(r, elevation)
        self.modules.append(n_point)
        self.n_sat += 1

    def addSymmetricalPlanes(self, a=r_moon, e=0, i=0, w=0, n_planes=1, n_sat_per_plane=1, dist_type=0, f=0, shift=0,
                             elevation=10):
        """Add  symmetrical orbit planes to the model.
        :param a: semi-major axis [m]
        :param e: eccentricity [-]
        :param i: inclination [deg]
        :param w: argument of periapsis [deg]
        :param n_planes: number of orbit planes
        :param n_sat_per_plane: number of satellites in each orbit plane
        :param dist_type: (optional) type of distribution of the orbit planes. 0- Divide over 180°, 1- Divide over 360°
        :param f: (optional) phase between planes (0-1)
        :param shift: (optional) shift of the satellites in the orbit plane [deg]
        :param elevation: (optional) elevation angle [deg]
        """
        if dist_type == 0:
            for n in range(n_planes):
                self.addOrbitPlane(a, e, i, w, 180 / n_planes * n, n_sat_per_plane, shift + f * 180 * n, elevation,
                                   id_start=n * n_sat_per_plane)
        elif dist_type == 1:
            for n in range(n_planes):
                self.addOrbitPlane(a, e, i, w, 360 / n_planes * n, n_sat_per_plane, shift + f * 360 * n, elevation,
                                   id_start=n * n_sat_per_plane)
        else:
            raise ValueError("Invalid distribution type.")

    def createMoon(self, resolution=100):
        """Add the Moon to the model.
        :param resolution: (optional) resolution of the meshgrid"""
        phi = np.linspace(0, 2 * np.pi, resolution)
        theta = np.linspace(0, np.pi, resolution)

        sphere_points = []
        for i in phi:
            for j in theta:
                x = r_moon * np.sin(j) * np.cos(i)
                y = r_moon * np.sin(j) * np.sin(i)
                z = r_moon * np.cos(j)
                sphere_points.append([x, y, z])
        return sphere_points

    def createOrbits(self, resolution=100):
        phi = np.linspace(0, 2 * np.pi, resolution)
        theta = np.linspace(0, np.pi, resolution)
        r = np.linspace(r_moon+resolution, r_orbit, resolution)

        orbit_points = []
        for m in r:
            for i in phi:
                for j in theta:
                    x = m * np.sin(j) * np.cos(i)
                    y = m * np.sin(j) * np.sin(i)
                    z = m * np.cos(j)
                    orbit_points.append([x, y, z])
        return orbit_points

    def setCoverage(self):
        """Set the coverage of the modules."""
        if self.modules == []:
            raise ValueError("No modules in the model.")
        for mod in self.modules:
            for i, point in enumerate(self.moon):
                if mod.isInView(point):
                    self.mod_inView[i] += 1
                    self.mod_inView_obj[i].append(mod)

    def setCoverageOrbits(self):
        if self.modules == []:
            raise ValueError("No modules in the model.")
        for mod in self.modules:
            for i, point in enumerate(self.orbits):
                if mod.isInView(point):
                    self.mod_inView_orbits[i] += 1
                    self.mod_inView_obj_orbits[i].append(mod)

    def plotCoverage(self):
        """Plot the coverage of the satellites."""
        if self.mod_inView.all() == 0:
            self.setCoverage()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot Moon
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        # Create a meshgrid of phi and theta values
        phi, theta = np.meshgrid(phi, theta)
        # Calculate the x, y, and z coordinates for each point on the sphere
        xM = r_moon * np.sin(theta) * np.cos(phi)
        yM = r_moon * np.sin(theta) * np.sin(phi)
        zM = r_moon * np.cos(theta)

        ax.plot_surface(xM, yM, zM, color='grey', alpha=0.2)

        # Plot modules
        mod_pos = [mod.r for mod in self.modules]
        ax.scatter(*zip(*mod_pos), s=10)

        # Plot satellites in view
        color_map = cm.ScalarMappable(cmap='PiYG')
        color_map.set_array(self.mod_inView)

        ax.scatter(*zip(*self.moon), marker='s', s=1, c=self.mod_inView, cmap='PiYG')
        plt.colorbar(color_map)

        # ax.set_title('Satellite coverage')
        ax.set_xlabel('x [$10^7$ m]')
        ax.set_ylabel('y [$10^7$ m]')
        ax.set_zlabel('z [$10^7$ m]')

        ax.set_xlim(-r_moon * 3, r_moon * 3)
        ax.set_ylim(-r_moon * 3, r_moon * 3)
        ax.set_zlim(-r_moon * 3, r_moon * 3)
        ax.set_aspect('equal')
        plt.show()

    def plotCoverageOrbits(self):
        """Plot the coverage of the satellites."""
        if self.mod_inView_orbits.all() == 0:
            self.setCoverageOrbits()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot Orbits
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        r = np.linspace(r_moon+100, r_orbit, 100)

        # Create a meshgrid of phi and theta values
        phi, theta = np.meshgrid(phi, theta)
        # Calculate the x, y, and z coordinates for each point on the sphere
        xO = r * np.sin(theta) * np.cos(phi)
        yO = r * np.sin(theta) * np.sin(phi)
        zO = r * np.cos(theta)

        ax.plot_surface(xO, yO, zO, color='grey', alpha=0.2)

        # Plot modules
        mod_pos = [mod.r for mod in self.modules]
        ax.scatter(*zip(*mod_pos), s=10)

        # Plot satellites in view
        color_map = cm.ScalarMappable(cmap='PiYG')
        color_map.set_array(self.mod_inView_orbits)

        ax.scatter(*zip(*self.orbits), marker='s', s=1, c=self.mod_inView_orbits, cmap='PiYG')
        plt.colorbar(color_map)

        # ax.set_title('Satellite coverage')
        ax.set_xlabel('x [$10^7$ m]')
        ax.set_ylabel('y [$10^7$ m]')
        ax.set_zlabel('z [$10^7$ m]')

        ax.set_xlim(-r_moon * 3, r_moon * 3)
        ax.set_ylim(-r_moon * 3, r_moon * 3)
        ax.set_zlim(-r_moon * 3, r_moon * 3)
        ax.set_aspect('equal')
        plt.show()

    def getSatellites(self):
        satellites_params = []
        for module in self.modules:
            if isinstance(module, Satellite):
                satellites_params.append(list(module.getParams()))
        return satellites_params

    def getParams(self):
        """Get the parameters of the model."""
        for module in self.modules:
            if isinstance(module, Satellite):
                print(module.getParams())

    def __repr__(self):
        l = f'Model with {self.n_sat} satellites and {self.n_orbit_planes} orbit planes. The modules are:\n'
        for module in self.modules:
            l += f'{module}\n'
        return l


if __name__ == '__main__':
    # Create model
    model = Model()
