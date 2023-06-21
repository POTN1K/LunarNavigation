
"""
Code in order to simulate multiple satellites and their positions(history) for a certain time period around the Moon.
The code is made user-friendly to allow the user to fill in an array of satellites, final time, resolution for integra-
ting, satellite mass, radiation area satellite to allow everything to be computed. The user is able to specify a certain
satellite to see its kepler characteristic history. The user can also receive the delta V required for orbit maintenance
between a self specified time period.

By Kyle Scherpenzeel

"""


# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import imageio



# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

c = 299792458

class PropagationTime:
    """Class to input satellite(s) and see their change of position over time"""
    def __init__(self, orbit_parameters=None, final_time=86400, resolution=900, mass_sat=250, area_sat=1, c_radiation=1.07*0.768, antenna_power=100, degree_earth =5, degree_moon=10):
        """Initialize the initial state of the satellite(s) with Keplerian elements,final time and resolution to see the final position
        :param orbit_parameters: array of Keplarian elements [[sat1],[sat2],[[sat3]] #Radians
        :param final_time: Time for the end of the simulation [s]
        :param resolution: Number of outputs between integration [-]
        :param mass_sat: Mass of each satellite [kg]
        :param area_sat: Radiation area sat [m^2]
        :param c_radiation: Coefficient radiation pressure [-]
        """

        if orbit_parameters is None:
            orbit_parameters = np.array([[20e6, 0, 0, 0, 0, 0], [20e6, 0, 0, 0, 0, 0]])
        self.resolution = resolution
        self.final_time = final_time
        self.orbit_parameters = np.array(orbit_parameters)
        self.mass_sat = mass_sat
        self.area_sat = area_sat
        self.antenna_power = antenna_power
        self.c_radiation = c_radiation
        self.fixed_step_size = resolution
        self.degree_earth = degree_earth
        self.degree_moon = degree_moon

        spice.load_standard_kernels()

        self.bodies, self.bodies_to_propagate, self.central_bodies = self.create_bodies()
        self.add_vehicle_radiation_pressure()
        self.dependent_variables_to_save = self.saving()
        self.states_array = None
        self.dep_vars_array = None

        self.acceleration_models = self.create_acceleration_models()
        self.initial_state = self.create_initial_state()
        self.propagator_settings = self.create_propagator_settings()
        self.kepler_elements = None
        self.simulate()

    @property
    def orbit_parameters(self):
        return self._orbit_parameters

    @orbit_parameters.setter
    def orbit_parameters(self, value):

        if len(value.shape) >= 2 and value.shape[0] >= 2:
            self._orbit_parameters = value
        else:
            raise ValueError("At least 2 satellites must be given")

    @property
    def final_time(self):
        return self._final_time

    @final_time.setter
    def final_time(self, value):
        if value >= 1:
            self._final_time = value
        else:
            raise ValueError("Time must be positive")

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        if value > 0:
            self._resolution = value
        else:
            raise ValueError("Resolution must be above 0")

    @property
    def mass_sat(self):
        return self._mass_sat

    @mass_sat.setter
    def mass_sat(self, value):
        if value > 0:
            self._mass_sat= value
        else:
            raise ValueError("Mass of satellites must be above 0")

    @property
    def area_sat(self):
        return self._area_sat

    @area_sat.setter
    def area_sat(self, value):
        if value >= 0:
            self._area_sat = value
        else:
            raise ValueError("Area of the satellites must be equal or above 0\n (can be 0 of radiation pressure is not considered)")

    @property
    def c_radiation(self):
        return self._c_radiation

    @c_radiation.setter
    def c_radiation(self, value):
        if value >= 0:
            self._c_radiation = value
        else:
            raise ValueError(
                "Coefficient of radiation must be above 0")

    def create_bodies(self):
        """
        Method to create celestial bodies and satellites for the simulation to use. \n
        :return (class,list,list): Class of bodies setup, list of propagating bodies (satellites), list of central bodies (Moon)
        """
        # Define string names for bodies to be created from default.
        bodies_to_create = ["Sun", "Earth", "Moon"]  # Perturbing bodies

        # Use "Earth"/"J2000" as global frame origin and orientation.
        global_frame_origin = "Moon"
        global_frame_orientation = "J2000"  # Starting time (January 2000)

        # Create default body settings, usually from `spice`.
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create,
            global_frame_origin,
            global_frame_orientation)

        # Create system of selected celestial bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)
        central_bodies = []
        # Create vehicle objects.
        bodies_to_propagate = []
        for i in range(len(self.orbit_parameters)):
            satellite_name = f"LunarSat{i + 1}"
            bodies.create_empty_body(satellite_name)
            bodies.get(satellite_name).mass = self.mass_sat
            bodies_to_propagate.append(satellite_name)
            central_bodies.append("Moon")
        return bodies, bodies_to_propagate, central_bodies

    def add_vehicle_radiation_pressure(self):
        """
        Function to add area and radiation pressure coefficient to the vehicles. Updates environment_setup and does not
        return anything.

        """
        occulting_bodies = ["Moon"]
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            "Sun", self.area_sat, self.c_radiation, occulting_bodies
        )
        for satellite_name in self.bodies_to_propagate:
            environment_setup.add_radiation_pressure_interface(
                self.bodies, satellite_name, radiation_pressure_settings
            )

    def add_antenna_thrust(self):
        a = self.antenna_power/(c*self.mass_sat)



    def create_acceleration_models(self):
        """
        Creates the acceleration settings and models for the system. The settings are assumed to be identical for each
        satellite. Assumptions: Sun = point mass, radiation = cannonball, Earth = spherical harmonic and moon =
        spherical harmonic. The code loops through all satellites and creates a dictionary with all satellites and their
        settings. Then makes the acceleration models using tudat-space. \n
        :return (object) : Propagation Setup
        """
        accelerations_settings_lunar_sats = dict(
            Sun=[
                propagation_setup.acceleration.cannonball_radiation_pressure(),
                propagation_setup.acceleration.point_mass_gravity()
            ],
            Earth=[
                propagation_setup.acceleration.spherical_harmonic_gravity(self.degree_earth, self.degree_earth)
            ],
            Moon=[
                propagation_setup.acceleration.spherical_harmonic_gravity(self.degree_moon, self.degree_moon),
                propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)
            ]

        )
        acceleration_settings = {}
        for satellite_name in self.bodies_to_propagate:
            acceleration_settings[satellite_name] = accelerations_settings_lunar_sats

        acceleration_models = propagation_setup.create_acceleration_models(
            self.bodies,
            acceleration_settings,
            self.bodies_to_propagate,
            self.central_bodies
        )
        return acceleration_models

    def create_initial_state(self):
        """
        Create the initial states of all the vehicles in cartesian coordinates.\n
        :return (array): 1D array of all the initial cartesian states.

        """
        moon_gravitational_parameter = self.bodies.get("Moon").gravitational_parameter
        cartesian_states = []
        for x in range(len(self.orbit_parameters)):

            cartesian_states.append(element_conversion.keplerian_to_cartesian_elementwise(
                gravitational_parameter=moon_gravitational_parameter,
                semi_major_axis=self.orbit_parameters[x][0],
                eccentricity=self.orbit_parameters[x][1],
                inclination=self.orbit_parameters[x][2],
                argument_of_periapsis=self.orbit_parameters[x][3],
                longitude_of_ascending_node=self.orbit_parameters[x][4],
                true_anomaly=self.orbit_parameters[x][5],

            )[:])
        return np.array(cartesian_states).reshape(len(self.orbit_parameters)*6, 1)

    def saving(self):
        """
        Function to add different variables to save within the simulation. Can uncomment the outputs but does require
        changes to the plotting. /n
        :return (list): 1D list with objects including outputs.
        """
        # Define list of dependent variables to save
        dependent_variables_to_save = []
        for i, satellite_name in enumerate(self.bodies_to_propagate):
            dependent_variables_to_save.append([
                # propagation_setup.dependent_variable.total_acceleration(satellite_name),
                # propagation_setup.dependent_variable.keplerian_state(satellite_name, "Moon")#,
                # propagation_setup.dependent_variable.latitude(satellite_name, "Moon"),
                # propagation_setup.dependent_variable.longitude(satellite_name, "Moon")
                propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, satellite_name, "Sun"
                ),
                propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.spherical_harmonic_gravity_type, satellite_name, "Moon"
                ),
                propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.spherical_harmonic_gravity_type, satellite_name, "Earth"
                ),
                propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.relativistic_correction_acceleration_type, satellite_name, "Moon"  #propagation_setup.acceleration.cannonball_radiation_pressure_type, satellite_name, "Sun"
                ),
                propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.cannonball_radiation_pressure_type, satellite_name, "Sun"
                )
            ])

        # Assign the dependent variables to save to the class attribute
        return list(np.array(dependent_variables_to_save).ravel())

        # Assign the dependent variables to save to the class attribute
        self.dependent_variables_to_save = list(np.array(dependent_variables_to_save).ravel())

    def create_propagator_settings(self):
        """
        Function to set up the initial simulation settings for tudat-space. User should not edit any variables. \n
        :return (object): returns the settings for the simulation

        """
        # Set simulation start and end epochs
        simulation_start_epoch = 0.0
        simulation_end_epoch = self.final_time
        termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

        # Create numerical integrator settings
        integrator_settings = propagation_setup.integrator.runge_kutta_4(self.fixed_step_size)

        # Create propagation settings

        propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            self.initial_state,
            simulation_start_epoch,
            integrator_settings,
            termination_settings,
            output_variables=self.dependent_variables_to_save
        )

        return propagator_settings

    def simulate(self):
        """
        Initializes and runs the simulation using all the settings from earlier stages. Updates states_array, dep_vars_
        array and kepler_elements for all time steps for later use in the plots.
        """
        # Create simulation object and propagate the dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            self.bodies, self.propagator_settings
        )

        # Extract the resulting state history and convert it to a ndarray
        states = dynamics_simulator.state_history

        self.states_array = result2array(states)
        dep_vars = dynamics_simulator.dependent_variable_history
        self.dep_vars_array = result2array(dep_vars)
        self.accelsun = self.dep_vars_array[:, -1]
        # self.kepler_elements = self.dep_vars_array[:, 1:-1]
        self.kepler_elements = np.delete(self.dep_vars_array, 0, axis=1)
        # Delete the initial position as this is not yet updated according to the simulation
        self.kepler_elements = np.delete(self.kepler_elements, 0, axis=0)

        indices_velocity = np.arange(4, self.states_array.shape[1], 6).reshape(-1, 1) + np.arange(3)
        # use advanced indexing
        self.velocity = self.states_array[:, indices_velocity.flatten()]


    def observation(self):
        """ I am trying to create a function to get observation time to earth and eclipse time for the power and thermal section
        maybe also add a method to obtain communication time between sats for updating
        """
        one_way_sat_link_ends = dict()
        one_way_sat_link_ends[observation.transmitter] = observation.body_reference_point_link_end_id("Sun")
        one_way_sat_link_ends[observation.receiver] = observation.body_origin_link_end_id("LunarSat")








    def inclination_change(self, v, delta_i):
        """
        Inclination change: calculates the delta V required for an inclination change.\n
        :param v (float): velocity of the spacecraft (m/s)
        :param delta_i (float): change in inclination (rad)
        """
        return 2 * v * np.sin(delta_i / 2)

    def hohmann_delta_v(self, a1, a2):
        """
        Function to calculate delta v from a hohmann transfer. \n
        :param a1 (float): semi major axis of initial orbit
        :param a2 (float): semi major axis of final orbit orbi
        :return delta_v (float): Returns delta_v for specific transfer
        """
        term1 = np.sqrt(self.bodies.get("Moon").gravitational_parameter / a1)
        term2 = np.sqrt(2 * a2 / (a1 + a2)) - 1
        return term1 * term2

    def delta_v_to_maintain_orbit(self, satellite_name, start_time, end_time):
        """
        Function to calculate delta_v for a specific satellite between certain time periods. Uses in plane hohmann and
        out of plane inclination change.
        :param satellite_name (string): name of the satellite
        :param start_time (int): start time for the time period
        :param end_time (int): final time for the time period
        :return delta_v (float): Returns the delta_v required for maintenance between 2 times
        """
        # Get the necessary orbital parameters
        satellite_name_index = self.bodies_to_propagate.index(satellite_name)
        a1 = self.kepler_elements[int(start_time/self.fixed_step_size)][satellite_name_index*6]
        a2 = self.kepler_elements[int(end_time/self.fixed_step_size)][satellite_name_index * 6]
        i1 = self.kepler_elements[int(start_time/self.fixed_step_size)][satellite_name_index*6 + 2]
        i2 = self.kepler_elements[int(end_time/self.fixed_step_size)][satellite_name_index*6 + 2]


        # Calculate delta V for the inclination change
        v2 = np.sqrt(self.bodies.get("Moon").gravitational_parameter/a2)
        delta_v_inclination = self.inclination_change(v2, abs(i2 - i1))

        # Calculate the delta-v for the Hohmann transfer
        delta_v1 = self.hohmann_delta_v(a1, a2)
        delta_v2 = self.hohmann_delta_v(a2, a1)
        return abs(delta_v1) + abs(delta_v2) + abs(delta_v_inclination)
    def complete_delta_v(self, start_time, end_time):
        """
        Function to calculate delta_v for the entire constellation between 2 times/n
        :param start_time (int): Starting time for the delta_v maintenance
        :param end_time (int): Final time for the delta_v maintenance
        :return delta_v_list (list): List with all delta_v for the satellites during th
        """
        delta_v_list = []
        for i, satellite_name in enumerate(self.bodies_to_propagate):
            delta_v = self.delta_v_to_maintain_orbit(satellite_name, start_time, end_time)
            # print(f"Delta-v for {satellite_name}: {delta_v}")
            delta_v_list.append(delta_v)
        delta_v_array = np.array(delta_v_list)
        # print(f" Range of Delta-v:{np.ptp(delta_v_array)}, max Delta-v {np.max(delta_v_array)}, min Delta-v "
        #       f"{np.min(delta_v_array)}, average Delta-v {np.mean(delta_v_array)}, SD Delta-v: {np.std(delta_v_array)}")
        return delta_v_list

    def min_max_kepler(self):
        """
        Function to calculate the range of Kepler elements for the whole simulation
        """
        # Delete the first column (time)
        kepler_elements = np.delete(self.dep_vars_array, 0, axis=1)
        # Delete the initial position as this is not yet updated according to the simulation
        kepler_elements = np.delete(kepler_elements, 0, axis=0)


        kepler_elements_3d = kepler_elements.reshape(kepler_elements.shape[0], -1, 6)

        # Compute the range (max - min) for each Kepler element for every satellite
        ranges = np.ptp(kepler_elements_3d, axis=1)

        # Compute the average of these ranges
        avg_ranges = np.mean(ranges, axis=0)

        # Compute the standard deviation of these average ranges
        std_dev = np.std(ranges, axis=0)
        max_ranges = np.max(ranges, axis=0)

        # print(f"Average range for Kepler element for all satellites: {avg_ranges}")
        # print(f"Standard deviation of the average ranges: {std_dev}")
        # print(f"Maximum range of each element across all groups: {max_ranges}")
        return(avg_ranges, std_dev, max_ranges)
    def plot_time(self):
        """
        Plots the history and current location for all bodies. First this function creates the lunar surface and then
        loops through the vehicles in order to plot their positions.
        """

        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, projection='3d')

        # Add the moon
        # generate sphere coordinates
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v) * 1737.4 * 1000
        y = np.sin(u) * np.sin(v) * 1737.4 * 1000
        z = np.cos(v) * 1737.4 * 1000

        # plot the sphere
        frames = []

        for i, body in enumerate(self.bodies_to_propagate):
            # Plot the 3D trajectory of each body
            ax1.plot(self.states_array[:, 6 * i + 1], self.states_array[:, 6 * i + 2],
                     self.states_array[:, 6 * i + 3],
                     label=body)
            # Plot the initial position of each body
            ax1.scatter(self.states_array[0, 6 * i + 1], self.states_array[0, 6 * i + 2],
                        self.states_array[0, 6 * i + 3],
                        marker='x')

        # Add the position of the central body: the Moon
        ax1.scatter(0, 0, 0, marker='x', label="Lunar Center", color='black')

        ax1.plot_surface(x, y, z, color='grey', alpha=0.3)

        # Set the limits of the plot
        max_range = 2*np.max(np.abs(self.states_array[:, 0:3]))
        ax1.set_xlim(-max_range, max_range)
        ax1.set_ylim(-max_range, max_range)
        ax1.set_zlim(-max_range, max_range)

        # Add a legend, labels, and use a tight layout to save space
        # ax1.legend()
        ax1.set_xlabel('x [$10^7$ m]')
        ax1.set_ylabel('y [$10^7$ m]')
        ax1.set_zlabel('z [$10^7$ m]')
        plt.tight_layout()

        for angle in range(0, 360, 5):
            ax1.view_init(elev=30, azim=angle)  # Set the camera angle
            fig1.canvas.draw()  # Redraw the figure
            frame = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8')  # Convert the figure to an RGB array
            frame = frame.reshape(fig1.canvas.get_width_height()[::-1] + (3,))  # Reshape the array
            frames.append(frame)  # Append the frame to the list

            # Create a GIF using imageio
        output_gif = "OrbitAnimation.gif"
        imageio.mimsave(output_gif, frames, duration=0.1)
        plt.show()

    def plot_kepler(self, satellite_number):
        """
        Plots the history of the 6 kepler elements over the entire orbit.
        :param satellite_number (int): Integer starting from 0 for the index of the satellite number
        """


        # Convert time from seconds to hours
        time_hours = self.states_array[:, 0] / 3600

        # Plot Kepler elements as a function of time
        kepler_elements = self.dep_vars_array[:, 1 + 6 * satellite_number:7 + 6 * satellite_number]
        # kepler_elements = dep_vars_array[:, 4 + 6 * self.satellite_number:10 + 6 * self.satellite_number]
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
        # Semi-major Axis
        semi_major_axis = kepler_elements[:, 0] / 1e3
        ax1.plot(time_hours, semi_major_axis)
        ax1.set_ylabel('Semi-major axis [km]')

        # Eccentricity
        eccentricity = kepler_elements[:, 1]
        ax2.plot(time_hours, eccentricity)
        ax2.set_ylabel('Eccentricity [-]')

        # Inclination
        inclination = np.rad2deg(kepler_elements[:, 2])
        ax3.plot(time_hours, inclination)
        ax3.set_ylabel('Inclination [deg]')

        # Argument of Periapsis
        argument_of_periapsis = np.rad2deg(kepler_elements[:, 3])
        ax4.plot(time_hours, argument_of_periapsis)
        ax4.set_ylabel('Argument of Periapsis [deg]')

        # Right Ascension of the Ascending Node
        raan = np.rad2deg(kepler_elements[:, 4])
        ax5.plot(time_hours, raan)
        ax5.set_ylabel('RAAN [deg]')

        # True Anomaly
        true_anomaly = np.rad2deg(kepler_elements[:, 5])
        ax6.scatter(time_hours, true_anomaly, s=1)
        ax6.set_ylabel('True Anomaly [deg]')

        for ax in fig.get_axes():
            ax.set_xlabel('Time [hr]')
            ax.set_xlim([min(time_hours), max(time_hours)])
            ax.grid()

        plt.tight_layout()
        plt.show()
miu_moon = 4.9048695e12
satellites = [[5701.2e3, 0.002, np.deg2rad(40.78), np.deg2rad(90),  0, 0], [5701.2e3, 0.002, np.deg2rad(40.78), np.deg2rad(90),  0, 0]]
propagation_time = PropagationTime(resolution=10, final_time= 2*np.pi * np.sqrt(10000000**3/miu_moon), orbit_parameters=satellites,degree_moon=5)
# # # print(np.average(np.array(propagation_time.complete_delta_v(0, 86400*14))))a
accell = propagation_time.dep_vars_array[:,1:6]
time = propagation_time.dep_vars_array[:, 0]
propagation_time = PropagationTime(resolution=10, final_time= 2*np.pi * np.sqrt(10000000**3/miu_moon), orbit_parameters=satellites,degree_earth =0,degree_moon=0)
accell2 = propagation_time.dep_vars_array[:,1:6]
propagation_time = PropagationTime(resolution=10, final_time= 2*np.pi * np.sqrt(10000000**3/miu_moon), orbit_parameters=satellites,degree_earth =10,degree_moon=10)
accell3 = propagation_time.dep_vars_array[:,1:6]




# plt.plot(time, accell[:, 0], label ='Sun Point Mass')
# plt.plot(time, accell[:, 1], label ='Moon 10th degree')
plt.plot(time, accell[:, 1], label ='Moon 5th Degree')
plt.plot(time, accell2[:, 1], label ='Moon Point Mass')
plt.plot(time, accell3[:, 1], label ='Moon 10th Degree')

# plt.plot(time, accell[:, 3], label ='Relativistic Correction')
# plt.plot(time, accell[:, 4], label ='Solar Radiation Pressure')
plt.legend(loc='lower right')
plt.show()

# propagation_time.plot_kepler(0)
propagation_time.plot_time()
