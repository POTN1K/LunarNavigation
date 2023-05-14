# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array


class PropagationTime:
    """Class to input satellite(s) and see their change of position over time"""
    def __init__(self, orbit_parameters, final_time, resolution, mass_sat, area_sat, c_radiation, satellite_number):
        """Initialize the initial state of the satellite(s) with Keplerian elements,final time and resolution to see the final position
        :param orbit_parameters: array of Keplarian elements [[sat1],[sat2],[[sat3]]
        :param final_time: Time for the end of the simulation [s]
        :param resolution: Number of outputs between start and end of simulation [-]
        :param mass_sat: Mass of each satellite [kg]
        :param mass_sat: Radiation area sat [m^2]
        :param mass_sat: Coefficient radiation pressure [-]
        """
        self.resolution = resolution
        self.final_time = final_time
        self.orbit_parameters = orbit_parameters
        self.mass_sat = mass_sat
        self.area_sat = area_sat
        self.c_radiation = c_radiation
        self.satellite_number = satellite_number
        self.fixed_step_size = resolution

        spice.load_standard_kernels()

        self.bodies, self.bodies_to_propagate, self.central_bodies = self.create_bodies()
        self.dependent_variables_to_save = self.saving()
        self.states_array = None
        self.dep_vars_array = None
        self.add_vehicle_radiation_pressure()
        self.acceleration_models = self.create_acceleration_models()
        self.initial_state = self.create_initial_state()
        self.propagator_settings = self.create_propagator_settings()
        self.kepler_elements = None




    def create_bodies(self):
        # Define string names for bodies to be created from default.
        bodies_to_create = ["Sun", "Earth", "Moon"]

        # Use "Earth"/"J2000" as global frame origin and orientation.
        global_frame_origin = "Moon"
        global_frame_orientation = "J2000"

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
        occulting_bodies = ["Earth"]
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            "Sun", self.area_sat, self.c_radiation, occulting_bodies
        )
        for satellite_name in self.bodies_to_propagate:
            environment_setup.add_radiation_pressure_interface(
                self.bodies, satellite_name, radiation_pressure_settings
            )

    def create_acceleration_models(self):
        accelerations_settings_lunar_sats = dict(
            Sun=[
                propagation_setup.acceleration.cannonball_radiation_pressure(),
                propagation_setup.acceleration.point_mass_gravity()
            ],
            Earth=[
                propagation_setup.acceleration.spherical_harmonic_gravity(5, 5)
            ],
            Moon=[
                propagation_setup.acceleration.spherical_harmonic_gravity(5, 5)
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
        # Define list of dependent variables to save
        dependent_variables_to_save = []
        for i, satellite_name in enumerate(self.bodies_to_propagate):
            dependent_variables_to_save.append([
                # propagation_setup.dependent_variable.total_acceleration(satellite_name),
                propagation_setup.dependent_variable.keplerian_state(satellite_name, "Moon") #,
                # propagation_setup.dependent_variable.latitude(satellite_name, "Moon"),
                # propagation_setup.dependent_variable.longitude(satellite_name, "Moon"),
                # propagation_setup.dependent_variable.single_acceleration_norm(
                #     propagation_setup.acceleration.point_mass_gravity_type, satellite_name, "Sun"
                # ),
                # propagation_setup.dependent_variable.single_acceleration_norm(
                #     propagation_setup.acceleration.spherical_harmonic_gravity_type, satellite_name, "Moon"
                # ),
                # propagation_setup.dependent_variable.single_acceleration_norm(
                #     propagation_setup.acceleration.spherical_harmonic_gravity_type, satellite_name, "Earth"
                # ),
                # propagation_setup.dependent_variable.single_acceleration_norm(
                #     propagation_setup.acceleration.cannonball_radiation_pressure_type, satellite_name, "Sun"
                # )
            ])

        # Assign the dependent variables to save to the class attribute
        return list(np.array(dependent_variables_to_save).ravel())

        # Assign the dependent variables to save to the class attribute
        self.dependent_variables_to_save = list(np.array(dependent_variables_to_save).ravel())

    def create_propagator_settings(self):
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
        # Create simulation object and propagate the dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            self.bodies, self.propagator_settings
        )

        # Extract the resulting state history and convert it to an ndarray
        states = dynamics_simulator.state_history

        self.states_array = result2array(states)
        dep_vars = dynamics_simulator.dependent_variable_history
        self.dep_vars_array = result2array(dep_vars)
        self.kepler_elements = self.dep_vars_array[:, 1:]


    def inclination_change(self, v, delta_i):
        """
        Inclination change: calculates the delta V required for an inclination change.
        v: velocity of the spacecraft (m/s)
        delta_i: change in inclination (rad)
        """
        return 2 * v * np.sin(delta_i / 2)
    def hohmann_delta_v(self, a1, a2):
        term1 = np.sqrt(self.bodies.get("Moon").gravitational_parameter / a1)
        term2 = np.sqrt(2 * a2 / (a1 + a2)) - 1
        return term1 * term2

    def delta_v_to_maintain_orbit(self, satellite_name, start_time, end_time):
        # Get the necessary orbital parameters
        satellite_name_index = self.bodies_to_propagate.index(satellite_name)
        a1 = self.kepler_elements[int(start_time/10)][satellite_name_index*6]
        a2 = self.kepler_elements[int(end_time/10)][satellite_name_index * 6]
        i1 = self.kepler_elements[int(start_time/10)][satellite_name_index*6 + 2]
        i2 = self.kepler_elements[int(end_time/10)][satellite_name_index*6 + 2]
        print(a1, a2, i1, i2)




        # Calculate delta V for the inclination change
        v2 = np.sqrt(self.bodies.get("Moon").mass/a2)
        delta_v_inclination = self.inclination_change(v2, abs(i2 - i1))

        print(self.bodies.get("Moon").mass)
        # Calculate the delta-v for the Hohmann transfer
        delta_v1 = self.hohmann_delta_v(a1, a2)
        delta_v2 = self.hohmann_delta_v(a2, a1)

        return abs(delta_v1) + abs(delta_v2) + abs(delta_v_inclination)



    def plot_time(self):

        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_title(f'System state evolution of all bodies w.r.t the Moon')

        # Add the moon
        # generate sphere coordinates
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v) * 1737.4 * 1000
        y = np.sin(u) * np.sin(v) * 1737.4 * 1000
        z = np.cos(v) * 1737.4 * 1000

        # plot the sphere


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
        max_range = np.max(np.abs(self.states_array[:, 0:3]))
        ax1.set_xlim(-max_range, max_range)
        ax1.set_ylim(-max_range, max_range)
        ax1.set_zlim(-max_range, max_range)

        # Add a legend, labels, and use a tight layout to save space
        ax1.legend()
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_zlabel('z [m]')
        plt.tight_layout()
        plt.show()

    def plot_kepler(self):

        # Convert time from seconds to hours
        time_hours = self.states_array[:, 0] / 3600

        # Plot Kepler elements as a function of time
        kepler_elements = self.dep_vars_array[:,1+  6 * self.satellite_number:7 + 6 * self.satellite_number]
        # kepler_elements = dep_vars_array[:, 4 + 6 * self.satellite_number:10 + 6 * self.satellite_number]
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
        fig.suptitle('Evolution of Kepler elements over the course of the propagation.')
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


satellites = [[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 0.17453292519943295],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 1.4311699866353502],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 2.6878070480712677],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 3.944444109507185],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 0.0, 5.201081170943102],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 0.17453292519943295],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 1.4311699866353502],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 2.6878070480712677],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 3.944444109507185],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 2.0943951023931953, 5.201081170943102],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 0.17453292519943295],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 1.4311699866353502],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 2.6878070480712677],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 3.944444109507185],
[24500000.0, 0, 1.2217304763960306, 0.39968039870670147, 4.1887902047863905, 5.201081170943102]]

if __name__ == '__main__':
    propagation_time = PropagationTime(satellites, 100000, 10, 250, 2, 0, 0)
    propagation_time.simulate()
    for i, satellite_name in enumerate(propagation_time.bodies_to_propagate):
        delta_v = propagation_time.delta_v_to_maintain_orbit(satellite_name, 0, 20)
        print(f"Delta-v for {satellite_name}: {delta_v}")
    propagation_time.plot_kepler()
    propagation_time.plot_time()

