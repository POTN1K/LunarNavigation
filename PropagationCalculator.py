# Load standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array
import modelClass

class PropagationTime:
    """Class to input satellite(s) and see their change of position over time"""
    def __init__(self, orbit_parameters, final_time, resolution, mass_sat, area_sat, c_radiation):
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

    spice.load_standard_kernels()


    def bodies(self):
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
        for x in range(len(self.orbit_parameters)):

            bodies.create_empty_body("LunarSat" + str(x+1))

            bodies.get("LunarSat" + str(x+1)).mass = self.mass_sat

            bodies_to_propagate.append("LunarSat" + str(x+1))
            central_bodies.append("Moon")

        return bodies, bodies_to_propagate, central_bodies
    def vehiclesettings(self, bodies):
        occulting_bodies = ["Earth"]
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            "Sun", self.area_sat, self.c_radiation, occulting_bodies
        )
        for x in range(len(self.orbit_parameters)):
            environment_setup.add_radiation_pressure_interface(
                bodies, "LunarSat" + str(x+1), radiation_pressure_settings)

    def accelerations(self):
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
        # Create global accelerations settings dictionary.
        acceleration_settings = {}
        for x in range(len(self.orbit_parameters)):
            acceleration_settings["LunarSat" + str(x + 1)] = accelerations_settings_lunar_sats

        # Create acceleration models.
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies,
            acceleration_settings,
            bodies_to_propagate,
            central_bodies)
        return acceleration_models

    def initialstate(self):
        moon_gravitational_parameter = bodies.get("Moon").gravitational_parameter
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

    # def saving(self): This can be used later for
    #     # Define list of dependent variables to save
    #     dependent_variables_to_save = [
    #         propagation_setup.dependent_variable.total_acceleration("LunarSat"),
    #         propagation_setup.dependent_variable.keplerian_state("LunarSat", "Moon"),
    #         propagation_setup.dependent_variable.latitude("LunarSat", "Moon"),
    #         propagation_setup.dependent_variable.longitude("LunarSat", "Moon"),
    #         propagation_setup.dependent_variable.single_acceleration_norm(
    #             propagation_setup.acceleration.point_mass_gravity_type, "LunarSat", "Sun"
    #         ),
    #         propagation_setup.dependent_variable.single_acceleration_norm(
    #             propagation_setup.acceleration.spherical_harmonic_gravity_type, "LunarSat", "Moon"
    #         ),
    #         propagation_setup.dependent_variable.single_acceleration_norm(
    #             propagation_setup.acceleration.spherical_harmonic_gravity_type, "LunarSat", "Earth"
    #         ),
    #         propagation_setup.dependent_variable.single_acceleration_norm(
    #             propagation_setup.acceleration.cannonball_radiation_pressure_type, "LunarSat", "Sun"
    #         )
    #     ]
    def propagationsettings(self):
        # Create termination settings
        # Load spice kernels
        spice.load_standard_kernels()

        # Set simulation start and end epochs
        simulation_start_epoch = 0.0
        simulation_end_epoch = self.final_time
        termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

        # Create numerical integrator settings
        fixed_step_size = 10.0
        integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)

        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            cartesian_states,
            simulation_start_epoch,
            integrator_settings,
            termination_settings
        )
        return propagator_settings

    def simulator(self):
        # Create simulation object and propagate the dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings
        )

        # Extract the resulting state history and convert it to an ndarray
        states = dynamics_simulator.state_history
        states_array = result2array(states)
        return states_array

    def createMoon(self):
        """Add the Moon to the model."""
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        r_moon = 1737.4
        sphere_points = []
        for i in phi:
            for j in theta:
                x = r_moon * np.sin(j) * np.cos(i)
                y = r_moon * np.sin(j) * np.sin(i)
                z = r_moon * np.cos(j)
                sphere_points.append([x, y, z])
        return sphere_points
    def plotTime(self):

        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_title(f'System state evolution of all bodies w.r.t SSB.')

        # Add the moon
        # generate sphere coordinates
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v) * 1737.4 * 1000
        y = np.sin(u) * np.sin(v) * 1737.4 * 1000
        z = np.cos(v) * 1737.4 * 1000

        # plot the sphere


        for i, body in enumerate(bodies_to_propagate):
            # Plot the 3D trajectory of each body
            ax1.plot(barycentric_system_state_array[:, 6 * i + 1], barycentric_system_state_array[:, 6 * i + 2],
                     barycentric_system_state_array[:, 6 * i + 3],
                     label=body)
            # Plot the initial position of each body
            ax1.scatter(barycentric_system_state_array[0, 6 * i + 1], barycentric_system_state_array[0, 6 * i + 2],
                        barycentric_system_state_array[0, 6 * i + 3],
                        marker='x')

        # Add the position of the central body: the Solar System Barycenter
        ax1.scatter(0, 0, 0, marker='x', label="SSB", color='black')

        ax1.plot_surface(x, y, z, color='grey',alpha =0.3)




        # Add a legend, labels, and use a tight layout to save space
        ax1.legend()
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_zlabel('z [m]')
        plt.tight_layout()
        plt.show()


sat = [[2000000, 0, 58.69, 22.9, 4, 6], [2000000, 0, 10, 22.9, 4, 6],[3000000, 0, 10, 22.9, 4, 6]]

if __name__ == '__main__':
    propagation_time = PropagationTime(sat, 1000, 10, 250, 2, 1)
    bodies, bodies_to_propagate, central_bodies = propagation_time.bodies()
    cartesian_states = propagation_time.initialstate()
    propagation_time.vehiclesettings(bodies)  # Add this line to set up radiation pressure settings
    acceleration_models = propagation_time.accelerations()
    initial_state = propagation_time.initialstate()
    propagator_settings = propagation_time.propagationsettings()  # Remove the previous assignment of propagator_settings
    barycentric_system_state_array = propagation_time.simulator()
    # sphere_moon = propagation_time.createMoon()

    propagation_time.plotTime()

