# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.data.horizons import HorizonsQuery
from tudatpy.util import result2array
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.data.sbdb import SBDBquery

# other useful modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from astropy.time import Time
from astroquery.mpc import MPC
from pathlib import Path
import re
import yaml
from matplotlib.lines import Line2D
import os

# import from the comegs library 
from comegs.accelerations import MarsdenAcceleration, COMarsdenAcceleration
from comegs.config_files import Configuration
from comegs.settings import Integrator
from comegs.observations import Observations
from comegs.plotting import PlotError

# Load the defaul SPICE kernels 
spice.load_standard_kernels()

# define the colour scheme for plotting 
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '9467bd', 'c5b0d5', '8c564b', 'c49c94', 'e377c2', 'f7b6d2', '7f7f7f', 'c7c7c7', 'bcbd22', 'dbdb8d', '17becf', '9edae5']

# set-up current directory 
current_dir = os.path.dirname(__file__)
workspace = Path.home()/f"comets/comets"
astrometry_results = workspace/ f"astrometry_data"

# define function to transform the target mpc code into the input of the configuration file 
def get_number(s: str) -> str:
    return re.sub(r'^[^\d]*|[^A-Za-z0-9]', '', s)

"""
USER INPUTS:
- code of the comet to be analysed (without the initial C/)
- sublimating volatile for the acceleration model
"""

target_mpc_code = '2014 UN271' 
mpc_code = 'C/' + target_mpc_code
element = 'H2O'

"""
Read the relative config file to define the horizons code, etc...
Make sure that the config file is updated (important for the Horizons code, as the orbit are updated in Horizons)
"""
config_number = get_number(target_mpc_code)
with open(f"{workspace}/configuration_files/config_{config_number}.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

"""
Retreive the attributes of the body - either from the SBDB or from the yaml dictionary
"""
horizons_code = config_dict['horizons_code']
target_sbdb = SBDBquery(target_mpc_code)

aud2_to_ms2 = 1.495978707e11/(86400*86400)
A1 = config_dict['marsden_params']['A1']*aud2_to_ms2
A2 = config_dict['marsden_params']['A2']*aud2_to_ms2
A3 = config_dict['marsden_params']['A3']*aud2_to_ms2

initial_parameters = [A1, A2, A3]

observations_start = target_sbdb.first_obs
observations_end = target_sbdb.last_obs
Dt = target_sbdb.Dt
time_perihelion = (target_sbdb.time_perihelion - 2451545.0)*86400

"""
Define settings for the estimation (start and end epochs, number of used iterations)
"""
number_of_pod_iterations = 8

# define the frame origin and orientation.
global_frame_origin = "SSB"
global_frame_orientation = "J2000"

epoch_start = (Time(observations_start).jd - 2451545.0)*86400 - 30*86400
epoch_end = (Time(observations_end).jd - 2451545.0)*86400 + 30*86400
epoch_middle = (config_dict['epoch_integration'] - 2451545.0)*86400 # epoch of integration of the Horizons orbit 

"""
Define the environment, settings controlled from the configuration file 
"""
configuration = Configuration(config_dict, str(target_mpc_code), horizons_code, epoch_start, epoch_end)
bodies, body_settings = configuration.system_of_bodies(f'{workspace}/SiMDA_250806.csv', comet_radius=None, density=None, satellites_codes=None, satellites_names=None)

# define the central body and body to be propagated (in this case the comet)
bodies_to_propagate = [str(target_mpc_code)]
central_bodies = ['Sun']

"""
Load the astrometrical measurements 
"""
observations = Observations(str(target_mpc_code), epoch_start, epoch_end, bodies)
# define the ground stations of the observatories 
observatories_table = MPC.get_observatory_codes().to_pandas()

observation_settings_list_astro, observation_collection_astro, astrometry_obs_dict = observations.load_observations_from_file(
    astrometry_results/f'{config_number}.psv', observatories_table, apply_weights = False
    )

"""
Define the acceleration settings; this is also defined in the configuration file and can be retrieved from the configuration class  
"""
dict_acc = dict()
nongrav_acc = COMarsdenAcceleration(A1, A2, A3, bodies, str(target_mpc_code), horizons_code, Dt, epoch_start, epoch_end, dict_acc)
acceleration_settings = configuration.acceleration_model(nongrav_acc)

# create the acceleration model
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

"""
Retrieve an initial guess for the comet's position - use the exact same as horizons ; this is wrt the Sun
"""
initial_states = HorizonsQuery(
            query_id= horizons_code,
            location="500@10",
            epoch_list = [epoch_start, epoch_end], 
            extended_query=True,
            )
initial_states = np.array(initial_states.cartesian(frame_orientation='J2000'))[0, 1:]

"""
Define the integration and propagation settings 
"""
integrator = Integrator(epoch_start, epoch_end, central_bodies, bodies_to_propagate, acceleration_models, initial_states)
termination_condition = propagation_setup.propagator.time_termination(epoch_end)

if config_dict['numerical']['propagator'] == 'propagation_setup.propagator.gauss_modified_equinoctial':
    propagator = propagation_setup.propagator.gauss_modified_equinoctial
elif config_dict['numerical']['propagator'] == 'propagation_setup.propagator.cowell':
    propagator = propagation_setup.propagator.cowell

if config_dict['numerical']['integrator']['coefficients'] == 'propagation_setup.integrator.CoefficientSets.rkf_56':
    coefficients = propagation_setup.integrator.CoefficientSets.rkf_56
elif config_dict['numerical']['integrator']['coefficients'] == 'propagation_setup.integrator.CoefficientSets.rkf_78':
    coefficients = propagation_setup.integrator.CoefficientSets.rkf_78

propagator_settings = integrator.fixed_step_size(termination_condition, propagator ,coefficients, float(config_dict['numerical']['integrator']['stepsize']))


"""
Perform the estimation using cartesian simulated obs 
"""
observations = Observations(target_mpc_code, epoch_start + 30*86400, epoch_end - 30*86400, bodies)
observation_settings_list, simulated_observations = observations.cartesian_observations()

parameter_settings = estimation_setup.parameter.initial_states(
    propagator_settings, bodies
)
parameters_to_estimate = estimation_setup.create_parameter_set(
    parameter_settings, bodies, propagator_settings
)

estimator = numerical_simulation.Estimator(
    bodies=bodies,
    estimated_parameters=parameters_to_estimate,
    observation_settings=observation_settings_list,
    propagator_settings=propagator_settings,
    integrate_on_creation=True,
)

pod_input = estimation.EstimationInput(
    simulated_observations,
    convergence_checker=estimation.estimation_convergence_checker(
        maximum_iterations=8,
    ),
)

pod_input.define_estimation_settings(reintegrate_variational_equations=True)
pod_output = estimator.perform_estimation(pod_input)

"""
Compare the estimated state with the Horizons JPL results 
"""
plotting = PlotError(bodies, epoch_start, epoch_end, horizons_code, target_mpc_code)
gaps = plotting.gaps(simulated_observations)
plot_error_total = plotting.plot_error_total(global_frame_origin, gaps, None)

"""
Simulate observations (from the ground) of the estimated  state 
"""
estimated_obs_dict = dict()
for observatory in astrometry_obs_dict.keys():
    obs_times = astrometry_obs_dict[observatory][0]
    if observatory not in bodies.get_body("Earth").ground_station_list:

        station_coord = observations.station_definition(observatories_table)
        row = station_coord.loc[observatories_table['Code'] == observatory].iloc[0]
        X = float(row['X'])
        Y = float(row['Y'])
        Z = float(row['Z'])

        ground_station_settings = numerical_simulation.environment_setup.ground_station.basic_station(
        station_name=observatory,
        station_nominal_position=[X,Y,Z,],)

        numerical_simulation.environment_setup.add_ground_station(
            bodies.get_body('Earth'), ground_station_settings
        )

    observation_settings_list = []
    observation_simulation_settings = []
    link_ends = dict()
    link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", observatory)  # the second name is the name of an observatory   
    link_ends[observation.transmitter] = observation.body_origin_link_end_id((str(target_mpc_code)))
    link_definition = observation.LinkDefinition(link_ends)
    observation_settings_list = [observation.angular_position(link_definition, bias_settings=None)]

    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.angular_position_type,
        link_definition,
        obs_times,
        ))

    ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
        observation_settings_list, bodies)

    # simulate the wanted observations 
    simulated_observations = estimation.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies)
    
    observations_list = np.array(simulated_observations.concatenated_observations)
    ra_prop = np.degrees([np.mod(el, 2*np.pi) for el in observations_list[::2]])
    dec_prop = np.degrees(observations_list[1::2])

    obs_array = []
    for i in range(len(ra_prop)):
        obs_array.append([ra_prop[i], dec_prop[i]])

    estimated_obs_dict[observatory] = [obs_times, np.array(obs_array)]


"""
Perform the propagation
"""
# as the propagation is performed from the middle epoch, re-define initial state and termination conditions 
initial_states = HorizonsQuery(
            query_id= horizons_code,
            location="500@10",
            epoch_list = [epoch_middle, epoch_end], 
            extended_query=True,
            )
initial_states = np.array(initial_states.cartesian(frame_orientation='J2000'))[0, 1:]

termination_condition = propagation_setup.propagator.non_sequential_termination(
        propagation_setup.propagator.time_termination(epoch_end),
        propagation_setup.propagator.time_termination(epoch_start))

propagator_settings = integrator.fixed_step_size(termination_condition, propagator ,coefficients, float(config_dict['numerical']['integrator']['stepsize']))

# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state and dependent variable history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)

"""
Compare the propagated state with the Horizons JPL results 
"""
times = states_array[:,0]
state_spice_query = HorizonsQuery(
            query_id= horizons_code,
            location="500@10",
            epoch_list = list(times),
            extended_query=True,
            )

state_spice = np.array(state_spice_query.cartesian(frame_orientation='J2000'))
state_spice_dict = {}
for i in range(len(state_spice)):
    state_spice_dict[np.ceil(state_spice[i,0])] = np.array(state_spice[i, 1:])

# cnvert the keys of the states_array to integers 
states = {np.ceil(k): v for k, v in states.items()}
# compare the obtained states with the JPL Horizons states 
difference = []
for key in state_spice_dict.keys():
    difference.append(np.linalg.norm(state_spice_dict[key][:3] - states[key][:3]))

plt.figure()
plt.plot(states.keys(), [el/1000 for el in difference])
plt.xlabel('Time since J2000 [sec]')
plt.ylabel('Difference JPL Horizons-propagated [km]')

"""
Assign the propagated state as the body ephemeris to then be able to simulate astrometric observations from it - get obs at the same time of the self measured ones 
"""
comet_ephemeris = {}
comet_ephemeris[str(target_mpc_code)] = numerical_simulation.environment_setup.ephemeris.tabulated(
    body_state_history = states, 
    frame_origin='Sun',
    frame_orientation=global_frame_orientation
    )
body_settings.get(str(target_mpc_code)).ephemeris_settings = comet_ephemeris[str(target_mpc_code)]
bodies = numerical_simulation.environment_setup.create_system_of_bodies(body_settings)

"""
Simulate observations (from the ground) of the propagated state 
"""
simulated_obs_dict = dict()
for observatory in astrometry_obs_dict.keys():
    obs_times = astrometry_obs_dict[observatory][0]
    if observatory not in bodies.get_body("Earth").ground_station_list:

        station_coord = observations.station_definition(observatories_table)
        row = station_coord.loc[observatories_table['Code'] == observatory].iloc[0]
        X = float(row['X'])
        Y = float(row['Y'])
        Z = float(row['Z'])

        ground_station_settings = numerical_simulation.environment_setup.ground_station.basic_station(
        station_name=observatory,
        station_nominal_position=[X,Y,Z,])

        numerical_simulation.environment_setup.add_ground_station(
            bodies.get_body('Earth'), ground_station_settings
        )

    observation_settings_list = []
    observation_simulation_settings = []
    link_ends = dict()
    link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", observatory)  # the second name is the name of an observatory   
    link_ends[observation.transmitter] = observation.body_origin_link_end_id((str(target_mpc_code)))
    link_definition = observation.LinkDefinition(link_ends)
    observation_settings_list = [observation.angular_position(link_definition, bias_settings=None)]

    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.angular_position_type,
        link_definition,
        obs_times,
        ))

    ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
        observation_settings_list, bodies)

    # simulate the wanted observations 
    simulated_observations = estimation.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies)
    
    observations_list = np.array(simulated_observations.concatenated_observations)
    ra_prop = np.degrees([np.mod(el, 2*np.pi) for el in observations_list[::2]])
    dec_prop = np.degrees(observations_list[1::2])

    obs_array = []
    for i in range(len(ra_prop)):
        obs_array.append([ra_prop[i], dec_prop[i]])

    simulated_obs_dict[observatory] = [obs_times, np.array(obs_array)]


"""
Simulate observations from JPL Horizons - assign JPL Horzions ephemeris to the comet's body 
"""
comet_ephemeris = {}
query_comet = HorizonsQuery(
    query_id = horizons_code,
    location=f"@{global_frame_origin}",
    epoch_start=epoch_start - 24 * 31 * 86400,
    epoch_end=epoch_end + 24 * 31 * 86400,
    epoch_step=f"{int(3600)}m",
    extended_query=True,
)
comet_ephemeris[str(target_mpc_code)] = query_comet.create_ephemeris_tabulated(
    frame_origin=global_frame_origin,
    frame_orientation=global_frame_orientation
)
body_settings.get(str(target_mpc_code)).ephemeris_settings = comet_ephemeris[str(target_mpc_code)]
bodies = numerical_simulation.environment_setup.create_system_of_bodies(body_settings)


jpl_sim_obs_dict = dict()
for observatory in astrometry_obs_dict.keys():
    obs_times = astrometry_obs_dict[observatory][0]
    if observatory not in bodies.get_body("Earth").ground_station_list:

        station_coord = observations.station_definition(observatories_table)
        row = station_coord.loc[observatories_table['Code'] == observatory].iloc[0]
        X = float(row['X'])
        Y = float(row['Y'])
        Z = float(row['Z'])

        ground_station_settings = numerical_simulation.environment_setup.ground_station.basic_station(
        station_name=observatory,
        station_nominal_position=[X,Y,Z,],)

        numerical_simulation.environment_setup.add_ground_station(
            bodies.get_body('Earth'), ground_station_settings
        )

    observation_settings_list = []
    observation_simulation_settings = []
    link_ends = dict()
    link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", observatory)  # the second name is the name of an observatory   
    link_ends[observation.transmitter] = observation.body_origin_link_end_id((str(target_mpc_code)))
    link_definition = observation.LinkDefinition(link_ends)
    observation_settings_list = [observation.angular_position(link_definition, bias_settings=None)]

    #observation_times = states_array[1:-1,0]

    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.angular_position_type,
        link_definition,
        obs_times,
        ))

    ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
        observation_settings_list, bodies)

    # simulate the wanted observations 
    simulated_observations = estimation.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies)
    
    observations_list = np.array(simulated_observations.concatenated_observations)
    ra_jpl = np.degrees([np.mod(el, 2*np.pi) for el in observations_list[::2]])
    dec_jpl = np.degrees(observations_list[1::2])

    obs_array = []
    for i in range(len(ra_jpl)):
        obs_array.append([ra_jpl[i], dec_jpl[i]])

    jpl_sim_obs_dict[observatory] = [obs_times, np.array(obs_array)]


"""
Retrieve observations directly from the JPL through a query 
"""
jpl_obs_dict = dict()
for observatory in astrometry_obs_dict.keys():
    obs_times = astrometry_obs_dict[observatory][0]

    state_spice_query_obs = HorizonsQuery(
                query_id= horizons_code,
                location= observatory,
                epoch_list = list(obs_times),
                extended_query=True,
                )

    jpl_observations = state_spice_query_obs.interpolated_observations(degrees=True, reference_system='J2000')

    ra_jpl = jpl_observations[:, 1]
    dec_jpl = jpl_observations[:, 2]

    obs_array = []
    for i in range(len(ra_jpl)):
        obs_array.append([ra_jpl[i], dec_jpl[i]])

    jpl_obs_dict[observatory] = [obs_times, np.array(obs_array)]


"""
Plot the various comparison by accessing the dictionaries 
"""
colors = {'F51': '#ff7f0e', 'W84':'#2ca02c', 'T14':'#1f77b4'}
# validation of the obs model - compared jpl simulated and queried 
fig, axes = plt.subplots(1, 2, figsize=(8,4))
for key in astrometry_obs_dict.keys():
    obs_times = astrometry_obs_dict[key][0]
    axes[0].scatter(obs_times, (jpl_obs_dict[key][1][:,0] - jpl_sim_obs_dict[key][1][:,0]) * 3600, color = colors[key])
    axes[0].set_ylabel("$\Delta$ RA [arcsec]")
    axes[0].set_xlabel("Time since J2000 [s]")
    axes[0].set_title("RA")
    #axes[0].tick_params(axis='both', which='major')
    axes[1].scatter(obs_times, (jpl_obs_dict[key][1][:,1] - jpl_sim_obs_dict[key][1][:,1]) * 3600, color = colors[key])
    axes[1].set_ylabel("$\Delta$ Dec [arcsec]")
    axes[1].set_xlabel("Time since J2000 [s]")
    axes[1].set_title("Dec")
    #axes[1].tick_params(axis='both', which='major')
custom_lines = [
    Line2D([0], [0], color='#ff7f0e', marker='o', linestyle='None', label='PAN-STARRS'),
    Line2D([0], [0], color='#2ca02c', marker='o', linestyle='None', label='DECam'),
    #Line2D([0], [0], color='#1f77b4', marker='o', linestyle='None', label='CFHT'), 
]
# Add the legend
plt.legend(handles=custom_lines, loc='best')
fig.suptitle("Queried - simulated astrometry residuals")
plt.tight_layout()
plt.savefig(f'{config_number}_obs_model_validation.png')

exit(0)

# compare the various results (JPL, remeasured astrometry, estimated orbit)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for key in astrometry_obs_dict.keys():
    obs_times = astrometry_obs_dict[key][0]
    axes[0,0].scatter(obs_times, (jpl_obs_dict[key][1][:,0] - estimated_obs_dict[key][1][:,0]) * 3600, color = colors[key])
    axes[0,0].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2)
    axes[0,0].set_ylabel("$\Delta$ RA [arcsec]")
    axes[0,0].set_xlabel("Time since J2000 [s]")
    axes[0,0].set_title("JPL-estimated")
    axes[0, 1].scatter(obs_times, (jpl_obs_dict[key][1][:,0] - np.rad2deg([np.mod(el, 2*np.pi) for el in astrometry_obs_dict[key][1][:,0]])) * 3600, c = colors[key])
    axes[0,1].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel("Time since J2000 [s]")
    axes[0, 1].set_title("JPL-measured")
    axes[0,2].scatter(obs_times, (estimated_obs_dict[key][1][:,0] - np.rad2deg([np.mod(el, 2*np.pi) for el in astrometry_obs_dict[key][1][:,0]])) * 3600, c = colors[key])
    axes[0,2].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2)
    axes[0,2].set_xlabel("Time since J2000 [s]")
    axes[0,2].set_title("Estimated-measured")

    axes[1,0].scatter(obs_times, (jpl_obs_dict[key][1][:,1] - estimated_obs_dict[key][1][:,1]) * 3600, color = colors[key])
    axes[1,0].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2)
    axes[1,0].set_ylabel("$\Delta$ Dec [arcsec]")
    axes[1,0].set_xlabel("Time since J2000 [s]")
    axes[1, 1].scatter(obs_times, (jpl_obs_dict[key][1][:,1] - np.rad2deg(astrometry_obs_dict[key][1][:,1])) * 3600, color = colors[key])
    axes[1, 1].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel("Time since J2000 [s]")
    axes[1,2].scatter(obs_times, (estimated_obs_dict[key][1][:,1] - np.rad2deg(astrometry_obs_dict[key][1][:,1])) * 3600, color = colors[key])
    axes[1,2].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2)
    axes[1,2].set_xlabel("Time since J2000 [s]")
custom_lines = [
    Line2D([0], [0], color='#ff7f0e', marker='o', linestyle='None', label='PAN-STARRS'),
    Line2D([0], [0], color='#2ca02c', marker='o', linestyle='None', label='DECam'),
    Line2D([0], [0], color='#1f77b4', marker='o', linestyle='None', label='CFHT'), 
    Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Time of perihelion')
]
# Add the legend
plt.tight_layout()
plt.legend(handles=custom_lines, loc='best')

plt.show()