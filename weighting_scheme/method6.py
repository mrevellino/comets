"""
Perform an estimation using both data coming from your 
own astrometric measureemnts and data coming from only selected observatories of the MPC
Apply a scheme for MPC obs = 1/sigma^2 sqrt(N), with sigma = veresx2 amd N = obs per motnh
ADES weights to astrometry obs
"""

# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup, propagation_setup, estimation, estimation_setup
from tudatpy.constants import GRAVITATIONAL_CONSTANT
from tudatpy.data.horizons import HorizonsQuery
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.data.mpc import BatchMPC
from tudatpy.data.sbdb import SBDBquery

# other useful modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd 
import os
import re 
import yaml
from astropy.time import Time
from astroquery.mpc import MPC
from pathlib import Path
from astropy import units as u
from matplotlib.lines import Line2D
from numpy.linalg import eigh, inv
from scipy.stats import chi2, kstest

# import the costum acceleration 
from comegs.accelerations import MarsdenAcceleration, COMarsdenAcceleration, YabushitaAcceleration
from comegs.settings import Integrator
from comegs.observations import Observations
from comegs.config_files import Configuration

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

target_mpc_code = '2019 U5' 
mpc_code = 'C/' + target_mpc_code
element = 'CO2'

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

"""
Define the environment, settings controlled from the configuration file 
"""
ra_residuals = {}
dec_residuals = {}
for i in range(2): 
    config_number = get_number(target_mpc_code)
    with open(f"{workspace}/configuration_files/config_{config_number}.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    configuration = Configuration(config_dict, str(target_mpc_code), horizons_code, epoch_start, epoch_end)
    bodies, body_settings = configuration.system_of_bodies(f'{workspace}/SiMDA_250806.csv', comet_radius=None, density=None, satellites_codes=None, satellites_names=None)

    # define the central body and body to be propagated (in this case the comet)
    bodies_to_propagate = [str(target_mpc_code)]
    central_bodies = ['Sun']


    """
    Load the astrometrical measurements 
    """
    # observations from the MPC
    batch = BatchMPC()
    batch.get_observations([str(mpc_code)])
    batch.filter(
        epoch_start=observations_start,
        epoch_end=observations_end,
        observatories=['568', 'Z84', 'H01', 'G37', 'F65', 'E10', 'F51', 'F52']
    )
    obs_df = batch.table
    obs_df['epochUTC'] = pd.to_datetime(obs_df['epochUTC'])
    # extract just the year and month
    obs_df['obs_month'] = obs_df['epochUTC'].dt.to_period('M')
    obs_per_month = obs_df.groupby('obs_month').size().reset_index(name='count')

    sigma_arcsec = np.select(
        [obs_df['observatory'] == '568',
        obs_df['observatory'] == 'H01',
        obs_df['observatory'].isin(['F51', 'F52'])],
        [0.4, 0.6, 0.4],
        default=2  
        ) * u.arcsec
    sigma = sigma_arcsec.to(u.rad).value

    obs_df['weight'] = 1 / (sigma**2 * np.sqrt(obs_df.groupby('obs_month')['obs_month'].transform('count')))
    weights_list = obs_df['weight'].tolist()
    batch.set_weights(weights_list)

    try:
        observation_collection_2 = batch.to_tudat(bodies, None, apply_weights_VFCC17=False)
        observation_settings_list_2 = list()
        link_list = list(
            observation_collection_2.get_link_definitions_for_observables(
                observable_type=observation.angular_position_type
            )
        )

        for link in link_list:
            # add optional bias settings here
            observation_settings_list_2.append(
                observation.angular_position(link, bias_settings=None)
            )
        
        # only get the MPC obs to compare if they are good
        observations_MPC = observation_collection_2.get_concatenated_computed_observations()
        obs_times_MPC = observation_collection_2.get_concatenated_observation_times()
        ra_MPC = observations_MPC[::2]
        dec_MPC = observations_MPC[1::2]
    except:
        print('No obs in MPC')

    # observations from your astrometry 
    observations = Observations(str(target_mpc_code), epoch_start, epoch_end, bodies)
    # define the ground stations of the observatories 
    observatories_table = MPC.get_observatory_codes().to_pandas()

    observation_settings_list_1, observation_collection_1, astrometry_obs_dict = observations.load_observations_from_file(
        astrometry_results/f'{config_number}.psv', observatories_table, apply_weights=True
        )

    obs_times_astro = []
    for key in astrometry_obs_dict.keys():
        obs_times_astro.extend(astrometry_obs_dict[key][0])

    try: 
        observation_settings_list = observation_settings_list_1 + observation_settings_list_2
        observation_collection = numerical_simulation.estimation.merge_observation_collections([observation_collection_1, observation_collection_2])
    except: 
        observation_collection = observation_collection_1
        observation_settings_list = observation_settings_list_1

    observations = observation_collection.get_concatenated_computed_observations()
    obs_times = observation_collection.get_concatenated_observation_times()

    epoch_start = sorted(obs_times)[0] - 30*86400
    epoch_end = sorted(obs_times)[-1] + 30*86400


    """
    Define the acceleration settings; this is also defined in the configuration file and can be retrieved from the configuration class
    """
    dict_acc = dict()
    nongrav_acc = MarsdenAcceleration(A1, A2, A3, bodies, str(target_mpc_code), horizons_code, Dt, epoch_start, epoch_end, dict_acc)

    acceleration_settings = configuration.acceleration_model(nongrav_acc)

    # create the acceleration model
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    """
    Retrieve an initial guess for the comet's position
    """ 
    initial_states = HorizonsQuery(
                query_id= horizons_code,
                location="500@10",
                epoch_list = [epoch_start ,epoch_end],
                extended_query=True,
                )

    initial_guess = np.array(initial_states.cartesian(frame_orientation='J2000'))[0, 1:]

    """
    Define the integrator/propagator 
    """
    integrator = Integrator(epoch_start, epoch_end, central_bodies, bodies_to_propagate, acceleration_models, initial_guess)
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
    Define the parameters to be estimated - other than the initial state it is also possible to estimate the Marsden model parameters and the Dt
    """
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies
    )
    if i == 0:
        parameter_settings.append(estimation_setup.parameter.custom_parameter(
                            "marsden_acc.custom_values", 3, nongrav_acc.get_custom_parameters, nongrav_acc.set_custom_parameters
                        )
        )

        parameter_settings[-1].custom_partial_settings = [
                        estimation_setup.parameter.custom_analytical_partial(
                            nongrav_acc.compute_parameter_partials, target_mpc_code, "Sun",
                            propagation_setup.acceleration.AvailableAcceleration.custom_acceleration_type
                        )
                    ]

    parameters_to_estimate = estimation_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )

    """
    Set up the estimation 
    """
    # Set up the estimator
    estimator = numerical_simulation.Estimator(
        bodies=bodies,
        estimated_parameters=parameters_to_estimate,
        observation_settings=observation_settings_list,
        propagator_settings=propagator_settings,
        integrate_on_creation=True,
    )

    # provide the observation collection as input, and limit number of iterations for estimation.
    pod_input = estimation.EstimationInput(
        observations_and_times=observation_collection,
        convergence_checker=estimation.estimation_convergence_checker(
            maximum_iterations=number_of_pod_iterations,
        ),
    )

    # Set methodological options
    pod_input.define_estimation_settings(reintegrate_variational_equations=True)

    """
    Perform the estimation 
    """
    pod_output = estimator.perform_estimation(pod_input)

    """
    Analayse the results 
    """
    results_final = pod_output.parameter_history[:, -1]    
    formal_errors = pod_output.formal_errors
    # compute the SNR of each computed parameter 
    SNR = results_final/formal_errors
    print(SNR)

    residual_history = pod_output.residual_history[:,-1]
    ra = residual_history[::2]
    dec = residual_history[1::2]

    ra_residuals[i] = ra
    dec_residuals[i] = dec


"""
Plot the residuals
"""
fig, ax = plt.subplots(2,2, figsize=(8,6))
ax[0,0].scatter(obs_times, (ra_residuals[0]* u.rad).to(u.arcsec).value)
ax[0,0].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2, label='Time of perihelion')
ax[0,0].set_title('RA')
ax[0,0].set_xlabel('Time since J2000 [sec]')
ax[0,0].set_ylabel('RA residuals [arcsec]')
ax[0,0].legend()
ax[0,1].scatter(obs_times, (dec_residuals[0]* u.rad).to(u.arcsec).value)
ax[0,1].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2)
ax[0,1].set_title('Dec')
ax[0,1].set_xlabel('Time since J2000 [sec]')
ax[0,1].set_ylabel('Dec residuals [arcsec]')
ax[1,0].scatter(obs_times, (ra_residuals[1]* u.rad).to(u.arcsec).value)
ax[1,0].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2, label='Time of perihelion')
ax[1,0].set_title('RA - estimating A1, A2, A3')
ax[1,0].set_xlabel('Time since J2000 [sec]')
ax[1,0].set_ylabel('RA residuals [arcsec]')
ax[1,0].legend()
ax[1,1].scatter(obs_times, (dec_residuals[1]* u.rad).to(u.arcsec).value)
ax[1,1].axvline(x=time_perihelion, color='red', linestyle='--', linewidth=2)
ax[1,1].set_title('Dec - estimating A1, A2, A3')
ax[1,1].set_xlabel('Time since J2000 [sec]')
ax[1,1].set_ylabel('Dec residuals [arcsec]')

obs_times = observation_collection.get_concatenated_observation_times()
for i, el in enumerate(obs_times):
    if el in obs_times_MPC:
        ax[0,0].scatter(el, (ra_residuals[0][i]* u.rad).to(u.arcsec).value, color = 'red')
        ax[0,1].scatter(el, (dec_residuals[0][i] * u.rad).to(u.arcsec).value, color = 'red')
        ax[1,0].scatter(el, (ra_residuals[1][i] * u.rad).to(u.arcsec).value, color = 'red')
        ax[1,1].scatter(el, (dec_residuals[1][i] * u.rad).to(u.arcsec).value, color = 'red')
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Measured astrometry',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='MPC astrometry',
           markerfacecolor='red', markersize=8)
]
plt.tight_layout()
plt.legend(handles=legend_elements, loc='best')


fig, ax = plt.subplots(1,2, figsize =(7,3), constrained_layout=True)
left, bottom, width, height = 0.1, 0.15, 0.38, 0.75  # normalized figure coordinates
ax[0].set_position([left, bottom, width, height])
ax[1].set_position([left + width + 0.05, bottom, width, height])  # 0.05 is spacing between plots

ax[0].scatter((ra_residuals[0]* u.rad).to(u.arcsec).value, (dec_residuals[0]* u.rad).to(u.arcsec).value)
ax[0].axhline(0, color='gray', linestyle='--')
ax[0].axvline(0, color='gray', linestyle='--')
ax[0].grid(True, linestyle=':', alpha=0.5)
ax[0].set_xlabel('RA residuals [arcsec]')   # X-axis label
ax[0].set_ylabel('Dec residuals [arcsec]')  # Y-axis label
ax[0].set_title(f'RA vs Dec')

ax[1].scatter((ra_residuals[1]* u.rad).to(u.arcsec).value, (dec_residuals[1]* u.rad).to(u.arcsec).value)
ax[1].axhline(0, color='gray', linestyle='--')
ax[1].axvline(0, color='gray', linestyle='--')
ax[1].set_xlabel('RA residuals [arcsec]')
ax[1].set_ylabel('Dec residuals [arcsec]')
ax[1].set_title(f'RA vs Dec - estimating A1,A2,A3')
ax[1].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()



