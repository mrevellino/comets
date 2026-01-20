"""
Comparison of weighting schemes 5 and 6 
"""

# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import propagation_setup, estimation, estimation_setup
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
import matplotlib.lines as mlines
from numpy.linalg import eigh, inv
from scipy.stats import chi2, kstest

# import the costum acceleration 
from comegs.accelerations import MarsdenAcceleration, COMarsdenAcceleration
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

# define the environment through the configuration file 
ra_residuals = {}
dec_residuals = {}
formal_errors_dict = {}
state_history_dict = {}
epoch_start_end = {}
covariance_dict = {}
snr_dict = {}
semi_major_dict = {}
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
    # observatorion from the MPC
    batch = BatchMPC()
    batch.get_observations([str(mpc_code)])
    batch.filter(
        epoch_start=observations_start,
        epoch_end=observations_end,
        observatories=['568', 'Z84', 'H01', 'G37', 'F65', 'E10', 'F51', 'F52']
    )
    obs_df = batch.table
    if i == 0:
        obs_df['epochUTC'] = pd.to_datetime(obs_df['epochUTC'])
        # extract just the date (no time)
        obs_df['obs_date'] = obs_df['epochUTC'].dt.date
        # keep only one obs per month 
        obs_df['epochUTC'] = pd.to_datetime(obs_df['epochUTC'])
        obs_df['year_month'] = obs_df['epochUTC'].dt.to_period('M')
        obs_df = obs_df.drop_duplicates(subset='year_month', keep='first')
        obs_df= obs_df.drop(columns=['year_month'])
        obs_df = obs_df.drop(columns=['obs_date'])
        obs_df = obs_df.drop(columns=['epochUTC'])
        obs_df['band'] = 0
        obs_df = obs_df.reset_index(drop=True)

        batch = BatchMPC()
        batch.from_pandas(obs_df, False, 'J2000')
        batch.summary()

        obs_df = batch.table
        sigma_arcsec = np.select(
            [obs_df['observatory'] == '568',
            obs_df['observatory'] == 'H01',
            obs_df['observatory'].isin(['F51', 'F52'])],
            [0.4, 0.6, 0.4],
            default=2  
            ) * u.arcsec
        sigma = sigma_arcsec.to(u.rad).value
        obs_df['weight'] = 1 / (sigma**2)
        weights_list = obs_df['weight'].tolist()
        batch.set_weights(weights_list)
    
    elif i == 1: 
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

    # observations from astrometry file
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

    epoch_start_end[i] = [epoch_start, epoch_end]


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
    Define the parameters to be estimated - other than the initial state it is also possible to estimate the Marsden model parameters 
    """
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies
    )

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

    # Create the parameters that will be estimated
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
    pod_input.define_estimation_settings(reintegrate_variational_equations=True, save_state_history_per_iteration=True)

    """
    Perform the estimation 
    """
    pod_output = estimator.perform_estimation(pod_input)

    """
    Analayse the results 
    """ 
    results_final = pod_output.parameter_history[:, -1]    
    formal_errors = pod_output.formal_errors
    formal_errors_dict[i] = formal_errors

    # propagate the covariance and save it for each method 
    covariance_propagated = estimation.propagate_covariance(pod_output.covariance, estimator.state_transition_interface, obs_times)
    covariance_dict[i] = covariance_propagated

    # compute the SNR of each computed parameter 
    SNR = results_final/formal_errors
    snr_dict[i] = SNR

    # compute the residuals in the 2 cases 
    residual_history = pod_output.residual_history[:,-1]
    ra = residual_history[::2]
    dec = residual_history[1::2]

    ra_residuals[i] = ra
    dec_residuals[i] = dec

    # save the reisudlas semi-major axis in the 2 cases 
    X = np.vstack([(ra* u.rad).to(u.arcsec).value, (dec* u.rad).to(u.arcsec).value]).T
    N = X.shape[0]
    mean = X.mean(axis=0)
    Xc = X - mean
    Sigma = np.cov(Xc, rowvar=False, bias=False)
    Sigma = 0.5*(Sigma + Sigma.T)
    # eigen-decompose (eigh returns ascending eigenvalues)
    vals, vecs = eigh(Sigma)
    vals = vals.real
    # order descending
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    # semi-axes for 68.3% joint
    CHI2_68 = chi2.ppf(0.683, df=2)
    a = np.sqrt(vals[0] * CHI2_68)

    # get the residulas only of the measured astroemtry
    ra_astro = []
    dec_astro = []
    for j, el in enumerate(obs_times):
        if el in obs_times_astro:
            ra_astro.append((ra[j]* u.rad).to(u.arcsec).value)
            dec_astro.append((dec[j]* u.rad).to(u.arcsec).value)
    X = np.vstack([ra_astro, dec_astro]).T
    N = X.shape[0]
    mean = X.mean(axis=0)
    Xc = X - mean
    # sample covariance (unbiased)
    Sigma = np.cov(Xc, rowvar=False, bias=False)
    # ensure symmetric
    Sigma = 0.5*(Sigma + Sigma.T)
    # eigen-decompose (eigh returns ascending eigenvalues)
    vals, vecs = eigh(Sigma)
    vals = vals.real
    # order descending
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    # semi-axes for 68.3% joint
    CHI2_68 = chi2.ppf(0.683, df=2)
    a_m = np.sqrt(vals[0] * CHI2_68)

    semi_major_dict[i] = {'a_tot': a, 'a_m': a_m}

    """
    Save the state history of the object 
    """
    times =  np.linspace(epoch_start_end[0][0], epoch_start_end[0][1], 2000)
    state_history = {}
    for j in range(len(times)):
        state_est = bodies.get(f'{target_mpc_code}').ephemeris.cartesian_state(times[j]) 
        state_history[times[j]] = state_est
    
    state_history_dict[i] = state_history


"""
Print the SNR and the semi-major axis of the residuals
"""
print(snr_dict)
print(semi_major_dict)

"""
Plot the difference in orbit between the 2 weighting algorithms
"""
state_history1 = state_history_dict[0]
state_history2 = state_history_dict[1]
error = []
for key in state_history1.keys():
    error.append(list(np.array(state_history1[key]) - np.array(state_history2[key])))

error = np.array(error)
times =  np.linspace(epoch_start_end[0][0], epoch_start_end[0][1], 2000)

colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '9467bd', 'c5b0d5', '8c564b', 'c49c94', 'e377c2', 'f7b6d2', '7f7f7f', 'c7c7c7', 'bcbd22', 'dbdb8d', '17becf', '9edae5']

fig, ax = plt.subplots(figsize = (4, 3))
ax.plot(times, error[:, 0]/1000, label="x", color = colors[0])
ax.plot(times, error[:, 1]/1000, label="y", color = colors[4])
ax.plot(times, error[:, 2]/1000, label="z", color = colors[2])
ax.set_ylabel('Position difference [km]')
ax.set_xlabel('Time since J2000 [sec]')

x_error = []
y_error = []
z_error = []
covariance_6 = covariance_dict[0]
for key, item in covariance_6.items():
    x_error.append(np.sqrt(np.diag(item)[0]))
    y_error.append(np.sqrt(np.diag(item)[1]))
    z_error.append(np.sqrt(np.diag(item)[2]))

# shade the covarinace values 
ax.fill_between([el for el in list(covariance_6.keys())], [el/1000 for el in x_error], [-el/1000 for el in x_error], alpha=0.2, color = colors[0])
ax.fill_between([el for el in list(covariance_6.keys())], [el/1000 for el in y_error], [-el/1000 for el in y_error], alpha=0.2, color = colors[4])
ax.fill_between([el for el in list(covariance_6.keys())], [el/1000 for el in z_error], [-el/1000 for el in z_error], alpha=0.2, color = colors[2])


# Combine existing legend + new handles
ax.set_ylabel('Position difference [km]')
ax.set_xlabel('Time since J2000 [s]')
ax.set_title(f'{mpc_code}')
ax.legend(ncol = 2)

plt.tight_layout()
plt.show()

