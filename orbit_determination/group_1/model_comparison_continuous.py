"""
Fit your selected empirical acceleration to astrometric data 
and compare with the residuals obtained applying the Marsden acceleration
Note differences 
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
import pandas as pd 
import os
import re 
import yaml
from astropy.time import Time
from astroquery.mpc import MPC
from pathlib import Path
from astropy import units as u
import matplotlib.lines as mlines

# import the costum acceleration 
from comegs.accelerations import ContinousAcceleration, COMarsdenAcceleration,  MarsdenAcceleration, PowerLawAcceleration
from comegs.settings import Integrator
from comegs.observations import Observations
from comegs.config_files import Configuration


# set-up current directory 
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
home = Path().home()
astrometry_results = home / f"Desktop/astrometry_results/Complete"

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

target_mpc_code = '2017 K2' 
mpc_code = 'C/' + target_mpc_code
element = 'CO'

residuals_ra = {}
residuals_dec = {}
obs_times_dict = {}
formal_errors_dict = {}
parameters_dict = {}
for i in range(2):
    """
    Read the relative config file to define the horizons code, etc...
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
    Set up parameters for the estimation
    """ 
    number_of_pod_iterations = 8

    # define the frame origin and orientation
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"

    epoch_start = (Time(observations_start).jd - 2451545.0)*86400 - 30*86400
    epoch_end = (Time(observations_end).jd - 2451545.0)*86400 + 30*86400

    """
    Define the environment (through the configuration file)
    """
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
    obs_times_astro = observation_collection_1.get_concatenated_observation_times()

    try: 
        observation_settings_list = observation_settings_list_1 + observation_settings_list_2
        observation_collection = numerical_simulation.estimation.merge_observation_collections([observation_collection_1, observation_collection_2])
    except: 
        observation_collection = observation_collection_1
        observation_settings_list = observation_settings_list_1

    observations = observation_collection.get_concatenated_computed_observations()
    obs_times = observation_collection.get_concatenated_observation_times()
    obs_times_dict[i] = obs_times

    epoch_start = sorted(obs_times)[0] - 30*86400
    epoch_end = sorted(obs_times)[-1] + 30*86400

    """
    Define the acceleration settings; this is also defined in the configuration file and can be retrieved from the configuration class
    """
    dict_acc = dict()

    if i == 0:
        r0_h20 = 2.8
        r0_co2 = 7
        r0_co = 30
        C = 0.5
        nongrav_acc = ContinousAcceleration(A1, A2, A3, bodies, str(target_mpc_code), 
                                            horizons_code, Dt, epoch_start, epoch_end, dict_acc, 
                                            element, r0_h20, r0_co2, r0_co, C)
    elif i == 1:
        nongrav_acc = COMarsdenAcceleration(A1, A2, A3, bodies, str(target_mpc_code), 
                                            horizons_code, Dt, epoch_start, epoch_end,
                                            dict_acc)
        

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
    integrator = Integrator(epoch_start, epoch_end, central_bodies, bodies_to_propagate, 
                            acceleration_models, initial_guess)
    termination_condition = propagation_setup.propagator.time_termination(epoch_end)

    if config_dict['numerical']['propagator'] == 'propagation_setup.propagator.gauss_modified_equinoctial':
        propagator = propagation_setup.propagator.gauss_modified_equinoctial
    elif config_dict['numerical']['propagator'] == 'propagation_setup.propagator.cowell':
        propagator = propagation_setup.propagator.cowell

    if config_dict['numerical']['integrator']['coefficients'] == 'propagation_setup.integrator.CoefficientSets.rkf_56':
        coefficients = propagation_setup.integrator.CoefficientSets.rkf_56
    elif config_dict['numerical']['integrator']['coefficients'] == 'propagation_setup.integrator.CoefficientSets.rkf_78':
        coefficients = propagation_setup.integrator.CoefficientSets.rkf_78

    propagator_settings = integrator.fixed_step_size(termination_condition, propagator,
                                                    coefficients, float(config_dict['numerical']['integrator']['stepsize']))


    """
    Define the parameters to be estimated - other than the initial state it is also possible to estimate the Marsden model parameters and the Dt
    """
    # Setup parameters settings to propagate the state transition matrix
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies
    )

    if i == 0: 
        # estimate the A1, A2, A3 and parameters
        parameter_settings.append(estimation_setup.parameter.custom_parameter(
                            "marsden_acc.custom_values", 3, nongrav_acc.get_custom_parameters, 
                            nongrav_acc.set_custom_parameters
                        )
        )

        # estimate the C parameter 
        parameter_settings.append(estimation_setup.parameter.custom_parameter(
                            "marsden_acc.C", 1, nongrav_acc.get_custom_C, 
                            nongrav_acc.set_custom_C
                        )
        )

        parameter_settings[-2].custom_partial_settings = [
                        estimation_setup.parameter.custom_analytical_partial(
                            nongrav_acc.compute_parameter_partials, target_mpc_code, "Sun",
                            propagation_setup.acceleration.AvailableAcceleration.custom_acceleration_type
                        )
                    ]

        parameter_settings[-1].custom_partial_settings = [
                        estimation_setup.parameter.custom_analytical_partial(
                            nongrav_acc.compute_C_partials, target_mpc_code, "Sun",
                            propagation_setup.acceleration.AvailableAcceleration.custom_acceleration_type
                        )
                    ]

    elif i == 1:
        parameter_settings.append(estimation_setup.parameter.custom_parameter(
                            "marsden_acc.custom_values", 3, nongrav_acc.get_custom_parameters, 
                            nongrav_acc.set_custom_parameters
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
    pod_input.define_estimation_settings(reintegrate_variational_equations=True)

    """
    Perform the estimation 
    """
    pod_output = estimator.perform_estimation(pod_input)

    """
    Retrieve estimation results
    """   
    formal_errors = pod_output.formal_errors
    final_parameters = pod_output.final_parameters

    parameters_dict[i] = final_parameters
    formal_errors_dict[i] = formal_errors


"""
For the 2 models applied evaluate the direction of the accelration vector defined by the estimated A1, A2 and A3 parameters 
"""
A1_pl = parameters_dict[0][-4]
A2_pl = parameters_dict[0][-3]
A3_pl = parameters_dict[0][-2]

sA1_pl = formal_errors_dict[0][-4]
sA2_pl = formal_errors_dict[0][-3]
sA3_pl = formal_errors_dict[0][-2]

A1_Mars = parameters_dict[1][-3]
A2_Mars = parameters_dict[1][-2]
A3_Mars = parameters_dict[1][-1]

sA1_Mars = formal_errors_dict[1][-3]
sA2_Mars = formal_errors_dict[1][-2]
sA3_Mars = formal_errors_dict[1][-1]

A_pl = np.array([A1_pl, A2_pl, A3_pl])
A_M  = np.array([A1_Mars, A2_Mars, A3_Mars])

sig_pl = np.array([sA1_pl, sA2_pl, sA3_pl])
sig_M  = np.array([sA1_Mars, sA2_Mars, sA3_Mars])

# monte carlo sampling of the A parameters in the 2 cases 
# each matrix is a 3 x 10000 size mayrtrix, with each row being a combination of A1, A2, A3 
# sampled from their uncertainties 

N = 10000 # number of Monte Carlo samples

Apl_samples = np.random.normal(
    loc=A_pl,
    scale=sig_pl,
    size=(N, 3)
)

AM_samples = np.random.normal(
    loc=A_M,
    scale=sig_M,
    size=(N, 3)
)

# normalize each row of the matrix = each acceleration vector 
# this allows to get the direction vector 
def normalize(v):
    norm = np.linalg.norm(v, axis=1)
    return v / norm[:, None]

Apl_hat = normalize(Apl_samples)
AM_hat  = normalize(AM_samples)

# get the distribution of angles obtained from the dot product of each line of the 2 matrices
cos_theta = np.sum(Apl_hat * AM_hat, axis=1)
theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

theta_med = np.median(theta)
theta_1s  = np.percentile(theta, [16, 84])
theta_2s  = np.percentile(theta, [2.5, 97.5])

# average the component of each vector (average A1, average A2, average A3) and then calculathe 
# the mean direction from the average components obtained 
# do not use the nominal direction estimated, cause it might be different from the mean if the errors have 
# different magnitudes in the different directions 
def mean_direction(vhat):
    m = np.mean(vhat, axis=0)
    return m / np.linalg.norm(m)

mean_pl = mean_direction(Apl_hat)
mean_M  = mean_direction(AM_hat)
print(mean_pl)
print(mean_M)

# calculate the abgular separation of each monte carlo vector from the mean 
def angular_distance(vhat, vmean):
    cosang = np.sum(vhat * vmean, axis=1)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

spread_pl = angular_distance(Apl_hat, mean_pl)
spread_M  = angular_distance(AM_hat, mean_M)

print(spread_pl)
print(spread_M)

# get the 1 sigma and 2 sigma distributions of the angular separation calculated 
cone_pl_1s = np.percentile(spread_pl, 68)
cone_pl_2s = np.percentile(spread_pl, 95)

cone_M_1s  = np.percentile(spread_M, 68)
cone_M_2s  = np.percentile(spread_M, 95)

# mu = angle between the calculated mean directions 
mu = np.degrees(np.arccos(
    np.clip(np.dot(mean_pl, mean_M), -1, 1)
))
# two cones overlap if the separation of their axes ≤ sum of their half-angles
overlap_1s = mu <= (cone_pl_1s + cone_M_1s)
overlap_2s = mu <= (cone_pl_2s + cone_M_2s)

print('overlap over 1 sigma', overlap_1s)
print('overlap over 2 sigma', overlap_2s)

# visualization 
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*Apl_hat.T, s=1, alpha=0.1, label='PL', color = colors[1])
ax.scatter(*AM_hat.T,  s=1, alpha=0.1, label='Marsden', color = colors[2])

ax.quiver(
    0, 0, 0,
    *mean_pl,
    linewidth=2,
    color=colors[1],
    arrow_length_ratio=0.05,
    pivot='tail'
)

ax.quiver(
    0, 0, 0,
    *mean_M,
    linewidth=2,
    color=colors[2],
    arrow_length_ratio=0.05,
    pivot='tail'
)

ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('R')
ax.set_ylabel('T')
ax.set_zlabel('N')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0],
           marker='o',
           linestyle='None',
           markersize=6,
           markerfacecolor=colors[1],
           markeredgecolor='none',
           label='Multi volatiles'),

    Line2D([0], [0],
           marker='o',
           linestyle='None',
           markersize=6,
           markerfacecolor=colors[2],
           markeredgecolor='none',
           label='Marsden')
]

ax.legend(handles=legend_elements)
ax.set_title(f'{config_number}')

fig.subplots_adjust(
    left=0.05,
    right=0.95,
    bottom=0.05,
    top=0.9
)

plt.savefig(f'direction/{config_number}_continuous_direction.png')
plt.show()



