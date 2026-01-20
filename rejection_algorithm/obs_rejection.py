"""
Check if observations from trusted observatories are rejected when the rejection algorithm is applied 
Raise a True/False flag and count the number of rejected obsevations
"""

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
import os
import re 
import yaml
from astropy.time import Time
from astroquery.mpc import MPC
from pathlib import Path
import pickle
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from astropy import units as u 

# import the costum acceleration 
from comegs.accelerations import  PowerLawAcceleration
from comegs.settings import Integrator
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
configuration = Configuration(config_dict, str(target_mpc_code), horizons_code, epoch_start, epoch_end)
bodies, body_settings = configuration.system_of_bodies(f'{workspace}/SiMDA_250806.csv', comet_radius=None, density=None, satellites_codes=None, satellites_names=None)

# define the central body and body to be propagated (in this case the comet)
bodies_to_propagate = [str(target_mpc_code)]
central_bodies = ['Sun']

"""
Load astrometric measurements from the MPC
"""
batch = BatchMPC()
batch.get_observations([str(mpc_code)])
obs_df = batch.table
obs_df['epochUTC'] = pd.to_datetime(obs_df['epochUTC'])
# extract just the date (no time)
obs_df['obs_date'] = obs_df['epochUTC'].dt.date
# count how many observations per date and define weights
obs_per_day = obs_df.groupby('obs_date').size().reset_index(name='count')

sigma_arcsec = np.select(
    [obs_df['observatory'] == '568',
    obs_df['observatory'] == 'H01',
    obs_df['observatory'].isin(['F51', 'F52']), 
    obs_df['observatory'].isin(['Z84', 'G37', 'F65', 'E10', 'W84', 'T14', '309'])],
    [0.4, 0.6, 0.4, 2],
    default=3  
    ) * u.arcsec
sigma = sigma_arcsec.to(u.rad).value

obs_df['weight'] = 1 / (sigma**2 *np.sqrt(obs_df.groupby('obs_date')['obs_date'].transform('count')))
weights_list = obs_df['weight'].tolist()
batch.set_weights(weights_list)


observation_collection = batch.to_tudat(bodies, None, apply_weights_VFCC17=False)
observation_settings_list = list()
link_list = list(
    observation_collection.get_link_definitions_for_observables(
        observable_type=observation.angular_position_type
    )
)

for link in link_list:
    # add optional bias settings here
    observation_settings_list.append(
        observation.angular_position(link, bias_settings=None)
    )


observations = observation_collection.get_concatenated_computed_observations()
obs_times = observation_collection.get_concatenated_observation_times()

# define star and end times based on the retrived observations
epoch_start = sorted(obs_times)[0] - 30*86400
epoch_end = sorted(obs_times)[-1] + 30*86400

"""
Define the acceleration settings; this is also defined in the configuration file and can be retrieved from the configuration class
"""
dict_acc = dict()
# element parameters 
if element == 'H2O':
    n = 2.3
    r0_pl = 2.8

elif element == 'CO2': 
    n = 2
    r0_pl = 7

nongrav_acc = PowerLawAcceleration(A1, A2, A3, bodies, str(target_mpc_code), 
                                            horizons_code, Dt, epoch_start, epoch_end,
                                            dict_acc, n, r0_pl)

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
parameter_settings = estimation_setup.parameter.initial_states(
    propagator_settings, bodies
)

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
Implement easy rejection algorithm - for 3 iteration discard all obs that create a residual > 5 sigma 
Not too harsh otherwise trends from non-grav might go away
"""
sigma = 1e3
for j in range(3): 
    
    """
    Filter the observations
    """
    filter_obj = numerical_simulation.estimation.observation_filter(
        numerical_simulation.estimation.residual_filtering,
        5*sigma,
        use_opposite_condition=False
    )

    observation_collection.filter_observations(filter_obj)
    print(len(observation_collection.get_concatenated_residuals()))

    """
    Set up the estimation 
    """
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
            maximum_iterations= number_of_pod_iterations,
        ),
    )
    # Set methodological options
    pod_input.define_estimation_settings(reintegrate_variational_equations=True)
    pod_output = estimator.perform_estimation(pod_input)
    residuals= pod_output.residual_history[:,-1]
    computed_obs = observation_collection.get_concatenated_computed_observations()
    dec = computed_obs[1::2]
    res_ra = residuals[::2]*np.cos(dec)
    res_dec = residuals[1::2]

    radial = np.sqrt(res_ra**2 + res_dec**2)
    sigma = np.std(radial, ddof=1)


observation_collection_filtered = observation_collection

obs_filtered_dict = {}
observations = observation_collection_filtered.get_concatenated_observations()
dec_filtered = observations[1::2]
ra_filtered = observations[::2]
for i, el in enumerate(observation_collection_filtered.get_concatenated_observation_times()):
    obs_filtered_dict[el] = np.array([ra_filtered[i], dec_filtered[i]])


# get the observations from the body obtaned from trusted observatories
batch = BatchMPC()
batch.get_observations([str(mpc_code)])
batch.filter(
    epoch_start=observations_start,
    epoch_end=observations_end,
    observatories=['568', 'Z84', 'H01', 'G37', 'F65', 'E10', 'F51', 'F52']
)
observation_collection_trusted = batch.to_tudat(bodies, None, apply_weights_VFCC17=False)
obs_trusted_dict = {}
observations = observation_collection_trusted.get_concatenated_observations()
dec_trusted = observations[1::2]
ra_trusted = observations[::2]
for i, el in enumerate(observation_collection_trusted.get_concatenated_observation_times()):
    obs_trusted_dict[el] = np.array([ra_trusted[i], dec_trusted[i]])


is_contained = all(
    key in obs_filtered_dict and np.allclose(obs_filtered_dict[key], value)
    for key, value in obs_trusted_dict.items()
)

not_contained = [
    key
    for key, value in obs_trusted_dict.items()
    if key not in obs_filtered_dict
    or not np.allclose(obs_filtered_dict[key], value)
]

print(not_contained)
print(obs_trusted_dict[float(not_contained)])

count_not_contained = len(not_contained)

print(len(observation_collection_trusted.get_concatenated_observations()))
print(is_contained)
print(count_not_contained)