"""
Orbit determination for analysed comets from group 2, for which all MPC data are analysed
Any of the implemented acceleration models can be applied
Estimation of the initial state and the A1, A2, A3 parameters 
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
from comegs.accelerations import COMarsdenAcceleration,  MarsdenAcceleration, PowerLawAcceleration, ContinousAcceleration
from comegs.settings import Integrator
from comegs.observations import Observations
from comegs.config_files import Configuration
from comegs.plotting import PlotError

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

target_mpc_code = '2021 D2' 
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
# set up data needed for the estimation 
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
Load the astrometrical measurements from the MPC
"""
# observations from the MPC
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

batch.summary()


try:
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
except:
    print('No obs in MPC')

# based on the available observations define the start and end epochs for the estimation
observations = observation_collection.get_concatenated_computed_observations()
obs_times = observation_collection.get_concatenated_observation_times()

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

elif element == 'CO': 
    n = 2
    r0_pl = 30       

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
# retrieve the initial state directly wrt Horizons 
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

# estimate the A1, A2, A3 parameters
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

# Iterative filtering of the available observations
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
    Estimate the 1-sigma value of residuals 
    """
    residuals= pod_output.residual_history[:,-1]
    computed_obs = observation_collection.get_concatenated_computed_observations()
    dec = computed_obs[1::2]
    res_ra = residuals[::2]*np.cos(dec)
    res_dec = residuals[1::2]

    radial = np.sqrt(res_ra**2 + res_dec**2)
    sigma = np.std(radial, ddof=1)

"""
Retrieve the results of the estimation
"""    
final_parameters = pod_output.final_parameters
formal_errors = pod_output.formal_errors

"""
RMS resdiuals 
"""
residuals= pod_output.residual_history[:,-1]
computed_obs = observation_collection.get_concatenated_computed_observations()
dec = computed_obs[1::2]
res_ra = residuals[::2]*np.cos(dec)
res_dec = residuals[1::2]
n = res_ra.size
residuals = np.empty(n*2)

residuals[0::2] = res_ra
residuals[1::2] = res_dec

weights = observation_collection.get_concatenated_weights()
weight_matrix = np.diag(np.array(weights))

chi_2 = residuals.T @ weight_matrix @ residuals
rms_tot = np.sqrt((1/len(residuals))*chi_2)

print('Final estimated parameter:',final_parameters)
print('Final estimated formal errors:',formal_errors)
print('Final normalised RMS residuals:', rms_tot)
exit(0)
"""
Plot the correlation matrix 
"""
fig, ax = plt.subplots(figsize=(8, 4))

estimated_param_names = ["x", "y", "z", "vx", "vy", "vz", "A1", "A2", "A3", 'r0']

correlations = pod_output.correlations
im = ax.imshow(
    np.abs(correlations),
    aspect='auto',
    interpolation='none',
    cmap='viridis_r'
)

fig.colorbar(im, ax=ax)

ax.set_title("Continuous")
ax.set_xlabel("Estimated Parameter")
ax.set_ylabel("Estimated Parameter")

ax.set_xticks(np.arange(len(estimated_param_names)))
ax.set_xticklabels(estimated_param_names)

ax.set_yticks(np.arange(len(estimated_param_names)))
ax.set_yticklabels(estimated_param_names)

#plt.savefig(f"corr_matrix_{config_number}.png")

"""
Plot the residuals
"""

fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].scatter(obs_times, (res_ra* u.rad).to(u.arcsec).value, color = colors[0])
ax[0].set_ylabel('RA res [arcsec]')
ax[0].set_xlabel('Time since J2000 [s]')

ax[1].scatter(obs_times, (res_dec* u.rad).to(u.arcsec).value, color = colors[0])
ax[1].set_ylabel('Dec res [arcsec]')
ax[1].set_xlabel('Time since J2000 [s]')

# horizontal line at y = 0 (grey)
ax[0].axhline(0, color='grey', linestyle='--', linewidth=1)
ax[1].axhline(0, color='grey', linestyle='--', linewidth=1)

# vertical line at x = time_perihelion (red)
ax[0].axvline(time_perihelion, color='red', linestyle='--', linewidth=1)
ax[1].axvline(time_perihelion, color='red', linestyle='--', linewidth=1)


plt.tight_layout()
plt.show()

