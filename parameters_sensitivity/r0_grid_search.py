"""
Analyze the effects of changing r0 on the residuals 
(fitting your own power law acceleration) and observe which value 
of r0 might help minimize the residuals - get a physical interpretation of this 
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
from comegs.accelerations import PowerLawAcceleration
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
Define the acceleration settings and set up the rid search for an optimal value of r0
"""
if element == 'CO2':
    r0_values = np.linspace(3, 10, 25)
elif element == 'H2O':
    r0_values = np.linspace(1, 5, 20)

perform_search = True
if perform_search:
    rms_ra = {}
    rms_dec = {}
    rms_tot = {}
    for r0 in r0_values:
        """
        Define the environment, settings controlled from the configuration file 
        """
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
                ))

            for link in link_list:
                # add optional bias settings here
                observation_settings_list_2.append(
                    observation.angular_position(link, bias_settings=None)
                )
        except:
            print('No obs in MPC')

        # observations from astrometry files 
        observations = Observations(str(target_mpc_code), epoch_start, epoch_end, bodies)
        # define the ground stations of the observatories 
        observatories_table = MPC.get_observatory_codes().to_pandas()

        observation_settings_list_1, observation_collection_1, astrometry_obs_dict = observations.load_observations_from_file(
            astrometry_results/f'{config_number}.psv', observatories_table, apply_weights=True
            )

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

        dict_acc = dict()
        if element == 'H2O':
            B = 0.32
            n = 2.3
            k = 3.1
        
        elif element == 'CO2':
            B = 0.978
            n = 2.08
            k = 0.68

        nongrav_acc = PowerLawAcceleration(A1, A2, A3, bodies, str(target_mpc_code), 
                                            horizons_code, Dt, epoch_start, epoch_end,
                                            dict_acc, n, r0)

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
        Define the parameters to be estimated - other than the initial state it is also possible to estimate the Marsden model parameters 
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
        try:
            pod_output = estimator.perform_estimation(pod_input)
            # process pod_output here if needed
            """
            Analayse the results 
            """
            results_final = pod_output.parameter_history[:, -1] 

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
            weight_matrix_ra = np.diag(np.array(weights[::2]))
            weight_matrix_dec = np.diag(np.array(weights[1::2]))

            chi_2_ra = res_ra.T @ weight_matrix_ra @ res_ra
            chi_2_dec = res_dec.T @ weight_matrix_dec @ res_dec
            chi_2 = residuals.T @ weight_matrix @ residuals

            key = r0

            rms_ra[key] = np.sqrt((1/len(res_ra))*(chi_2_ra))
            rms_dec[key] = np.sqrt((1/len(res_dec))*(chi_2_dec))

            rms_iteration = np.sqrt((1/len(residuals))*chi_2)
            rms_tot[key] = rms_iteration

        except:
            print(f"Estimation failed, skipping this input.")
            continue  # skip to next iteration
    
    # save the heliocentric distance of the first and last observations to shade them in the plots (highlights where data is present and where it is not)
    obs_times = observation_collection.get_concatenated_observation_times()
    distance_far = HorizonsQuery(
                    query_id= horizons_code,
                    location="500@10",
                    epoch_list = [obs_times[0], obs_times[-1]],
                    extended_query=True,
                    )
    distance_far_pre = (np.linalg.norm(np.array(initial_states.cartesian(frame_orientation='J2000'))[0, 1:3])*u.m).to(u.au).value
    distance_far_post = (np.linalg.norm(np.array(initial_states.cartesian(frame_orientation='J2000'))[1, 1:3])*u.m).to(u.au).value
    distance_close = (target_sbdb.perihelion *u.m).to(u.au).value

    if distance_far_pre > distance_far_post:
        distances = {1: distance_close, 2:distance_far_pre}
    else: 
        distances = {1: distance_close, 2:distance_far_post}

    # save results in files for future use 
    with open(f'acceleration_vs_r0/rms_ra_{config_number}.pkl', 'wb') as f:
        pickle.dump(rms_ra, f)

    with open(f'acceleration_vs_r0/rms_dec_{config_number}.pkl', 'wb') as f:
        pickle.dump(rms_dec, f)

    with open(f'acceleration_vs_r0/rms_tot_{config_number}.pkl', 'wb') as f:
        pickle.dump(rms_tot, f)
    
    with open(f'acceleration_vs_r0/distances_{config_number}.pkl', 'wb') as f:
        pickle.dump(distances, f)

# load files and plot results
with open(f'acceleration_vs_r0/rms_ra_{config_number}.pkl', "rb") as f:
    rms_ra = pickle.load(f)

with open(f'acceleration_vs_r0/rms_dec_{config_number}.pkl', "rb") as f:
    rms_dec = pickle.load(f)

with open(f'acceleration_vs_r0/rms_tot_{config_number}.pkl', "rb") as f:
    rms_tot = pickle.load(f)

with open(f'acceleration_vs_r0/distances_{config_number}.pkl', 'rb') as f:
    distances = pickle.load(f)


# Unpack the dict
r0 = rms_ra.keys()
res_ra = rms_ra.values()

fig, ax = plt.subplots(1,2,figsize=(8, 3))
ax[0].scatter(r0, res_ra, label='RA res')
ax[0].set_xlabel('r_0')
ax[0].set_ylabel('RMS RA')
ax[0].set_yscale('log')

r0 = rms_dec.keys()
res_dec = rms_dec.values()
ax[1].scatter(r0, res_dec, label='Dec res')
ax[1].set_xlabel('r_0')
ax[1].set_ylabel('RMS Dec')
ax[1].set_yscale('log')

plt.tight_layout()
#plt.savefig(f'acceleration_vs_r0/r0_gridsearch_residuals_{config_number}.png')

r0 = rms_tot.keys()
res_tot = rms_tot.values()
print(rms_tot)
plt.figure(figsize=(5,3))
plt.scatter(r0, res_tot, label='Res total')
# shade the part without data 
plt.axvspan(min(r0), distances[1], color='grey', alpha=0.3)
plt.axvspan(distances[2], max(r0), color='grey', alpha=0.3)

# limit x axis between min and max of r0
plt.xlim(min(r0), max(r0))

plt.xlabel('r0 [au]')
plt.ylabel('RMS tot')
plt.yscale('log')
plt.title(f'{target_mpc_code}')
plt.tight_layout()
#plt.savefig(f'acceleration_vs_r0/r0_gridsearch_total_residuals_{config_number}.png')
plt.show()

"""
Calculate the error bars for the estimated r_0 value 
"""
# retrive obs from MPC
config_number = get_number(target_mpc_code)
with open(f"{current_dir}/configuration_files/config_{config_number}.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
configuration = Configuration(config_dict, str(target_mpc_code), horizons_code, epoch_start, epoch_end)
bodies, body_settings = configuration.system_of_bodies(f'{current_dir}/SiMDA_250806.csv', comet_radius=None, density=None, satellites_codes=None, satellites_names=None)
batch = BatchMPC()
batch.get_observations([str(mpc_code)])
batch.filter(
    epoch_start=observations_start,
    epoch_end=observations_end,
    observatories=['568', 'Z84', 'H01', 'G37', 'F65', 'E10', 'F51', 'F52']
)

observation_collection_2 = batch.to_tudat(bodies, None, apply_weights_VFCC17=False)

# observations from astrometry files  
observations = Observations(str(target_mpc_code), epoch_start, epoch_end, bodies)
# define the ground stations of the observatories 
observatories_table = MPC.get_observatory_codes().to_pandas()

observation_settings_list_1, observation_collection_1, astrometry_obs_dict = observations.load_observations_from_file(
    astrometry_results/f'{config_number}.psv', observatories_table, apply_weights=True
    )

observation_collection = numerical_simulation.estimation.merge_observation_collections([observation_collection_1, observation_collection_2])
N = len(observation_collection.get_concatenated_computed_observations())

# calculate the error bars 
r0_vals = np.array(list(rms_tot.keys()), dtype=float)
rms_vals = np.array(list(rms_tot.values()), dtype=float)
# Sort by r_0
idx = np.argsort(r0_vals)
r0_vals = r0_vals[idx]
rms_vals = rms_vals[idx]

# find the minimum RMS between those obtained 
idx_min = np.argmin(rms_vals)
r0_star = r0_vals[idx_min]
rms_min = rms_vals[idx_min]

# express chi^2 sigma = chi^2 min + 1 in terms of RMS
rms_sigma = np.sqrt(rms_min**2 + 1.0/(2*N))   # N = number of observations, saved after filtering 

# indices left of the minimum
left = np.where(rms_vals[:idx_min] > rms_sigma)[0]

if len(left) == 0:
    r0_left = r0_vals[0]   # interval not closed on left
    print('Interval not closed on the left')
else:
    i = left[-1]
    r0_left = np.interp(
        rms_sigma,
        [rms_vals[i], rms_vals[i+1]],
        [r0_vals[i], r0_vals[i+1]]
    )


right = np.where(rms_vals[idx_min:] > rms_sigma)[0]

if len(right) == 0:
    r0_right = r0_vals[-1]   # interval not closed on right
    print('Interval not closed on the right')
else:
    i = right[0] + idx_min - 1
    r0_right = np.interp(
        rms_sigma,
        [rms_vals[i], rms_vals[i+1]],
        [r0_vals[i], r0_vals[i+1]]
    )

sigma_minus = r0_star - r0_left
sigma_plus  = r0_right - r0_star

if abs(r0_left - r0_star) < 1e-12:
    print("Lower bound coincides with minimum (steep or grid-limited)")

if abs(r0_star - r0_right) < 1e-12:
    print("Lower bound coincides with minimum (steep or grid-limited)")

print('Error bar minus:', sigma_minus)
print('Error bar plus:',  sigma_plus)



