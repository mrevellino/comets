"""
Compare the estimated orbit applying the implemented Marden model 
with the JPL Horizons ephemeris 
"""
# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.data.sbdb import SBDBquery

# other useful modules
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from astropy import units as u
import pandas as pd 
from astropy.time import Time
import os
import re  
import yaml
from pathlib import Path

from comegs.accelerations import MarsdenAcceleration, COMarsdenAcceleration
from comegs.plotting import PlotError
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

target_mpc_code = '2006 S3' 
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
Simulate the observations
"""
observations = Observations(target_mpc_code, epoch_start, epoch_end, bodies)
observation_settings_list, simulated_observations = observations.cartesian_observations()

"""
Create the acceleration settings
"""
dict_acc = dict()
nongrav_acc = MarsdenAcceleration(A1, A2, A3, bodies, str(target_mpc_code), 
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
initial_states = bodies.get(str(target_mpc_code)).ephemeris.cartesian_state(epoch_start) - bodies.get("Sun").ephemeris.cartesian_state(epoch_start)

# Add random offset for initial guess
rng = np.random.default_rng(seed=1)
initial_position_offset = 1e6 * 1000
initial_velocity_offset = 100
# add offset 
initial_guess = initial_states.copy()
initial_guess[0:3] += (2 * rng.random(3) - 1) * initial_position_offset 
initial_guess[3:6] += (2 * rng.random(3) - 1) * initial_velocity_offset 


"""
Define integrator and propoagator
"""

integrator = Integrator(epoch_start, epoch_end, central_bodies, bodies_to_propagate, acceleration_models, initial_guess)
termination_condition = propagation_setup.propagator.time_termination(epoch_end)
propagator_settings = integrator.fixed_step_size(termination_condition, propagation_setup.propagator.cowell, propagation_setup.integrator.CoefficientSets.rkf_56, 400000)

"""
Set up the estimation
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
Perform the estimation
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
    simulated_observations,
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
Retrieve the estimation results
"""
results_final = pod_output.parameter_history[:, -1] 
print(results_final)


vector_error_initial = (np.array(initial_guess) - initial_states)[0:3]
error_magnitude_initial = np.sqrt(np.square(vector_error_initial).sum()) / 1000

vector_error_final = (np.array(results_final[:6]) - initial_states)[0:3]
error_magnitude_final = np.sqrt(np.square(vector_error_final).sum()) / 1000

print(
    f"{target_mpc_code} initial guess radial error to spice: {round(error_magnitude_initial, 2)} km"
)
print(
    f"{target_mpc_code} final radial error to spice: {round(error_magnitude_final, 2)} km"
)

"""
Plot the error with respect to spice
"""
plotting = PlotError(bodies, epoch_start, epoch_end, horizons_code, target_mpc_code)

gaps = plotting.gaps(simulated_observations)
plot_error_total = plotting.plot_error_total(global_frame_origin, gaps, f"{current_dir}/cartesian_error_parameters_{config_number}.png")
plot_error_single = plotting.plot_error_single(global_frame_origin, gaps, None)

"""
Calculate the difference between the estimated non-gravitational parameters
"""
A1 = config_dict['marsden_params']['A1']*aud2_to_ms2
A2 = config_dict['marsden_params']['A2']*aud2_to_ms2
A3 = config_dict['marsden_params']['A3']*aud2_to_ms2

A1_estimated = results_final[-4]
A2_estimated = results_final[-3]
A3_estimated = results_final[-2]

diff_A1 = (A1 - A1_estimated)/aud2_to_ms2 
diff_A2 = (A2 - A2_estimated)/aud2_to_ms2 
diff_A3 = (A3 - A3_estimated)/aud2_to_ms2 
print('difference in the estimated parameters:')
print(diff_A1)
print(diff_A2)
print(diff_A3)

print('Estimated Dt', results_final[-1]/86400)

plt.show()
