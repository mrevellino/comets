"""
Get residuals of MPC data wrt to JPL horizons orbit 
Get residuals of your own data wrt JPL horizons orbit 
analyze the residuals 
"""
# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy.data.horizons import HorizonsQuery
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
import pickle
from astropy.time import Time
from astroquery.mpc import MPC
from pathlib import Path
from collections import defaultdict

# import the costum acceleration 
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
configuration = Configuration(config_dict, target_mpc_code, horizons_code, epoch_start, epoch_end)
bodies, body_settings = configuration.system_of_bodies(f'{workspace}/SiMDA_250806.csv', comet_radius=None, density=None, satellites_codes=None, satellites_names=None)

# define the central body and body to be propagated (in this case the comet)
bodies_to_propagate = [str(target_mpc_code)]
central_bodies = ['Sun']

"""
Get the observations from the MPC 
"""
batch = BatchMPC()
batch.get_observations([str(mpc_code)])
batch.filter(
    epoch_start=observations_start,
    epoch_end=observations_end,
    observatories_exclude=['C51', 'C57', 'C53']
)

observations_mpc = batch.table

temp_dict = defaultdict(list)
# Collect all [epoch, RA, DEC] per observatory
for _, row in observations_mpc.iterrows():
    temp_dict[row['observatory']].append([row['epochJ2000secondsTDB'], row['RA'], row['DEC']])

# Build the final dictionary
mpc_obs_dict = {}
for obs, values in temp_dict.items():
    values = np.array(values)  # convert to NumPy array for easy slicing
    times = values[:, 0].tolist()        # list of epochs
    coords = values[:, 1:3]              # array of RA, DEC
    mpc_obs_dict[obs] = [times, coords]

"""
Retrieve observations directly from the JPL through a query 
"""
jpl_mpc_obs_dict = dict()
for observatory in mpc_obs_dict.keys():
    print(observatory)
    obs_times = mpc_obs_dict[observatory][0]

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

    jpl_mpc_obs_dict[observatory] = [obs_times, np.array(obs_array)]


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
Retrieve observations directly from the JPL through a query 
"""
jpl_astro_obs_dict = dict()
for observatory in astrometry_obs_dict.keys():
    print(observatory)
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

    jpl_astro_obs_dict[observatory] = [obs_times, np.array(obs_array)]


"""
Get the residuals in the 2 cases 
"""
# mpc 
RA_mpc = []
Dec_mpc = []
obs_times_mpc = []
for key in mpc_obs_dict.keys():
    obs_times = mpc_obs_dict[key][0]
    mpc_coords = mpc_obs_dict[key][1]
    jpl_coords = jpl_mpc_obs_dict[key][1]

    for i in range(len(obs_times)):
        ra_diff = (jpl_coords[i, 0] - np.rad2deg(np.mod(mpc_coords[i, 0], 2*np.pi))) * 3600
        dec_diff = (jpl_coords[i, 1] - np.rad2deg(mpc_coords[i, 1])) * 3600

        # Keep only if both differences are within (-3, 3)
        if -3 < ra_diff < 3 and -3 < dec_diff < 3:
            obs_times_mpc.append(obs_times[i])
            RA_mpc.append(ra_diff)
            Dec_mpc.append(dec_diff)

# astrometry 
RA_ast = []
Dec_ast = []
obs_times_ast = []
for key in astrometry_obs_dict.keys():
    obs_times = astrometry_obs_dict[key][0]
    ast_coords = astrometry_obs_dict[key][1]
    jpl_coords = jpl_astro_obs_dict[key][1]

    for i in range(len(obs_times)):
        ra_diff = (jpl_coords[i, 0] - np.rad2deg(np.mod(ast_coords[i, 0], 2*np.pi))) * 3600
        dec_diff = (jpl_coords[i, 1] - np.rad2deg(ast_coords[i, 1])) * 3600

        # Keep only if both differences are within (-3, 3)
        if -3 < ra_diff < 3 and -3 < dec_diff < 3:
            obs_times_ast.append(obs_times[i])
            RA_ast.append(ra_diff)
            Dec_ast.append(dec_diff)

"""
Plot the residuals 
"""
fig,ax = plt.subplots(2,2, figsize = (8,7))
ax[0,0].scatter(obs_times_mpc, RA_mpc, color = colors[0])
ax[1,0].scatter(obs_times_mpc, Dec_mpc, color = colors[1])

ax[0,1].scatter(obs_times_ast, RA_ast, color = colors[2])
ax[1,1].scatter(obs_times_ast, Dec_ast, color = colors[3])
ax[0,0].set_ylabel("$\Delta$ RA [arcsec]")
ax[0,0].set_xlabel("Time since J2000 [s]")
ax[0,0].set_title("Difference jpl-mpc")
ax[1,0].set_ylabel("$\Delta$ Dec [arcsec]")
ax[1,0].set_xlabel("Time since J2000 [s]")
ax[0,1].set_xlabel("Time since J2000 [s]")
ax[0,1].set_title("Difference jpl-astro")
ax[1,1].set_xlabel("Time since J2000 [s]")
plt.show()