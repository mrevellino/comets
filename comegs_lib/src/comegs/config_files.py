"""
Python code to read the config files and adjuct the evironment accordingly - 
this functionalities can be used to test dufferent dynamical models without having 
the different models hard-coded into the script; this allows for versatility and 
readability of the files 

Different config files will define the different bodies to have in the system
and the different acceleration settinsg to be used during the estimation
"""

from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro import frame_conversion
from tudatpy.util import result2array
from tudatpy.math import interpolators
from tudatpy.astro import element_conversion
from tudatpy.constants import GRAVITATIONAL_CONSTANT

# import MPC interface
from tudatpy.data.mpc import BatchMPC

# import SBDB interface
from tudatpy.data.sbdb import SBDBquery

# import Horizons interface 
from tudatpy.data.horizons import HorizonsQuery

# other useful modules
import numpy as np
import datetime
import pandas as pd 
import yaml 
from pathlib import Path
import pickle
from astropy.time import Time


current_dir = Path(__file__).resolve().parent
home_dir = current_dir.parent.parent.parent


class Configuration: 
    def __init__(self, config_dict, target, horizons_code, epoch_start, epoch_end):
        self.target = target 
        self.horizons_code = horizons_code 
        self.start = epoch_start 
        self.end = epoch_end 
        self.config = config_dict
        self.asteroids = None
    
    
    def system_of_bodies(self, simda_file_path, comet_radius = None, density = None, satellites_codes = None, satellites_names = None):
        
        global_frame_origin = self.config['frame']['global_frame_origin']
        global_frame_orientation = self.config['frame']['global_frame_orientation']
        
        # creation of bodies included in the yaml file 
        bodies_to_create = self.config['bodies']

        # additional bodies not contained in standard spice kernels 
        extra_bodies = self.config['extra_bodies'].keys()
        extra_bodies_masses = self.config['extra_bodies'].values()
        extra_ephemeris = {}
        for code in extra_bodies:
            query = HorizonsQuery(
                query_id=f"{code}",
                location=f"@{global_frame_origin}",
                epoch_start=self.start - 12 * 31 * 86400,
                epoch_end=self.end + 12 * 31 * 86400,
                epoch_step=f"{int(3600)}m",
                extended_query=True,
            )

            extra_ephemeris[code] = query.create_ephemeris_tabulated(
                frame_origin=global_frame_origin,
                frame_orientation=global_frame_orientation,
            )

        # additional massive asteroids, with mass > than the minimum specified in the config file 
        min_asteroid_mass = float(self.config['asteroids']['min_mass'])
        simda = (
            pd.read_csv(simda_file_path)
            .iloc[18:] # the first 18 rows contain comets, which are omitted
            .assign(NUM=lambda x: np.int32(x.NUM))
            .query("(MASS > @min_asteroid_mass)") # filter relevant bodies
            .query("NUM != [1, 4]") # remove Ceres and Vesta which are retrieved through spice kernel
            .loc[:, ["NUM", "DESIGNATION", "DIAM", "DYN", "MASS"]]
        )
        asteroids = simda.NUM.to_list()
        asteroids_masses = simda.MASS.to_list()
        self.asteroids = asteroids
        ast_ephemeris = {}
        for code in asteroids:
            query = HorizonsQuery(
                query_id=f"{code};",
                location=f"@{global_frame_origin}",
                epoch_start=self.start - 12 * 31 * 86400,
                epoch_end=self.end + 12 * 31 * 86400,
                epoch_step=f"{int(3600)}m",
                extended_query=True,
            )

            ast_ephemeris[code] = query.create_ephemeris_tabulated(
                frame_origin=global_frame_origin,
                frame_orientation=global_frame_orientation,
            )

        # satellites ephemeris if satellites observations available
        if satellites_names is not None: 
            main_satellites = {'WISE':['-163', 'C51'], 'TESS':['-95', 'C57'], 'NEOSSat':['-139089', 'C53']}
            sat_ephemeris = {}
            t_j2000 = Time("2000-01-01T12:00:00", scale="tdb")
            t_tess = (Time("2018-05-01T00:00:00", scale="utc").tdb - t_j2000).to_value('sec')# UTC input
            t_wise = (Time("2024-10-01T00:00:00", scale="utc").tdb - t_j2000).to_value('sec') # UTC input

            used_satellites = dict()

            for code, name in zip(satellites_codes, satellites_names):
                print(code, name)
                if name == 'TESS' and self.epoch_start < t_tess:
                    used_satellites[main_satellites[name][0]] = name
                    query = HorizonsQuery(
                        query_id=code,
                        location=f"@{self.global_frame_origin}",
                        epoch_start= t_tess,
                        epoch_end=self.epoch_end,
                        epoch_step=f"{int(3600)}m", # Horizons does not permit a stepsize in seconds
                        extended_query=True, # extended query allows for more data to be retrieved.
                    )
                    sat_ephemeris[name] = query.create_ephemeris_tabulated(
                        frame_origin=self.global_frame_origin,
                        frame_orientation=self.global_frame_orientation,
                    )
                elif name == 'WISE' and self.epoch_end > t_wise:
                    used_satellites[main_satellites[name][0]] = name
                    query = HorizonsQuery(
                        query_id=code,
                        location=f"@{self.global_frame_origin}",
                        epoch_start= self.epoch_start,
                        epoch_end=t_wise,
                        epoch_step=f"{int(3600)}m", # Horizons does not permit a stepsize in seconds
                        extended_query=True, # extended query allows for more data to be retrieved.
                    )
                    sat_ephemeris[name] = query.create_ephemeris_tabulated(
                        frame_origin=self.global_frame_origin,
                        frame_orientation=self.global_frame_orientation,
                    )
                else:
                    used_satellites[main_satellites[name][0]] = name
                    query = HorizonsQuery(
                        query_id=code,
                        location=f"@{self.global_frame_origin}",
                        epoch_start= self.epoch_start,
                        epoch_end=self.epoch_end,
                        epoch_step=f"{int(3600)}m", # Horizons does not permit a stepsize in seconds
                        extended_query=True, # extended query allows for more data to be retrieved.
                    )

                    sat_ephemeris[name] = query.create_ephemeris_tabulated(
                        frame_origin=self.global_frame_origin,
                        frame_orientation=self.global_frame_orientation,
                    )
                
                with open("dict.pkl", "wb") as f:
                    pickle.dump(used_satellites, f)
        

        # create standard body settings 
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation
        )

        # Add extra bodies and their ephemerides and gravity field to body settings
        for extra_code, extra_mass in zip(extra_bodies, extra_bodies_masses):
            body_settings.add_empty_settings(str(extra_code))
            body_settings.get(str(extra_code)).ephemeris_settings = extra_ephemeris[extra_code]
            body_settings.get(str(extra_code)).gravity_field_settings = (
                environment_setup.gravity_field.central(float(extra_mass) * GRAVITATIONAL_CONSTANT)
            )   

        # Add asteroids, their ephemerides and gravity field to body settings
        for asteroid_code, asteroid_mass in zip(asteroids, asteroids_masses):
            body_settings.add_empty_settings(str(asteroid_code))
            body_settings.get(str(asteroid_code)).ephemeris_settings = ast_ephemeris[asteroid_code]
            body_settings.get(str(asteroid_code)).gravity_field_settings = (
                environment_setup.gravity_field.central(asteroid_mass * GRAVITATIONAL_CONSTANT)
            )

        # add the target and its ephemeris to the system of bodies 
        body_settings.add_empty_settings(str(self.target))
        comet_ephemeris = {}
        query_comet = HorizonsQuery(
            query_id=self.horizons_code,
            location=f"@{global_frame_origin}",
            epoch_start=self.start - 24 * 31 * 86400,
            epoch_end=self.end + 24 * 31 * 86400,
            epoch_step=f"{int(3600)}m",
            extended_query=True,
        )
        comet_ephemeris[str(self.target)] = query_comet.create_ephemeris_tabulated(
            frame_origin=global_frame_origin,
            frame_orientation=global_frame_orientation
        )
        body_settings.get(str(self.target)).ephemeris_settings = comet_ephemeris[str(self.target)]    

        #eventual modifications / additions to the system of bodies - to be read from the config file 
        for setting_type in self.config['body_settings'].keys():
            for body, setting in self.config['body_settings'][setting_type].items():
                if setting_type == 'rotation_model':
                    body_settings.get(body).rotation_model_settings = exec(setting)
                elif setting_type == 'radiation_pressure_settings':
                    body_settings.get(body).radiation_pressure_target_settings = exec(setting)
                elif setting_type == 'gravitational_parameter': 
                    body_settings.get(body).gravity_field_settings.gravitational_parameter  = float(setting)#environment_setup.gravity_field.central(float(setting)) # this should be a float

        # add Earth shape model, common to all cases 
        body_settings.get("Earth").shape_settings = (
            environment_setup.shape.oblate_spherical_spice()
        )

        # create the Bodies objects with all the updated and complete settings 
        bodies = environment_setup.create_system_of_bodies(body_settings)

        return bodies, body_settings
                             

    def acceleration_model(self, nongrav_acc):

        function_mapping = {
           'propagation_setup.acceleration.point_mass_gravity()' : propagation_setup.acceleration.point_mass_gravity(),
           'propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)': propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
           'propagation_setup.acceleration.spherical_harmonic_gravity(3,3)' : propagation_setup.acceleration.spherical_harmonic_gravity(3,3),
           'propagation_setup.acceleration.spherical_harmonic_gravity(6,6)' : propagation_setup.acceleration.spherical_harmonic_gravity(6,6),
           'propagation_setup.acceleration.custom_acceleration(nongrav_acc.compute_acc)' : propagation_setup.acceleration.custom_acceleration(nongrav_acc.compute_acc)
        }

        if self.config is None: 
            self.config = self.read_file()

        accelerations = self.config['accelerations']
        for key, value in accelerations.items():
            accelerations[key] = [function_mapping.get(el) for el in value]

        extra_accelerations = {str(num):[propagation_setup.acceleration.point_mass_gravity()] for num in self.config['extra_bodies'].keys()}
        
        asteroid_accelerations = {str(num):[propagation_setup.acceleration.point_mass_gravity()] for num in self.asteroids} 

        acceleration_settings = {str(self.target) : {**accelerations,
                                            **asteroid_accelerations, 
                                            **extra_accelerations,                                                                                                   
                                        }}
        return acceleration_settings
    

