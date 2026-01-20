# Tudat imports for propagation and estimation
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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import pandas as pd 
import pickle

from astropy import units as u
from astropy.time import Time

from time import sleep

import os 

current_dir = os.path.dirname(__file__)

class Environment:
    def __init__(self, target, horizons_code, global_frame_origin, global_frame_orientation, epoch_start, epoch_end):
        self.target = target
        self.horizons_code = horizons_code
        self.global_frame_origin = global_frame_origin
        self.global_frame_orientation = global_frame_orientation
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end

    def system_of_bodies(self, extra_bodies, extra_bodies_masses, asteroids, asteroids_masses, comet_radius, density, satellites_codes= None, satellites_names= None):
        # List the bodies for our environment
        bodies_to_create = [
            "Sun",
            "Mercury",
            "Venus",
            "Earth",
            "Moon",
            "Mars",
            "Ceres",
            "Vesta",
            "Jupiter",
            "Io",
            "Europa",
            "Ganymede",
            "Callisto",
            "Saturn",
            "Titan",
            "Uranus",
            "Neptune",]

        # add ephemeris for both extra bodies and asteroids 
        ast_ephemeris = {}
        for code in asteroids:
            query = HorizonsQuery(
                query_id=f"{code};",
                location=f"@{self.global_frame_origin}",
                epoch_start=self.epoch_start - 12 * 31 * 86400,
                epoch_end=self.epoch_end + 12 * 31 * 86400,
                epoch_step=f"{int(360000/60)}m",
                extended_query=True,
            )

            ast_ephemeris[code] = query.create_ephemeris_tabulated(
                frame_origin=self.global_frame_origin,
                frame_orientation=self.global_frame_orientation,
            )

        # Ephemeris for Pluto, Triton and Titania
        extra_ephemeris = {}
        for code in extra_bodies:
            query = HorizonsQuery(
                query_id=f"{code}",
                location=f"@{self.global_frame_origin}",
                epoch_start=self.epoch_start - 12 * 31 * 86400,
                epoch_end=self.epoch_end + 12 * 31 * 86400,
                epoch_step=f"{int(3600000/60)}m",
                extended_query=True,
            )

            extra_ephemeris[code] = query.create_ephemeris_tabulated(
                frame_origin=self.global_frame_origin,
                frame_orientation=self.global_frame_orientation,
            )

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
                        epoch_step=f"{int(3600000/60)}m", # Horizons does not permit a stepsize in seconds
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
                        epoch_step=f"{int(3600000/60)}m", # Horizons does not permit a stepsize in seconds
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
                        epoch_step=f"{int(3600000/60)}m", # Horizons does not permit a stepsize in seconds
                        extended_query=True, # extended query allows for more data to be retrieved.
                    )

                    sat_ephemeris[name] = query.create_ephemeris_tabulated(
                        frame_origin=self.global_frame_origin,
                        frame_orientation=self.global_frame_orientation,
                    )
                
                with open("dict.pkl", "wb") as f:
                    pickle.dump(used_satellites, f)

        # Create system of bodies through SPICE
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, self.global_frame_origin, self.global_frame_orientation
        )

        # Add asteroids, their ephemerides and gravity field to body settings
        for asteroid_code, asteroid_mass in zip(asteroids, asteroids_masses):
            body_settings.add_empty_settings(str(asteroid_code))
            body_settings.get(str(asteroid_code)).ephemeris_settings = ast_ephemeris[asteroid_code]
            body_settings.get(str(asteroid_code)).gravity_field_settings = (
                environment_setup.gravity_field.central(asteroid_mass * GRAVITATIONAL_CONSTANT)
            )

        # Add Pluto, Triton and Titania and their ephemerides and gravity field to body settings
        for extra_code, extra_mass in zip(extra_bodies, extra_bodies_masses):
            body_settings.add_empty_settings(str(extra_code))
            body_settings.get(str(extra_code)).ephemeris_settings = extra_ephemeris[extra_code]
            body_settings.get(str(extra_code)).gravity_field_settings = (
                environment_setup.gravity_field.central(extra_mass * GRAVITATIONAL_CONSTANT)
            )
        
        if satellites_names is not None: 
            for name in used_satellites.values():
                body_settings.add_empty_settings(name)
                body_settings.get(name).ephemeris_settings = sat_ephemeris[name]

        # add settings (mass, radiation pressure interface) and ephemeris to the comet analyzed 
        body_settings.add_empty_settings(str(self.target))
        comet_ephemeris = {}

        query_comet = HorizonsQuery(
            query_id=self.horizons_code,
            location=f"@{self.global_frame_origin}",
            epoch_start=self.epoch_start - 24 * 31 * 86400,
            epoch_end=self.epoch_end + 24 * 31 * 86400,
            epoch_step=f"{int(360000/60)}m",
            extended_query=True,
        )
        comet_ephemeris[str(self.target)] = query_comet.create_ephemeris_tabulated(
            frame_origin=self.global_frame_origin,
            frame_orientation=self.global_frame_orientation
        )
        body_settings.get(str(self.target)).ephemeris_settings = comet_ephemeris[str(self.target)]

        # add the radiation pressure coefficient
        reference_area_radiation = comet_radius**2*np.pi  
        # the coeff depends on the reflectivity of the nucleus, usually they're dark 
        radiation_pressure_coefficient = 1.2 
        comet_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
            reference_area_radiation, radiation_pressure_coefficient)
        body_settings.get(str(self.target)).radiation_pressure_target_settings = comet_target_settings

        # add the mass: calculated as V*rho 
        body_settings.get(str(self.target)).constant_mass = comet_radius**3*(4/3)*np.pi*density

        # set up the accurate rotation model for the Earth 
        precession_nutation_theory = environment_setup.rotation_model.IAUConventions.iau_2006
        original_frame = "J2000"
        body_settings.get( "Earth" ).rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
        precession_nutation_theory, original_frame)

        # create the Bodies objects with all the updated and complete settings 
        bodies = environment_setup.create_system_of_bodies(body_settings)

        return bodies, body_settings
    
    def acceleration_grav(self, extra_bodies, asteroids, nongrav_acc):
        acceleration_settings = {}

        accelerations = {
            "Sun": [
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
                # propagation_setup.acceleration.radiation_pressure(),
            ],
            "Mercury": [propagation_setup.acceleration.point_mass_gravity()],
            "Venus": [propagation_setup.acceleration.point_mass_gravity()],
            "Earth": [
                propagation_setup.acceleration.point_mass_gravity(),
            ],
            "Moon": [propagation_setup.acceleration.point_mass_gravity()],

            "Mars": [propagation_setup.acceleration.point_mass_gravity()],

            "Ceres": [propagation_setup.acceleration.point_mass_gravity()],
            "Vesta": [propagation_setup.acceleration.point_mass_gravity()],

            "Jupiter": [propagation_setup.acceleration.spherical_harmonic_gravity(3,3)],
            "Io": [propagation_setup.acceleration.point_mass_gravity()],
            "Europa": [propagation_setup.acceleration.point_mass_gravity()],
            "Ganymede": [propagation_setup.acceleration.point_mass_gravity()],
            "Callisto": [propagation_setup.acceleration.point_mass_gravity()],

            "Saturn": [propagation_setup.acceleration.point_mass_gravity()],
            "Titan": [propagation_setup.acceleration.point_mass_gravity()],

            "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
            "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
        }

        # For each asteroid + Pluto, Titania and Triton we create a point mass gravity.
        asteroid_accelerations = {str(num):[propagation_setup.acceleration.point_mass_gravity()] for num in asteroids}
        extra_accelerations = {str(num):[propagation_setup.acceleration.point_mass_gravity()] for num in extra_bodies}

        acceleration_settings = {str(self.target) : {**accelerations,
                                            **asteroid_accelerations, 
                                            **extra_accelerations,                                                                                                   
                                        }}
        
        return acceleration_settings

    def acceleration_nongrav(self, extra_bodies, asteroids, nongrav_acc):
        acceleration_settings = {}

        accelerations = {
            "Sun": [
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
                # propagation_setup.acceleration.radiation_pressure(),
                propagation_setup.acceleration.custom_acceleration(nongrav_acc.compute_acc)
            ],
            "Mercury": [propagation_setup.acceleration.point_mass_gravity()],
            "Venus": [propagation_setup.acceleration.point_mass_gravity()],
            "Earth": [
                propagation_setup.acceleration.point_mass_gravity(),
            ],
            "Moon": [propagation_setup.acceleration.point_mass_gravity()],

            "Mars": [propagation_setup.acceleration.point_mass_gravity()],

            "Ceres": [propagation_setup.acceleration.point_mass_gravity()],
            "Vesta": [propagation_setup.acceleration.point_mass_gravity()],

            "Jupiter": [propagation_setup.acceleration.spherical_harmonic_gravity(3,3)],
            "Io": [propagation_setup.acceleration.point_mass_gravity()],
            "Europa": [propagation_setup.acceleration.point_mass_gravity()],
            "Ganymede": [propagation_setup.acceleration.point_mass_gravity()],
            "Callisto": [propagation_setup.acceleration.point_mass_gravity()],

            "Saturn": [propagation_setup.acceleration.point_mass_gravity()],
            "Titan": [propagation_setup.acceleration.point_mass_gravity()],

            "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
            "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
        }

        # For each asteroid + Pluto, Titania and Triton we create a point mass gravity.
        asteroid_accelerations = {str(num):[propagation_setup.acceleration.point_mass_gravity()] for num in asteroids}
        extra_accelerations = {str(num):[propagation_setup.acceleration.point_mass_gravity()] for num in extra_bodies}

        acceleration_settings = {str(self.target) : {**accelerations,
                                            **asteroid_accelerations, 
                                            **extra_accelerations,                                                                                                   
                                        }}
        
        return acceleration_settings


class Integrator:
    def __init__(self,epoch_start,  epoch_end, central_bodies, bodies_to_propagate, acceleration_models, initial_guess):
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.central_bodies = central_bodies 
        self.bodies_to_propagate = bodies_to_propagate 
        self.acceleration_models = acceleration_models 
        self.initial_guess = initial_guess 

    def variable_step_size(self, termination_condition, propagator, integrator_coeff, current_tol, abs_tol):
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            1.0,
            integrator_coeff,
            0.01,
            1000000,
            current_tol,
            abs_tol)
        
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies=self.central_bodies,
            acceleration_models=self.acceleration_models,
            bodies_to_integrate=self.bodies_to_propagate,
            initial_states=self.initial_guess,
            initial_time=self.epoch_start,
            integrator_settings=integrator_settings,
            termination_settings=termination_condition,
            propagator=propagator,
        )

        return propagator_settings
    
    def fixed_step_size(self, termination_condition, propagator, integrator_coeff, step_size):
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
                step_size, integrator_coeff)

        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies=self.central_bodies,
            acceleration_models=self.acceleration_models,
            bodies_to_integrate=self.bodies_to_propagate,
            initial_states=self.initial_guess,
            initial_time=self.epoch_start,
            integrator_settings=integrator_settings,
            termination_settings=termination_condition,
            propagator=propagator,
        )

        return propagator_settings
