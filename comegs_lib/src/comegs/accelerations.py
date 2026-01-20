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
import math
from astropy import units as u
from time import sleep
import os 

current_dir = os.path.dirname(__file__)


class Acceleration:
    def __init__(self, A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end):
        self.custom_values = [A1, A2, A3]
        self.bodies = bodies 
        self.target = target
        self.code_horizons = code_horizons
        self.Dt = np.array([Dt]) #time lag in seconds
        self.start = epoch_start
        self.end = epoch_end
        self.state_interpolator = None
        self._cached_value = None 

    def fake_comet_settings(self, current_time):
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
            "Neptune"]

        body_settings_fakecomet = environment_setup.get_default_body_settings(
            bodies_to_create, 'SSB', 'J2000')      

        body_settings_fakecomet.add_empty_settings('fake_comet')

        bodies_fakecomet = environment_setup.create_system_of_bodies(body_settings_fakecomet)
        # Define bodies that are propagated
        bodies_to_propagate_fakecomet = ["fake_comet"]
        # Define central bodies of propagation
        central_bodies_fakecomet = ["Sun"]

        accelerations = {
            "Sun": [
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
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

        # # For each asteroid + Pluto, Titania and Triton we create a point mass gravity.
        # asteroid_accelerations = {str(num):[propagation_setup.acceleration.point_mass_gravity()] for num in self.asteroids}
        # extra_accelerations = {str(num):[propagation_setup.acceleration.point_mass_gravity()] for num in self.extra_bodies}

        acceleration_settings_fakecomet = {"fake_comet" : {**accelerations}}

        acceleration_models_fakecomet = propagation_setup.create_acceleration_models(
            bodies_fakecomet,
            acceleration_settings_fakecomet,
            bodies_to_propagate_fakecomet,
            central_bodies_fakecomet)

        if float(self.Dt) < 0:
            initial_state_fakecomet = self.bodies.get(self.target).ephemeris.cartesian_state(self.end) - self.bodies.get("Sun").ephemeris.cartesian_state(self.end)
            termination_condition_fakecomet = propagation_setup.propagator.time_termination(self.end + abs(2*float(self.Dt)))
        
            integrator_settings_fakecomet = propagation_setup.integrator.runge_kutta_fixed_step(
                abs(float(self.Dt))/500, coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_56
            )

            # Create propagation settings
            propagator_settings = propagation_setup.propagator.translational(
                ["Sun"],
                acceleration_models_fakecomet,
                ["fake_comet"],
                initial_state_fakecomet,
                self.end,
                integrator_settings_fakecomet,
                termination_condition_fakecomet,                
            )
            dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                bodies_fakecomet, propagator_settings
            )

            # Extract the resulting state and dependent variable history and convert it to an ndarray
            states = dynamics_simulator.propagation_results.state_history

        elif float(self.Dt) > 0:
            initial_state_fakecomet = self.bodies.get(self.target).ephemeris.cartesian_state(self.start) - self.bodies.get("Sun").ephemeris.cartesian_state(self.start)
            termination_condition_fakecomet = propagation_setup.propagator.time_termination(self.start - abs(2*float(self.Dt)))
        
            integrator_settings_fakecomet = propagation_setup.integrator.runge_kutta_fixed_step(
               - abs(float(self.Dt))/500, coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_56
            )

            # Create propagation settings
            propagator_settings = propagation_setup.propagator.translational(
                ["Sun"],
                acceleration_models_fakecomet,
                ["fake_comet"],
                initial_state_fakecomet,
                self.start,
                integrator_settings_fakecomet,
                termination_condition_fakecomet,                
            )
            dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                bodies_fakecomet, propagator_settings
            )

            # Extract the resulting state and dependent variable history and convert it to an ndarray
            states = dynamics_simulator.propagation_results.state_history

        body_settings_fakecomet.get('fake_comet').ephemeris_settings = environment_setup.ephemeris.tabulated(
            states,
            'Sun', 
            'J2000')
        bodies_fakecomet = environment_setup.create_system_of_bodies(body_settings_fakecomet)

        # self.bodies_fakecomet = bodies_fakecomet
        # bodies_fakecomet = environment_setup.create_system_of_bodies(body_settings_fakecomet)

        return bodies_fakecomet    

    def get_cartesian_state(self, current_time):
        if float(self.Dt) != 0:
            if (self.start - 100) <= current_time <= (self.start + 100) or self._cached_value is None:
                self._cached_value = self.fake_comet_settings(current_time)
            try: 
                cartesian_state = self.bodies.get(self.target).ephemeris.cartesian_state(current_time = (current_time - float(self.Dt))) - self.bodies.get("Sun").ephemeris.cartesian_state(current_time = (current_time - float(self.Dt)))
            except:
                cartesian_state = self._cached_value.get('fake_comet').ephemeris.cartesian_state(current_time = (current_time - float(self.Dt)))
        else: 
           cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        return cartesian_state 
    
    
class MarsdenAcceleration(Acceleration):
    def __init__(self, A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end, dict):

        super().__init__(A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end)

        self.dict = dict
        # self._cached_value = None

    def compute_acc(self, current_time):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        r_au = helio_distance.to(u.au).value 
        alpha =  0.1112620426
        r0 = 2.808
        m = 2.15
        n = 5.093
        k = 4.6142

        g_r = alpha*((r_au/r0)**(-m))*(1 + (r_au/r0)**n)**(-k)

        acc = np.array([self.custom_values[0] * g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        self.dict[current_time] = list([r_au, self.custom_values[0] * g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        acc_inertial = rotation_matrix @ acc.tolist()

        return acc_inertial         

    def compute_parameter_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state
        position = cartesian_state[:3]
        velocity = cartesian_state[3:]
        position_norm = np.linalg.norm(position)

        R = position/position_norm 
        h = np.cross(position, velocity)
        h_norm = np.linalg.norm(h)
        N = h/h_norm 
        T = np.cross(N, R)
        
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            position_norm = np.linalg.norm(cartesian_state[:3])
        r = position_norm * u.m
        # r is in meters, in the formula it has to be in au - use astropy for the conversion
        r_au = r.to(u.au).value
        #rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        g_r = 0.1112620426*((r_au/2.808)**(-2.15))*((1+(r_au/2.808)**5.093)**(-4.6142))

        matrix = np.array([np.array([g_r*R[0],  g_r*T[0], g_r*N[0]]), np.array([g_r*R[1],  g_r*T[1], g_r*N[1]]), np.array([g_r*R[2],  g_r*T[2], g_r*N[2]])])

        return matrix 

    def compute_Dt_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state

        cartesian_state_Dt = self.get_cartesian_state(current_time)
        position = cartesian_state[:3]
        velocity = cartesian_state[3:]

        helio_distance = np.linalg.norm(cartesian_state_Dt[:3]) * u.m
        r_au = helio_distance.to(u.au).value 
        alpha =  0.1112620426
        r0 = 2.808
        m = 2.15
        n = 5.093
        k = 4.6142

        g_r = alpha*((r_au/r0)**(-m))*((1 + (r_au/r0)**n)**(-k))     

        # g_r_prime = (-1/r_au) * alpha * ((1 + (r_au/r0)**n)**(-1-k)) * ((r_au/r0)**(-m)) * (m + ((r_au/r0)**n)*m + n*((r_au/r0)**n)*k)
        g_r_prime = ((((-m* alpha)/r0)*((r_au/r0)**(-m-1))*((1 + (r_au/r0)**n)*+(-k))) + (((-k*alpha*n)/r0)*((r_au/r0)**(-m))*((r_au/r0)**(n-1))+((1 + (r_au/r0)**n)**(-k-1)))) *(1/(1.4945*10**11))
        if float(self.Dt) > 0: 
            velocity = - velocity
        dg_ddt = g_r_prime * np.dot(position,velocity)/np.linalg.norm(position)
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)

        da_dt = np.array([self.custom_values[0]*dg_ddt, self.custom_values[1]*dg_ddt, self.custom_values[2]*dg_ddt])
        matrix = rotation_matrix @ da_dt.to_list()

        return matrix 

    def get_custom_parameters(self):
        return self.custom_values

    def set_custom_parameters(self, custom_parameters):
        self.custom_values = custom_parameters

    def get_custom_Dt(self):
        return self.Dt

    def set_custom_Dt(self, Dt_estimated):
        print(self.Dt)
        self.Dt = np.array([Dt_estimated])

class COMarsdenAcceleration(Acceleration):
    def __init__(self, A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end, dict):

        super().__init__(A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end)

        self.dict = dict

    def compute_acc(self, current_time):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        r_au = helio_distance.to(u.au).value 
        alpha =  0.04083733261
        r0 = 5
        m = 2
        n = 3
        k = 2.6

        g_r = alpha*((r_au/r0)**(-m))*(1 + (r_au/r0)**n)**(-k)

        acc = np.array([self.custom_values[0] * g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        # with open(f"{current_dir}/{self.target}/rsw_costumacc.txt", "a") as f:
        #     f.write(f'{current_time} - {acc}\n')
        self.dict[current_time] = list([r_au, self.custom_values[0] * g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        acc_inertial = rotation_matrix @ acc.tolist()
        return acc_inertial

    def compute_parameter_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state
        position = cartesian_state[:3]
        velocity = cartesian_state[3:]
        position_norm = np.linalg.norm(position)

        R = position/position_norm 
        h = np.cross(position, velocity)
        h_norm = np.linalg.norm(h)
        N = h/h_norm 
        T = np.cross(N, R)
        
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            position_norm = np.linalg.norm(cartesian_state[:3])
        r = position_norm * u.m
        # r is in meters, in the formula it has to be in au - use astropy for the conversion
        r_au = r.to(u.au).value
        #rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        alpha =  0.04083733261
        r0 = 5
        m = 2
        n = 3
        k = 2.6

        g_r = alpha*((r_au/r0)**(-m))*(1 + (r_au/r0)**n)**(-k)

        matrix = np.array([np.array([g_r*R[0],  g_r*T[0], g_r*N[0]]), np.array([g_r*R[1],  g_r*T[1], g_r*N[1]]), np.array([g_r*R[2],  g_r*T[2], g_r*N[2]])])
        return matrix     

    def compute_Dt_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state

        cartesian_state_Dt = self.get_cartesian_state(current_time)
        position = cartesian_state[:3]
        velocity = cartesian_state[3:]

        helio_distance = np.linalg.norm(cartesian_state_Dt[:3]) * u.m
        r_au = helio_distance.to(u.au).value 
        alpha =  0.04083733261
        r0 = 5
        m = 2
        n = 3
        k = 2.6

        g_r_prime = ((((-m* alpha)/r0)*((r_au/r0)**(-m-1))*((1 + (r_au/r0)**n)*+(-k))) + (((-k*alpha*n)/r0)*((r_au/r0)**(-m))*((r_au/r0)**(n-1))+((1 + (r_au/r0)**n)**(-k-1)))) *(1/(1.4945*10**11))
        if self.Dt > 0: 
            velocity = - velocity
        dg_ddt = g_r_prime * np.dot(position,velocity)/np.linalg.norm(position)
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)

        da_dt = np.array([self.custom_values[0]*dg_ddt, self.custom_values[1]*dg_ddt, self.custom_values[2]*dg_ddt])
        matrix = rotation_matrix @ da_dt#.to_list()

        return matrix

    def get_custom_parameters(self):
        return self.custom_values

    def set_custom_parameters(self, custom_parameters):
        self.custom_values = custom_parameters

    def get_custom_Dt(self):
        return self.Dt

    def set_custom_Dt(self, Dt):
        print(self.Dt)
        self.Dt = np.array([Dt])

class YabushitaAcceleration(Acceleration):
    def __init__(self, A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end, dict):

        super().__init__(A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end)

        self.dict = dict

    def compute_acc(self, current_time):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        r_au = helio_distance.to(u.au).value 

        f_r = (1.0006 / (r_au**2)) * (10**(-0.07395*(r_au -1))) * ((1 + 0.0006*r_au**5)**(-1))

        acc = np.array([self.custom_values[0] * f_r, self.custom_values[1]*f_r, self.custom_values[2]*f_r])
        # with open(f"{current_dir}/{self.target}/rsw_costumacc.txt", "a") as f:
        #     f.write(f'{current_time} - {acc}\n')
        self.dict[current_time] = list([r_au, self.custom_values[0] * f_r, self.custom_values[1]*f_r, self.custom_values[2]*f_r])
        acc_inertial = rotation_matrix @ acc.tolist()
        # print(acc_inertial)
        return acc_inertial

    def compute_parameter_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state
        position = cartesian_state[:3]
        velocity = cartesian_state[3:]
        position_norm = np.linalg.norm(position)

        R = position/position_norm 
        h = np.cross(position, velocity)
        h_norm = np.linalg.norm(h)
        N = h/h_norm 
        T = np.cross(N, R)
        
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            position_norm = np.linalg.norm(cartesian_state[:3])
        r = position_norm * u.m
        # r is in meters, in the formula it has to be in au - use astropy for the conversion
        r_au = r.to(u.au).value
        #rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        f_r = (1.0006 / (r_au**2)) * (10**(-0.07395*(r_au -1))) * ((1 + 0.0006*r_au**5)**(-1))

        matrix = np.array([np.array([f_r*R[0],  f_r*T[0], f_r*N[0]]), np.array([f_r*R[1],  f_r*T[1], f_r*N[1]]), np.array([f_r*R[2],  f_r*T[2], f_r*N[2]])])
        return matrix 
    
    def get_custom_parameters(self):
        return self.custom_values

    def set_custom_parameters(self, custom_parameters):
        self.custom_values = custom_parameters

class PowerLawAcceleration(Acceleration):
    def __init__(self, A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end, dict, n, r0):
        
        super().__init__(A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end)

        self.dict = dict
        self.n = n
        self.r0 = np.array([r0]) 

    def compute_acc(self, current_time):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        r_au = helio_distance.to(u.au).value 

        r_0 = float(self.r0)
        
        S = 1 / (1 + np.exp(-(r_au - r_0)/(1e-2))) 
        g_r = (1-S)*(1/r_au**self.n) + (S)*(np.exp(-r_au/(1e-2)))


        acc = np.array([self.custom_values[0]*g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        self.dict[current_time] = list([r_au, self.custom_values[0] * g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        acc_inertial = rotation_matrix @ acc

        return acc_inertial    
    
    def compute_parameter_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state
        position = cartesian_state[:3]
        velocity = cartesian_state[3:]
        position_norm = np.linalg.norm(position)

        R = position/position_norm 
        h = np.cross(position, velocity)
        h_norm = np.linalg.norm(h)
        N = h/h_norm 
        T = np.cross(N, R)
        
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            position_norm = np.linalg.norm(cartesian_state[:3])

        r = position_norm * u.m
        r_au = r.to(u.au).value

        r_0 = float(self.r0)
        
        S = 1 / (1 + np.exp(-(r_au - r_0)/(1e-2))) 
        g_r = (1-S)*(1/r_au**self.n) + (S)*(np.exp(-r_au/(1e-2)))

        matrix = np.array([np.array([g_r*R[0],  g_r*T[0], g_r*N[0]]), np.array([g_r*R[1],  g_r*T[1], g_r*N[1]]), np.array([g_r*R[2],  g_r*T[2], g_r*N[2]])])

        return matrix   

    def compute_r0_partial(self, current_time, current_pos): 
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        r_au = helio_distance.to(u.au).value 

        r_0 = float(self.r0)

        dS_dr0 = - ((1e2)*np.exp((1e2)*(r_0-r_au))) / ((np.exp((1e2)*(r_0-r_au)) + 1)**2)
        dg_dr0 = (1/r_au**self.n)*(-dS_dr0) + dS_dr0*(np.exp(-r_au/(1e-2)))

        acc =  np.array([self.custom_values[0]*dg_dr0, self.custom_values[1]*dg_dr0, self.custom_values[2]*dg_dr0])
        da_dr0 = rotation_matrix @ acc

        return da_dr0 

    def get_custom_parameters(self):
        return self.custom_values

    def set_custom_parameters(self, custom_parameters):
        self.custom_values = custom_parameters 
    
    def get_custom_r0(self):
        return self.r0

    def set_custom_r0(self, estimated_r0):
        self.r0 = np.array([estimated_r0])

class ContinousAcceleration(Acceleration):
    def __init__(self, A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end, dict, element, r0_h2O, r0_CO2, r0_CO, C):
        
        super().__init__(A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end)

        self.dict = dict
        self.element = element 
        self.r0_h2o = r0_h2O
        self.r0_co2 = r0_CO2      
        self.r0_co = r0_CO    
        self.C = np.array([C])

    def compute_acc(self, current_time):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        r_au = helio_distance.to(u.au).value 

        r0_h2o = float(self.r0_h2o)
        if self.element == 'CO2':
            r0_sec= float(self.r0_co2)
            n = 2
        elif self.element == 'CO':
            r0_sec = float(self.r0_co)
            n = 2
        else:
            print('NO ELEMENT MATCH')

        S1 = 1 / (1 + np.exp(-(r_au - r0_h2o)/(1e-2))) 
        S2 = 1 / (1 + np.exp(-(r_au - r0_sec)/(1e-2))) 

        C = float(self.C)
        
        g_r = (1-S2)*((1-S1)*1/r_au**2.3 + C*S1*1/r_au**n) + (S2)*(np.exp(-r_au/(1e-2)))


        acc = np.array([self.custom_values[0]*g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        self.dict[current_time] = list([r_au, self.custom_values[0] * g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        acc_inertial = rotation_matrix @ acc

        return acc_inertial    
    
    def compute_parameter_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state
        position = cartesian_state[:3]
        velocity = cartesian_state[3:]
        position_norm = np.linalg.norm(position)

        R = position/position_norm 
        h = np.cross(position, velocity)
        h_norm = np.linalg.norm(h)
        N = h/h_norm 
        T = np.cross(N, R)
        
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            position_norm = np.linalg.norm(cartesian_state[:3])

        r = position_norm * u.m
        r_au = r.to(u.au).value

        r0_h2o = float(self.r0_h2o)
        r0_h2o = float(self.r0_h2o)
        if self.element == 'CO2':
            r0_sec= float(self.r0_co2)
            n = 2
        elif self.element == "CO":
            r0_sec = float(self.r0_co)
            n = 2

        C = float(self.C)

        S1 = 1 / (1 + np.exp(-(r_au - r0_h2o)/1e-2)) 
        S2 = 1 / (1 + np.exp(-(r_au - r0_sec)/1e-2)) 
        
        g_r = (1-S2)*((1-S1)*1/r_au**2.3 + C*S1*1/r_au**n) + (S2)*(np.exp(-r_au/1e-2))

        matrix = np.array([np.array([g_r*R[0],  g_r*T[0], g_r*N[0]]), np.array([g_r*R[1],  g_r*T[1], g_r*N[1]]), np.array([g_r*R[2],  g_r*T[2], g_r*N[2]])])

        return matrix   

    def compute_C_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        r_au = helio_distance.to(u.au).value 

        r0_h2o = float(self.r0_h2o)
        if self.element == 'CO2':
            r0_sec= float(self.r0_co2)
            n = 2
        elif self.element == "CO":
            r0_sec = float(self.r0_co)
            n = 2

        S1 = 1 / (1 + np.exp(-(r_au - r0_h2o)/1e-2)) 
        S2 = 1 / (1 + np.exp(-(r_au - r0_sec)/1e-2)) 

        dg_dC = (1-S2) * S1* (1/(r_au**n))

        acc = np.array([self.custom_values[0]*dg_dC, self.custom_values[1]*dg_dC, self.custom_values[2]*dg_dC])
        da_dC = rotation_matrix @ acc

        return da_dC

    def get_custom_parameters(self):
        return self.custom_values

    def set_custom_parameters(self, custom_parameters):
        self.custom_values = custom_parameters     

    def get_custom_C(self):
        return self.C

    def set_custom_C(self, estimated_C):
        self.C = np.array([estimated_C])    

class PowerLawAcceleration_kgrid(Acceleration):
    def __init__(self, A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end, dict, n, r0, k):
        
        super().__init__(A1, A2, A3, bodies, target, code_horizons, Dt, epoch_start, epoch_end)

        self.dict = dict
        self.n = n
        self.r0 = r0 
        self.k = k
        self.B = k

    def compute_acc(self, current_time):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state 
        rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(cartesian_state)
        helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            helio_distance = np.linalg.norm(cartesian_state[:3]) * u.m
        r_au = helio_distance.to(u.au).value 

        r_0 = float(self.r0)
        
        S = 1 / (1 + np.exp(-(r_au - r_0)/self.k)) 
        g_r = (1-S)*(1/r_au**self.n) + (S)*(np.exp(-r_au/self.B))


        acc = np.array([self.custom_values[0]*g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        self.dict[current_time] = list([r_au, self.custom_values[0] * g_r, self.custom_values[1]*g_r, self.custom_values[2]*g_r])
        acc_inertial = rotation_matrix @ acc

        return acc_inertial    
    
    def compute_parameter_partials(self, current_time, current_pos):
        cartesian_state = self.bodies.get(self.target).state - self.bodies.get('Sun').state
        position = cartesian_state[:3]
        velocity = cartesian_state[3:]
        position_norm = np.linalg.norm(position)

        R = position/position_norm 
        h = np.cross(position, velocity)
        h_norm = np.linalg.norm(h)
        N = h/h_norm 
        T = np.cross(N, R)
        
        if float(self.Dt) != 0: 
            cartesian_state = self.get_cartesian_state(current_time)
            position_norm = np.linalg.norm(cartesian_state[:3])

        r = position_norm * u.m
        r_au = r.to(u.au).value

        r_0 = float(self.r0)
        
        S = 1 / (1 + np.exp(-(r_au - r_0)/self.k)) 
        g_r = (1-S)*(1/r_au**self.n) + (S)*(np.exp(-r_au/self.B))

        matrix = np.array([np.array([g_r*R[0],  g_r*T[0], g_r*N[0]]), np.array([g_r*R[1],  g_r*T[1], g_r*N[1]]), np.array([g_r*R[2],  g_r*T[2], g_r*N[2]])])

        return matrix   

    def get_custom_parameters(self):
        return self.custom_values

    def set_custom_parameters(self, custom_parameters):
        self.custom_values = custom_parameters 
