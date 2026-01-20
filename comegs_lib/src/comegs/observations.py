# Tudat imports for propagation and estimation
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.constants import GRAVITATIONAL_CONSTANT
from tudatpy.data.horizons import HorizonsQuery

# other useful modules
import numpy as np
import pickle
from astropy.time import Time
import pandas as pd 
from astropy.coordinates import SkyCoord
import os 
from pathlib import Path
import astropy.units as u 

current_dir= Path.cwd()
parent = current_dir.parent.parent.parent

# parent directory
parent = os.path.dirname(current_dir)


class Observations:
    def __init__(self, target, epoch_start, epoch_end,bodies):
        self.target = str(target)
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end 
        self.bodies = bodies
  
    def cartesian_observations(self):
        link_ends = dict()
        link_ends[observation.observed_body] = observation.body_origin_link_end_id(str(self.target))

        # Create observation settings for each link/observable
        link_definition = observation.LinkDefinition(link_ends)
        observation_settings_list = [observation.cartesian_position(link_definition)]

        #Define observation simulation times for each link (separated by steps of 1 minute)
        observation_simulation_settings = []
        observation_times = np.arange(self.epoch_start , self.epoch_end, 3600000.0/2)
        observation_simulation_settings.append(observation.tabulated_simulation_settings(
            observation.position_observable_type,
            link_definition,
            observation_times,
            reference_link_end_type=estimation_setup.observation.observed_body
        ))

        ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
            observation_settings_list, self.bodies)

        # simulate the wanted observations 
        # Simulate required observations

        simulated_observations = estimation.simulate_observations(
            observation_simulation_settings,
            ephemeris_observation_simulators,
            self.bodies)
        
        # observation_times = np.array(simulated_observations.concatenated_times)
        # observations_list = np.array(simulated_observations.concatenated_observations)
        # plt.figure(figsize=(9, 5))
        # plt.title("Observations as a function of time")
        # plt.scatter(observation_times / 3600.0, observations_list)
                
        return observation_settings_list, simulated_observations

    def astrometric_observations(self, batch):
        observation_collection = batch.to_tudat(bodies=self.bodies, included_satellites=None, apply_weights_VFCC17=True, apply_star_catalog_debias = True)

        # Define the uplink link ends for one-way observable
        observation_settings_list = []
        observation_simulation_settings = []
        space_telescope = batch.observatories_table(only_space_telescopes=True).Code.to_list()
        batch.filter(observatories_exclude=space_telescope)
        observatories = batch.observatories
        for el in observatories:
            link_ends = dict()
            link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", el)  # the second name is the name of an observatory   
            link_ends[observation.transmitter] = observation.body_origin_link_end_id((str(self.target)))
            link_definition = observation.LinkDefinition(link_ends)
            observation_settings_list.append(observation.angular_position(link_definition, bias_settings=None))

            batch_table = batch.table[batch.table['observatory'] == el]
            observation_times = batch_table.epochJ2000secondsTDB.to_list()

            observation_simulation_settings.append(observation.tabulated_simulation_settings(
                observation.angular_position_type,
                link_definition,
                observation_times,
                ))


        ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
            observation_settings_list, self.bodies)

        # simulate the wanted observations 
        # Simulate required observations
        simulated_observations = estimation.simulate_observations(
            observation_simulation_settings,
            ephemeris_observation_simulators,
            self.bodies)


        # observations_list = np.array(simulated_observations.concatenated_observations)
        # simulated_RA = [np.mod(el, 2*np.pi) for el in observations_list[::2]]
        # simulated_DEC = observations_list[1::2]

        return observation_settings_list, simulated_observations
    
    def MPC_observations(self, batch, include_satellites: False):
        if include_satellites == True:
            with open("dict.pkl", "rb") as f:
                used_satellites = pickle.load(f)
            observation_collection = batch.to_tudat(bodies=self.bodies, included_satellites=used_satellites, apply_weights_VFCC17=True, apply_star_catalog_debias = True)
            file_path = f"{current_dir}/dict.pkl"
            # check if file exists, then remove it
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            observation_collection = batch.to_tudat(bodies=self.bodies, included_satellites=None, apply_weights_VFCC17=True, apply_star_catalog_debias = True)

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

        return observation_settings_list, observation_collection
    
    def observations_from_state(self,states:np.ndarray):
        link_ends = dict()
        link_ends[observation.observed_body] = observation.body_origin_link_end_id(self.target)
        link_definition = observation.LinkDefinition(link_ends)
        observation_settings_list = [observation.cartesian_position(link_definition)]
        
        observations = states[:,1:4]
        observation_times = states[:,0]
        single_obs_set = numerical_simulation.estimation.single_observation_set(
            observable_type=observation.position_observable_type,
            link_definition=link_definition, 
            observations=observations,
            observation_times= observation_times, 
            reference_link_end=estimation_setup.observation.observed_body,            
        )

        observation_collection = numerical_simulation.estimation.ObservationCollection([single_obs_set])

        return observation_settings_list, observation_collection
    
    def observatory_info(self, observatory_name, observatory_file):
        # make sure 098 and 98 are the same
        if len(observatory_name) == 2:                   
            Observatory = '0' + Observatory
        # make sure 098 and 98 are the same
        elif len(observatory_name) == 1:                  
            Observatory = '00' + Observatory
        with open(f'{observatory_file}', 'r') as file:    #https://www.projectpluto.com/obsc.htm, https://www.projectpluto.com/mpc_stat.txt
            lines = file.readlines()
            for line in lines[1:]:  # Ignore the first line
                columns = line.split()
                if columns[1] == observatory_name:
                    longitude = float(columns[2])
                    latitude = float(columns[3])
                    altitude = float(columns[4])
                    print(observatory_name, longitude, latitude, altitude)
                    return np.deg2rad(longitude),  np.deg2rad(latitude), altitude
                    #return longitude, latitude, altitude
            print('No matching Observatory found') 
          
    def station_definition(self, observatories_table):
        r_earth = self.bodies.get('Earth').shape_model.average_radius

        observatories_table = (
            observatories_table.assign(X=lambda x: x.cos * r_earth * np.cos(np.radians(x.Longitude)))
            .assign(Y=lambda x: x.cos * r_earth * np.sin(np.radians(x.Longitude)))
            .assign(Z=lambda x: x.sin * r_earth)
        )
    
        return observatories_table

    def load_observations_from_file(self, astro_file_name, observatories_table, apply_weights: bool):

        df = pd.read_csv(f"{astro_file_name}", sep="|")
        df.columns = df.columns.str.strip()

        observation_set_list = []
        obs_dict = {}

        # create smaller dataframes for each station 
        df['stn'] = df['stn'].str.strip()
        groups = {stn: group for stn, group in df.groupby('stn')}

        # loop over the stations dataframe 
        observation_settings_list = []
        for observatory, group in groups.items(): 
            obs_times = group["obsTime"].astype(str).str.strip().str.rstrip("Z").tolist()

            # transform ime in seconds after J2000 TDB 
            j2000 = Time("2000-01-01T12:00:00", scale="tdb")
            t_utc = Time(obs_times, scale="utc")
            t_tdb = t_utc.tdb

            # Compute seconds after J2000 TDB
            times_tdb = (t_tdb - j2000).to("s").value
            angles_ICRF = [np.array([ra, dec]) for ra, dec in zip(group["ra"], group["dec"])]

            # read the residuals to build the weights table - put them in radians
            rms_RA = (group["rmsRA"].values * u.arcsec)/(np.cos(np.deg2rad(group["dec"].values)))  # weights are given in RAcos(Dec); we work in RA and Dec thus the weghts are all divided by cos(Dec)
            rms_RA_rad = rms_RA.to(u.rad)
            RA_weights = 1 / (rms_RA_rad.value ** 2)
            rms_Dec = (group["rmsDec"].values * u.arcsec)
            rms_Dec_rad = rms_Dec.to(u.rad)
            Dec_weights = 1 / (rms_Dec_rad.value ** 2)

            weights = [val for pair in zip(RA_weights, Dec_weights) for val in pair]
            errors = [val for pair in zip(rms_RA_rad.value, rms_Dec_rad.value) for val in pair]
            
            # transform the angles from ICRF to J2000 (bias of milliarcseconds)
            angles = []
            for el in angles_ICRF:
                ra = el[0]
                dec = el[1]
                coord = SkyCoord(ra = ra, dec = dec, frame = 'icrs', unit='deg').transform_to('fk5')
                #angles.append(np.array([coord.ra.radian, coord.dec.radian]))
                angles.append(np.mod(np.array([coord.ra.radian, coord.dec.radian]) + np.pi, 2 * np.pi) - np.pi)

            if observatory not in obs_dict:
                obs_dict[str(observatory)] = []
            if observatory not in self.bodies.get_body("Earth").ground_station_list:
                print(observatory)
                station_coord = self.station_definition(observatories_table)
                row = station_coord.loc[observatories_table['Code'] == observatory].iloc[0]
                X = float(row['X'])
                Y = float(row['Y'])
                Z = float(row['Z'])

                ground_station_settings = environment_setup.ground_station.basic_station(
                station_name=observatory,
                station_nominal_position=[X,Y,Z,])

                environment_setup.add_ground_station(
                    self.bodies.get_body('Earth'), ground_station_settings
                )

            # define the link ends 
            link_ends = dict()
            link_ends[observation.transmitter] = observation.body_origin_link_end_id(str(self.target))
            link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", observatory)
            link_definition = observation.LinkDefinition(link_ends)
            observation_settings_list.append(observation.angular_position(link_definition))

            # create observation set
            single_observation_set = numerical_simulation.estimation.single_observation_set(
                observation.angular_position_type,
                link_definition,
                angles,
                times_tdb,
                observation.receiver
                )
            
            if apply_weights: 
                single_observation_set.set_tabulated_weights(weights)
            
            observation_set_list.append(single_observation_set)

            obs_dict[str(observatory)] = [single_observation_set.observation_times, np.array(single_observation_set.list_of_observations), errors, weights]

        observation_collection = estimation.ObservationCollection(observation_set_list)
        return observation_settings_list, observation_collection, obs_dict


    def load_observations_from_file_with_roving_obs(self, body_settings, global_frame_origin, global_frame_orientation, astro_file_name, observatories_table, satellite_list, apply_weights: bool):

        df = pd.read_csv(f"{astro_file_name}", sep="|")
        df.columns = df.columns.str.strip()

        observation_set_list = []
        obs_dict = {}

        # create smaller dataframes for each station 
        df['stn'] = df['stn'].astype(str).str.strip()
        groups = {stn: group for stn, group in df.groupby('stn')}
        
        for observatory in groups.keys():
            # add satellite to system of bodies and use it as the receiver directly 
            if observatory in satellite_list and observatory not in self.bodies.list_of_bodies():
                print('satellite:', observatory)
                satellite_dict = {'250' : '-48', '339': '-143', '338':'-255', '28':'-28', '274':'-170'}
                body_settings.add_empty_settings(str(observatory))
                ephemeris_obs = {}
                query_obs = HorizonsQuery(
                    query_id=satellite_dict[observatory],
                    location=f"@{global_frame_origin}",
                    epoch_start=self.epoch_start - 30*86400,
                    epoch_end=self.epoch_end + 30*86400,
                    epoch_step=f"{int(60)}m",
                    extended_query=True,
                )
                ephemeris_obs[str(observatory)] = query_obs.create_ephemeris_tabulated(
                    frame_origin=global_frame_origin,
                    frame_orientation=global_frame_orientation
                )
                body_settings.get(str(observatory)).ephemeris_settings = ephemeris_obs[str(observatory)]   
        
        self.bodies = environment_setup.create_system_of_bodies(body_settings)

        # loop over the stations dataframe 
        observation_settings_list = []
        for observatory, group in groups.items(): 
            
            obs_times = group["obsTime"].astype(str).str.strip().str.rstrip("Z").tolist()

            # transform time in seconds after J2000 TDB 
            j2000 = Time("2000-01-01T12:00:00", scale="tdb")
            t_utc = Time(obs_times, scale="utc")
            t_tdb = t_utc.tdb

            # Compute seconds after J2000 TDB
            times_tdb = (t_tdb - j2000).to("s").value
            angles_ICRF = [np.array([ra, dec]) for ra, dec in zip(group["ra"], group["dec"])]

            # read the residuals to build the weights table - put them in radians
            rms_RA = (group["rmsRA"].values * u.arcsec)#/(np.cos(np.deg2rad(group["dec"].values)))  # weights are given in RAcos(Dec); we work in RA and Dec thus the weghts are all divided by cos(Dec)
            rms_RA_rad = rms_RA.to(u.rad)
            if observatory in satellite_list: 
                RA_weights = 1 / (rms_RA_rad.value ** 2)
            else:
                RA_weights = 1 / (rms_RA_rad.value ** 2)
            
            rms_Dec = (group["rmsDec"].values * u.arcsec)
            rms_Dec_rad = rms_Dec.to(u.rad)
            if observatory in satellite_list: 
                Dec_weights = 1 / (rms_Dec_rad.value ** 2)
            else: 
                Dec_weights = 1 / (rms_Dec_rad.value ** 2)
            
            weights = [val for pair in zip(RA_weights, Dec_weights) for val in pair]
            errors = [val for pair in zip(rms_RA_rad.value, rms_Dec_rad.value) for val in pair]
            
            # transform the angles from ICRF to J2000 (bias of milliarcseconds)
            angles = []
            for el in angles_ICRF:
                ra = el[0]
                dec = el[1]
                coord = SkyCoord(ra = ra, dec = dec, frame = 'icrs', unit='deg').transform_to('fk5')
                #angles.append(np.array([coord.ra.radian, coord.dec.radian]))
                angles.append(np.mod(np.array([coord.ra.radian, coord.dec.radian]) + np.pi, 2 * np.pi) - np.pi)

            if observatory not in obs_dict:
                obs_dict[str(observatory)] = []

            if observatory not in self.bodies.get_body("Earth").ground_station_list and observatory not in satellite_list:
                print(observatory)
                station_coord = self.station_definition(observatories_table)
                row = station_coord.loc[observatories_table['Code'] == observatory].iloc[0]
                X = float(row['X'])
                Y = float(row['Y'])
                Z = float(row['Z'])

                ground_station_settings = environment_setup.ground_station.basic_station(
                station_name=observatory,
                station_nominal_position=[X,Y,Z,])

                environment_setup.add_ground_station(
                    self.bodies.get_body('Earth'), ground_station_settings
                )

            if observatory not in satellite_list:
                # define the link ends 
                link_ends = dict()
                # define a ground station
                link_ends[observation.transmitter] = observation.body_origin_link_end_id(str(self.target))
                link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", observatory)
                link_definition = observation.LinkDefinition(link_ends)
                observation_settings_list.append(observation.angular_position(link_definition))
            else:
                link_ends = dict()
                # define a ground station
                link_ends[observation.transmitter] = observation.body_origin_link_end_id(str(self.target))
                link_ends[observation.receiver] = observation.body_origin_link_end_id(str(observatory))
                link_definition = observation.LinkDefinition(link_ends)
                observation_settings_list.append(observation.angular_position(link_definition))


            # create observation set
            single_observation_set = numerical_simulation.estimation.single_observation_set(
                observation.angular_position_type,
                link_definition,
                angles,
                times_tdb,
                observation.receiver
                )
            
            if apply_weights: 
                single_observation_set.set_tabulated_weights(weights)
            
            observation_set_list.append(single_observation_set)

            obs_dict[str(observatory)] = [single_observation_set.observation_times, np.array(single_observation_set.list_of_observations), errors, weights]
        
            #else: 
                # loop over all observations taken by a single spacecraft
                # for idx, row in group.iterrows():
                #     obs_time = str(row["obsTime"]).strip().rstrip("Z")
                #     # Transform time in seconds after J2000 TDB
                #     j2000 = Time("2000-01-01T12:00:00", scale="tdb")
                #     t_utc = Time(obs_time, scale="utc")
                #     t_tdb = t_utc.tdb
                #     times_tdb = (t_tdb - j2000).to("s").value

                #     # ICRF angles
                #     ra, dec = row["ra"], row["dec"]
                #     coord = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg').transform_to('fk5')
                #     angles = np.mod(np.array([coord.ra.radian, coord.dec.radian]) + np.pi, 2 * np.pi) - np.pi

                #     # RMS weights (convert to radians)
                #     rms_RA_rad = (row["rmsRA"] * u.arcsec / np.cos(np.deg2rad(dec))).to(u.rad).value
                #     RA_weight = 1 / rms_RA_rad**2

                #     rms_Dec_rad = (row["rmsDec"] * u.arcsec).to(u.rad).value
                #     Dec_weight = 1 / rms_Dec_rad**2

                #     # Combine weights and errors
                #     weights = [RA_weight, Dec_weight]
                #     errors = [rms_RA_rad, rms_Dec_rad]

                #     rot_matrix = self.bodies.get('Earth').rotation_model.inertial_to_body_fixed_rotation(times_tdb)

                #     if row['sys'] == 'ICRF_AU':
                #         x = (float(row["pos1"]) * u.au).to(u.m).value
                #         y = (float(row["pos2"]) * u.au).to(u.m).value
                #         z = (float(row["pos3"]) * u.au).to(u.m).value
                #         position = np.array([x,y,z])
                #     if row['sys'] == 'ICRF_KM':
                #         x = (float(row["pos1"]) * u.km).to(u.m).value
                #         y = (float(row["pos2"]) * u.km).to(u.m).value
                #         z = (float(row["pos3"]) * u.km).to(u.m).value
                #         position = np.array([x,y,z])
                    
                #     position_itrf = rot_matrix@position
                    
                #     # define the single observatory ground station
                #     ground_station_settings = environment_setup.ground_station.basic_station(
                #         station_name=f"{observatory}_{idx}",
                #         station_nominal_position=[position_itrf[0],position_itrf[1],position_itrf[2],])

                #     environment_setup.add_ground_station(
                #         self.bodies.get_body('Earth'), ground_station_settings
                #     )

                #     link_ends = dict()
                #     # define a ground station
                #     link_ends[observation.transmitter] = observation.body_origin_link_end_id(str(self.target))
                #     link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", f"{observatory}_{idx}")
                #     link_definition = observation.LinkDefinition(link_ends)
                #     observation_settings_list.append(observation.angular_position(link_definition))

                #     # create observation set
                #     single_observation_set = numerical_simulation.estimation.single_observation_set(
                #         observation.angular_position_type,
                #         link_definition,
                #         np.array([angles]),
                #         [times_tdb],
                #         observation.receiver
                #         )
                    
                #     if apply_weights: 
                #         single_observation_set.set_tabulated_weights(weights)
                    
                #     observation_set_list.append(single_observation_set)

                #     obs_dict[f"{observatory}_{idx}"] = [single_observation_set.observation_times, np.array(single_observation_set.list_of_observations), errors, weights]
                 
        bodies  = self.bodies                       
        observation_collection = estimation.ObservationCollection(observation_set_list)
        return bodies, body_settings, observation_settings_list, observation_collection, obs_dict

    def load_obs_spacecraft(self, body_settings, global_frame_origin, global_frame_orientation, astro_file_name, observatories_table, satellite_list, apply_weights: bool):

        df = pd.read_csv(f"{astro_file_name}", sep="|")
        df.columns = df.columns.str.strip()

        observation_set_list = []
        obs_dict = {}

        # create smaller dataframes for each station 
        df['stn'] = df['stn'].astype(str).str.strip()
        groups = {stn: group for stn, group in df.groupby('stn')}

        # loop over the stations dataframe 
        observation_settings_list = []
        for observatory, group in groups.items(): 
            if observatory in satellite_list: 
                state_history = {}
                print('satellite', observatory)
                for idx, row in group.iterrows():
                    if row['sys'] == 'ICRF_AU':
                        x = (float(row["pos1"]) * u.au).to(u.m).value
                        y = (float(row["pos2"]) * u.au).to(u.m).value
                        z = (float(row["pos3"]) * u.au).to(u.m).value
                        vx = (float(row["vel1"]) * u.au/u.day).to(u.m/u.s).value
                        vy = (float(row["vel2"]) * u.au/u.day).to(u.m/u.s).value
                        vz = (float(row["vel3"]) * u.au/u.day).to(u.m/u.s).value
                        state = np.array([x,y,z, vx, vy, vz])
                    elif row['sys'] == 'ICRF_KM':
                        x = (float(row["pos1"]) * u.km).to(u.m).value
                        y = (float(row["pos2"]) * u.km).to(u.m).value
                        z = (float(row["pos3"]) * u.km).to(u.m).value
                        vx = (float(row["vel1"]) * u.km/u.h).to(u.m/u.s).value
                        vy = (float(row["vel2"]) * u.km/u.h).to(u.m/u.s).value
                        vz = (float(row["vel3"]) * u.km/u.h).to(u.m/u.s).value
                        state = np.array([x,y,z, vx, vy, vz])

                    obs_time = str(row["obsTime"]).strip().rstrip("Z")
                    # Transform time in seconds after J2000 TDB
                    j2000 = Time("2000-01-01T12:00:00", scale="tdb")
                    t_utc = Time(obs_time, scale="utc")
                    t_tdb = t_utc.tdb
                    time_tdb = (t_tdb - j2000).to("s").value
                    state_history[time_tdb] = state
                
                # define the satellite ephemeris 
                frame_origin = "Earth"
                frame_orientation = "J2000"

                if len(state_history.keys()) < 6: 
                    satellite_dict = {'250' : '-48', '339': '-143', '338':'-255', '28':'-28', '274':'-170'}
                    for i in range(7-len(state_history.keys())):
                        query_time = float(time_tdb) + i*100000
                        # retrieve the state directly wrt Horizons 
                        query_states = HorizonsQuery(
                                    query_id= satellite_dict[observatory],
                                    location="500@399",
                                    epoch_list = [query_time],
                                    extended_query=True,
                                    )
                        state_history[query_time] = np.array(query_states.cartesian(frame_orientation='J2000'))[0, 1:]

                # Create the tabulated ephemeris settings and add them to the body observatory
                body_settings.add_empty_settings(str(observatory))
                body_settings.get(str(observatory)).ephemeris_settings = environment_setup.ephemeris.tabulated(state_history,
                    frame_origin,
                    frame_orientation)

        self.bodies = environment_setup.create_system_of_bodies(body_settings)

        for observatory, group in groups.items(): 
            # add all observations 
            obs_times = group["obsTime"].astype(str).str.strip().str.rstrip("Z").tolist()

            # transform time in seconds after J2000 TDB 
            j2000 = Time("2000-01-01T12:00:00", scale="tdb")
            t_utc = Time(obs_times, scale="utc")
            t_tdb = t_utc.tdb

            # Compute seconds after J2000 TDB
            times_tdb = (t_tdb - j2000).to("s").value
            angles_ICRF = [np.array([ra, dec]) for ra, dec in zip(group["ra"], group["dec"])]

            # read the residuals to build the weights table - put them in radians
            rms_RA = (group["rmsRA"].values * u.arcsec)/(np.cos(np.deg2rad(group["dec"].values)))  # weights are given in RAcos(Dec); we work in RA and Dec thus the weghts are all divided by cos(Dec)
            rms_RA_rad = rms_RA.to(u.rad)
            RA_weights = 1 / (rms_RA_rad.value ** 2)
            
            rms_Dec = (group["rmsDec"].values * u.arcsec)
            rms_Dec_rad = rms_Dec.to(u.rad)
            Dec_weights = 1 / (rms_Dec_rad.value ** 2)
            
            weights = [val for pair in zip(RA_weights, Dec_weights) for val in pair]
            errors = [val for pair in zip(rms_RA_rad.value, rms_Dec_rad.value) for val in pair]
            
            # transform the angles from ICRF to J2000 (bias of milliarcseconds)
            angles = []
            for el in angles_ICRF:
                ra = el[0]
                dec = el[1]
                coord = SkyCoord(ra = ra, dec = dec, frame = 'icrs', unit='deg').transform_to('fk5')
                #angles.append(np.array([coord.ra.radian, coord.dec.radian]))
                angles.append(np.mod(np.array([coord.ra.radian, coord.dec.radian]) + np.pi, 2 * np.pi) - np.pi)

            if observatory not in obs_dict:
                obs_dict[str(observatory)] = []

            if observatory not in self.bodies.get_body("Earth").ground_station_list and observatory not in satellite_list:
                print(observatory)
                station_coord = self.station_definition(observatories_table)
                row = station_coord.loc[observatories_table['Code'] == observatory].iloc[0]
                X = float(row['X'])
                Y = float(row['Y'])
                Z = float(row['Z'])

                ground_station_settings = environment_setup.ground_station.basic_station(
                station_name=observatory,
                station_nominal_position=[X,Y,Z,])

                environment_setup.add_ground_station(
                    self.bodies.get_body('Earth'), ground_station_settings
                )

            if observatory not in satellite_list: 
                # define the link ends 
                link_ends = dict()
                # define a ground station
                link_ends[observation.transmitter] = observation.body_origin_link_end_id(str(self.target))
                link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", observatory)
                link_definition = observation.LinkDefinition(link_ends)
                observation_settings_list.append(observation.angular_position(link_definition))
            else:
                link_ends = dict()
                # define a ground station
                link_ends[observation.transmitter] = observation.body_origin_link_end_id(str(self.target))
                link_ends[observation.receiver] = observation.body_origin_link_end_id(str(observatory))
                link_definition = observation.LinkDefinition(link_ends)
                observation_settings_list.append(observation.angular_position(link_definition))
                
            # create observation set
            single_observation_set = numerical_simulation.estimation.single_observation_set(
                observation.angular_position_type,
                link_definition,
                angles,
                times_tdb,
                observation.receiver
                )
            
            if apply_weights: 
                single_observation_set.set_tabulated_weights(weights)
            
            observation_set_list.append(single_observation_set)

            obs_dict[str(observatory)] = [single_observation_set.observation_times, np.array(single_observation_set.list_of_observations), errors, weights]
        
        bodies = self.bodies
        observation_collection = estimation.ObservationCollection(observation_set_list)
        return bodies, body_settings, observation_settings_list, observation_collection, obs_dict

            