"""
Define a plotting class, to be called and used when wanting to visualize results in a certain way 
"""

# Tudat imports for propagation and estimation
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro import frame_conversion

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

from astropy import units as u


class PlotError:
    def __init__(self, bodies, epoch_start_buffer, epoch_end_buffer, query_code, mpc_code):
        self.bodies = bodies
        self.start = epoch_start_buffer
        self.end = epoch_end_buffer
        self.code_horizons = query_code
        self.code_mpc = mpc_code

    def gaps(self,observation_collection):
        # lets get ranges for all gaps larger than 6 months:
        gap_in_months = 6
        # get the time in years instead of sec since J2000
        times = (np.array(observation_collection.concatenated_times) / (86400 * 365.25) + 2000)

        gaps = np.abs(np.diff(sorted(times)))
        num_gaps = (
            gaps > (gap_in_months / 12)
        ).sum()  # counts the number of gaps larger than 0.5 years
        indices_of_largest_gaps = np.argsort(gaps)[-num_gaps:]

        # (start, end) for each of the gaps
        gap_ranges = [
            (sorted(times)[idx - 1], sorted(times)[idx + 1])
            for idx in indices_of_largest_gaps
        ]

        return gap_ranges


    def plot_error_total(self, global_frame_origin, gap_ranges, filename):
        # Now lets plot the orbit error
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        estimation_states = []

        # retrieve the states for a list of times.
        times = np.linspace(self.start, self.end, 1000)
        times_plot = times #times / (86400 * 365.25) + 2000  # approximate

        #from spice - this state is defined wrt the SSB - the ephemeris produced by the estimation are in SSB-J2000
        state_spice = HorizonsQuery(
            query_id=  self.code_horizons,
            location=f"@{global_frame_origin}",
            epoch_list = list(times),
            extended_query=True,
            )

        state_spice = np.array(state_spice.cartesian(frame_orientation='J2000'))
        state_spice_error = np.delete(state_spice, 0, axis=1)

        # state from estimation - this state is defined wrt the SSB
        for i in range(len(times)):
            state_est = self.bodies.get(self.code_mpc).ephemeris.cartesian_state(times[i]) 
            estimation_states.append(state_est)
        
        # Error in kilometers
        error = (np.array(state_spice_error) - np.array(estimation_states)) # in meters 
        error_norm = np.linalg.norm(error, axis=1)
        # print(f'Max error: {max(error_norm)}')
        # plot
        ax.plot(times_plot[10:-5], error[10:-5, 0]/1000, label="x")
        ax.plot(times_plot[10:-5], error[10:-5, 1]/1000, label="y")
        ax.plot(times_plot[10:-5], error[10:-5, 2]/1000, label="z")

        ax.grid()
        ax.legend(ncol=1)

        plt.tight_layout()

        ax.set_ylabel("Cartesian Error [km]")
        ax.set_xlabel("Time since J2000 [s]")

        fig.suptitle(f"Error vs SPICE over time for {self.code_mpc}")
        fig.set_tight_layout(True)

        if filename != None: 
            plt.savefig(filename)


    def plot_error_total_iterations(self, global_frame_origin, gap_ranges, fig, ax, iter):
            styles = [':', '-', '-.', '--', (0, (3, 1, 1, 1, 1, 1))]

            estimation_states = []

            # retrieve the states for a list of times.
            times = np.linspace(self.start, self.end, 1000)
            times_plot = times / (86400 * 365.25) + 2000  # approximate

            #from spice - this state is defined wrt the SSB - the ephemeris produced by the estimation are in SSB-J2000
            state_spice = HorizonsQuery(
                query_id=  self.code_horizons,
                location=f"@{global_frame_origin}",
                epoch_list = list(times),
                extended_query=True,
                )

            state_spice = np.array(state_spice.cartesian(frame_orientation='J2000'))
            state_spice_error = np.delete(state_spice, 0, axis=1)

            # state from estimation - this state is defined wrt the SSB
            for i in range(len(times)):
                state_est = self.bodies.get(self.code_mpc).ephemeris.cartesian_state(times[i]) 
                estimation_states.append(state_est)
            
            # Error in kilometers
            error = (np.array(state_spice_error) - np.array(estimation_states)) # in meters 
            error_norm = np.linalg.norm(error, axis=1)
            # print(f'Max error: {max(error_norm)}')
            # plot
            ax.plot(times_plot[:], error[:, 0]/1000, label=f"x - {iter}", linestyle=styles[iter])
            ax.plot(times_plot[:], error[:, 1]/1000, label=f"y - {iter}", linestyle=styles[iter])
            ax.plot(times_plot[:], error[:, 2]/1000, label=f"z - {iter}", linestyle=styles[iter])

    def plot_error_norm_iterations(self, global_frame_origin, gap_ranges, fig, ax, iter):
            styles = [':', '-', '-.', '--', (0, (3, 1, 1, 1, 1, 1))]

            estimation_states = []

            # retrieve the states for a list of times.
            times = np.linspace(self.start, self.end, 1000)
            times_plot = times / (86400 * 365.25) + 2000  # approximate

            #from spice - this state is defined wrt the SSB - the ephemeris produced by the estimation are in SSB-J2000
            state_spice = HorizonsQuery(
                query_id=  self.code_horizons,
                location=f"@{global_frame_origin}",
                epoch_list = list(times),
                extended_query=True,
                )

            state_spice = np.array(state_spice.cartesian(frame_orientation='J2000'))
            state_spice_error = np.delete(state_spice, 0, axis=1)

            # state from estimation - this state is defined wrt the SSB
            for i in range(len(times)):
                state_est = self.bodies.get(self.code_mpc).ephemeris.cartesian_state(times[i]) 
                estimation_states.append(state_est)
            
            # Error in kilometers
            error = (np.array(state_spice_error) - np.array(estimation_states)) # in meters 
            error_norm = np.linalg.norm(error, axis=1)
            # print(f'Max error: {max(error_norm)}')
            # plot
            ax.plot(times_plot[3:-3], error_norm[3:-3]/1000, label=f"{iter}", linestyle=styles[iter])


    def plot_error_single(self, global_frame_origin, gap_ranges, filename):
        # retrieve the states for a list of times.
        times = np.linspace(self.start, self.end, 1000)
        times_plot = times / (86400 * 365.25) + 2000  # approximate

        estimation_states_sun = []

        # from spice
        state_spice = HorizonsQuery(
            query_id=  self.code_horizons,#target_mpc_code,
            location=f"@Sun",
            epoch_list = list(times),
            extended_query=True,
            )

        state_spice = np.array(state_spice.cartesian(frame_orientation='J2000'))
        spice_states_sun = np.delete(state_spice, 0, axis=1)


        for i in range(len(times)):
            state_est_sun = self.bodies.get(self.code_mpc).ephemeris.cartesian_state(times[i]) - self.bodies.get('Sun').ephemeris.cartesian_state(times[i]) 
            estimation_states_sun.append(state_est_sun)
        

        # convert to RSW frame
        error_spice= [
            (frame_conversion.inertial_to_rsw_rotation_matrix(state_S) @ (state_S[:3] - state_E[:3]))
            for (state_E, state_S) in zip(estimation_states_sun, spice_states_sun)
        ]

        # spice_rsw = [
        #     (frame_conversion.inertial_to_rsw_rotation_matrix(state_S) @ (state_S[:3]))
        #     for (state_S) in spice_states_sun
        # ]

        # error_spice = (np.array(spice_rsw) - np.array(estimated_rsw))/1000
        error_spice = np.array(error_spice)/1000
        frame_name = "RSW"
        
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.plot(times_plot[3:-3], error_spice[3:-3, 0], label="Radial" )
        ax.plot(times_plot[3:-3], error_spice[3:-3, 1], label="Along-Track")
        ax.plot(times_plot[3:-3], error_spice[3:-3, 2], label="Cross-Track")
        ax.plot(
            times_plot[3:-3],
            np.linalg.norm(error_spice[3:-3, :3], axis=1),
            label="magnitude",
            linestyle="--",
            color="k",
        )


        # show areas where there are no observations:
        for i, gap in enumerate(gap_ranges):
            ax.axvspan(
                xmin=gap[0],
                xmax=gap[1],
                alpha=0.1,
                label="Large gap in observations" if i == 0 else None,
            )
        ax.grid()
        ax.legend(ncol=5)
        ax.set_ylabel(f"{frame_name} Error [km]")
        ax.set_xlabel("Year")

        ax.set_title(f"Error vs SPICE over time for {self.code_mpc}")
        fig.set_tight_layout(True)
        if filename != None: 
            plt.savefig(filename)



    def plot_cartesian(self, global_frame_origin):
            times = np.linspace(self.start, self.end, 1000)
            times_plot = times / (86400 * 365.25) + 2000  # approximate

            estimation_states = []
            spice_states = []

            # from spice
            state_spice = HorizonsQuery(
                query_id=  self.code_horizons,#target_mpc_code,
                location=f"@{global_frame_origin}",
                epoch_list = list(times),
                extended_query=True,
                )

            state_spice = np.array(state_spice.cartesian(frame_orientation='J2000'))
            #spice cartesian state in km
            state_spice_error = np.delete(state_spice, 0, axis=1)
            spice_states = state_spice_error/1000


            for i in range(len(times)):
                state_est = self.bodies.get(self.code_mpc).ephemeris.cartesian_state(times[i])
                estimation_states.append(state_est) 
            
            # estimated cartesian state in km 
            estimation_states = np.array(estimation_states )/1000

            fig, axs = plt.subplots(1, 3, figsize=(12, 9))
            axs[0].plot(times_plot, spice_states[:, 0], label="X", color='red')
            axs[1].plot(times_plot, spice_states[:, 1], label="Y", color='red')
            axs[2].plot(times_plot, spice_states[:, 2], label="Z", color='red')

            axs[0].plot(times_plot, estimation_states[:, 0], label="X", color='blue')
            axs[1].plot(times_plot, estimation_states[:, 1], label="Y", color='blue')
            axs[2].plot(times_plot, estimation_states[:, 2], label="Z", color='blue')


            axs[0].set_ylabel(f"Error [km]")
            axs[0].set_xlabel("Year")
            axs[1].set_ylabel(f"Error [km]")
            axs[1].set_xlabel("Year")
            axs[2].set_ylabel(f"Error [km]")
            axs[2].set_xlabel("Year")
            axs[0].set_title(f"Estimated (blue) vs SPICE (red) X component over time for {self.code_mpc}")
            axs[1].set_title(f"Estimated (blue) vs SPICE (red) Y component over time for {self.code_mpc}")
            axs[2].set_title(f"Estimated (blue) vs SPICE (red) Z component over time for {self.code_mpc}")
            fig.set_tight_layout(True)


    