"""
This module contains functions for plotting the results of estimation scripts,
e.g. est_w_mpc_single.py and est_w_mpc_large.py.

generate_estimation_plots() function can be called with the path to the saved
output files of one body's estimation as the only argument and the relevant
plots will be generated. This file can also be run as a script.
"""

#%%%%%%# A. Imports
######## A. Imports
#region A. Imports

# Directory business
import os
# local_path = os.path.dirname(__file__)
# os.chdir(local_path) # ensure that the cwd is the subfolder where this file is

# Tudat imports
import tudatpy.constants as cnst
from tudatpy.interface import spice
from tudatpy.astro import frame_conversion
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics import environment_setup
from tudatpy.estimation import (
    observable_models_setup,
    observations_setup,
    observations,
)

# MPC, JPL SBDB, Horizons interfaces
from tudatpy.data.mpc import BatchMPC
from tudatpy.data.horizons import HorizonsQuery

# Other Python imports
# import sys
# import click
import json
import numpy as np
import datetime as dttm
import matplotlib
import matplotlib.cm as colormaps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# Local imports
import IS_utils as isu




#%%%%%%################   B. Plotting Functions   ##############################
#######################   B. Plotting Functions   ##############################
#region B. Plotting Functions

def output_plot(
    fig_name,
    plots_save_path,
    pb = 'show',
    colormode = 0,
    format = 'pdf', # only 'pdf' or 'png'
    dpi = 200,
):
    if (pb == 'show'):
        plt.show(block=False)
        plt.pause(0.001)
    elif (pb == 'save'):
        save_path = plots_save_path + f"{format.upper()}s/" if format != 'pdf' else plots_save_path
        save_path += "darkmode/" if colormode else ""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_name += f".{format}"
        print(save_path + fig_name)
        plt.savefig(save_path + fig_name, format=format, dpi=dpi)
        plt.close()

def get_gap_ranges(
    first_obs_J2000,
    last_obs_J2000,
    observation_collection_concatenated_times,
    gap_in_months=6,
):
    # Get ranges for all gaps in observations larger than gap_in_months (default
    # is 6 months)
    residual_times = (
        np.array([first_obs_J2000, first_obs_J2000, *observation_collection_concatenated_times, last_obs_J2000, last_obs_J2000]) / cnst.JULIAN_YEAR + 2000
    )
    gaps = np.abs(np.diff(sorted(residual_times)))
    # Count the number of gaps larger than gap_in_months
    num_gaps = ( gaps > (gap_in_months / 12) ).sum()
    indices_of_largest_gaps = np.argsort(gaps)[-num_gaps:]
    # (start, end) for each of the gaps
    gap_ranges = [
        (sorted(residual_times)[idx - 1], sorted(residual_times)[idx + 1])
        for idx in indices_of_largest_gaps
    ]

    return gap_ranges

def get_inverted_gap_ranges(
    first_obs_J2000,
    last_obs_J2000,
    gap_ranges,
):
    # invert gap ranges (get periods when observations ARE available, rather
    # than where they're NOT available)
    gap_ranges.sort()
    inv_gap_ranges = np.zeros((len(gap_ranges)+1, 2))
    inv_gap_ranges[0,0] = (first_obs_J2000 / cnst.JULIAN_YEAR) + 2000
    inv_gap_ranges[1:,0] = [gap[1] for gap in gap_ranges]
    inv_gap_ranges[:-1,1] = [gap[0] for gap in gap_ranges]
    inv_gap_ranges[-1,1] = (last_obs_J2000 / cnst.JULIAN_YEAR) + 2000

    return inv_gap_ranges

def check_comparison_reference(comparison_reference: str):

    comparison_reference = comparison_reference.lower()
    available_references = ["spk", "horizons"]

    if comparison_reference not in available_references:
        raise ValueError(
            f"Comparison reference must be one of {available_references}, got {comparison_reference}."
        )

    return comparison_reference

def get_reference_states(
    comparison_reference,
    body,
    central_body,
    global_frame_orientation,
    epoch_list,
):
    comparison_reference = check_comparison_reference(comparison_reference)

    # retrieve the states for a list of times in:
    # SPICE
    if comparison_reference == "spk":
        reference_states = np.array([
            spice.get_body_cartesian_state_at_epoch(
                body['SPKID'],
                central_body,
                global_frame_orientation,
                "NONE",
                timee,
            ) for timee in epoch_list
        ])

    # Horizons
    elif comparison_reference == "horizons":
        horizons_query = HorizonsQuery(
            query_id=f"{body['provisional_designation']};",
            location=f"@{central_body}",
            epoch_list=list(epoch_list),
            extended_query=True,
        )
        reference_states = horizons_query.cartesian(
            frame_orientation=global_frame_orientation
        )[:, 1:]

    return reference_states

#region B1.1a Pos. diff.
def plot_position_error_2_by_2(
    state_history_array,
    reference_states,
    output_times,
    obs_ranges,
    in_RSW: bool = False,
    plot_formal_error: bool = False,
    formal_errors_KM = None,
    formal_errors_n_sigma: int = 1, # must be 1, 2, or 3
    plot_JPL_uncertainty: bool = False,
    ephem_unc_table = None,
    colormode: int = 0,
    plot_behaviour: str = "show",
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    # Error in kilometers (and optional conversion to RSW)
    error_to_reference_KM = (reference_states - state_history_array) / 1000
    if in_RSW:
        error_to_reference_KM = np.array(
            [frame_conversion.inertial_to_rsw_rotation_matrix(reference_state) @ error[:3]
                for reference_state, error in zip(reference_states, error_to_reference_KM)]
        )

    # Plot
    times_plot = (output_times / cnst.JULIAN_YEAR) + 2000 # approximate for plot ticks
    fig, axs = plt.subplots(
        2, 2,
        figsize = np.array([1.0,0.67])*isu.full_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )

    # Plot XYZ / RSW true error components, each component gets its own subplot
    component_names = ["R", "S", "W"] if in_RSW else [r"$x$", r"$y$", r"$z$"]
    for i, ax in enumerate(axs.flatten()[:3]):
        ax.plot(
            times_plot,
            error_to_reference_KM[:, i],
            color = isu.cmap_custom10[colormode](i),
            label = fr"True error in {component_names[i]} ($\varepsilon_{component_names[i]}$)",
            linewidth = 1.2,
            alpha = 0.8,
        )
    # Plot true error magnitude in the last subplot
    error_to_reference_KM_mag = np.linalg.norm(error_to_reference_KM[:, :3], axis=1)
    axs[1,1].plot(
            times_plot,
            error_to_reference_KM_mag,
            color = isu.text_color[colormode],
            label = fr"True error of pos. mag. ($\varepsilon_{{r}}$)",
            linewidth = 1.2,
            alpha = 0.8,
        )

    # Plot JPL ephemeris uncertainty (band of +/- 3 sigma from zero line)
    if plot_JPL_uncertainty:
        # Convert times to J2000, then to years
        ephem_unc_epochs_YEAR = (ephem_unc_table["ephemeris_time_J2000"] / cnst.JULIAN_YEAR) + 2000

        if in_RSW:
            ephem_unc_KM = np.array([
                    ephem_unc_table["r_s"],
                    ephem_unc_table["t_s"],
                    ephem_unc_table["n_s"],
            ], dtype=float).T / 1000
        else:
            ephem_unc_KM = np.array([
                    ephem_unc_table["x_s"],
                    ephem_unc_table["y_s"],
                    ephem_unc_table["z_s"],
            ], dtype=float).T / 1000

        # XYZ / RSW components
        for i, ax in enumerate(axs.flatten()[:3]):
            ax.fill_between(
                ephem_unc_epochs_YEAR,
                ephem_unc_KM[:, i],
                -ephem_unc_KM[:, i],
                color = isu.text_color[colormode],
                label = fr"JPLH $\pm 3\sigma$ uncertainty ({component_names[i]})",
                alpha = 0.4,
            )
        # Magnitude
        ephem_unc_KM_mag = np.linalg.norm(ephem_unc_KM, axis=1)
        axs[1,1].fill_between(
            ephem_unc_epochs_YEAR,
            ephem_unc_KM_mag,
            color = isu.text_color[colormode],
            label = fr"JPLH $3\sigma$ uncertainty (pos. mag.)",
            alpha = 0.4,
        )

    # Plot formal error
    if plot_formal_error:
        formal_errors_n_sigma_str = str(formal_errors_n_sigma) if formal_errors_n_sigma != 1 else ""
        # XYZ / RSW components
        for i, ax in enumerate(axs.flatten()[:3]):
            ax.plot(
                times_plot,
                formal_errors_KM[:, i] * formal_errors_n_sigma,
                color = isu.text_color[colormode],
                label = fr"Formal error in {component_names[i]} (${formal_errors_n_sigma_str}\sigma_{component_names[i]}$)",
                alpha = 0.8,
                linewidth = 1,
                linestyle = '--',
            )
        # Magnitude
        formal_errors_KM_mag = np.linalg.norm(formal_errors_KM[:, :3], axis=1)
        axs[1,1].plot(
            times_plot,
            formal_errors_KM_mag * formal_errors_n_sigma,
            color = isu.text_color[colormode],
            label = fr"Formal error of pos. mag. (${formal_errors_n_sigma_str}\sigma_{{r}}$)",
            alpha = 0.8,
            linewidth = 1,
            linestyle = '--',
            # edgecolor = isu.text_color[colormode],
        )

    for ax_idx, ax in enumerate(axs.flatten()):
        # Show areas where there are no observations:
        for i, gap in enumerate(obs_ranges):
            ax.axvspan(
                xmin = gap[0],
                xmax = gap[1],
                color = isu.cmap_custom10[colormode](9),
                alpha = 0.4,
                label = "Periods with observations" if (i==0 and ax_idx==0) else None,
            )
        # Legend
        ax.legend(
            loc = 'best',
            # bbox_to_anchor = (1.1, 1.05),
            markerscale = 1.2,
            fontsize = isu.small_font_size,
            labelcolor = isu.text_color[colormode],
            facecolor = isu.legend_bg_color[colormode],
            # ncol = 3,
        )
        # Make zero-lines more obvious
        ax.axhline(linewidth=1, color=isu.text_color[colormode])
        # Set default settings
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        # Y labels
        ax.set_ylabel("Error [km]")
        # Shrink axis to fit legend at top
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])

    # Figure title
    frame_name = "RSW" if in_RSW else "Cartesian"
    fig.suptitle(
        f"Position Difference ({frame_name} Frame) w.r.t. JPL Horizons: {body_longname}",
        fontsize = isu.medium_font_size,
        weight = 'bold',
    )
    # Subplot titles
    axs[0,0].set_title(
        r"$x$ Cartesian Component" if not in_RSW else "Radial (R) Component",
        fontsize = isu.medium_font_size,
    )
    axs[0,1].set_title(
        r"$y$ Cartesian Component" if not in_RSW else "Transverse (S) Component",
        fontsize = isu.medium_font_size,
    )
    axs[1,0].set_title(
        r"$z$ Cartesian Component" if not in_RSW else "Cross-Track (W) Component",
        fontsize = isu.medium_font_size,
    )
    axs[1,1].set_title(
        "Position Error Magnitude",
        fontsize = isu.medium_font_size,
    )
    # X labels
    for ax in axs.flatten()[2:]:
        ax.set_xlabel("Year")

    plt.tight_layout(h_pad=2, rect=[0,0,1,0.97])

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     formal_error_addition = f"{'_w-formal-errors'*plot_formal_error}"
    #     fig_name = f"pos-error-wrt-JPLH_subplots_{frame_name}{formal_error_addition}.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Position error wrt JPL Horizons {'with formal errors '*plot_formal_error}(subplots) vs. time {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    formal_error_addition = f"{'_w-formal-errors'*plot_formal_error}"
    fig_name = f"pos-error-wrt-JPLH_subplots_{frame_name}{formal_error_addition}"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B1.1a Pos. diff.
def plot_position_error(
    state_history_array,
    reference_states,
    output_times,
    obs_ranges,
    in_RSW: bool = False,
    plot_formal_error: bool = False,
    formal_errors_KM = None,
    formal_errors_n_sigma: int = 1, # must be 1, 2, or 3
    plot_JPL_uncertainty: bool = False,
    ephem_unc_table = None,
    colormode: int = 0,
    plot_behaviour: str = "show",
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    # Error in kilometers (and optional conversion to RSW)
    error_to_reference_KM = (reference_states - state_history_array) / 1000
    if in_RSW:
        error_to_reference_KM = np.array(
            [frame_conversion.inertial_to_rsw_rotation_matrix(reference_state) @ error[:3]
                for reference_state, error in zip(reference_states, error_to_reference_KM)]
        )

    # Plot
    times_plot = (output_times / cnst.JULIAN_YEAR) + 2000 # approximate for plot ticks
    fig, axs = plt.subplots(
        4, 1,
        figsize = np.array([1.0,0.75])*isu.full_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
        sharex = True,
    )

    # Show areas where there are observations:
    for ax_idx, ax in enumerate(axs.flatten()):
        for gap_idx, gap in enumerate(obs_ranges):
            ax.axvspan(
                xmin = gap[0],
                xmax = gap[1],
                color = isu.cmap_custom10[colormode](9),
                alpha = 0.4,
                # label = "Periods with observations" if (gap_idx==0 and ax_idx==0) else None,
                label = "Obs." if (gap_idx==0 and ax_idx==0) else None,
            )

    # Plot XYZ / RSW true error components, each component gets its own subplot
    component_names = ["R", "S", "W"] if in_RSW else [r"$x$", r"$y$", r"$z$"]
    for ax_idx, (ax, comp_name) in enumerate(zip(axs.flatten()[:3], component_names)):
        ax.plot(
            times_plot,
            error_to_reference_KM[:, ax_idx],
            color = isu.cmap_custom10[colormode](ax_idx),
            # label = fr"True error, {comp_name} ($\varepsilon_{comp_name}$)",
            label = fr"$ε_{comp_name}$",
            linewidth = 1.5,
            alpha = 1.0,
            zorder = 100,
        )

    # Plot true error magnitude in the last subplot
    error_to_reference_KM_mag = np.linalg.norm(error_to_reference_KM[:, :3], axis=1)
    axs[3].plot(
            times_plot,
            error_to_reference_KM_mag,
            color = isu.text_color[colormode],
            # label = fr"True error, pos. ($\varepsilon_{{r}}$)",
            label = fr"$ε_{{r}}$",
            linewidth = 1.5,
            alpha = 1.0,
        )

    # Plot formal error
    if plot_formal_error:
        formal_errors_n_sigma_str = str(formal_errors_n_sigma) if formal_errors_n_sigma != 1 else ""
        # XYZ / RSW components
        for ax_idx, (ax, comp_name) in enumerate(zip(axs.flatten()[:3], component_names)):
            ax.plot(
                times_plot,
                formal_errors_KM[:, ax_idx] * formal_errors_n_sigma,
                color = isu.cmap_custom10[colormode](ax_idx),
                # label = fr"Formal error, {comp_name} (${formal_errors_n_sigma_str}σ_{comp_name}$)",
                label = fr"${formal_errors_n_sigma_str}σ_{comp_name}$",
                alpha = 0.7,
                linewidth = 1.5,
                linestyle = '--',
            )

        # Magnitude
        formal_errors_KM_mag = np.linalg.norm(formal_errors_KM[:, :3], axis=1)
        axs[3].plot(
            times_plot,
            formal_errors_KM_mag * formal_errors_n_sigma,
            color = isu.text_color[colormode],
            # label = fr"Formal error, pos. (${formal_errors_n_sigma_str}σ_{{r}}$)",
            label = fr"${formal_errors_n_sigma_str}S_{{r}}$", # rename!!!
            alpha = 0.7,
            linewidth = 1.5,
            linestyle = '--',
        )

    # Plot JPL ephemeris uncertainty (band of +/- 3 sigma from zero line)
    if plot_JPL_uncertainty:
        # Convert times to J2000, then to years
        ephem_unc_epochs_YEAR = (ephem_unc_table["ephemeris_time_J2000"] / cnst.JULIAN_YEAR) + 2000

        if in_RSW:
            ephem_unc_KM = np.array([
                    ephem_unc_table["r_s"],
                    ephem_unc_table["t_s"],
                    ephem_unc_table["n_s"],
            ], dtype=float).T / 1000
        else:
            ephem_unc_KM = np.array([
                    ephem_unc_table["x_s"],
                    ephem_unc_table["y_s"],
                    ephem_unc_table["z_s"],
            ], dtype=float).T / 1000

        # XYZ / RSW components
        for ax_idx, (ax, comp_name) in enumerate(zip(axs.flatten()[:3], component_names)):
            ax.fill_between(
                ephem_unc_epochs_YEAR,
                ephem_unc_KM[:, ax_idx],
                -ephem_unc_KM[:, ax_idx],
                color = isu.cmap_custom10[colormode](ax_idx),
                # label = fr"JPLH $\pm 3σ$ unc. ({comp_name})",
                label = fr"$3σ_{comp_name}^{{JPLH}}$",
                alpha = 0.3,
            )
        # Magnitude
        ephem_unc_KM_mag = np.linalg.norm(ephem_unc_KM, axis=1)
        axs[3].fill_between(
            ephem_unc_epochs_YEAR,
            ephem_unc_KM_mag,
            color = isu.text_color[colormode],
            # label = fr"JPLH $3σ$ unc. (pos.)",
            label = fr"$3S_{{r}}^{{JPLH}}$", # rename!!!
            alpha = 0.3,
        )

    for ax_idx, ax in enumerate(axs.flatten()):
        # Make zero-lines more obvious
        ax.axhline(linewidth=1, color=isu.text_color[colormode], zorder=5)
        # Set default settings
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        # Reduce horizontal margin (default is 0.05)
        ax.margins(x=0.01)
        # Shrink axis to fit legend at top
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])

    # Figure title
    frame_name = "RSW" if in_RSW else "Cartesian"
    fig.suptitle(
        f"Position Difference w.r.t. JPL Horizons: {body_longname}",
        fontsize = isu.medium_font_size,
        # weight = 'bold',
    )
    # X/Y labels, ticks
    axs[3].set_xlabel("Year")
    axs[3].set_ylabel(r"$r$ error [km]", fontsize=isu.medium_font_size)
    for ax, comp_name in zip(axs[:3], component_names):
        ax.set_ylabel(f"{comp_name} error [km]", fontsize=isu.medium_font_size)
        ax.tick_params(labelbottom=False)

    ## Legend business: single legend for all subplots
    # Get all lines and their labels
    lines_labels = [axis.get_legend_handles_labels() for axis in fig.axes]
    lines, labels = [sum(list_of_lists, []) for list_of_lists in zip(*lines_labels)]
    leg = fig.legend(
        lines,
        labels,
        # loc = 'outside upper center',
        loc = 'center right',
        bbox_to_anchor = (0.99, 0.5),
        markerscale = 2,
        fontsize = isu.medium_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
        ncol = 1,
    )
    # Make legend items full opacity (except for regions)
    for lh in leg.legend_handles:
        if not isinstance(lh, matplotlib.patches.Rectangle):
            lh.set_alpha(1)
    # Make space for legend by shrinking the main part of the figure
    fig.tight_layout(rect=[0,0,0.82,1])

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     formal_error_addition = f"{'_w-formal-errors'*plot_formal_error}"
    #     JPL_eph_addition = f"{'_w-JPL-unc'*plot_JPL_uncertainty}"
    #     fig_name = f"pos-error-wrt-JPLH_subplots_{frame_name}{formal_error_addition}{JPL_eph_addition}.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Position error wrt JPL Horizons {'with formal errors '*plot_formal_error}(subplots) vs. time {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    formal_error_addition = f"{'_w-formal-errors'*plot_formal_error}"
    JPL_eph_addition = f"{'_w-JPL-unc'*plot_JPL_uncertainty}"
    fig_name = f"pos-error-wrt-JPLH_subplots_{frame_name}{formal_error_addition}{JPL_eph_addition}"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B1.1b Pos. diff.
def plot_cartesian_single(
    state_history_array,
    reference_states,
    output_times,
    obs_ranges,
    in_RSW: bool = False,
    plot_error_norm: bool = True,
    plot_formal_error: bool = False,
    formal_errors_KM = None,
    formal_errors_n_sigma: int = 1, # must be 1, 2, or 3
    plot_JPL_uncertainty: bool = False,
    ephem_unc_table = None,
    colormode: int = 0,
    plot_behaviour: str = "show",
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    # Error in kilometers (and optional conversion to RSW)
    error_to_reference_KM = (reference_states - np.array(state_history_array)) / 1000
    if in_RSW:
        error_to_reference_KM = np.array([
            frame_conversion.inertial_to_rsw_rotation_matrix(reference_state) @ error[:3]
            for reference_state, error in zip(reference_states, error_to_reference_KM)
        ])

    # Plot
    times_plot = (output_times / cnst.JULIAN_YEAR) + 2000 # approximate for plot ticks
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.50])*isu.full_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )

    # Plot ranges where observations are present
    for i, gap in enumerate(obs_ranges):
        ax.axvspan(
            xmin = gap[0],
            xmax = gap[1],
            color = isu.cmap_custom10[colormode](9),
            alpha = 0.4,
            # label = "Periods with observations" if i == 0 else None,
            label = "Obs." if i == 0 else None,
        )

    # XYZ / RSW error component curves
    # labels = [" — Radial", " — Transverse", " — Cross-track"] if in_RSW else ["", "", ""]
    component_names = ["R", "S", "W"] if in_RSW else [r"$x$", r"$y$", r"$z$"]
    for i, comp_name in enumerate(component_names):
        ax.plot(
            times_plot,
            error_to_reference_KM[:, i],
            # label = fr"True error in {comp_name} ($\varepsilon_{comp_name}$)",
            label = fr"$ε_{comp_name}$",
            color = isu.cmap_custom10[colormode](i),
            linewidth = 1.5,
            alpha = 0.8,
        )
    # Error magnitude curve
    if plot_error_norm:
        error_to_reference_KM_mag = np.linalg.norm(error_to_reference_KM[:, :3], axis=1)
        ax.plot(
            times_plot,
            error_to_reference_KM_mag,
            # label = fr"True error of pos. mag. ($\varepsilon_{{r}}$)",
            label = fr"$ε_{{r}}$",
            linestyle = isu.linestyles[0],
            color = isu.text_color[colormode],
            linewidth = 2.0,
            alpha = 0.8,
            zorder = 100,
        )

    # Plot formal error curves
    if plot_formal_error:
        formal_errors_n_sigma_str = str(formal_errors_n_sigma) if formal_errors_n_sigma != 1 else ""
        if plot_error_norm:
            # Magnitude
            formal_errors_KM_mag = np.linalg.norm(formal_errors_KM[:, :3], axis=1)
            ax.plot(
                times_plot,
                formal_errors_KM_mag * formal_errors_n_sigma,
                color = isu.text_color[colormode],
                linestyle = "--",
                linewidth = 1.5,
                # label = fr"Formal error of pos. mag. (${formal_errors_n_sigma_str}\sigma_{{r}}$)",
                label = fr"${formal_errors_n_sigma_str}S_{{r}}$", # rename!!!
                alpha = 0.5,
            )
        for i, comp_name in enumerate(component_names):
            ax.plot(
                times_plot,
                formal_errors_KM[:, i] * formal_errors_n_sigma,
                color = isu.cmap_custom10[colormode](i),
                linestyle = "--",
                linewidth = 1.5,
                # label = fr"Formal error in {component_names[i]} (${formal_errors_n_sigma_str}\sigma_{component_names[i]}$)",
                label = fr"${formal_errors_n_sigma_str}σ_{comp_name}$", # rename!!!
                alpha = 0.5,
            )

    # Plot JPL ephemeris uncertainty (band of +3 sigma from zero line)
    if plot_JPL_uncertainty:
        # Convert times to J2000, then to years
        ephem_unc_epochs_YEAR = (ephem_unc_table["ephemeris_time_J2000"] / cnst.JULIAN_YEAR) + 2000

        if in_RSW:
            ephem_unc_KM = np.array([
                    ephem_unc_table["r_s"],
                    ephem_unc_table["t_s"],
                    ephem_unc_table["n_s"],
            ], dtype=float).T / 1000
        else:
            ephem_unc_KM = np.array([
                    ephem_unc_table["x_s"],
                    ephem_unc_table["y_s"],
                    ephem_unc_table["z_s"],
            ], dtype=float).T / 1000

        # Magnitude
        ephem_unc_KM_mag = np.linalg.norm(ephem_unc_KM, axis=1)
        ax.fill_between( # +3 sigma
            ephem_unc_epochs_YEAR,
            ephem_unc_KM_mag,
            color = isu.text_color[colormode],
            # linestyle = "--",
            # linewidth = 2.0,
            # label = f"JPLH $3\sigma$ uncertainty (pos. mag.)",
            label = fr"$3S_{{r}}^{{JPLH}}$", # rename!!!
            alpha = 0.3,
        )

    # Set default settings
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    # Make zero-lines more obvious
    ax.axhline(linewidth=1, color=isu.text_color[colormode])
    # Reduce horizontal margin (default is 0.05)
    ax.margins(x=0.01)
    # Legend
    leg = fig.legend(
        loc = 'outside center right',
        markerscale = 1.2,
        fontsize = isu.medium_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    for lh in leg.legend_handles:
        # Make legend items full opacity (except for regions)
        if not isinstance(lh, matplotlib.patches.Rectangle):
            lh.set_alpha(1)
    # Labels, title
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Error [km]")
    frame_name = "RSW" if in_RSW else "Cartesian"
    fig.suptitle(
        f"Position Difference w.r.t. JPL Horizons: {body_longname}",
        # weight = 'bold',
        fontsize = isu.medium_font_size,
    )

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     formal_error_addition = f"{'_w-formal-errors'*plot_formal_error}"
    #     fig_name = f"pos-error-wrt-JPLH_single_{frame_name}{formal_error_addition}.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Position error wrt JPL Horizons {'with formal errors '*plot_formal_error}(single) vs. time {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    formal_error_addition = f"{'_w-formal-errors'*plot_formal_error}"
    fig_name = f"pos-error-wrt-JPLH_single_{frame_name}{formal_error_addition}"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B1.2 Formal error
def plot_formal_errors(
    output_times,
    formal_errors_KM,
    obs_ranges,
    in_RSW: bool = False,
    plot_error_norm: bool = True,
    colormode: int = 0,
    plot_behaviour: str = "show",
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    # Plot
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.50])*isu.full_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )

    # Plot ranges where observations are present
    for i, gap in enumerate(obs_ranges):
        ax.axvspan(
            xmin = gap[0],
            xmax = gap[1],
            color = isu.cmap_custom10[colormode](9),
            alpha = 0.4,
            # label="Periods with observations" if i == 0 else None,
            label = "Obs." if i == 0 else None,
        )

    # Formal error XYZ / RSW component curves
    times_plot = (output_times / cnst.JULIAN_YEAR) + 2000
    component_names = ["R", "S", "W"] if in_RSW else [r"$x$", r"$y$", r"$z$"]
    for i, comp_name in enumerate(component_names):
        ax.plot(
            times_plot,
            formal_errors_KM[:, i],
            # label = fr"Formal error in {comp_name} ($\sigma_{comp_name}$)",
            label = fr"$σ_{comp_name}$",
            color = isu.cmap_custom10[colormode](i),
            linewidth = 1.5,
            alpha = 0.8,
        )
    # Plot formal error in position magnitude curve
    if plot_error_norm:
        ax.plot(
            times_plot,
            np.linalg.norm(formal_errors_KM[:, :3], axis=1),
            # label = fr"Formal error of pos. mag. ($\sigma_{{r}}$)",
            label = fr"$S_{{r}}$", # rename!!!
            linestyle = isu.linestyles[0],
            color = isu.text_color[colormode],
            linewidth = 2.0,
            alpha = 0.8,
        )

    # Default axis settings
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    # Make zero line more obvious
    ax.axhline(linewidth=1, color=isu.text_color[colormode])
    # Reduce horizontal margin (default is 0.05)
    ax.margins(x=0.01)    # Legend, axis labels, figure title
    leg = fig.legend(
        loc = 'outside center right',
        markerscale = 1.2,
        fontsize = isu.medium_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    for lh in leg.legend_handles:
        # Make legend items full opacity (except for regions)
        if not isinstance(lh, matplotlib.patches.Rectangle):
            lh.set_alpha(1)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Error [km]")
    frame_name = "RSW" if in_RSW else "Cartesian"
    fig.suptitle(
        fr"Formal Position Errors ($1\sigma$): {body_longname}",
        # weight = 'bold',
        fontsize = isu.medium_font_size
    )

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"formal-errors_{frame_name}.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Formal error ({frame_name}) vs. time {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    fig_name = f"formal-errors_{frame_name}"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B2.1 Iter.residuals
def plot_residuals(
    number_of_pod_iterations: int,
    est_out_residual_history: list,
    obs_col_concatenated_times: list,
    best_iteration: int,
    colormode: int = 0,
    plot_behaviour: str = "show",
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    fig, axs = plt.subplots(
        1, number_of_pod_iterations,
        figsize = (number_of_pod_iterations * 3.5, 4),
        sharex = True,
        sharey = False,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )

    # Get an approx. year from our times (which are in seconds since J2000)
    residual_times = (np.array(obs_col_concatenated_times) / cnst.JULIAN_YEAR) + 2000

    # convert residuals from radians to arcseconds
    residual_history = np.rad2deg(est_out_residual_history) * 3600

    # Plot the residuals, split between RA and DEC types
    for i, ax in enumerate(axs.flatten()):
        # print(len(residual_times[::2]))
        # print(len(residual_history[::2, i]))

        # Residual scatters
        ax.scatter(
            residual_times[::2],
            residual_history[::2, i],
            color = isu.cmap_custom10[colormode](0),
            marker = "+",
            alpha = 0.3,
            s = 30,
            label = "RA",
        )
        ax.scatter(
            residual_times[1::2],
            residual_history[1::2, i],
            color = isu.cmap_custom10[colormode](1),
            marker = "+",
            alpha = 0.3,
            s = 30,
            label = "DEC",
        )
        # Make the horizontal zero-line more obvious
        ax.axhline(linewidth=1, color=isu.text_color[colormode])
        # Set default settings
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        # Set y-axis limits
        # axs[i, setup_idx].set_ylim(-5, 5)
        # Set log scale for y-axis (to be used in combination with absolute values of residuals)
        # axs[i, setup_idx].set_yscale('log')

        # Subplot titles, axis labels, legend
        if i == best_iteration:
            ax.set_title(
                f"Iteration {i+1} (best)",
                fontsize = isu.medium_font_size,
            )
        else:
            ax.set_title(
                f"Iteration {i+1}",
                fontsize = isu.medium_font_size,
            )
        ax.set_xlabel("Year")#, fontsize = isu.small_font_size)
        if i == 0:
            ax.set_ylabel("Residual [arcsec]")#, fontsize = isu.small_font_size)
            leg = ax.legend(
                loc = 'best',
                markerscale = 1.2,
                fontsize = isu.small_font_size,
                labelcolor = isu.text_color[colormode],
                facecolor = isu.legend_bg_color[colormode],
            )
            for lh in leg.legend_handles:
                lh.set_alpha(1) # makes the markers in legend full opacity

    # Figure title
    fig.suptitle(
        f"Observation Residuals by Iteration: {body_longname}",
        fontsize = isu.medium_font_size,
        # weight = 'bold',
    )

    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"obs-residuals-by-iteration.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Observation residuals vs. time {'darkmode '*colormode}figure
    #     # saved: \n{save_path + fig_name}")

    # Plot output
    fig_name = f"obs-residuals-by-iteration"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B2.2 Sim.residuals
def plot_simulated_residuals(
    batch_table_filtered,
    residual_filter_cutoff_ARCSEC,
    plot_outliers: bool = True,
    batch_table_outliers = None,
    colormode: int = 0,
    plot_behaviour: str = 'show',
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    # Plot residuals over time
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.67])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = 'constrained',
    )
    ax = isu.set_default_2d_ax_settings(ax, colormode)

    # Kept observations
    kept_times = (np.array(batch_table_filtered.epoch_seconds_TDB) / cnst.JULIAN_YEAR) + 2000
    ax.scatter(
        kept_times,
        batch_table_filtered.simulated_residual_RA_ARCSEC,
        color = isu.cmap_custom10[colormode](0),
        marker = "+",
        alpha = 0.3,
        s = 30,
        label = "RA",
    )
    ax.scatter(
        kept_times,
        batch_table_filtered.simulated_residual_DEC_ARCSEC,
        color = isu.cmap_custom10[colormode](1),
        marker = "+",
        alpha = 0.3,
        s = 30,
        label = "DEC",
    )

    if plot_outliers and (len(batch_table_outliers) > 0):
        # Outliers / removed observations
        outlier_times = (np.array(batch_table_outliers.epoch_seconds_TDB) / cnst.JULIAN_YEAR) + 2000
        ax.scatter(
            outlier_times,
            batch_table_outliers.simulated_residual_RA_ARCSEC,
            color = isu.cmap_custom10[colormode](3),
            marker = "x",
            alpha = 0.3,
            s = 30,
            label = "Rej. RA",
        )
        ax.scatter(
            outlier_times,
            batch_table_outliers.simulated_residual_DEC_ARCSEC,
            color = isu.cmap_custom10[colormode](4),
            marker = "x",
            alpha = 0.3,
            s = 30,
            label = "Rej. DEC",
        )

    # Draw horizontal lines representing the residual cutoff
    ax.axhline(
        residual_filter_cutoff_ARCSEC,
        label = "Cutoff",
        color = isu.text_color[colormode],
        alpha = 0.8,
        linewidth = 1,
        linestyle = '--',
    )
    ax.axhline(
        -residual_filter_cutoff_ARCSEC,
        color = isu.text_color[colormode],
        alpha = 0.8,
        linewidth = 1,
        linestyle = '--',
    )

    # Make the horizontal zero-line more obvious
    ax.axhline(linewidth=1, color=isu.text_color[colormode])

    # Legend, labels, title
    leg = fig.legend(
        loc = 'outside center right',
        markerscale = 1.2,
        fontsize = isu.tiny_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1) # makes the markers in legend full opacity

    ax.set_ylabel("Residual [arcsec]")
    ax.set_xlabel("Year")
    ax.set_title(
        f"Simulated Residuals: {body_longname}",
        fontsize = isu.medium_font_size,
    )

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     outlier_addition = f"{'_w-outliers'*plot_outliers}"
    #     fig_name = f"simulated-residuals{outlier_addition}.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")
    #     plt.close()

    # Plot output
    outlier_addition = f"{'_w-outliers'*plot_outliers}"
    fig_name = f"simulated-residuals{outlier_addition}"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B2.3 Norm.residuals
def plot_final_residuals(
    final_residuals: list,
    obs_col_concatenated_times: list,
    ylabel = None,
    fig_name = None,
    colormode: int = 0,
    plot_behaviour: str = "show",
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.67])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )
    ax = isu.set_default_2d_ax_settings(ax, colormode)

    # Get an approx. year from our times (which are in seconds since J2000)
    residual_times = (np.array(obs_col_concatenated_times) / cnst.JULIAN_YEAR) + 2000

    # Residual scatters
    ax.scatter(
        residual_times[::2],
        final_residuals[::2],
        color = isu.cmap_custom10[colormode](0),
        marker = "+",
        alpha = 0.3,
        s = 30,
        label = "RA",
    )
    ax.scatter(
        residual_times[1::2],
        final_residuals[1::2],
        color = isu.cmap_custom10[colormode](1),
        marker = "+",
        alpha = 0.3,
        s = 30,
        label = "DEC",
    )
    # Make the horizontal zero-line more obvious
    ax.axhline(linewidth=1, color=isu.text_color[colormode])
    # Set default settings
    # axs[i, setup_idx].set_ylim(-5, 5)
    # Set log scale for y-axis (to be used in combination with absolute values of residuals)
    # axs[i, setup_idx].set_yscale('log')

    # Subplot titles, axis labels, legend
    ax.set_xlabel("Year")#, fontsize = isu.small_font_size)
    ax.set_ylabel(ylabel)#, fontsize = isu.small_font_size)
    leg = ax.legend(
        loc = 'best',
        markerscale = 1.2,
        fontsize = isu.tiny_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1) # makes the markers in legend full opacity

    # Figure title
    fig.suptitle(
        f"Final Residuals: {body_longname}",
        fontsize = isu.medium_font_size,
        # weight = 'bold',
    )

    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()

    # Plot output
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B3.1 Pre/Post Fit
def get_observatory_data_for_plot(
    batch_observatories_table,
    obs_col_concatenated_link_definition_ids,
    obs_col_link_definition_ids,
    num_observatories: int,
):
    # Collect the num_observatories largest observatories
    observatory_names = (
        batch_observatories_table
        .sort_values("count", ascending=False)
        .iloc[0:num_observatories]
        .set_index("Code")
    )
    top_observatories = observatory_names.index.tolist()

    # Create a concatenated_receiving_observatories map to identify the
    # observatories by their MPC code instead of an internally used id
    residuals_observatories = obs_col_concatenated_link_definition_ids
    unique_observatories = set(residuals_observatories)
    observatory_link_to_mpccode = {
        idx: obs_col_link_definition_ids[idx][
            observable_models_setup.links.receiver
        ].reference_point
        for idx in unique_observatories
    }

    # the resulting map (MPC code for each item in the residuals_history):
    concatenated_receiving_observatories = np.array(
        [observatory_link_to_mpccode[idx] for idx in residuals_observatories]
    )

    # mask for the observatories not in top num_observatories:
    mask_not_top = [
        (False if observatory in top_observatories else True)
        for observatory in concatenated_receiving_observatories
    ]

    # get the number of observations by the other observatories
    # (divide by two because the observations are concatenated RA,DEC in this list)
    n_obs_not_top = int(sum(mask_not_top) / 2)

    return observatory_names, top_observatories, unique_observatories, \
        concatenated_receiving_observatories, mask_not_top, n_obs_not_top

def plot_residuals_by_observatories(
    batch_observatories_table,
    obs_col_concatenated_link_definition_ids,
    obs_col_link_definition_ids,
    obs_col_concatenated_times,
    residual_data_ARCSEC,
    title_prefix: str,
    num_observatories: int = 25,
    colormode: int = 0,
    plot_behaviour: str = 'show',
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    observatory_names, top_observatories, \
        unique_observatories, concatenated_receiving_observatories, \
            mask_not_top, n_obs_not_top = get_observatory_data_for_plot(
            batch_observatories_table = batch_observatories_table,
            obs_col_concatenated_link_definition_ids = obs_col_concatenated_link_definition_ids,
            obs_col_link_definition_ids = obs_col_link_definition_ids,
            num_observatories = num_observatories,
    )

    residual_times_YEAR = (np.array(obs_col_concatenated_times) / cnst.JULIAN_YEAR) + 2000

    ## Plot
    fig, axs = plt.subplots(
        2, 1,
        figsize = np.array([1.0,0.67])*isu.full_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )

    # Plot remaining observatories first
    if n_obs_not_top > 0:
        # RA
        axs[0].scatter(
            residual_times_YEAR[mask_not_top][::2],
            residual_data_ARCSEC[mask_not_top][::2],
            marker = ".",
            s = 30,
            color = isu.text_color[colormode],
            alpha = 0.2,
        )
        # DEC
        axs[1].scatter(
            residual_times_YEAR[mask_not_top][1::2],
            residual_data_ARCSEC[mask_not_top][1::2],
            marker = ".",
            s = 30,
            label = f"{len(unique_observatories) - num_observatories} Other | RMS RA (DEC): {isu.rms(residual_data_ARCSEC[mask_not_top][::2]):.2f} ({isu.rms(residual_data_ARCSEC[mask_not_top][1::2]):.2f})",
            # label = f"{len(unique_observatories) - num_observatories} Other Observatories | RMS_RA: {isu.rms(residual_data_ARCSEC[mask_not_top][::2]):.3f} | RMS_DEC: {isu.rms(residual_data_ARCSEC[mask_not_top][1::2]):.3f}",
            color = isu.text_color[colormode],
            alpha = 0.2,
        )

    # plots the highlighted top num_observatories observatories
    for i, observatory in enumerate(top_observatories):
        # name_ra_dec = f"{observatory} | {observatory_names.loc[observatory].Name}"
        name_ra_dec = f"{observatory} | RMS RA (DEC): {isu.rms(residual_data_ARCSEC[concatenated_receiving_observatories == observatory][::2]):.2f} ({isu.rms(residual_data_ARCSEC[concatenated_receiving_observatories == observatory][1::2]):.2f})"
        # name_ra_dec = f"{observatory} | {observatory_names.loc[observatory].Name} | RMS_RA: {isu.rms(residual_data_ARCSEC[concatenated_receiving_observatories == observatory][::2]):.3f} | RMS_DEC: {isu.rms(residual_data_ARCSEC[concatenated_receiving_observatories == observatory][1::2]):.3f}"
        axs[0].scatter(
            residual_times_YEAR[concatenated_receiving_observatories == observatory][::2],
            residual_data_ARCSEC[concatenated_receiving_observatories == observatory][::2],
            color = isu.cmap_custom10[colormode](i%10),
            marker = isu.markers[i//10],
            s = 30,
            alpha = 0.7,
            zorder = 100,
        )
        axs[1].scatter(
            residual_times_YEAR[concatenated_receiving_observatories == observatory][1::2],
            residual_data_ARCSEC[concatenated_receiving_observatories == observatory][1::2],
            color = isu.cmap_custom10[colormode](i%10),
            marker = isu.markers[i//10],
            s = 30,
            label = name_ra_dec,
            alpha = 0.7,
            zorder = 100,
        )

    axs[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.47, -0.24),
        ncols = 3,
        markerscale = 1.2,
        fontsize = isu.tiny_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )

    for ax in fig.get_axes():
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        ax.set_ylabel("Residual [arcsec]")
        ax.axhline(linewidth=1, color=isu.text_color[colormode])
        # ax.set_ylim((-5,5))

    axs[0].set_title(
        f"{title_prefix} Residuals: Right Ascension — {body_longname}",
        fontsize = isu.medium_font_size,
    )
    axs[1].set_title(
        f"{title_prefix} Residuals: Declination — {body_longname}",
        fontsize = isu.medium_font_size,
    )
    axs[1].set_xlabel("Year")
    # fig.suptitle(
    #     f" {body_longname}",
    #     weight = 'bold',
    #     fontsize = isu.medium_font_size,
    # )

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"{title_prefix.lower().replace(' ', '-')}-residuals-by-observatory_n{num_observatories}.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"{title_prefix} residuals by observatory plot {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    fig_name = f"{title_prefix.lower().replace(' ', '-')}-residuals-by-observatory_n{num_observatories}"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B3.2 W/o gaps
def plot_residuals_by_observatories_without_time_gaps(
    obs_ranges,
    batch_observatories_table,
    obs_col_concatenated_link_definition_ids,
    obs_col_link_definition_ids,
    obs_col_concatenated_times,
    residual_data_ARCSEC,
    title_prefix: str,
    num_observatories: int = 25,
    colormode: int = 0,
    plot_behaviour: str = 'show',
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    # Get observation gap ranges
    n_obs_ranges = len(obs_ranges)
    obs_range_sizes_YEAR = [gap[1] - gap[0] for gap in obs_ranges]
    sum_obs_range_sizes_YEAR = sum(obs_range_sizes_YEAR)

    # Add a buffer to the observation ranges for nicer plotting
    plot_buffer_proportion = 0.25
    plot_buffer_total_YEAR = (plot_buffer_proportion * sum_obs_range_sizes_YEAR) / (1 - plot_buffer_proportion)
    plot_buffer_YEAR = plot_buffer_total_YEAR / (n_obs_ranges * 2)
    plot_obs_ranges = np.array([
        [gap[0] - plot_buffer_YEAR, gap[1] + plot_buffer_YEAR]
        for gap in obs_ranges
    ])
    plot_obs_range_sizes_YEAR = [gap[1] - gap[0] for gap in plot_obs_ranges]

    # Find the range that is the most central in the plot (for plot titles, etc.)
    plot_obs_range_midpoint = sum(plot_obs_range_sizes_YEAR) / 2
    current_position = 0
    central_idx = -1
    while current_position < plot_obs_range_midpoint:
        central_idx += 1
        current_position += plot_obs_range_sizes_YEAR[central_idx]

    # Convert x axis times from J2000 epoch to datetime
    J2000_datetime = dttm.datetime(2000, 1, 1, 12, 0, 0)
    residual_times_DTTM = np.array([
        J2000_datetime + dttm.timedelta(seconds=t)
        for t in obs_col_concatenated_times
    ])
    plot_obs_ranges_J2000 = (plot_obs_ranges - 2000) * cnst.JULIAN_YEAR

    observatory_names, top_observatories, \
        unique_observatories, concatenated_receiving_observatories, \
            mask_not_top, n_obs_not_top = get_observatory_data_for_plot(
            batch_observatories_table = batch_observatories_table,
            obs_col_concatenated_link_definition_ids = obs_col_concatenated_link_definition_ids,
            obs_col_link_definition_ids = obs_col_link_definition_ids,
            num_observatories = num_observatories,
    )

    ## Plot
    fig, axs = plt.subplots(
        2, n_obs_ranges,
        figsize = (14, 11),
        width_ratios = plot_obs_range_sizes_YEAR,
        facecolor = isu.fig_background[colormode],
        # layout = "constrained",
    )
    fig.subplots_adjust(wspace=0.1)  # adjust space between Axes
    fig.subplots_adjust(hspace=0.3)  # adjust space between Axes

    for obs_range_idx, obs_range in enumerate(obs_ranges):
        # Make a mask to plot only the observations within the current obs_range
        mask_after_window_start = np.where(obs_col_concatenated_times > plot_obs_ranges_J2000[obs_range_idx][0], True, False)
        mask_before_window_end = np.where(obs_col_concatenated_times < plot_obs_ranges_J2000[obs_range_idx][1], True, False)
        mask_within_obs_range = np.logical_and(mask_after_window_start, mask_before_window_end)

        # Plot remaining observatories first
        if n_obs_not_top > 0:
            # Make a mask to only plot observations from non-top observatories
            # which are within the current obs_range
            mask_to_plot = np.logical_and(mask_within_obs_range, mask_not_top)

            # RA
            axs[0, obs_range_idx].scatter(
                residual_times_DTTM[mask_to_plot][::2],
                residual_data_ARCSEC[mask_to_plot][::2],
                marker = ".",
                s = 30,
                color = isu.text_color[colormode],
                alpha = 0.2,
            )
            # DEC
            axs[1, obs_range_idx].scatter(
                residual_times_DTTM[mask_to_plot][1::2],
                residual_data_ARCSEC[mask_to_plot][1::2],
                marker = ".",
                s = 30,
                label = f"{len(unique_observatories) - num_observatories} Other Observatories | RMS_RA: {isu.rms(residual_data_ARCSEC[mask_not_top][::2]):.3f} | RMS_DEC: {isu.rms(residual_data_ARCSEC[mask_not_top][1::2]):.3f}",
                color = isu.text_color[colormode],
                alpha = 0.2,
            )

        # plot the highlighted top num_observatories observatories
        for i, observatory in enumerate(top_observatories):
            name_ra_dec = f"{observatory} | {observatory_names.loc[observatory].Name} | RMS_RA: {isu.rms(residual_data_ARCSEC[concatenated_receiving_observatories == observatory][::2]):.3f} | RMS_DEC: {isu.rms(residual_data_ARCSEC[concatenated_receiving_observatories == observatory][1::2]):.3f}"

            # Make a mask to only plot observations from the current observatory
            # AND within the current obs_range
            mask_correct_observatory = np.where(concatenated_receiving_observatories == observatory, True, False)
            mask_to_plot = np.logical_and(mask_within_obs_range, mask_correct_observatory)

            axs[0, obs_range_idx].scatter(
                residual_times_DTTM[mask_to_plot][::2],
                residual_data_ARCSEC[mask_to_plot][::2],
                color = isu.cmap_custom10[colormode](i%10),
                marker = isu.markers[i//10],
                s = 30,
                alpha = 0.7,
                zorder = 100,
            )
            axs[1, obs_range_idx].scatter(
                residual_times_DTTM[mask_to_plot][1::2],
                residual_data_ARCSEC[mask_to_plot][1::2],
                color = isu.cmap_custom10[colormode](i%10),
                marker = isu.markers[i//10],
                s = 30,
                label = name_ra_dec,
                alpha = 0.7,
                zorder = 100,
            )

    # This is for the lines that separate the broken axes
    kwargs = dict(
        marker = [(0, -1), (0, 1)],
        markersize = 12,
        linestyle = "none",
        color = isu.text_color[colormode],
        mec = isu.text_color[colormode],
        mew = 1,
        clip_on = False,
    )

    # Settings for the subplot axes
    for sub_fig_idx, sub_fig in enumerate(axs):
        for ax_idx, ax in enumerate(sub_fig):
            # Default axis settings
            ax = isu.set_default_2d_ax_settings(ax, colormode)
            # Horizontal zero line
            ax.axhline(linewidth=1, color=isu.text_color[colormode])
            # Make all RA axes share a common y axis, ditto with DEC
            ax.sharey(axs[sub_fig_idx, 0])
            # Set x limits according to the periods where observations exist
            ax.set_xlim(
                J2000_datetime + dttm.timedelta(seconds=cnst.JULIAN_YEAR*(plot_obs_ranges[ax_idx][0]-2000)),
                J2000_datetime + dttm.timedelta(seconds=cnst.JULIAN_YEAR*(plot_obs_ranges[ax_idx][1]-2000)),
            )
            # Remove unwanted ticks and spines. Set y labels. Add vertical lines
            # where axes are broken
            if ax_idx == 0:
                # Left-most obs. range
                ax.spines.right.set_visible(False)
                ax.set_ylabel("Residual [arcsec]")
                ax.plot(
                    [1, 1], # x coordinates
                    [0, 1], # y coordinates
                    transform=ax.transAxes,
                    **kwargs
                )
            elif ax_idx == n_obs_ranges-1:
                # Right-most obs. range
                ax.spines.left.set_visible(False)
                ax.tick_params(left=False, labelleft=False)
                ax.plot(
                    [0, 0], # x coordinates
                    [0, 1], # y coordinates
                    transform=ax.transAxes,
                    **kwargs
                )
            else:
                # Middle obs. ranges
                ax.spines.left.set_visible(False)
                ax.spines.right.set_visible(False)
                ax.tick_params(left=False, labelleft=False)
                ax.plot(
                    [0, 0, 1, 1], # x coordinates
                    [0, 1, 0, 1], # y coordinates
                    transform=ax.transAxes,
                    **kwargs
                )
            # Ticks, tick labels, and grid
            ax.tick_params(
                axis = 'both',
                which = 'both',
                labelsize = isu.small_font_size,
                # labelcolor = text_color[colormode],
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
            if sum_obs_range_sizes_YEAR < 5.0:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_minor_locator(mdates.DayLocator((8,15,22,29)))
            else:
                ax.xaxis.set_major_locator(mdates.MonthLocator([1,4,7,10]))
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.grid(visible=True, which='major', color=isu.text_color[colormode], alpha=0.5, linestyle='-')
            ax.grid(visible=True, which='minor', color=isu.text_color[colormode], alpha=0.2, linestyle='--')
            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=90, horizontalalignment='right')


    axs[1, 0].legend(
        loc = "upper left",
        bbox_to_anchor = (-0.1, -0.2),
        ncols = 2,
        markerscale = 1.2,
        fontsize = isu.tiny_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    axs[0, central_idx].set_title(
        "Right Ascension",
        fontsize = isu.medium_font_size,
    )
    axs[1, central_idx].set_title(
        "Declination",
        fontsize = isu.medium_font_size,
    )
    # axs[1, central_idx].set_xlabel("Year")
    fig.suptitle(
        f"{title_prefix} Residuals: {body_longname}",
        weight = 'bold',
        fontsize = isu.medium_font_size,
    )
    # plt.tight_layout(rect=[0, 0, 1.0, 1.0])

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"{title_prefix.lower().replace(' ', '-')}-residuals-by-observatory-without-time-gaps_n{num_observatories}.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf", bbox_inches='tight')
    #     plt.close()
    #     # print(f"{title_prefix} residuals by observatory without time gaps plot {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    fig_name = f"{title_prefix.lower().replace(' ', '-')}-residuals-by-observatory-without-time-gaps_n{num_observatories}"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B4.Corr. Matrix
def plot_correlation_matrix(
    correlations,
    colormode: int = 0,
    plot_behaviour: str = 'show',
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    estimated_param_names = [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$A_2$"]

    # Plot
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.75])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )
    im = ax.imshow(
        correlations,
        cmap = colormaps.coolwarm,
        vmin = -1,
        vmax = 1,
    )

    # add numbers to each of the boxes
    for i in range(len(estimated_param_names)):
        for j in range(len(estimated_param_names)):
            text = ax.text(
                j, i,
                f"{correlations[i, j]:.2f}",
                ha = "center",
                va = "center",
                color = "k",
                fontsize = isu.tiny_font_size,
            )
    # colorbar
    cb = plt.colorbar(im)
    cb.outline.set_edgecolor(isu.text_color[colormode])
    cb.ax.tick_params(
        color = isu.text_color[colormode],
        labelsize = isu.tiny_font_size,
        labelcolor = isu.text_color[colormode],
    )

    ax = isu.set_default_2d_ax_settings(ax, colormode)
    ax.set_xticks(
        np.arange(len(estimated_param_names)),
        labels = estimated_param_names,
        fontsize = isu.medium_font_size,
    )
    ax.set_yticks(
        np.arange(len(estimated_param_names)),
        labels = estimated_param_names,
        fontsize = isu.medium_font_size,
    )
    ax.tick_params(left=False, bottom=False)
    ax.grid()
    ax.set_xlabel("Estimated parameter")
    ax.set_ylabel("Estimated parameter")
    fig.suptitle(
        f"Correlations: {body_longname}",
        # weight = 'bold',
        fontsize = isu.medium_font_size,
    )

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"correlation-matrix.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Correlation matrix plot {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    fig_name = f"correlation-matrix"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B5.Hist. by obs.
def get_squarest_dimensions(n):
    nrows = int(np.floor(np.sqrt(n)))
    ncols = int(np.ceil(np.sqrt(n)))
    if n > nrows*ncols:
        nrows += 1
    return nrows, ncols

def plot_histograms_per_observatory(
    batch_observatories_table,
    final_residuals_ARCSEC,
    concatenated_receiving_observatories,
    num_observatories: int = 12,
    colormode: int = 0,
    plot_behaviour: str = 'show',
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):
    nbins = 20
    opacity = 0.6
    number_of_rows, number_of_columns = get_squarest_dimensions(num_observatories)

    # Retrieve the observatory names again
    observatory_names_hist = (
        batch_observatories_table
        .set_index("Code")
        .sort_values("count", ascending=False)
        .iloc[0:num_observatories]
    )
    top_observatories_hist = observatory_names_hist.index.tolist()

    # Plot
    fig, axs = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize = (3.5 * number_of_columns, 3 * number_of_rows),
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )
    for ax in axs.flatten():
        ax = isu.set_default_2d_ax_settings(ax, colormode)

    # Add the labels to edge x- and y-axes
    for col in range(number_of_columns):
        axs[int(number_of_rows - 1), col].set_xlabel(
            "Residual [arcsec]",
            fontsize = isu.small_font_size,
        )
    for row in range(number_of_rows):
        axs[row, 0].set_ylabel(
            "No. observations",
            fontsize = isu.small_font_size,
        )
    axs = axs.flatten()

    for idx, observatory in enumerate(top_observatories_hist):
        name = f"{observatory_names_hist.loc[observatory].Name}\nCode: {observatory} | No. observations: {int(observatory_names_hist.loc[observatory]['count'])}"
        current_observatory_RA_ARCSEC = final_residuals_ARCSEC[concatenated_receiving_observatories == observatory][0::2]
        axs[idx].hist(
            current_observatory_RA_ARCSEC,
            bins = nbins,
            alpha = opacity + 0.05,
            label = "Right ascension",
            color = isu.cmap_custom10[colormode](0)
        )
        current_observatory_DEC_ARCSEC = final_residuals_ARCSEC[concatenated_receiving_observatories == observatory][1::2]
        axs[idx].hist(
            current_observatory_DEC_ARCSEC,
            bins = nbins,
            alpha = opacity,
            label = "Declination",
            color = isu.cmap_custom10[colormode](1)
        )
        axs[idx].axvline(linewidth=1, color=isu.text_color[colormode])
        axs[idx].set_title(name, fontsize=isu.small_font_size)
        if len(current_observatory_RA_ARCSEC) > 0:
            # Make xlims centred on zero
            xlim = np.max([
                np.max(np.abs(current_observatory_RA_ARCSEC)),
                np.max(np.abs(current_observatory_DEC_ARCSEC)),
            ])
            axs[idx].set_xlim(-xlim*1.05, xlim*1.05)

    axs[0].legend(
        loc="best",
        markerscale = 1.2,
        fontsize = isu.tiny_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    fig.suptitle(
        f"Final Iteration Residual Histograms\nof the {num_observatories} Highest-Contributing Observatories: {body_longname}",
        weight = 'bold',
        fontsize = isu.big_font_size,
    )
    plt.tight_layout(rect=[0.01,0,0.97,1])

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"residual-histograms-by-observatory_n{num_observatories}.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Residual histograms by observatory plot {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    fig_name = f"residual-histograms-by-observatory_n{num_observatories}"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B6a.Debias
def plot_star_catalog_corrections(
    batch_table: BatchMPC,
    include_satellites: bool = True,
    colormode: int = 0,
    plot_behaviour: str = 'show',
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):

    if include_satellites:
        epochs = batch_table["epoch_seconds_UTC"].values
        epochs = (epochs / cnst.JULIAN_YEAR) + 2000
        ra_corrections = batch_table["corr_RA_EFCC18"].values
        dec_corrections = batch_table["corr_DEC_EFCC18"].values
    else:
        epochs = batch_table[batch_table["note2"] != "S"]["epoch_seconds_UTC"].values
        epochs = (epochs / cnst.JULIAN_YEAR) + 2000
        ra_corrections = batch_table[batch_table["note2"] != "S"][
            "corr_RA_EFCC18"
        ].values
        dec_corrections = batch_table[batch_table["note2"] != "S"][
            "corr_DEC_EFCC18"
        ].values

    fig = plt.figure(
        figsize = np.array([1.0,0.67])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )
    gs = gridspec.GridSpec(
        1, 2,
        figure = fig,
        width_ratios = [4, 1],
        # height_ratios = [1, 1],
        hspace = 0.07,
        wspace = 0.03
    )

    # Set up scatter plots
    ax1 = fig.add_subplot(gs[0, 0])
    # Set up histograms on the right
    hist1 = fig.add_subplot(gs[0, 1], sharey=ax1)

    # Populate scatter plots
    ax1.scatter(
        epochs,
        np.rad2deg(ra_corrections) * 3600,
        label = "RA",
        color = isu.cmap_custom10[colormode](0),
        marker = "+",
        alpha = 0.3,
    )
    ax1.scatter(
        epochs,
        np.rad2deg(dec_corrections) * 3600,
        label = "DEC",
        color = isu.cmap_custom10[colormode](1),
        marker = "+",
        alpha = 0.3,
    )
    # Populate histograms
    hist1.hist(
        np.rad2deg(ra_corrections) * 3600,
        bins = 30,
        orientation = "horizontal",
        color = isu.cmap_custom10[colormode](0),
        alpha = 0.6,
    )
    hist1.hist(
        np.rad2deg(dec_corrections) * 3600,
        bins = 30,
        orientation = "horizontal",
        color = isu.cmap_custom10[colormode](1),
        alpha = 0.6,
    )

    # Legend
    leg = ax1.legend(
        loc = 'best',
        markerscale = 1.2,
        fontsize = isu.small_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1) # makes the markers in legend full opacity

    # Axis settings, labels, title
    for ax in [ax1, hist1]:#[ax1, ax2, hist1, hist2]:
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        ax.axhline(linewidth=1, color=isu.text_color[colormode])
    ax1.set_ylabel("Correction [arcsec]")#, fontsize=isu.small_font_size)
    ax1.set_xlabel("Year")#, fontsize=isu.small_font_size)
    hist1.set_xlabel("Count")#, fontsize=isu.small_font_size)
    hist1.tick_params(labelleft=False)
    fig.suptitle(
        f"Star Catalog Corrections: {body_longname}",
        # weight = 'bold',
        fontsize = isu.medium_font_size,
    )

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"star-catalog-corrections.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Star catalog corrections plot {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    fig_name = f"star-catalog-corrections"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B6b.Weight
def plot_observation_weights(
    batch_table: BatchMPC,
    include_satellites: bool = True,
    colormode: int = 0,
    plot_behaviour: str = 'show',
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):
    sat_epochs = batch_table[batch_table["note2"] == "S"]["epoch_seconds_UTC"].values
    sat_epochs = (sat_epochs / cnst.JULIAN_YEAR) + 2000
    sat_weights = batch_table[batch_table["note2"] == "S"]["weight"].values
    reg_epochs = batch_table[batch_table["note2"] != "S"]["epoch_seconds_UTC"].values
    reg_epochs = (reg_epochs / cnst.JULIAN_YEAR) + 2000
    reg_weights = batch_table[batch_table["note2"] != "S"]["weight"].values

    # Set up figure and GridSpec for scatter + histogram
    fig = plt.figure(
        figsize = np.array([1.0,0.67])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = "constrained",
    )
    gs = gridspec.GridSpec(
        1, 2,
        figure = fig,
        width_ratios = [4, 1],
        wspace = 0.03,
    )

    # Set up plots
    ax = fig.add_subplot(gs[0])
    hist_ax = fig.add_subplot(gs[1], sharey=ax)

    # Populate plots
    ax.scatter(
        reg_epochs,
        reg_weights,
        # np.rad2deg( np.sqrt(np.reciprocal(reg_weights)) )*3600, # to show uncertainty in arcseconds
        label = "Ground observatory",
        color = isu.cmap_custom10[colormode](0),
        marker = "+",
        alpha = 0.3,
    )
    if include_satellites:
        ax.scatter(
            sat_epochs,
            sat_weights,
            # np.rad2deg( np.sqrt(np.reciprocal(sat_epochs)) )*3600,
            label = "Satellite",
            color = isu.cmap_custom10[colormode](3),
            marker = "+",
            alpha = 0.3,
        )
    if include_satellites:
        hist_range = (
            min(reg_weights.min(), sat_weights.min()),
            max(reg_weights.max(), sat_weights.max()),
        )
    else:
        hist_range = (reg_weights.min(), reg_weights.max())
        print(hist_range)

    logbins = np.logspace(np.log10(hist_range[0]), np.log10(hist_range[1]), 30)

    hist_ax.hist(
        reg_weights,
        # np.rad2deg( np.sqrt(np.reciprocal(reg_weights)) )*3600,
        range = hist_range,
        bins = logbins,
        orientation = "horizontal",
        color = isu.cmap_custom10[colormode](0),
        alpha = 0.8,
        # log = False,
    )
    if include_satellites:
        hist_ax.hist(
            sat_weights,
            # np.rad2deg( np.sqrt(np.reciprocal(sat_epochs)) )*3600,
            range = hist_range,
            bins = logbins,
            orientation = "horizontal",
            color = isu.cmap_custom10[colormode](3),
            alpha = 0.8,
            # log = False,
        )

    for axis in [ax, hist_ax]:
        axis = isu.set_default_2d_ax_settings(axis, colormode)
    ax.set_ylabel(r"Weight $[rad^{-2}]$")#, fontsize=isu.small_font_size)
    ax.set_xlabel("Year")#, fontsize=isu.small_font_size)
    hist_ax.set_xlabel("Count")#, fontsize=isu.small_font_size)
    hist_ax.tick_params(labelleft=False)
    ax.set_yscale('log')
    # hist_ax.set_xscale('log')

    if include_satellites:
        leg = ax.legend(
            loc = 'best',
            markerscale = 1.2,
            fontsize = isu.small_font_size,
            labelcolor = isu.text_color[colormode],
            facecolor = isu.legend_bg_color[colormode],
        )
        for lh in leg.legend_handles:
            lh.set_alpha(1) # makes the markers in legend full opacity

    fig.suptitle(
        f"Observation Weights per RA/DEC Pair:\n{body_longname}",
        # weight = 'bold',
        fontsize = isu.medium_font_size,
    )
    # fig.tight_layout()
    # plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"observation-weights.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    #     plt.close()
    #     # print(f"Observation weights plot {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

    # Plot output
    fig_name = f"observation-weights"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#region B7.
def plot_distance_from_bodies(
    output_times,
    position_of_target_relative_to_bodies,
    relative_position_bodies,
    obs_ranges,
    colormode: int = 0,
    plot_behaviour: str = "show",
    output_format: str = "pdf",
    plots_save_path: str = None,
    body_longname: str = None,
):
    # Plot distance between Earth and target over time
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.67])*isu.twothirds_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = 'constrained',
    )
    ax = isu.set_default_2d_ax_settings(ax, colormode)

    # Plot
    for rel_pos_bod in position_of_target_relative_to_bodies:
        target_body_distance_AU = np.linalg.norm(
            position_of_target_relative_to_bodies[rel_pos_bod], axis=1
        ) / cnst.ASTRONOMICAL_UNIT
        min_target_body_distance_AU = min(target_body_distance_AU)
        print(f"Initial distance to {rel_pos_bod}: {target_body_distance_AU[0]:.6f} AU")
        print(f"Minimum distance to {rel_pos_bod}: {min_target_body_distance_AU:.6f} AU\n")

        # only include body in plot if minimum distance is <0.1 AU
        if min_target_body_distance_AU > 0.1:
            continue
        if rel_pos_bod == "Moon":
            continue

        ax.plot(
            (output_times / cnst.JULIAN_YEAR) + 2000,
            target_body_distance_AU,
            color = isu.cmap_custom10[colormode](relative_position_bodies[rel_pos_bod][1]),
            alpha = 0.8,
            linestyle = relative_position_bodies[rel_pos_bod][2],
            label = rel_pos_bod,
        )

        # Make the horizontal zero-line more obvious
        # ax.axhline(linewidth=1, color=isu.text_color[colormode])

        # Plot line at min. distance
        ax.axhline(
            min_target_body_distance_AU,
            linewidth = 1,
            color = isu.cmap_custom10[colormode](relative_position_bodies[rel_pos_bod][1]),
            alpha = 0.8,
            linestyle = 'dotted',
            label = f"Min.={min_target_body_distance_AU:.4f} AU"
        )

    # Observation ranges
    for gap_idx, gap in enumerate(obs_ranges):
        ax.axvspan(
            xmin = gap[0],
            xmax = gap[1],
            color = isu.cmap_custom10[colormode](9),
            alpha = 0.4,
            label = "Obs." if (gap_idx==0) else None,
        )

    # Legend, labels, title
    ax.margins(x=0.01)
    ax.set_yscale('log')
    fig.legend(
        loc = 'outside center right',
        markerscale = 1.2,
        fontsize = isu.small_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    ax.set_ylabel(f"Distance [AU]")
    ax.set_xlabel("Year")
    ax.set_title(f"Distance from {body_longname}", fontsize=isu.medium_font_size)

    # # Plot output
    # if (plot_behaviour == 'show'):
    #     plt.show(block=False)
    #     plt.pause(0.001)
    # elif (plot_behaviour == 'save'):
    #     save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     fig_name = f"distance-from-bodies.pdf"
    #     plt.savefig(save_path + fig_name, format="pdf")
    #     plt.close()

    # Plot output
    fig_name = f"distance-from-bodies"
    output_plot(
        fig_name = fig_name,
        plots_save_path = plots_save_path,
        pb = plot_behaviour,
        colormode = colormode,
        format = output_format,
    )

#%%%%%%################   C. Generator Function   ##############################
#######################   C. Generator Function   ##############################
#region C. Generator Function

# @click.command()
# @click.option(
#     "--est_w_mpc_result_path",
#     "-p",
#     # prompt = "Path to output data directory",
#     help = "The (path to the) directory containing the estimation output data from which to generate plots",
# )
# e.g. for results from a single run:
# "yarkovsky-est-w-mpc/_individual-runs/488803-2005-GB120/260422-012953/"
# e.g. for results of one body from a large run:
# "yarkovsky-est-w-mpc/_large-runs/260422-015108/488803-2005-GB120/"

def generate_estimation_plots(
    est_w_mpc_result_path: str,
    overriding_plot_behaviour: str = None,
    output_format: str = 'pdf',
):
    ### Directories
    output_data_path = isu.output_data_dir + est_w_mpc_result_path
    plots_save_path = isu.plot_dir + est_w_mpc_result_path


    ### Read inputs
    #region C.1 Read inputs
    # Read and extract inputs from "_inputs.json"
    with open(output_data_path + "_inputs.json", 'r') as json_fp:
        input_dict = json.load(json_fp)[0]
    # Extract dict values into their own variables
    number_of_pod_iterations = input_dict["number_of_pod_iterations"]
    use_catalog_cor = input_dict["use_catalog_cor"]
    use_obs_weights = input_dict["use_obs_weights"]
    residual_filter_cutoff_ARCSEC = input_dict["residual_filter_cutoff_ARCSEC"]
    use_SC_data = input_dict["use_SC_data"]
    propagation_start_J2000 = input_dict["propagation_start_J2000"]
    propagation_end_J2000 = input_dict["propagation_end_J2000"]
    central_body = input_dict["central_body"]
    global_frame_origin = input_dict["global_frame_origin"]
    global_frame_orientation = input_dict["global_frame_orientation"]
    comparison_reference = input_dict["comparison_reference"]
    plots_in_RSW = input_dict["plots_in_RSW"]
    # Plotting behaviour
    colormode = input_dict["colormode"]
    if overriding_plot_behaviour:
        plot_behaviour = overriding_plot_behaviour
    else:
        plot_behaviour = input_dict["plot_behaviour"]
    plot_behaviour_list, colormode_list = isu.define_plot_behaviour_lists(
        plot_behaviour, colormode,
    )

    ### Load output data
    #region C.2 Load output data
    # Body dict
    body = isu.load_with_pickle(output_data_path, "body.p")
    spice.load_kernel(isu.SPK_file_dir + body['SPKID'] + ".bsp")

    # Load attributes of BatchMPC object
    batch_data_path = output_data_path + "batch/"
    batch_table = isu.load_with_pickle(batch_data_path, "table.p")
    batch_observatories_table = isu.load_with_pickle(batch_data_path, "observatories_table.p")

    batch_table_filtered = isu.load_with_pickle(output_data_path, "batch_table_filtered.p")
    batch_table_outliers = isu.load_with_pickle(output_data_path, "batch_table_outliers.p")
    final_normalised_residuals = isu.load_with_pickle(output_data_path, "final_residuals_normalised.p")

    # Load attributes of EstimationOutput object
    EO_data_path = output_data_path + "estimation_output/"
    EO_best_iteration = isu.load_with_pickle(EO_data_path, "best_iteration.p")
    EO_correlations = isu.load_with_pickle(EO_data_path, "correlations.p")
    EO_residual_history = isu.load_with_pickle(EO_data_path, "residual_history.p")
    # Load attributes of each estimation iteration
    EO_iterations = {}
    for iter_idx in range(number_of_pod_iterations):
        # Dictionary to store all attributes of the iteration
        iter_dict = {}
        # Load attributes of SingleArcVariationalSimulationResults object
        iter_data_path = EO_data_path + f'simulation_results_per_iteration_{iter_idx}/'
        # Load attributes of SingleArcSimulationResults object
        dyn_result_data_path = iter_data_path + f'dynamics_results/'
        iter_dict['state_history'] = isu.load_with_pickle(dyn_result_data_path, "state_history.p")
        iter_dict['dependent_variable_history'] = isu.load_with_pickle(dyn_result_data_path, "dependent_variable_history.p")
        # Append dictionary
        EO_iterations[iter_idx] = iter_dict

    # Formal errors
    formal_errors_XYZ_KM = isu.load_with_pickle(output_data_path, "formal_errors_XYZ_KM.p")
    formal_errors_RSW_KM = isu.load_with_pickle(output_data_path, "formal_errors_RSW_KM.p")

    ### Get ephemeris uncertainty data
    body_ephem_unc_table = isu.load_with_pickle(isu.ephem_unc_dir, f"{body['SPKID']}.p")
    body_ephem_unc_table["ephemeris_time"] = [
        DateTime.from_julian_day(jd)
        for jd in body_ephem_unc_table["datetime_jd"]
    ]
    body_ephem_unc_table["ephemeris_time_J2000"] = [
        dt.to_epoch()
        for dt in body_ephem_unc_table["ephemeris_time"]
    ]
    # Keep only the rows of the table corresponding to the target body's propagation period
    mask_after_window_start = np.where(
        body_ephem_unc_table["ephemeris_time_J2000"] > propagation_start_J2000, True, False
    )
    mask_before_window_end = np.where(
        body_ephem_unc_table["ephemeris_time_J2000"] < propagation_end_J2000, True, False
    )
    mask_within_obs_range = np.logical_and(mask_after_window_start, mask_before_window_end)
    body_ephem_unc_table = body_ephem_unc_table[mask_within_obs_range]


    ### Extract some useful variables data from output data
    EO_best_dict = EO_iterations[EO_best_iteration]
    output_times = np.array(sorted(list(EO_best_dict['state_history'].keys())))
    state_history_array = np.array(list(EO_best_dict['state_history'].values()))
    dependent_variables_array = np.array(list(EO_best_dict['dependent_variable_history'].values()))
    relative_position_bodies = isu.load_with_pickle(output_data_path, "relative_position_bodies.p")
    position_of_target_relative_to_bodies = {
        relative_position_bodies[rel_pos_bod][0]: dependent_variables_array[:, (3*idx):(3*idx + 3)]
        for idx, rel_pos_bod in enumerate(relative_position_bodies)
    }
    prefit_residuals_ARCSEC = np.rad2deg(np.array(EO_residual_history[:, 0])) * 3600
    final_residuals_ARCSEC = np.rad2deg(np.array(EO_residual_history[:, EO_best_iteration])) * 3600


    ## Create observation collection from saved batch table
    new_batch = BatchMPC()
    new_batch.from_pandas(
        # batch_table,
        batch_table_filtered,
        in_degrees = False,
        custom_name=body['SPKID'],
    )
    observation_collection = new_batch.create_observations_from_astropy_table(
        table = batch_table_filtered,
        apply_star_catalog_debias = use_catalog_cor,
        apply_weights_VFCC17 = use_obs_weights,
    )
    first_obs_J2000 = min(observation_collection.concatenated_times)
    last_obs_J2000 = max(observation_collection.concatenated_times)


    ### Call plot functions
    #region C.4 Call plot funcs
    for colormode, pb in zip(colormode_list, plot_behaviour_list):
        gap_ranges = get_gap_ranges(
            first_obs_J2000,
            last_obs_J2000,
            observation_collection.concatenated_times,
            gap_in_months = 6,
        )
        obs_ranges = get_inverted_gap_ranges(
            first_obs_J2000,
            last_obs_J2000,
            gap_ranges,
        )
        reference_states = get_reference_states(
            comparison_reference = comparison_reference,
            body = body,
            central_body = central_body,
            global_frame_orientation = global_frame_orientation,
            epoch_list = output_times,
        )
        ### P1.1 Position difference w.r.t. JPL Horizons
        #region P1.1 Pos. diff.
        # 4 subplots of error components and error magnitude, with JPL ephemeris uncertainty
        plot_position_error(
            state_history_array = state_history_array,
            reference_states = reference_states,
            output_times = output_times,
            obs_ranges = obs_ranges,
            in_RSW = plots_in_RSW,
            plot_JPL_uncertainty = True,
            ephem_unc_table = body_ephem_unc_table,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        # Single plot with error components and error magnitude together, with JPL ephemeris uncertainty band
        plot_cartesian_single(
            state_history_array = state_history_array,
            reference_states = reference_states,
            output_times = output_times,
            obs_ranges = obs_ranges,
            in_RSW = plots_in_RSW,
            plot_error_norm = True,
            plot_JPL_uncertainty = True,
            ephem_unc_table = body_ephem_unc_table,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )


        ### P1.2 Propagated formal error over time
        #region P1.2 Formal error
        formal_errors_KM = formal_errors_RSW_KM if plots_in_RSW else formal_errors_XYZ_KM
        # Just formal errors over time (1 sigma)
        plot_formal_errors(
            output_times = output_times,
            formal_errors_KM = formal_errors_KM,
            obs_ranges = obs_ranges,
            in_RSW = plots_in_RSW,
            plot_error_norm = True,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        # Subplots of position error wrt JPLH, with formal error regions added
        plot_position_error(
            state_history_array = state_history_array,
            reference_states = reference_states,
            output_times = output_times,
            obs_ranges = obs_ranges,
            in_RSW = plots_in_RSW,
            plot_formal_error = True,
            formal_errors_KM = formal_errors_KM,
            formal_errors_n_sigma = 3,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        # Subplots of position error wrt JPLH, with formal error regions AND JPL ephemeris uncertainty band
        plot_position_error(
            state_history_array = state_history_array,
            reference_states = reference_states,
            output_times = output_times,
            obs_ranges = obs_ranges,
            in_RSW = plots_in_RSW,
            plot_formal_error = True,
            plot_JPL_uncertainty = True,
            ephem_unc_table = body_ephem_unc_table,
            formal_errors_KM = formal_errors_KM,
            formal_errors_n_sigma = 3,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        # Single plot of position error wrt JPLH, with formal error regions added
        plot_cartesian_single(
            state_history_array = state_history_array,
            reference_states = reference_states,
            output_times = output_times,
            obs_ranges = obs_ranges,
            in_RSW = plots_in_RSW,
            plot_error_norm = False,
            plot_formal_error = True,
            formal_errors_KM = formal_errors_KM,
            formal_errors_n_sigma = 3,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )


        ### P2. Residuals by iteration/setup
        #region P2.Residuals
        # Residuals by iterations
        plot_residuals(
            number_of_pod_iterations = number_of_pod_iterations,
            est_out_residual_history = EO_residual_history,
            obs_col_concatenated_times = observation_collection.concatenated_times,
            best_iteration = EO_best_iteration,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        # Simulated residuals
        plot_simulated_residuals(
            batch_table_filtered = batch_table_filtered,
            residual_filter_cutoff_ARCSEC = residual_filter_cutoff_ARCSEC,
            plot_outliers = True,
            batch_table_outliers = batch_table_outliers,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        # Final residuals (arcsec)
        plot_final_residuals(
            final_residuals = final_residuals_ARCSEC,
            obs_col_concatenated_times = observation_collection.concatenated_times,
            ylabel = "Residual [arcsec]",
            fig_name = "final-residuals",
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        # Final residuals (normalised)
        plot_final_residuals(
            final_residuals = final_normalised_residuals,
            obs_col_concatenated_times = observation_collection.concatenated_times,
            ylabel = "Normalised residual",
            fig_name = "final-residuals-normalised",
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )

        ### P3.1 Pre/Post-Fit residuals highlighted per observatory
        #region P3.1 Pre/Post Fit
        num_observatories = 20 #+66 # 25 default
        plot_residuals_by_observatories(
            batch_observatories_table = batch_observatories_table,
            obs_col_concatenated_link_definition_ids = observation_collection.concatenated_link_definition_ids,
            obs_col_link_definition_ids = observation_collection.link_definition_ids,
            obs_col_concatenated_times = observation_collection.concatenated_times,
            residual_data_ARCSEC = prefit_residuals_ARCSEC,
            title_prefix = 'Pre-Fit',
            num_observatories = num_observatories,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        plot_residuals_by_observatories(
            batch_observatories_table = batch_observatories_table,
            obs_col_concatenated_link_definition_ids = observation_collection.concatenated_link_definition_ids,
            obs_col_concatenated_times = observation_collection.concatenated_times,
            obs_col_link_definition_ids = observation_collection.link_definition_ids,
            residual_data_ARCSEC = final_residuals_ARCSEC,
            title_prefix = 'Post-Fit',
            num_observatories = num_observatories,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )


        ### P3.2 Pre/Post-Fit residuals highlighted per observatory without time gaps
        #region P3.2 W/o gaps
        plot_residuals_by_observatories_without_time_gaps(
            obs_ranges = obs_ranges,
            batch_observatories_table = batch_observatories_table,
            obs_col_concatenated_link_definition_ids = observation_collection.concatenated_link_definition_ids,
            obs_col_link_definition_ids = observation_collection.link_definition_ids,
            obs_col_concatenated_times = observation_collection.concatenated_times,
            residual_data_ARCSEC = prefit_residuals_ARCSEC,
            title_prefix = 'Pre-Fit',
            num_observatories = num_observatories,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        plot_residuals_by_observatories_without_time_gaps(
            obs_ranges = obs_ranges,
            batch_observatories_table = batch_observatories_table,
            obs_col_concatenated_link_definition_ids = observation_collection.concatenated_link_definition_ids,
            obs_col_link_definition_ids = observation_collection.link_definition_ids,
            obs_col_concatenated_times = observation_collection.concatenated_times,
            residual_data_ARCSEC = final_residuals_ARCSEC,
            title_prefix = 'Post-Fit',
            num_observatories = num_observatories,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )


        ### P4. Correlation Matrix
        #region P4.Corr. Matrix
        plot_correlation_matrix(
            correlations = EO_correlations,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )


        ### P5. Histograms per observatory
        #region P5.Hist. by obs.
        _, _, _, concatenated_receiving_observatories, _, _ = \
            get_observatory_data_for_plot(
                batch_observatories_table = batch_observatories_table,
                obs_col_concatenated_link_definition_ids = observation_collection.concatenated_link_definition_ids,
                obs_col_link_definition_ids = observation_collection.link_definition_ids,
                num_observatories = num_observatories,
        )
        plot_histograms_per_observatory(
            batch_observatories_table = batch_observatories_table,
            final_residuals_ARCSEC = final_residuals_ARCSEC,
            concatenated_receiving_observatories = concatenated_receiving_observatories,
            num_observatories = 12,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )


        ### P6. Star catalogue corrections & weights
        #region P6.Debi.+weigh
        plot_star_catalog_corrections(
            batch_table = batch_table_filtered,
            include_satellites = use_SC_data,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )
        plot_observation_weights(
            batch_table = batch_table_filtered,
            include_satellites = use_SC_data,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )


        ### P7. Distance from Selected Bodies
        #region P7. Dist. Bodies
        plot_distance_from_bodies(
            output_times = output_times,
            position_of_target_relative_to_bodies = position_of_target_relative_to_bodies,
            relative_position_bodies = relative_position_bodies,
            obs_ranges = obs_ranges,
            colormode = colormode,
            plot_behaviour = pb,
            output_format = output_format,
            plots_save_path = plots_save_path,
            body_longname = body['longname'],
        )

    if 'save' in plot_behaviour_list:
        print(f"\nAll plots for {body['longname']} saved to: \n{plots_save_path}")

#%%%%%%##################   D. Run as Script   #################################
#########################   D. Run as Script   #################################
#region D. Run as Script

if __name__ == "__main__":
    # ENTER MANUALLY
    # Directories that specify the estimation run to be plotted
    #
    # Draft results folders:
    # timestamp_dir = '260517-045652_0-3/'
    # timestamp_dir = '260517-051531_4/'
    # timestamp_dir = '260517-053036_5-154/'
    # timestamp_dir = '260517-213027_155-347/'
    #
    # Final results folders:
    timestamp_dir = '260611-172545_0-4/'
    timestamp_dir = '260611-185119_5-52/'
    # timestamp_dir = '260612-005119_53-158/'
    # timestamp_dir = '260612-214607_159-171/'
    # timestamp_dir = '260613-020313_172-300/'
    # timestamp_dir = '260614-032235_301-347/'
    # timestamp_dir = '260614-132323_161_223/'
    # timestamp_dir = ''260616-182532_10_57/'

    body_dir = '2011-EP51/'
    overriding_plot_behaviour = 'save'
    # overriding_plot_behaviour = 'show'
    # overriding_plot_behaviour = None

    # Individual run:
    # est_w_mpc_plot_dir = 'yarkovsky-est-w-mpc/_individual-runs/'
    # est_w_mpc_result_path = est_w_mpc_plot_dir + body_dir + timestamp_dir

    # Large run:
    est_w_mpc_plot_dir = 'yarkovsky-est-w-mpc/_large-runs/'
    est_w_mpc_result_path = est_w_mpc_plot_dir + timestamp_dir + body_dir
    print(est_w_mpc_result_path)

    generate_estimation_plots(
        est_w_mpc_result_path = est_w_mpc_result_path,
        overriding_plot_behaviour = overriding_plot_behaviour,
        # output_format = 'pdf',
        output_format = 'png',
    )

    # print(f"__name__:               {__name__}")
    # print(f"length of sys.argv:     {len(sys.argv)}")
    # generate_estimation_plots()


""""
Bodies for which plots have been saved for final results:
- 69230-Hermes      260612-005119_53-158    ∆arc = -62.9y due to filtering
- 138175-2000-EE104 260611-185119_5-52      ∆arc =  -0.92y due to filtering
- 66400-1999-LT7    260612-005119_53-158    ∆arc = -12.1y due to filtering
- 1566-Icarus       260611-185119_5-52      16% of observations filtered out (all in ~1968)
- 1994-GL           260612-005119_53-158    13.3% of observations filtered out (mostly near beginning of arc)
- 2010-VK139        260612-005119_53-158    21.1% of observations filtered out (mostly near beginning of arc)
- 2002-QQ40         260613-020313_172-300   initial position change > 1000 km
- 2022-QX4          260612-005119_53-158    JPLH ephemeris uncertainty unusually high
- 612050 (1997 GL3) 260612-214607_159-171   RMS FE/TE_pos. very close to 1
- 2008-LG2          260612-005119_53-158    High RMS FE/TE_pos. (~11.4)
- 2011-EP51         260611-185119_5-52      Example for nice low residuals
- 2020-PP1          260613-020313_172-300   Example where pre- and post-fit residuals are quite different
- 2011-CE50         260614-032235_301-347   Example of relatively high RMSNR
-
"""


#%% Extra code
"""
FULL VERSION OF LOADING ATTRIBUTES OF BatchMPC OBJECT
    # Load attributes of BatchMPC object
    batch_data_path = output_data_path + "batch/"
    # batch_MPC_space_telescopes = isu.load_with_pickle(batch_data_path, "_MPC_space_telescopes.p")
    # batch_size = isu.load_with_pickle(batch_data_path, "size.p")
    batch_epoch_end = isu.load_with_pickle(batch_data_path, "epoch_end.p")
    batch_epoch_start = isu.load_with_pickle(batch_data_path, "epoch_start.p")
    # batch_MPC_objects = isu.load_with_pickle(batch_data_path, "MPC_objects.p")
    # batch_bands = isu.load_with_pickle(batch_data_path, "bands.p")
    # batch_observatories = isu.load_with_pickle(batch_data_path, "observatories.p")
    # batch_space_telescopes = isu.load_with_pickle(batch_data_path, "space_telescopes.p")
    # batch_bodies_created = isu.load_with_pickle(batch_data_path, "bodies_created.p")
    batch_table = isu.load_with_pickle(batch_data_path, "table.p")
    batch_observatories_table = isu.load_with_pickle(batch_data_path, "observatories_table.p")

FULL VERSION OF LOADING ATTRIBUTES OF EstimationOutput OBJECT
    # Load attributes of EstimationOutput object
    EO_data_path = output_data_path + "estimation_output/"
    EO_best_iteration = isu.load_with_pickle(EO_data_path, "best_iteration.p")
    # EO_consider_covariance = isu.load_with_pickle(EO_data_path, "consider_covariance.p")
    # EO_consider_covariance_contribution = isu.load_with_pickle(EO_data_path, "consider_covariance_contribution.p")
    # EO_consider_normalization_terms = isu.load_with_pickle(EO_data_path, "consider_normalization_terms.p")
    EO_correlations = isu.load_with_pickle(EO_data_path, "correlations.p")
    # EO_covariance = isu.load_with_pickle(EO_data_path, "covariance.p")
    # EO_covariance_with_consider_parameters = isu.load_with_pickle(EO_data_path, "covariance_with_consider_parameters.p")
    # # EO_design_matrix = isu.load_with_pickle(EO_data_path, "design_matrix.p")
    # EO_design_matrix_consider_parameters = isu.load_with_pickle(EO_data_path, "design_matrix_consider_parameters.p")
    # EO_exception_during_inversion = isu.load_with_pickle(EO_data_path, "exception_during_inversion.p")
    # EO_exception_during_propagation = isu.load_with_pickle(EO_data_path, "exception_during_propagation.p")
    # EO_final_parameters = isu.load_with_pickle(EO_data_path, "final_parameters.p")
    # EO_final_residuals = isu.load_with_pickle(EO_data_path, "final_residuals.p")
    # EO_formal_errors = isu.load_with_pickle(EO_data_path, "formal_errors.p")
    # EO_inverse_covariance = isu.load_with_pickle(EO_data_path, "inverse_covariance.p")
    # EO_inverse_normalized_covariance = isu.load_with_pickle(EO_data_path, "inverse_normalized_covariance.p")
    # EO_normalization_terms = isu.load_with_pickle(EO_data_path, "normalization_terms.p")
    # EO_normalized_covariance = isu.load_with_pickle(EO_data_path, "normalized_covariance.p")
    # EO_normalized_covariance_with_consider_parameters = isu.load_with_pickle(EO_data_path, "normalized_covariance_with_consider_parameters.p")
    # # EO_normalized_design_matrix = isu.load_with_pickle(EO_data_path, "normalized_design_matrix.p")
    # EO_normalized_design_matrix_consider_parameters = isu.load_with_pickle(EO_data_path, "normalized_design_matrix_consider_parameters.p")
    # EO_parameter_history = isu.load_with_pickle(EO_data_path, "parameter_history.p")
    EO_residual_history = isu.load_with_pickle(EO_data_path, "residual_history.p")
    # # EO_weighted_design_matrix = isu.load_with_pickle(EO_data_path, "weighted_design_matrix.p")
    # # EO_weighted_normalized_design_matrix = isu.load_with_pickle(output_data_path, "weighted_normalized_design_matrix.p")
    # Load attributes of each estimation iteration
    EO_iterations = {}
    for iter_idx in range(number_of_pod_iterations):
        # Dictionary to store all attributes of the iteration
        iter_dict = {}

        # Load attributes of SingleArcVariationalSimulationResults object
        iter_data_path = EO_data_path + f'simulation_results_per_iteration_{iter_idx}/'
        # # EOVSR_sensitivity_matrix_history = isu.load_with_pickle(iter_data_path, "sensitivity_matrix_history.p")#
        # # EOVSR_state_transition_matrix_history = isu.load_with_pickle(iter_data_path, "state_transition_matrix_history.p")#

        # Load attributes of SingleArcSimulationResults object
        dyn_result_data_path = iter_data_path + f'dynamics_results/'
        # # iter_dict['cumulative_computation_time_history'] = isu.load_with_pickle(dyn_result_data_path, "cumulative_computation_time_history.p")#
        # # iter_dict['cumulative_number_of_function_evaluations_history'] = isu.load_with_pickle(dyn_result_data_path, "cumulative_number_of_function_evaluations_history.p")#
        # iter_dict['dependent_variable_history'] = isu.load_with_pickle(dyn_result_data_path, "dependent_variable_history.p")
        # iter_dict['dependent_variable_history_time_object'] = isu.load_with_pickle(dyn_result_data_path, "dependent_variable_history_time_object.p")
        # iter_dict['dependent_variable_ids'] = isu.load_with_pickle(dyn_result_data_path, "dependent_variable_ids.p")
        # iter_dict['initial_and_final_times'] = isu.load_with_pickle(dyn_result_data_path, "initial_and_final_times.p")
        # iter_dict['integration_completed_successfully'] = isu.load_with_pickle(dyn_result_data_path, "integration_completed_successfully.p")
        # iter_dict['ordered_dependent_variable_settings'] = isu.load_with_pickle(dyn_result_data_path, "ordered_dependent_variable_settings.p")
        # iter_dict['processed_state_ids'] = isu.load_with_pickle(dyn_result_data_path, "processed_state_ids.p")
        # iter_dict['propagated_state_ids'] = isu.load_with_pickle(dyn_result_data_path, "propagated_state_ids.p")
        # iter_dict['propagated_state_vector_length'] = isu.load_with_pickle(dyn_result_data_path, "propagated_state_vector_length.p")
        # iter_dict['propagation_is_performed'] = isu.load_with_pickle(dyn_result_data_path, "propagation_is_performed.p")
        iter_dict['state_history'] = isu.load_with_pickle(dyn_result_data_path, "state_history.p")
        # # iter_dict['state_history_time_object'] = isu.load_with_pickle(dyn_result_data_path, "state_history_time_object.p")#
        # iter_dict['total_computation_time'] = isu.load_with_pickle(dyn_result_data_path, "total_computation_time.p")
        # iter_dict['total_number_of_function_evaluations'] = isu.load_with_pickle(dyn_result_data_path, "total_number_of_function_evaluations.p")
        # iter_dict['unordered_dependent_variable_settings'] = isu.load_with_pickle(dyn_result_data_path, "unordered_dependent_variable_settings.p")
        # # iter_dict['unprocessed_state_history'] = isu.load_with_pickle(dyn_result_data_path, "unprocessed_state_history.p")#
        # # iter_dict['unprocessed_state_history_time_object'] = isu.load_with_pickle(dyn_result_data_path, "unprocessed_state_history_time_object.p")#
"""

