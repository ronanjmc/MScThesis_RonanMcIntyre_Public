"""
This script is for comparing the performance of several different numerical
integration methods, similar to Question 2 of PropOp Assignment 1. The body is
propagated for the same duration with a set of different numerical integration
methods and a set of different time steps.

PER-INTEGRATOR PLOTS:
- One of each of these plots is generated for each integrator
- Plot A: isu.plot_position_error_per_timestep() --- shows the position error
  over time, with a separate line for each Δt except for the smallest. Error for
  a propagation with time step Δt is approximated as the position difference
  compared to the propagation that used Δt/2. Filename is
  error_vs_time_[integrator].pdf
- Plot B: isu.plot_max_and_RMS_error_vs_delta_t() --- shows the RMS of the
  position error over time and the maximum position error against the Δt used.
  Filename is error_vs_timestep_[integrator].pdf

PER-BODY PLOTS:
- One of each of these plots is generated for the selected target body
- Plot 1: plot_position_error_subplots_per_integrator_per_timestep() ---
  position error over time subplots. Essentially combines all Plot A plots into
  one figure, with a subplot for each integrator. Filename is
  error_vs_time_[integrator_1]_..._[integrator_n].pdf
- Plot 2: plot_RMS_error_vs_function_evaluations_per_integrator() --- RMS of
  position error over time vs. number of function evaluations, with a separate
  line for each integrator. Each point on the line represents an integrator / Δt
  pair. Filename is error_vs_funcevals_[integrator_1]_..._[integrator_n].pdf
- Plot 3: plot_max_and_RMS_error_vs_delta_t_per_integrator() --- Maximum (and
  RMS of) position error over time vs. Δt, with a separate line for each
  integrator. Each point on the line represents an integrator / Δt pair.
  Filename is error_vs_timestep_[integrator_1]_..._[integrator_n].pdf

INTEGRATOR ANALYSIS SUMMARY PLOTS:
- Max. error vs. eccentricity # !!!
"""

#%%%%%%# 0. Inputs
######## 0. Inputs
#region 0. Inputs

## CHANGING INPUTS
# Target body
bodies_to_propagate = [
    # str : accepts designation number, common name, or provisional designation of asteroid
    # #               # ARC [y]   e [-]       a [AU]
    # "595116",       # 23.96     0.0281      1.3107    # DONE
    # "2016 CO246",   #  4.02     0.1262      0.9948    # DONE - 12 mins
    # "2009 JR2",     # 14.50     0.1790      1.2498    # DONE - 42 mins
    # "499998",       #  6.30     0.2166      1.3159    # DONE - 24 mins
    # "152563",       # 71.15     0.2717      0.9079    # DONE - 100 mins
    # "2063",         # 48.14     0.3494      1.0780    # DONE - 184 mins
    # "7350",         # 59.74     0.3912      1.3560    # DONE - 103 mins
    # "2018 GY",      #  4.04     0.4063      0.7667    # DONE
    # "1221",         # 92.95     0.4346      1.9198    # DONE - 130 mins
    # "8014",         # 35.61     0.4560      1.7476    # DONE - 99 mins
    # "7336",         # 42.24     0.4820      2.3055    # DONE
    # "480808",       # 29.02     0.5262      0.6709    # DONE
    # "177016",       # 21.92     0.5822      1.1601    # DONE - 61 mins
    # "513126",       # 19.52     0.5826      1.7859    # DONE - 55 mins
    # "4179",         # 92.11     0.6247      2.5430    # DONE - 127 mins
    # "152664",       # 31.57     0.7140      2.5202    # DONE - 86 mins
    # "364136",       # 20.09     0.7549      0.6766    # DONE - 58 mins
    # "2017 DN109",   #  6.20     0.8268      1.1958    # DONE
    # "1995 CR",      # 31.04     0.8684      0.9069    # DONE - 85 mins
    # "137924",       # 29.00     0.8950      0.8764    # DONE - 111 mins
    # "1566",         # 76.15     0.8270      1.0780    # DONE - 124 minutes
    # "2101",         # 89.43     0.7641      1.8740    # DONE - 177 mins
]
bodies_to_propagate = [
    "2016 CO246",   #  4.02     0.1262      0.9948    # DONE - 12 mins
]

# Integrator
integrator_plan =         ["RK6", "RK8", "RK10","RK12"]
step_size_hi_power_plan = [   6 ,    6 ,     6 ,    6 ] # full set
# step_size_lo_power_plan = [  -7 ,   -6 ,    -5 ,   -5 ] # full set
step_size_lo_power_plan = [  -6 ,   -5 ,    -4 ,   -4 ] # slightly reduced set
# step_size_hi_power_plan = [   4 ,    4 ,     4 ,    4 ] # for quick checks
# step_size_lo_power_plan = [   1 ,    1 ,     1 ,    1 ] # for quick checks
integrator_error_threshold_KM = 10**-2 # in km. just used in plotting to indicate the threshold for "acceptable" error
# integrator_error_threshold_KM = None # in km. just used in plotting to indicate the threshold for "acceptable" error

# Plotting & Printing
verbose = False                 # bool : choose True to include Tudat propagation printing
quiet = False                   # bool : choose True to suppress all print statements
colormode = 0                # bool : choose True to use dark mode for plots
# plot_behaviour = 'skip'
# plot_behaviour = 'show'
plot_behaviour = 'save'
integrator_selection_dir = "integrator-selection/"
save_state_histories_to_files = False # Recommended: False
save_plot_data_to_files = True # Recommended: True

## DEFAULT INPUTS
# Observations
obs_range_start = [1900, 1, 1]    # list : start of time range to consider for astrometric observations of body_to_propagate. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.
obs_range_end = [2026, 7, 1]      # list : end of time range to consider for astrometric observations of body_to_propagate. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.
# Dynamic model
# define the acceleration model (start with "LB", add "+EIH", "+LMBA", "+CP", "+AB", "+Yark" as desired)
dynamic_model_def = "LB+LMBA+AB+Yark" # "LB+Yark" # "LB+EIH+LMBA+AB+Yark" #
n_LMBAs = 16                      # number of MBAs whose gravitation is to be included in dynamics (takes the n_LMBAs most massive MBAs). Only relevant if 'LMBA' is included in dynamic_model_def
# Frames
central_body = "Sun"
global_frame_origin = "SSB"
global_frame_orientation = "J2000"


#%%%%%%# 1. Imports
######## 1. Imports
#region 1. Imports
## OLD INTEGRATOR PLANS
"""
integrator_plan =         ["RK4", "RK6", "RK8", "RK10","RK12","RK14"]
step_size_hi_power_plan = [   8 ,    9 ,    9 ,     9 ,    9 ,    9 ]
step_size_lo_power_plan = [  -8 ,   -7 ,   -6 ,    -5 ,   -4 ,   -4 ]
step_size_lo_power_plan = [  -6 ,   -5 ,   -4 ,    -3 ,   -2 ,   -2 ] #!

integrator_plan =         ["Euler", "RK2"]
step_size_hi_power_plan = [     7 ,    7 ]
step_size_lo_power_plan = [    -10,   -10]

# Changing inputs (SIMPLE)
integrator_plan =         ["Euler", "RK2", "RK4"]
step_size_hi_power_plan = [     5 ,    5 ,    6 ]
step_size_lo_power_plan = [    -3 ,   -3 ,   -2 ]
"""

## IMPORTS
import os
import time as tm
import numpy as np
import pandas as pd
import IS_utils as isu
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
import tudatpy.constants as cnst


# import importlib
# isu = importlib.reload(isu)

local_path = os.path.dirname(__file__)
os.chdir(local_path) # ensure that the cwd is the subfolder where this file is

## Plotting behaviour
plot_behaviour_list, colormode_list = isu.define_plot_behaviour_lists(
    plot_behaviour, colormode,
)

# Dict of SPKID: obs_arc_est_YEAR for the selected asteroids for integrator analysis
arc_dict = {
    '20595116': 23.95108625564658,
    '3743897': 4.008189271937371,
    '20499998': 6.296942271640043,
    '20002063': 48.14183613153758,
    '3802409': 4.026430198495798,
    '20001221': 92.93967153560186,
    '20480808': 29.011748959872293,
    '20004179': 92.10510858737986,
    '20152664': 31.56480076147991,
    '3770288': 6.189811603024783,
    '20001566': 76.14188745223277,
    '3005973': 31.027648321956534,
    '20137924': 28.9978041369416,
    '20152563': 71.15167375018528,
    '20007350': 60.73924781222205,
    '20007336': 42.24437218674436,
    '20008014': 35.60530741653137,
    '3459812': 14.502085305112004,
    '20513126': 19.51600451600879,
    '20177016': 21.91929442809116,
    '20364136': 20.092304159626416,
    '20002101': 89.4318457222343,
}

# body_dir and timestamp_dir of successful full runs for each body selected for
# integrator analysis
ia_bodies_dir_dict = {
    # perm/prov. designation : (body_dir, timestamp_dir)
    "1221": ("1221-Amor/", "260510-032838/"),
    "595116": ("595116-2001-QE96/", "260508-040010/"),
    "499998": ("499998-2011-PT/", "260510-152019/"),
    "480808": ("480808-1994-XL1/", "260508-023618/"),
    "152664": ("152664-1998-FW4/", "260508-052247/"),
    "137924": ("137924-2000-BD19/", "260510-132908/"),
    "4179": ("4179-Toutatis/", "260510-053838/"),
    "2063": ("2063-Bacchus/", "260510-173316/"),
    "2018 GY": ("2018-GY/", "260507-000917/"),
    "2017 DN109": ("2017-DN109/", "260508-050552/"),
    "1995 CR": ("1995-CR/", "260508-064928/"),
    "2009 JR2": ("2009-JR2/", "260511-124958/"),
    "2016 CO246": ("2016-CO246/", "260511-233928/"),
    "7336": ("7336-Saunders/", "260513-035431/"),
    "8014": ("8014-1990-MF/", "260512-005211/"),
    "177016": ("177016-2003-BM47/", "260511-235052/"),
    "364136": ("364136-2006-CJ/", "260511-133149/"),
    "513126": ("513126-1998-QP/", "260511-142954/"),
    "7350": ("7350-1993-VA/", "260514-164719/"),
    "152563": ("152563-1992-BF/", "260514-184302/"),
    "1566": ("1566-Icarus/", "260515-200050/"),
    "2101": ("2101-Adonis/", "260515-140637/"),
}

#%%%%%%# 2. Plotting functions
######## 2. Plotting functions

### Plot 1: Position error vs. time
#region P.1 Err vs time
def plot_position_error_subplots_per_integrator_per_timestep(
    epoch_sets_per_int,
    pos_error_mag_sets_per_int,
    integrator_plan,
    step_size_hi_power_plan,
    step_size_lo_power_plan,
    body,
    colormode: int = 0,
    plot_behaviour: str = "show",
    plots_save_path: str = None,
):
    # Define style_dict so each timestep gets the same colour/linestyle in all plots
    highest_power = max(step_size_hi_power_plan)
    lowest_power = min(step_size_lo_power_plan)
    full_step_size_range = np.logspace(lowest_power, highest_power, highest_power-lowest_power+1, base=2)
    step_size_style_dict = dict()
    offset = max(step_size_lo_power_plan) - min(step_size_lo_power_plan) + 1
    for idx, current_step_size_DAY in enumerate(full_step_size_range):
        step_size_style_dict[current_step_size_DAY] = (
            # offset so blue solid line is visible in all plots
            isu.cmap_custom10[colormode]((idx-offset)%10),
            # offset so only smallest timesteps get dashdot linestyle
            isu.linestyles[(idx-offset)//10],
        )

    # Plot position error vs. time subplots
    # n_cols = int(np.ceil(np.sqrt(len(integrator_plan))))
    n_cols = 2
    n_rows = int(np.ceil( len(integrator_plan) / n_cols ))
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize = np.array([1.0,0.33*n_rows])*isu.full_width_INCH*0.9,
        facecolor = isu.fig_background[colormode],
        layout = 'constrained'
    )

    for int_idx, (integrator, ax) in enumerate(zip(integrator_plan, axs.flatten())):
        # ax = plt.subplot(n_cols, n_rows, int_idx+1)
        row = int_idx // n_cols
        col = int_idx % n_cols

        step_sizes_DAY = np.logspace(
            step_size_lo_power_plan[int_idx],
            step_size_hi_power_plan[int_idx],
            step_size_hi_power_plan[int_idx] - step_size_lo_power_plan[int_idx] + 1,
            base = 2,
        )

        for step_idx, current_step_size_DAY in enumerate(step_sizes_DAY[1:]):
            # Make Δt more readable
            step_size_unit_str = isu.convert_step_size_DAY_to_str_with_unit(
                current_step_size_DAY,
            )

            # Approximate years for plot ticks
            epochs_plot = (epoch_sets_per_int[int_idx][step_idx] / cnst.JULIAN_YEAR) + 2000
            ax.plot(
                epochs_plot,
                np.array(pos_error_mag_sets_per_int[int_idx][step_idx]) / 1000, # in km
                label = f"Δt = {step_size_unit_str}",
                color = step_size_style_dict[current_step_size_DAY][0],
                alpha = 0.8,
                linestyle = step_size_style_dict[current_step_size_DAY][1],
                linewidth = 2.0,
                zorder = 2.5,
            )

        # Axis settings, subplot title
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        ax.set_yscale('log')
        if integrator_error_threshold_KM:
            ax.axhline(
                integrator_error_threshold_KM,
                color = isu.text_color[colormode],
                linewidth = 2.0,
                linestyle = 'dotted',
                label = "Threshold",
                zorder = 2.1,
            )
        ax.set_title(
            f"Integrator: {integrator}",
            fontsize = isu.medium_font_size,
        )

        # Y-axis labels on left-most plots
        if col == 0:
            ax.set_ylabel(f"Error [km]")
        if (
            # X-axis labels on bottom row plots
            (row == (n_rows-1)) or
            # X-axis labels for hanging plots with no bottom row plot beneath them
            ((int_idx + n_cols) > len(integrator_plan)-1)
        ):
            ax.set_xlabel("Year")

    ## Legend business: single legend for all subplots
    # Get all lines and their labels
    lines_labels = [axis.get_legend_handles_labels() for axis in fig.axes]
    lines, labels = [sum(list_of_lists, []) for list_of_lists in zip(*lines_labels)]
    # Many labels are duplicates and will show multiple times in the legend. Find
    # them and remove from the 'labels', as well as the corresponding line from 'lines'
    for idx in reversed(range(len(labels)-1)): # reversed so indices don't change as we pop them out
        if labels[idx] in labels[idx+1:]:
            labels.pop(idx)
            lines.pop(idx)
    fig.legend(
        lines,
        labels,
        loc = 'outside center right',
        markerscale = 1.2,
        fontsize = isu.small_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
        reverse = True,
    )
    title_tex_Body = f"$\\bf{{Body:}}$"
    fig.suptitle(
        fr"{title_tex_Body} {body['longname']}:  $e$ = {body['eccentricity']:.3f},  $a$ = {body['semimajoraxis_AU']:.3f} AU,  arc = {arc_dict[body['SPKID']]:.3g} y",
        # f"Position Error over Time: {body['longname']}",
        fontsize = isu.medium_font_size,
        # weight = 'bold',
    )

    if plot_behaviour == 'show':
        plt.show(block=False)
        plt.pause(0.001)
    elif plot_behaviour == 'save':
        save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_name = f"error_vs_time_{'_'.join(integrator_plan)}.pdf"
        plt.savefig(save_path + fig_name, format="pdf")
        plt.close()
        print(f"Error vs. time subplots {'darkmode '*colormode}figure saved: \n{save_path+fig_name}")


### Plot 2: RMS of error vs. function evaluations
#region P.2 Err vs func.eval
def plot_RMS_error_vs_function_evaluations_per_integrator(
    function_evaluations_per_int,
    pos_error_RMSs_per_int,
    integrator_plan,
    body,
    colormode: int = 0,
    plot_behaviour: str = "show",
    plots_save_path: str = None,
):
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.85])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = 'constrained'
    )
    for idx, integrator in enumerate(integrator_plan):
        ax.plot(
            function_evaluations_per_int[idx][1:],
            np.array(pos_error_RMSs_per_int[idx]) / 1000, # in km
            label = integrator,
            color = isu.cmap_custom10[colormode](idx%10),
            alpha = 0.70,
            linestyle = isu.linestyles[idx//10],
            linewidth = 2.0,
            marker = isu.markers[idx],
            markersize = 8,
            zorder = 2.5,
        )

    # Axis settings
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    ax.set_yscale('log')
    ax.set_xscale('log')
    if integrator_error_threshold_KM:
        ax.axhline(
            integrator_error_threshold_KM,
            color = isu.text_color[colormode],
            linewidth = 2.0,
            linestyle = 'dotted',
            label = "Threshold",
            zorder = 2.1,
        )
    # Legend, labels, title
    ax.legend(
        loc = 'best',
        markerscale = 1.2,
        fontsize = isu.small_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    ax.set_ylabel(f"RMS of error [km]")
    ax.set_xlabel("Function evaluations [-]")
    fig.suptitle(
        # f"RMS of Position Error vs. Function Evaluations:\n{body['longname']}",
        # f"Body: {body['longname']}",
        body['longname'],
        fontsize = isu.medium_font_size,
        # weight = 'bold',
    )
    ax.set_title(
        fr"$e$ = {body['eccentricity']:.3f},  $a$ = {body['semimajoraxis_AU']:.3f} AU,  arc = {arc_dict[body['SPKID']]:.3g} y",
        fontsize = isu.small_font_size,
    )

    if plot_behaviour == 'show':
        plt.show(block=False)
        plt.pause(0.001)
    elif plot_behaviour == 'save':
        save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_name = f"error_vs_funcevals_{'_'.join(integrator_plan)}.pdf"
        plt.savefig(save_path + fig_name, format="pdf")
        plt.close()
        print(f"Error vs. function evaluations {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")


### Plot 3: Max/RMS of error vs. Δt per integrator
#region P.3 Max err vs Δt
def plot_max_and_RMS_error_vs_delta_t_per_integrator(
    pos_error_maxes_per_int,
    pos_error_RMSs_per_int,
    integrator_plan,
    step_size_hi_power_plan,
    step_size_lo_power_plan,
    body,
    colormode: int = 0,
    plot_behaviour: str = "show",
    plots_save_path: str = None,
):
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.85])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = 'constrained',
    )
    all_step_sizes_DAY = np.logspace(
        min(step_size_lo_power_plan),
        max(step_size_hi_power_plan),
        max(step_size_hi_power_plan) - min(step_size_lo_power_plan) + 1,
        base = 2,
    )
    highest_points = {step: 0.0 for step in all_step_sizes_DAY[1:]}
    for int_idx, integrator in enumerate(integrator_plan):
        # List of Δt
        step_sizes_DAY = np.logspace(
            step_size_lo_power_plan[int_idx],
            step_size_hi_power_plan[int_idx],
            step_size_hi_power_plan[int_idx] - step_size_lo_power_plan[int_idx] + 1,
            base = 2,
        )
        # Plot
        pos_error_maxes = np.array(pos_error_maxes_per_int[int_idx]) / 1000 # in km
        pos_error_RMSs = np.array(pos_error_RMSs_per_int[int_idx]) / 1000 # in km
        ax.plot(
            step_sizes_DAY[1:] * 24, # in h
            pos_error_maxes,  # in km
            label = integrator,
            color = isu.cmap_custom10[colormode](int_idx),
            alpha = 0.7,
            marker = isu.markers[int_idx],
            markersize = 8,
            linestyle = isu.linestyles[int_idx//10],
            linewidth = 2.5,
            zorder = 2.5,
        )
        # ax.plot(
        #     step_sizes_DAY[1:] * 24, # in h
        #     pos_error_RMSs,   # in km
        #     label = f"{integrator}: RMS of pos. error",
        #     marker = 's',
        #     markersize = 8,
        #     linewidth = 2.5,
        #     linestyle = '--',
        #     color = isu.cmap_custom10[colormode](int_idx),
        #     alpha = 0.7,
        #     zorder = 2.5,
        # )
        # Collect highest points for each Δt
        for step, step_max in zip(step_sizes_DAY[1:], pos_error_maxes):
            if step_max > highest_points[step]:
                highest_points[step] = step_max

    # Set default ax settings
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    # Other axis settings
    ax.set_yscale('log')
    ax.set_xscale('log')
    if integrator_error_threshold_KM:
        ax.axhline(
            integrator_error_threshold_KM,
            color = isu.text_color[colormode],
            linewidth = 2.0,
            linestyle = 'dotted',
            label = "Threshold",
            zorder = 2.1,
        )
    ax.margins(x=0.10)
    lowest_point = min(map(min, pos_error_maxes_per_int)) / 1000
    y_min, y_max = ax.get_ylim()
    default_y_margin_log = np.log10(lowest_point) - np.log10(y_min)
    new_y_margin_log = 6.0 * default_y_margin_log
    new_y_min_log = np.log10(lowest_point) - new_y_margin_log
    ax.set_ylim((10**new_y_min_log, y_max))

    # Annotate all points with time steps with readable units
    high_text_point = 10**(np.log10(lowest_point) - (0.50 * new_y_margin_log))
    low_text_point = 10**(np.log10(lowest_point) - (0.85 * new_y_margin_log))
    # List that alternates between high_text_point and low_text_point
    text_points = [None]*len(all_step_sizes_DAY)
    text_points[::2] = [high_text_point] * len(text_points[::2])
    text_points[1::2] = [low_text_point] * len(text_points[1::2])
    for (step,x,y) in zip(
        all_step_sizes_DAY[1:],
        np.array(all_step_sizes_DAY[1:]) * 24, # in h
        text_points,
    ):
        step_size_unit_str = isu.convert_step_size_DAY_to_str_with_unit(
            step,
            minute_as_m = True,
        )
        ax.annotate(
            step_size_unit_str, # text
            (x,y),              # coordinates
            textcoords = "offset points", # how to position the text
            xytext = (0,0), # distance from text to point (x,y)
            # xytext = (x,y*0.3), # distance from text to point (x,y)
            ha = 'center',      # horizontal alignment can be left, right or center
            fontsize = isu.tiny_font_size,
            color = isu.text_color[colormode],
        )
        ax.plot(
            [x,x],
            [y*6, highest_points[step]],
            color = isu.text_color[colormode],
            linestyle = '--',
            linewidth = 1.0,
            alpha = 0.5,
            zorder = 2.1,
        )

    # Legend, labels, titles
    ax.legend(
        loc = 'upper left',
        markerscale = 1.2,
        fontsize = isu.small_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )
    ax.set_ylabel("Max. pos. error [km]")
    ax.set_xlabel("Δt [h]")
    fig.suptitle(
        body['longname'],
        fontsize = isu.medium_font_size,
    )
    ax.set_title(
        fr"$e$ = {body['eccentricity']:.3f},  $a$ = {body['semimajoraxis_AU']:.3f} AU,  arc = {arc_dict[body['SPKID']]:.3g} y",
        fontsize = isu.small_font_size,
    )
    # fig.set_tight_layout(True)
    if plot_behaviour == 'show':
        plt.show()
    elif plot_behaviour == 'save':
        save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_name = f"error_vs_timestep_{'_'.join(integrator_plan)}.pdf"
        plt.savefig(save_path + fig_name, format="pdf")
        # plt.close()
        print(f"Error vs. Δt per integrator {'darkmode '*colormode}figure saved: \n{save_path+fig_name}")
        #
        # extra_save_path = 'plots/integrator-selection/error_vs_timestep_per_integrator/'
        # extra_fig_name = f"{body['filing_name_short']}_error_vs_timestep_{'_'.join(integrator_plan)}.png"
        # plt.savefig(extra_save_path + extra_fig_name, format="png")
        plt.close()


#%%%%%%# 3. Perform propagations & plot
######## 3. Perform propagations & plot
#region 3. Prop. & plot

T_START_body_loop = tm.perf_counter()
timestamp_list = []
runtime_history = {}

for body_idx, body_to_propagate in enumerate(bodies_to_propagate):
    T_START_integrator_loop = tm.perf_counter()
    runtime_history[body_to_propagate] = {}

    # Timestamp string to uniquely identify this run
    # timestamp_str = dttm.datetime.now().strftime("%y%m%d-%H%M%S") # e.g. formats 28 November 2025, 1:57:01pm as '251128-135701'
    timestamp_str = "260515-140637" #!!! remove later
    timestamp_dir = f"{timestamp_str}/"
    timestamp_list.append(timestamp_str)
    body = isu.query_SBDB(body_to_propagate)
    body_dir = f"{body['filing_name_long']}/"
    # Save integrator selection inputs to files
    if save_plot_data_to_files:
        plot_data_save_path = isu.output_data_dir + integrator_selection_dir + \
            body_dir + timestamp_dir
        if not os.path.exists(plot_data_save_path):
            os.makedirs(plot_data_save_path)
        isu.save_with_pickle(body, plot_data_save_path, "body.p")
        isu.save_with_pickle(integrator_plan, plot_data_save_path, "integrator_plan.p")
        isu.save_with_pickle(step_size_hi_power_plan, plot_data_save_path, "step_size_hi_power_plan.p")
        isu.save_with_pickle(step_size_lo_power_plan, plot_data_save_path, "step_size_lo_power_plan.p")

    function_evaluations_per_int = []
    epoch_sets_per_int = []
    pos_error_mag_sets_per_int = []
    pos_error_RMSs_per_int = []
    pos_error_maxes_per_int = []

    for int_idx, integrator in enumerate(integrator_plan):
        T_START_current_integrator = tm.perf_counter()
        print('\n\n')
        print( '###################################################')
        print(f'####### BODY:       {body_to_propagate}	({body_idx+1} of {len(bodies_to_propagate)})')
        print(f'####### INTEGRATOR: {integrator}	({int_idx+1} of {len(integrator_plan)})')
        print( '###################################################\n')

        body, function_evaluations, epoch_sets, pos_error_mag_sets, \
            pos_error_RMSs, pos_error_maxes, plots_save_path = \
            isu.propagate_body_with_integrator(
                # Target body
                body_to_propagate = body_to_propagate,
                # Integrator
                integrator = integrator,
                step_size_hi_power = step_size_hi_power_plan[int_idx],
                step_size_lo_power = step_size_lo_power_plan[int_idx],
                # Observations
                obs_range_start = obs_range_start,
                obs_range_end = obs_range_end,
                # Dynamic model
                dynamic_model_def = dynamic_model_def,
                n_LMBAs = n_LMBAs,
                # Frames
                central_body = central_body,
                global_frame_origin = global_frame_origin,
                global_frame_orientation = global_frame_orientation,
                # Plotting & Printing
                verbose = verbose,
                quiet = quiet,
                colormode = colormode,
                plot_behaviour = plot_behaviour,
                integrator_selection_dir = integrator_selection_dir,
                timestamp_dir = timestamp_dir,
                save_state_histories_to_files = save_state_histories_to_files,
        )

        # Generate plots for current integrator
        for cm, pb in zip(colormode_list, plot_behaviour_list):
            isu.plot_position_error_per_timestep(
                epoch_sets = epoch_sets,
                pos_error_mag_sets = pos_error_mag_sets,
                step_size_hi_power = step_size_hi_power_plan[int_idx],
                step_size_lo_power = step_size_lo_power_plan[int_idx],
                integrator = integrator,
                body = body,
                arc_YEAR = arc_dict[body['SPKID']],
                integrator_error_threshold_KM = integrator_error_threshold_KM,
                colormode = cm,
                plot_behaviour = pb,
                plots_save_path = plots_save_path,
            )
            isu.plot_max_and_RMS_error_vs_delta_t(
                step_size_hi_power = step_size_hi_power_plan[int_idx],
                step_size_lo_power = step_size_lo_power_plan[int_idx],
                pos_error_maxes = pos_error_maxes,
                pos_error_RMSs = pos_error_RMSs,
                integrator = integrator,
                body = body,
                arc_YEAR = arc_dict[body['SPKID']],
                integrator_error_threshold_KM = integrator_error_threshold_KM,
                colormode = cm,
                plot_behaviour = pb,
                plots_save_path = plots_save_path,
            )

        # Save outputs to files
        if save_plot_data_to_files:
            int_data_save_path = plot_data_save_path + f"{integrator}/"
            if not os.path.exists(int_data_save_path):
                os.makedirs(int_data_save_path)
            isu.save_with_pickle(function_evaluations, int_data_save_path, "function_evaluations.p")
            isu.save_with_pickle(epoch_sets, int_data_save_path, "epoch_sets.p")
            isu.save_with_pickle(pos_error_mag_sets, int_data_save_path, "pos_error_mag_sets.p")
            isu.save_with_pickle(pos_error_RMSs, int_data_save_path, "pos_error_RMSs.p")
            isu.save_with_pickle(pos_error_maxes, int_data_save_path, "pos_error_maxes.p")
            # also save step_size_hi_power_plan[int_idx]
            # also save step_size_lo_power_plan[int_idx]

        # Save outputs to lists
        function_evaluations_per_int.append(function_evaluations)
        epoch_sets_per_int.append(epoch_sets)
        pos_error_mag_sets_per_int.append(pos_error_mag_sets)
        pos_error_RMSs_per_int.append(pos_error_RMSs)
        pos_error_maxes_per_int.append(pos_error_maxes)

        # Record and print timing
        T_END_current_integrator = tm.perf_counter()
        print(f"CUMULATIVE TIME: {(T_END_current_integrator - T_START_integrator_loop)/60:.3f} mins \n")
        current_integrator_runtime = T_END_current_integrator - T_START_current_integrator
        runtime_history[body_to_propagate][integrator] = current_integrator_runtime

    # Save output lists to files
    if save_plot_data_to_files:
        isu.save_with_pickle(function_evaluations_per_int, plot_data_save_path, "function_evaluations_per_int.p")
        isu.save_with_pickle(epoch_sets_per_int, plot_data_save_path, "epoch_sets_per_int.p")
        isu.save_with_pickle(pos_error_mag_sets_per_int, plot_data_save_path, "pos_error_mag_sets_per_int.p")
        isu.save_with_pickle(pos_error_RMSs_per_int, plot_data_save_path, "pos_error_RMSs_per_int.p")
        isu.save_with_pickle(pos_error_maxes_per_int, plot_data_save_path, "pos_error_maxes_per_int.p")
        isu.save_with_pickle(plots_save_path, plot_data_save_path, "plots_save_path.p")

    print(f"\n\nALL INTEGRATORS DONE | {body['longname']} | Timestamp: {timestamp_str} | {(T_END_current_integrator - T_START_integrator_loop)/60:.3f} mins\n")


    ### Generate summary plots for target body
    #region Plot loop
    for cm, pb in zip(colormode_list, plot_behaviour_list):
        # Plot directories
        plots_save_path = isu.plot_dir + integrator_selection_dir + body_dir + timestamp_dir
        if not os.path.exists(plots_save_path):
            os.makedirs(plots_save_path)

        plot_position_error_subplots_per_integrator_per_timestep(
            epoch_sets_per_int = epoch_sets_per_int,
            pos_error_mag_sets_per_int = pos_error_mag_sets_per_int,
            integrator_plan = integrator_plan,
            step_size_hi_power_plan = step_size_hi_power_plan,
            step_size_lo_power_plan = step_size_lo_power_plan,
            body = body,
            colormode = cm,
            plot_behaviour = pb,
            plots_save_path = plots_save_path,
        )
        plot_RMS_error_vs_function_evaluations_per_integrator(
            function_evaluations_per_int = function_evaluations_per_int,
            pos_error_RMSs_per_int = pos_error_RMSs_per_int,
            integrator_plan = integrator_plan,
            body = body,
            colormode = cm,
            plot_behaviour = pb,
            plots_save_path = plots_save_path,
        )
        plot_max_and_RMS_error_vs_delta_t_per_integrator(
            pos_error_RMSs_per_int = pos_error_RMSs_per_int,
            pos_error_maxes_per_int = pos_error_maxes_per_int,
            integrator_plan = integrator_plan,
            step_size_hi_power_plan = step_size_hi_power_plan,
            step_size_lo_power_plan = step_size_lo_power_plan,
            body = body,
            colormode = cm,
            plot_behaviour = pb,
            plots_save_path = plots_save_path,
        )

T_END_body_loop = tm.perf_counter()
if len(bodies_to_propagate) > 1:
    print(f"\n\n\nALL BODIES DONE | {(T_END_body_loop - T_START_body_loop)/60:.3f} mins\n")
    print(f"Body 		Timestamp ")
    for body_propagated, timestamp in zip(bodies_to_propagate, timestamp_list):
        print(f"{body_propagated}		{timestamp} ")

print("\nRUNTIMES:")
for body_propagated in bodies_to_propagate:
    print(body_propagated)
    for integrat in integrator_plan:
        rt = runtime_history[body_propagated][integrat]
        if rt < 60.0:
            print(f"    {integrat}:	{rt:.2f} seconds")
        else:
            print(f"    {integrat}:	{rt/60:.2f} minutes")


#%%%%%%# 4. Plot from files
######## 4. Plot from files
#region 4. Plot from files

import importlib
isu = importlib.reload(isu)

# # # Plot behaviour
# cm = 0
pb = "show"

# # Load plot data
body_to_plot = "1995 CR"
integrator_selection_dir = "integrator-selection/"
body_dir = ia_bodies_dir_dict[body_to_plot][0]
timestamp_dir = ia_bodies_dir_dict[body_to_plot][1]
plot_data_save_path = isu.output_data_dir + integrator_selection_dir + \
    body_dir + timestamp_dir

# Integrator selection inputs
loaded_integrator_plan = isu.load_with_pickle(plot_data_save_path, "integrator_plan.p")
loaded_step_size_hi_power_plan = isu.load_with_pickle(plot_data_save_path, "step_size_hi_power_plan.p")
loaded_step_size_lo_power_plan = isu.load_with_pickle(plot_data_save_path, "step_size_lo_power_plan.p")
# Output lists
loaded_body = isu.load_with_pickle(plot_data_save_path, "body.p")
loaded_function_evaluations_per_int = isu.load_with_pickle(plot_data_save_path, "function_evaluations_per_int.p")
loaded_epoch_sets_per_int = isu.load_with_pickle(plot_data_save_path, "epoch_sets_per_int.p")
loaded_pos_error_mag_sets_per_int = isu.load_with_pickle(plot_data_save_path, "pos_error_mag_sets_per_int.p")
loaded_pos_error_RMSs_per_int = isu.load_with_pickle(plot_data_save_path, "pos_error_RMSs_per_int.p")
loaded_pos_error_maxes_per_int = isu.load_with_pickle(plot_data_save_path, "pos_error_maxes_per_int.p")
loaded_plots_save_path = isu.load_with_pickle(plot_data_save_path, "plots_save_path.p")
# loaded_plots_save_path = isu.plot_dir + integrator_selection_dir + \
#     body_dir + timestamp_dir

# loaded_integrator_plan =         ["RK6", "RK8", "RK10","RK12"]
# loaded_step_size_hi_power_plan = [   6 ,    6 ,     6 ,    6 ] # full set
# loaded_step_size_lo_power_plan = [  -6 ,   -5 ,    -4 ,   -5 ] # slightly reduced set

for cm, pb in zip(colormode_list, plot_behaviour_list):
    # Plots per integrator
    for int_idx, integrator in enumerate(loaded_integrator_plan):
        isu.plot_position_error_per_timestep(
            epoch_sets = loaded_epoch_sets_per_int[int_idx],
            pos_error_mag_sets = loaded_pos_error_mag_sets_per_int[int_idx],
            step_size_hi_power = loaded_step_size_hi_power_plan[int_idx],
            step_size_lo_power = loaded_step_size_lo_power_plan[int_idx],
            integrator = integrator,
            body = loaded_body,
            arc_YEAR = arc_dict[loaded_body['SPKID']],
            integrator_error_threshold_KM = integrator_error_threshold_KM,
            colormode = cm,
            plot_behaviour = pb,
            plots_save_path = loaded_plots_save_path,
        )
        isu.plot_max_and_RMS_error_vs_delta_t(
            step_size_hi_power = loaded_step_size_hi_power_plan[int_idx],
            step_size_lo_power = loaded_step_size_lo_power_plan[int_idx],
            pos_error_RMSs = loaded_pos_error_RMSs_per_int[int_idx],
            pos_error_maxes = loaded_pos_error_maxes_per_int[int_idx],
            integrator = integrator,
            body = loaded_body,
            arc_YEAR = arc_dict[loaded_body['SPKID']],
            integrator_error_threshold_KM = integrator_error_threshold_KM,
            colormode = cm,
            plot_behaviour = pb,
            plots_save_path = loaded_plots_save_path,
        )

    # Summary plots
    plot_position_error_subplots_per_integrator_per_timestep(
        epoch_sets_per_int = loaded_epoch_sets_per_int,
        pos_error_mag_sets_per_int = loaded_pos_error_mag_sets_per_int,
        integrator_plan = loaded_integrator_plan,
        step_size_hi_power_plan = loaded_step_size_hi_power_plan,
        step_size_lo_power_plan = loaded_step_size_lo_power_plan,
        body = loaded_body,
        colormode = cm,
        plot_behaviour = pb,
        plots_save_path = loaded_plots_save_path,
    )
    plot_RMS_error_vs_function_evaluations_per_integrator(
        function_evaluations_per_int = loaded_function_evaluations_per_int,
        pos_error_RMSs_per_int = loaded_pos_error_RMSs_per_int,
        integrator_plan = loaded_integrator_plan,
        body = loaded_body,
        colormode = cm,
        plot_behaviour = pb,
        plots_save_path = loaded_plots_save_path,
    )
    plot_max_and_RMS_error_vs_delta_t_per_integrator(
        pos_error_RMSs_per_int = loaded_pos_error_RMSs_per_int,
        pos_error_maxes_per_int = loaded_pos_error_maxes_per_int,
        integrator_plan = loaded_integrator_plan,
        step_size_hi_power_plan = loaded_step_size_hi_power_plan,
        step_size_lo_power_plan = loaded_step_size_lo_power_plan,
        body = loaded_body,
        colormode = cm,
        plot_behaviour = pb,
        plots_save_path = loaded_plots_save_path,
    )


#%% Get data for max. error vs. eccentricity
### Get data for max. error vs. eccentricity
#region Data for err vs. e

# Load data relevant to plot into dataframe
selected_integrators = ["RK6", "RK8", "RK10","RK12"]
ia_body_dfs = [pd.DataFrame()]*len(selected_integrators)
lowest_available_step_size_power_to_plot = np.zeros(len(selected_integrators))

for selected_int_idx, selected_integrator in enumerate(selected_integrators):
    for body_idx, ia_body in enumerate(ia_bodies_dir_dict.keys()):
        # Paths
        ia_data_save_path = isu.output_data_dir + integrator_selection_dir + \
            ia_bodies_dir_dict[ia_body][0] + ia_bodies_dir_dict[ia_body][1]
        plots_save_path = isu.plot_dir + integrator_selection_dir + \
            ia_bodies_dir_dict[ia_body][0] + ia_bodies_dir_dict[ia_body][1]

        # Integrator selection inputs
        loaded_integrator_plan = isu.load_with_pickle(ia_data_save_path, "integrator_plan.p")
        loaded_step_size_hi_power_plan = isu.load_with_pickle(ia_data_save_path, "step_size_hi_power_plan.p")
        loaded_step_size_lo_power_plan = isu.load_with_pickle(ia_data_save_path, "step_size_lo_power_plan.p")
        # Output lists
        loaded_pos_error_maxes_per_int = isu.load_with_pickle(ia_data_save_path, "pos_error_maxes_per_int.p")
        loaded_body = isu.load_with_pickle(ia_data_save_path, "body.p")

        ia_body_dict = dict(
            desig = ia_body,
            longname = loaded_body['longname'],
            # filing_name_long = loaded_body['filing_name_long'],
            # filing_name_short = loaded_body['filing_name_short'],
            eccentricity = loaded_body['eccentricity'],
            semimajoraxis_AU = loaded_body['semimajoraxis_AU'],
            obs_arc_YEAR = arc_dict[loaded_body['SPKID']],
            selected_integrator = selected_integrator,
        )

        # Max. position errors are assigned to columns corresponding to their respective ∆t
        int_plan_idx = loaded_integrator_plan.index(selected_integrator)
        step_sizes_powers = np.arange(
            loaded_step_size_lo_power_plan[int_plan_idx],
            loaded_step_size_hi_power_plan[int_plan_idx] + 1,
            dtype = int,
        )
        for step_idx, step_size in enumerate(step_sizes_powers[1:]):
            ia_body_dict[step_size] = loaded_pos_error_maxes_per_int[int_plan_idx][step_idx]

        if step_sizes_powers[1] < lowest_available_step_size_power_to_plot[selected_int_idx]:
            lowest_available_step_size_power_to_plot[selected_int_idx] = step_sizes_powers[1]

        # Add data for current body to dataframe
        df = pd.DataFrame(data=ia_body_dict, index=[body_idx])
        ia_body_dfs[selected_int_idx] = pd.concat([ia_body_dfs[selected_int_idx], df])

    # Sort df in order of eccentricity
    ia_body_dfs[selected_int_idx] = \
        ia_body_dfs[selected_int_idx].sort_values('eccentricity').reset_index(drop=True)

#%% PX.1 Max. error vs. eccentricity
### PX.1 Max. error vs. eccentricity
#region PX.1 err vs. e

colormode = 0
pb = 'show'
# pb = 'save'

plot_x_axis = 'e'
fig_name = f"max-error_vs_eccentricity.pdf"

# plot_x_axis = 'a'
# fig_name = f"max-error_vs_semimajoraxis.pdf"

# plot_x_axis = 'arc'
# fig_name = f"max-error_vs_obs-arc.pdf"

plots_save_path = isu.plot_dir + integrator_selection_dir + "max-error_vs_parameter/"

# define ∆ts to plot
# step_sizes_power_plot_hi = [0, 1, 2, 2]
step_sizes_power_plot_hi = [0, 0, 0, 0]
# lowest_available_step_size_power_to_plot.astype(int) = array([-6, -5, -4, -4])
step_sizes_power_plot_lo = [-5, -5, -4, -4]
integrator_error_threshold_KM = 10**-2 # in km. used in plotting to indicate the threshold for "acceptable" error
# integrator_error_threshold_KM = None

# Define style_dict so each timestep gets the same colour/linestyle in all plots
highest_power = max(step_sizes_power_plot_hi)
lowest_power = min(step_sizes_power_plot_lo)
full_step_size_range = np.logspace(lowest_power, highest_power, highest_power-lowest_power+1, base=2)
cmap_sample_indices = np.linspace(0,225,len(full_step_size_range)).astype(int)
step_size_style_dict = dict()
offset = max(step_size_lo_power_plan) - min(step_size_lo_power_plan) + 1
markers = ['v', (5, 1, 0),  'P', (6, 1, 0),
           '^', (6, 2, 0),  'X', (5, 2, 0),
           '<', (10, 1, 0), 'o', 'x',
           '>', 'h',        'd', '2',       ]
for idx, current_step_size_DAY in enumerate(full_step_size_range):
    # step_size_DAY: (color, linestyle, marker)
    step_size_style_dict[current_step_size_DAY] = (
        # offset so blue solid line is visible in all plots
        # isu.cmap_custom10[colormode]((idx-offset)%10),
        colormaps.gnuplot(cmap_sample_indices[idx]),        # colour
        # offset so only smallest timesteps get dashdot linestyle
        # isu.linestyles[(idx+offset)//10],
        isu.linestyles[0],               # linestyle
        markers[(idx)%15],                          # marker
    )


# create the figure
fig, axs = plt.subplots(
    2, 2, # !! this isn't flexible if length of selected_integrators != 4
    figsize = np.array([1.0,0.85])*isu.full_width_INCH,
    sharex = True,
    sharey = False,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
for selected_int_idx, (selected_integrator, ax) in enumerate(
    zip(selected_integrators, axs.flatten())
):
    # Common axis settings
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    ax.set_yscale('log')
    ax.tick_params(right=True)
    ax.set_title(
        f"Integrator: {selected_integrator}",
        fontsize = isu.medium_font_size,
    )

    step_size_plot_lo = step_sizes_power_plot_lo[selected_int_idx]
    step_size_plot_hi = step_sizes_power_plot_hi[selected_int_idx]
    step_sizes_plot_powers = np.arange(
        step_size_plot_lo,
        step_size_plot_hi + 1,
        dtype = int,
    )
    step_sizes_plot_DAY = np.logspace(
        step_size_plot_lo,
        step_size_plot_hi,
        step_size_plot_hi - step_size_plot_lo + 1,
        base = 2,
    )
    for step_idx, (step_power, step_DAY) in enumerate(
        zip(step_sizes_plot_powers, step_sizes_plot_DAY)
    ):
        step_size_unit_str = isu.convert_step_size_DAY_to_str_with_unit(
                step_DAY,
                minute_as_m = True,
        )

        if plot_x_axis == 'e':
            # Sort by eccentricity
            ia_body_dfs[selected_int_idx] = \
                ia_body_dfs[selected_int_idx].sort_values('eccentricity').reset_index(drop=True)
            # Get eccentricities
            x_data = ia_body_dfs[selected_int_idx].eccentricity
            # Axis limits
            ax.set_xlim((0.0,1.0))
            # ymin, ymax = ax.get_ylim()
            # ax.set_ylim(( (10**-4)/3.16, (10**0)*3.16 ))
            # Axis labels (bottom subplots only)
            axs[1,0].set_xlabel(r"Eccentricity $e$")
            axs[1,1].set_xlabel(r"Eccentricity $e$")

        elif plot_x_axis == 'a':
            # Sort by semi-major axis
            ia_body_dfs[selected_int_idx] = \
                ia_body_dfs[selected_int_idx].sort_values('semimajoraxis_AU').reset_index(drop=True)
            # Get semi-major axes
            x_data = ia_body_dfs[selected_int_idx].semimajoraxis_AU
            # Axis limits
            # ...
            # Axis labels (bottom subplots only)
            axs[1,0].set_xlabel(r"Semi-major axis $a$ [AU]")
            axs[1,1].set_xlabel(r"Semi-major axis $a$ [AU]")

        elif plot_x_axis == 'arc':
            # Sort by observation arc
            ia_body_dfs[selected_int_idx] = \
                ia_body_dfs[selected_int_idx].sort_values('obs_arc_YEAR').reset_index(drop=True)
            # Get observation arcs
            x_data = ia_body_dfs[selected_int_idx].obs_arc_YEAR
            # Axis limits
            # ...
            # Axis labels (bottom subplots only)
            axs[1,0].set_xlabel(r"Observation arc [year]")
            axs[1,1].set_xlabel(r"Observation arc [year]")

        # !! do I need the scatter?
        ax.scatter(
            x_data,
            ia_body_dfs[selected_int_idx][step_power] / 1000, # in km
            color = step_size_style_dict[step_DAY][0],
            alpha = 0.1,
            marker = step_size_style_dict[step_DAY][2],
            s = 36,
            zorder = 2.5,
        )
        ax.plot(
            x_data,
            ia_body_dfs[selected_int_idx][step_power] / 1000, # in km
            label = f"Δt = {step_size_unit_str}",
            color = step_size_style_dict[step_DAY][0],
            alpha = 0.3,
            marker = step_size_style_dict[step_DAY][2],
            markersize = 6,
            linestyle = step_size_style_dict[step_DAY][1],
            linewidth = 2.0,
            zorder = 2.5,
        )

    if integrator_error_threshold_KM:
        ax.axhline(
            integrator_error_threshold_KM,
            color = isu.text_color[colormode],
            linewidth = 2.0,
            linestyle = 'dotted',
            label = "Threshold",
            zorder = 2.1,
        )

## Legend business: single legend for all subplots
# Get all lines and their labels
lines_labels = [axis.get_legend_handles_labels() for axis in fig.axes]
lines, labels = [sum(list_of_lists, []) for list_of_lists in zip(*lines_labels)]
# Many labels are duplicates and will show multiple times in the legend. Find
# them and remove from the 'labels', as well as the corresponding line from 'lines'
for idx in reversed(range(len(labels)-1)): # reversed so indices don't change as we pop them out
    if labels[idx] in labels[idx+1:]:
        labels.pop(idx)
        lines.pop(idx)
leg = fig.legend(
    lines,
    labels,
    loc = 'outside center right',
    markerscale = 1.5,
    fontsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
    reverse = True,
)
for lh in leg.legend_handles:
    # Make legend items full opacity
    lh.set_alpha(0.7)


# if plot_x_axis == 'e':
if plot_x_axis:
    # Highlight points that correspond to the integrator/Δt scheme
    for selected_int_idx, (selected_integrator, ax) in enumerate(
        zip(selected_integrators, axs.flatten())
    ):
        n_special_lines = 0
        for step_idx, (step_power, step_DAY) in enumerate(
            zip(step_sizes_plot_powers, step_sizes_plot_DAY)
        ):
            if (selected_integrator == "RK8") and (step_power == -2):
                e_range = [0, 0.6]
                special_label = fr"Selection for $e < {e_range[1]}$",
            # elif (selected_integrator == "RK8") and (step_power == -1):
            #     e_range = [0.4, 0.6]
            #     special_label = fr"Selection for ${e_range[0]} \leq e < {e_range[1]})$",
            elif (selected_integrator == "RK10") and (step_power == -3):
                e_range = [0.6, 1]
                special_label = fr"Selection for $e \geq {e_range[0]}$",
            else:
                e_range = None

            if e_range:
                n_special_lines += 1
                special_points = ia_body_dfs[selected_int_idx].loc[
                    (ia_body_dfs[selected_int_idx].eccentricity >= e_range[0]) &
                    (ia_body_dfs[selected_int_idx].eccentricity < e_range[1])
                ]

                if plot_x_axis == 'e':
                    special_x_data = special_points.eccentricity
                elif plot_x_axis == 'a':
                    special_x_data = special_points.semimajoraxis_AU
                elif plot_x_axis == 'arc':
                    special_x_data = special_points.obs_arc_YEAR

                ax.plot(
                    special_x_data,
                    special_points[step_power] / 1000, # in km
                    label = special_label,
                    color = step_size_style_dict[step_DAY][0],
                    alpha = 1.0,
                    marker = step_size_style_dict[step_DAY][2],
                    markersize = 10,
                    markeredgecolor = isu.text_color[colormode],
                    markeredgewidth = 0.5,
                    linestyle = step_size_style_dict[step_DAY][1],
                    linewidth = 3.0,
                    zorder = 2.6,
                )
        if n_special_lines:
            lines_labels = ax.get_legend_handles_labels()
            lines = [tup[0] for tup in zip(*lines_labels)]
            labels = [tup[1] for tup in zip(*lines_labels)]
            # Legend for only the 'special' points
            ax.legend(
                lines[-n_special_lines:],
                labels[-n_special_lines:],
                loc = 'best',
                markerscale = 1.0,
                fontsize = isu.small_font_size,
                labelcolor = isu.text_color[colormode],
                facecolor = isu.legend_bg_color[colormode],
                reverse = True,
            )

# Labels, ticks
axs[0,0].set_ylabel("Max. pos. error [km]")
axs[1,0].set_ylabel("Max. pos. error [km]")
# axs[0,1].tick_params(labelleft=False)
# axs[1,1].tick_params(labelleft=False)

if (pb == 'show'):
    plt.show(block=False)
    plt.pause(0.001)
elif (pb == 'save'):
    save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + fig_name, format="pdf")#, bbox_inches='tight')
    plt.close()

#%% Collect and save _per_int data sets from integrator directories
### Collect and save _per_int data sets from integrator directories
#region Collect & save data

# # Directories
# body_to_plot = "2101"
# body_dir = ia_bodies_dir_dict[body_to_plot][0]
# timestamp_dir = ia_bodies_dir_dict[body_to_plot][1]
# ia_data_save_path = isu.output_data_dir + integrator_selection_dir + \
#     body_dir + timestamp_dir
# plots_save_path = isu.plot_dir + integrator_selection_dir + \
#     body_dir + timestamp_dir

# # Load data
# l_integrator_plan = isu.load_with_pickle(ia_data_save_path, "integrator_plan.p")
# l_integrator_plan =         ["RK6", "RK8", "RK10","RK12"]
# # l_step_size_hi_power_plan = isu.load_with_pickle(ia_data_save_path, "step_size_hi_power_plan.p")
# # l_step_size_lo_power_plan = isu.load_with_pickle(ia_data_save_path, "step_size_lo_power_plan.p")
# # l_pos_error_mag_sets_per_int = isu.load_with_pickle(ia_data_save_path, "pos_error_mag_sets_per_int.p")

# epoch_sets_per_int = []
# function_evaluations_per_int = []
# pos_error_mag_sets_per_int = []
# pos_error_maxes_per_int = []
# pos_error_RMSs_per_int = []

# for int_idx, integrator in enumerate(l_integrator_plan):
#     int_data_save_path = ia_data_save_path + f"{integrator}/"
#     epoch_sets = isu.load_with_pickle(int_data_save_path, "epoch_sets.p")
#     function_evaluations = isu.load_with_pickle(int_data_save_path, "function_evaluations.p")
#     pos_error_mag_sets = isu.load_with_pickle(int_data_save_path, "pos_error_mag_sets.p")
#     pos_error_maxes = isu.load_with_pickle(int_data_save_path, "pos_error_maxes.p")
#     pos_error_RMSs = isu.load_with_pickle(int_data_save_path, "pos_error_RMSs.p")

#     epoch_sets_per_int.append(epoch_sets)
#     function_evaluations_per_int.append(function_evaluations)
#     pos_error_mag_sets_per_int.append(pos_error_mag_sets)
#     pos_error_maxes_per_int.append(pos_error_maxes)
#     pos_error_RMSs_per_int.append(pos_error_RMSs)

#     # pos_error_RMSs = []
#     # pos_error_maxes = []
#     # step_sizes_DAY = np.logspace(
#     #     l_step_size_hi_power_plan[int_idx],
#     #     l_step_size_hi_power_plan[int_idx],
#     #     l_step_size_hi_power_plan[int_idx] - l_step_size_lo_power_plan[int_idx] + 1,
#     #     base = 2,
#     # )
#     # for step_idx, current_step_size_DAY in enumerate(step_sizes_DAY[1:]):
#     #     pos_error_mags = l_pos_error_mag_sets_per_int[int_idx][step_idx]
#     #     pos_error_RMS = isu.rms(np.array(pos_error_mags))
#     #     pos_error_max = max(pos_error_mags)
#     #     pos_error_RMSs.append(pos_error_RMS)
#     #     pos_error_maxes.append(pos_error_max)
#     # pos_error_RMSs_per_int.append(pos_error_RMSs)
#     # pos_error_maxes_per_int.append(pos_error_maxes)

# isu.save_with_pickle(epoch_sets_per_int, ia_data_save_path, "epoch_sets_per_int.p")
# isu.save_with_pickle(function_evaluations_per_int, ia_data_save_path, "function_evaluations_per_int.p")
# isu.save_with_pickle(pos_error_mag_sets_per_int, ia_data_save_path, "pos_error_mag_sets_per_int.p")
# isu.save_with_pickle(pos_error_maxes_per_int, ia_data_save_path, "pos_error_maxes_per_int.p")
# isu.save_with_pickle(pos_error_RMSs_per_int, ia_data_save_path, "pos_error_RMSs_per_int.p")
# isu.save_with_pickle(plots_save_path, ia_data_save_path, "plots_save_path.p")

# integrator_plan =         ["RK6", "RK8", "RK10","RK12"]
# step_size_hi_power_plan = [   6 ,    6 ,     6 ,    6 ] # full set
# step_size_lo_power_plan = [  -6 ,   -5 ,    -4 ,   -5 ] # slightly reduced set
# isu.save_with_pickle(integrator_plan, ia_data_save_path, "integrator_plan.p")
# isu.save_with_pickle(step_size_hi_power_plan, ia_data_save_path, "step_size_hi_power_plan.p")
# isu.save_with_pickle(step_size_lo_power_plan, ia_data_save_path, "step_size_lo_power_plan.p")