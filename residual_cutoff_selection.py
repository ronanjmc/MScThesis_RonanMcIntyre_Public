"""
# !!! describe script

"""

#%%%%%%# 0. Imports
######## 0. Imports
#region 0.Imports

import time as tm
T_START_imports = tm.perf_counter()
print("Importing packages and loading SPICE kernels...")

# Directory business
import os
local_path = os.path.dirname(__file__)
os.chdir(local_path) # ensure that the cwd is the subfolder where this file is

# Tudat imports
from tudatpy.dynamics import environment_setup
import tudatpy.constants as cnst
from tudatpy.interface import spice
from tudatpy.astro.time_representation import DateTime
from tudatpy.data.mpc import BatchMPC
from tudatpy.estimation import (
    observable_models_setup,
    observations_setup,
    observations
)

# Other Python imports
import numpy as np
import pandas as pd
import datetime as dttm
import matplotlib.pyplot as plt

# Local imports
import IS_utils as isu

# Load SPICE kernels
spice.clear_kernels()
spice.load_standard_kernels()

T_END_imports = tm.perf_counter()
print(f"Done ({T_END_imports - T_START_imports:.3f}s) \n")

#%%%%%%# 1. Inputs
######## 1. Inputs
#region 1.Inputs

## Observations
obs_range_start = [1900, 1, 1]    # list : start of time range to consider for astrometric observations of body_to_propagate. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.
obs_range_end = [2026, 7, 1]      # list : end of time range to consider for astrometric observations of body_to_propagate. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.

## Frames
central_body = "Sun"
global_frame_origin = "SSB"
global_frame_orientation = "J2000"

## Plotting & Printing
verbose = False                 # bool : choose True to include additional supplementary print statements
quiet = False                   # bool : choose True to suppress all print statements
colormode = 0           # int : 0 for lightmode, 1 for darkmode when showing plots (both colormodes are saved when plot_behaviour is 'save' anyway)
# plot_behaviour = 'skip'         # 'skip' to skip all plotting
# plot_behaviour = 'save'       # 'save' to save plots to files (lightmode and darkmode)
plot_behaviour = 'show'       # 'show' to show plots in (interactive) terminal (specified colormode only)
# plot_behaviour = 'both'       # 'both' to show plots in (interactive) terminal (specified colormode only) AND save plots to files (lightmode and darkmode)
# plot_behaviour = 'showboth'   # 'showboth' to show plots in (interactive) terminal (lightmode and darkmode)
residual_cutoff_selection_dir = "residual-cutoff-selection/"

# Load Yarkovsky data
Fenucci_etal2024_tabB1 = isu.load_Fenucci_etal2024_tabB1()
body_to_test_list = Fenucci_etal2024_tabB1.Asteroid.to_list()
bodies_to_skip = [
    '101955', # Bennu (SPK file not available)
]

# Define set of residual filter cutoffs to test
residual_filter_cutoff_ARCSEC_list = [3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]
residual_filter_cutoff_ARCSEC_list = [1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  8.0, 10.0, 15.0, 20.0, 30.0, 50.0]

#%%%%%%# 2. Input processing
######## 2. Input processing
#region 2.Input process

# Printing setup
if verbose and quiet:
    raise ValueError("Verbose mode and quiet mode have both been selected. Choose one or neither.")
# Function that prints args if quiet=False and does nothing if quiet=True
print_default = print if not quiet else lambda *a, **k: None
# Function that prints args if verbose=True and does nothing if verbose=False
print_verbose = print if verbose else lambda *a, **k: None

## Observation range time variables
obs_range_start_J2000 = DateTime(*obs_range_start).epoch() # in seconds since J2000
obs_range_end_J2000 = DateTime(*obs_range_end).epoch() # in seconds since J2000


#%%%%%%# 3. Simulate observations in loop
######## 3. Simulate observations in loop
#region 3. Loop

# Shorter lists for testing
# body_to_test_list = body_to_test_list[:7]
# # residual_filter_cutoff_ARCSEC_list = residual_filter_cutoff_ARCSEC_list[:1]
# residual_filter_cutoff_ARCSEC_list = [2.0, 3.0]

# Timestamp string to uniquely identify this run
timestamp_str = dttm.datetime.now().strftime("%y%m%d-%H%M%S") # e.g. formats 28 November 2025, 1:57:01pm as '251128-135701'
timestamp_dir = f"{timestamp_str}/"

# Directory for saving outputs
run_output_data_save_path = isu.output_data_dir + residual_cutoff_selection_dir + timestamp_dir
if not os.path.exists(run_output_data_save_path):
    os.makedirs(run_output_data_save_path)
isu.save_with_pickle(residual_filter_cutoff_ARCSEC_list, run_output_data_save_path, "residual_filter_cutoff_ARCSEC_list.p")

# Initialise DataFrames to store outputs
simulated_residual_df = pd.DataFrame()
T_START_loop = tm.perf_counter()
print_default(f"STARTING RUN WITH {len(body_to_test_list)} BODIES AND {len(residual_filter_cutoff_ARCSEC_list)} CUTOFF VALUES")
print_default(f"TIMESTAMP: {timestamp_str}")

for body_idx, body_to_test in enumerate(body_to_test_list):
    T_START_body = tm.perf_counter()

    # Get body data from JPL SBDB and print selected info
    body = isu.query_SBDB(body_to_test, print_body_info=verbose)
    print_default(f"\nBODY: {body['longname']} | {body_idx+1}/{len(body_to_test_list)}")
    print_default(f"Setting up...")

    if body_to_test in bodies_to_skip:
        print(f"Skipping asteroid {body['longname']}.")
        continue

    # Get literature Yarkovsky/body data
    A2_target_body, A2_target_body_unc, Fenucci_body_data, _ = \
        isu.get_Yarkovsky_parameter(body, verbose)


    ######## 3.1 Retrieve MPC observations
    #region 3.1 Retrieve MPC

    ## Retrieve observations
    if body['designation_number'] == '[unnumbered]':
        target_body_name = body['provisional_designation']
    else:
        target_body_name = body['designation_number']
    MPC_codes = [target_body_name]       # BatchMPC requires a list
    batch = BatchMPC()
    batch.get_observations(MPC_codes, custom_name = body['SPKID'])

    # Filter by date
    batch.filter(
        epoch_start = obs_range_start_J2000,
        epoch_end = obs_range_end_J2000,
    )
    n_obs_available = len(batch.table)

    # Filter out spacecraft observations
    spacecraft_in_batch = batch.observatories_table(only_space_telescopes=True).Name.to_list()
    spacecraft_to_exclude_MPC_codes = [
        isu.spacecraft_codes[name][0] for name in spacecraft_in_batch
    ]
    batch.filter(
        observatories_exclude = spacecraft_to_exclude_MPC_codes,
    )

    # Observation numbers
    n_space_obs_dropped = n_obs_available - len(batch.table)
    n_ground_obs_available = len(batch.table)

    ## Time variables
    first_obs_J2000 = batch.epoch_start
    last_obs_J2000 = batch.epoch_end
    propagation_start_DateTime, propagation_start_J2000 = \
        isu.set_propagation_start_from_first_observation_epoch(first_obs_J2000)
    propagation_end_DateTime, propagation_end_J2000 = \
        isu.set_propagation_end_from_last_observation_epoch(last_obs_J2000)

    # Other time variables
    obs_arc_YEAR = (last_obs_J2000 - first_obs_J2000) / cnst.JULIAN_YEAR
    propagation_duration_SEC = propagation_end_J2000 - propagation_start_J2000
    propagation_duration_YEAR = propagation_duration_SEC / cnst.JULIAN_YEAR
    time_buffer = 2 * 31 * cnst.JULIAN_DAY  # 2 month time buffer to avoid interpolation errors

    ## Print some info
    if verbose:
        batch.summary()
    print_verbose(f"Propagation start and end dates: "
        f"{propagation_start_DateTime.year}-{propagation_start_DateTime.month}-{propagation_start_DateTime.day} "
        f"to {propagation_end_DateTime.year}-{propagation_end_DateTime.month}-{propagation_end_DateTime.day} "
        f"({propagation_duration_YEAR:.4g} years)")
    print_verbose(f"Observation arc: {obs_arc_YEAR:.4g} years")
    print_verbose(f"N observations available: {n_obs_available}")
    print_verbose(f"N space observations excluded: {n_space_obs_dropped} "
                f"(from {spacecraft_in_batch})")
    print_verbose(f"N observations to be used: {n_ground_obs_available} "
                f"({n_ground_obs_available} ground + 0 space)")

    # Save data
    body_res_dict = dict(
        # BODY DATA
        desig = body_to_test,
        longname = body['longname'],
        # filing_name_long = loaded_body['filing_name_long'],
        # filing_name_short = loaded_body['filing_name_short'],
        eccentricity = body['eccentricity'],
        semimajoraxis_AU = body['semimajoraxis_AU'],
        # LITERATURE DATA
        obs_arc_lit_YEAR = Fenucci_body_data['deltat'].item(),
        n_opt_obs_lit = Fenucci_body_data['NOptObs'].item(),
        n_opt_obs_rej_lit = Fenucci_body_data['NRejOpt'].item(),
        n_rad_obs_lit = Fenucci_body_data['NRadObs'].item(),
        n_rad_obs_rej_lit = Fenucci_body_data['NRejRad'].item(),
        n_opt_old_lit = Fenucci_body_data['NOptOld'].item(),
        SNR_lit = Fenucci_body_data['SNR'].item(),
        RMS_norm_res_lit = Fenucci_body_data['RMS'].item(),
        # AVAILABLE OBSERVATIONS
        obs_arc_YEAR = obs_arc_YEAR,
        n_obs_available = n_obs_available,
        n_space_obs_dropped = n_space_obs_dropped,
        n_ground_obs_available = n_ground_obs_available,
    )

    ######## 3.2 Environment setup (bodies, ephemerides)
    #region 3.2 Env. bodies

    # Load ephemeris for target body
    spice.load_kernel(isu.SPK_file_dir + body['SPKID'] + ".bsp")
    print_verbose(f"SPK file of {body['longname']} loaded. (SPKID: {body['SPKID']})")
    target_body_ephemeris = environment_setup.ephemeris.direct_spice(
        global_frame_origin, global_frame_orientation, body['SPKID']
    )
    target_body_ephemeris = environment_setup.ephemeris.tabulated_from_existing(
        target_body_ephemeris,
        start_time = propagation_start_J2000 - time_buffer,
        end_time = propagation_end_J2000 + time_buffer,
        time_step = 1.0 * cnst.JULIAN_DAY,
    )

    # Create body settings
    body_settings = environment_setup.get_default_body_settings(
        ["Earth"], global_frame_origin, global_frame_orientation
    )
    # Add target body with name = SPKID
    body_settings.add_empty_settings(body['SPKID'])
    body_settings.get(body['SPKID']).ephemeris_settings = target_body_ephemeris

    # Create SystemOfBodies object
    bodies = environment_setup.create_system_of_bodies(body_settings)

    T_END_setup = tm.perf_counter()
    print_default(f"Setup done ({T_END_setup - T_START_body:.3f}s)")



    ######## 3.3 Simulate residuals
    #region 3.3 Simulate residuals

    print_default(f"Simulating residuals...")
    for cutoff_idx, residual_filter_cutoff_ARCSEC in enumerate(residual_filter_cutoff_ARCSEC_list):
        T_START_cutoff = tm.perf_counter()
        print_verbose(f"Cutoff value: {residual_filter_cutoff_ARCSEC} arcsec | "
                      f"{cutoff_idx+1}/{len(residual_filter_cutoff_ARCSEC_list)}")
        # The observation collection is created with BatchMPC.to_tudat().
        # Internally, to_tudat() links a space telescope's observatory code to the
        # spacecraft's dynamics
        observation_collection = batch.to_tudat(
            bodies = bodies,
            included_satellites = None,
            apply_star_catalog_debias = True,
            apply_weights_VFCC17 = True,
        )

        # Set create angular_position settings for each link in the list
        observation_settings_list = []
        link_list = list(
            observation_collection.get_link_definitions_for_observables(
                observable_type = observable_models_setup.model_settings.angular_position_type
            )
        )
        for link in link_list:
            observation_settings_list.append(
                observable_models_setup.model_settings.angular_position(
                    link,
                    bias_settings = None
                )
            )

        # Use simulated observations to calculate residuals and filter out outliers
        # Create simulators
        observation_simulators = \
            observations_setup.observations_simulation_settings.create_observation_simulators(
                observation_settings_list, bodies
        )

        # Compute residuals
        observations.compute_residuals_and_dependent_variables(
            observation_collection, observation_simulators, bodies
        )
        prefiltered_residuals_RAD = observation_collection.get_concatenated_residuals()
        prefiltered_residuals_ARCSEC = np.rad2deg(prefiltered_residuals_RAD) * 3600.0

        # # Normalise (prefit) residuals
        # assumed_obs_err_RAD = np.sqrt(np.reciprocal(observation_collection.concatenated_weights))
        # assumed_obs_err_ARCSEC = np.rad2deg(assumed_obs_err_RAD) * 3600
        # prefiltered_normalised_residuals = prefiltered_residuals_ARCSEC / assumed_obs_err_ARCSEC
        # RMS_prefiltered_normalised_residuals = isu.rms(prefiltered_normalised_residuals)

        # Filter outliers from ObservationCollection object
        residual_filter_cutoff_RAD = np.deg2rad(residual_filter_cutoff_ARCSEC / 3600.0)
        residual_filter = observations.observations_processing.observation_filter(
            observations.observations_processing.ObservationFilterType.residual_filtering,
            residual_filter_cutoff_RAD,
        )
        observation_collection.filter_observations(residual_filter)
        observation_collection.remove_empty_observation_sets()
        postfiltered_residuals_RAD = observation_collection.get_concatenated_residuals()
        postfiltered_residuals_ARCSEC = np.rad2deg(postfiltered_residuals_RAD) * 3600.0

        # Normalise (postfit) residuals
        # ... get weights from filtered observation collection
        # postfiltered_normalised_residuals = postfiltered_residuals_ARCSEC /
        # RMS_postfiltered_normalised_residuals = isu.rms(postfiltered_normalised_residuals)

        # Filtering / pre-fit data
        RMS_prefiltered_residuals_ARCSEC = isu.rms(np.rad2deg(prefiltered_residuals_RAD) * 3600)
        RMS_postfiltered_residuals_ARCSEC = isu.rms(np.rad2deg(postfiltered_residuals_RAD) * 3600)
        n_obs_filtered_out = int((
            len(prefiltered_residuals_RAD) - len(observation_collection.concatenated_weights)
            ) / 2 )
        percent_obs_filtered_out = (n_obs_filtered_out / (len(prefiltered_residuals_RAD)/2)) * 100
        filtered_obs_arc_YEAR = (
            max(observation_collection.concatenated_times) - \
            min(observation_collection.concatenated_times)
            ) / cnst.JULIAN_YEAR
        change_in_obs_arc_due_to_filtering_YEAR = filtered_obs_arc_YEAR - obs_arc_YEAR

        # Save filter data for current cutoff value
        if cutoff_idx == 0:
            body_res_dict[f'RMS_prefiltered'] = RMS_prefiltered_residuals_ARCSEC,
        body_res_dict[f'{cutoff_idx}_cutoff'] = residual_filter_cutoff_ARCSEC,
        body_res_dict[f'{cutoff_idx}_arc'] = filtered_obs_arc_YEAR,
        body_res_dict[f'{cutoff_idx}_delta_arc'] = change_in_obs_arc_due_to_filtering_YEAR,
        body_res_dict[f'{cutoff_idx}_n_obs_filtered_out'] = n_obs_filtered_out,
        body_res_dict[f'{cutoff_idx}_pc_obs_filtered_out'] = percent_obs_filtered_out,
        body_res_dict[f'{cutoff_idx}_RMS_postfiltered'] = RMS_postfiltered_residuals_ARCSEC,

        T_END_cutoff = tm.perf_counter()
        print_verbose(f"Done ({T_END_cutoff - T_START_cutoff:.3f}s)")

    # Add data for current body to dataframe
    df = pd.DataFrame(data=body_res_dict, index=[body_idx])
    simulated_residual_df = pd.concat([simulated_residual_df, df])
    print_default(f"Residual simulations done ({T_END_cutoff - T_END_setup:.3f}s)")

    T_END_body = tm.perf_counter()
    print_default(f"BODY DONE: {T_END_body - T_START_body:.3f}s")

    if ((body_idx % 8) == 0) and (body_idx != 0):
        update_timestamp = dttm.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        cumulative_time = T_END_body - T_START_loop
        average_time_per_body = cumulative_time / (body_idx+1)
        bodies_remaining = len(body_to_test_list) - (body_idx+1)
        predicted_time_remaining = bodies_remaining * average_time_per_body
        print_default(f"\n\nUPDATE: | {update_timestamp}")
        print_default(f"Cumulative time: {(cumulative_time)/60:.2f} min")
        print_default(f"Bodies remaining: {bodies_remaining}")
        print_default(f"Predicted time remaining: {predicted_time_remaining/60:.2f} min\n")

# Save df to a file
simulated_residual_df.to_csv(
    run_output_data_save_path + f"outputs.csv"
)

# Timing, printing
print_default(f"\n\n\n### LOOP DONE | {(T_END_body - T_START_loop)/60:.2f} min")



#%% P.1 Histogram of % rejected observations
### P.1 Histogram of % rejected observations
#region P.1 Hist. rej. obs.

# Use data stored in local variables
# data_load_path = isu.output_data_dir + residual_cutoff_selection_dir + timestamp_dir
# outputs_df = simulated_residual_df
# cutoff_list = residual_filter_cutoff_ARCSEC_list

# Load data from files
timestamp_dir = "260517-191551/"
data_load_path = isu.output_data_dir + residual_cutoff_selection_dir + timestamp_dir
outputs_df = pd.read_csv(
    data_load_path + f"outputs.csv",
    index_col = 0
)
cutoff_list = isu.load_with_pickle(data_load_path, "residual_filter_cutoff_ARCSEC_list.p")


## PLOT
# Plot output
colormode = 0
pb = 'show'
# pb = 'save'
plots_save_path = isu.plot_dir + residual_cutoff_selection_dir + timestamp_dir
if not os.path.exists(plots_save_path):
    os.makedirs(plots_save_path)

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram: % of observations filtered out by cutoff value
max_pc_cutoff = max(outputs_df['0_pc_obs_filtered_out'])
opt_bins = np.arange(0, max_pc_cutoff+0.5, 0.5)
# opt_log_bins = np.logspace(
#     np.log10(min(opt_parameter)),
#     np.log10(max(opt_parameter)),
#     20
# )

# Plot curve for literature
outputs_df['lit_pc_rej'] = (outputs_df.n_opt_obs_rej_lit / outputs_df.n_opt_obs_lit) * 100
ax.hist(
    outputs_df['lit_pc_rej'],
    bins = opt_bins,
    label = f'NEOCC',
    ec = isu.cmap_custom10[colormode](9),
    alpha = 1.0,
    histtype = 'step',
    # linestyle = isu.linestyles[cutoff_idx//10],
    linewidth = 3,
    zorder = 100,
)
mean_pc_opt_rej = np.mean(outputs_df['lit_pc_rej'])
opt_parameter_nz = outputs_df['lit_pc_rej'].loc[outputs_df['lit_pc_rej'] != 0]
n_opt_rej_zero = len(outputs_df) - len(opt_parameter_nz)
print(f"NEOCC:")
print(f"N for which ZERO observations were rejected: {n_opt_rej_zero} ({(n_opt_rej_zero/len(outputs_df))*100:.3f}%)")
print(f"Mean % of observations rejected: {mean_pc_opt_rej:.3f}%")

selected_values_to_plot = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
selected_values_to_highlight = [3.0]

plot_idx = 0
for cutoff_idx, cutoff_ARCSEC in enumerate(residual_filter_cutoff_ARCSEC_list):
    opt_parameter = outputs_df[f'{cutoff_idx}_pc_obs_filtered_out']
    mean_pc_opt_rej = np.mean(opt_parameter)
    opt_parameter_nz = opt_parameter.loc[opt_parameter != 0]
    n_opt_rej_zero = len(outputs_df) - len(opt_parameter_nz)

    print(f"\nCutoff value: {cutoff_ARCSEC} arcsec")
    print(f"N for which ZERO observations were rejected: {n_opt_rej_zero} ({(n_opt_rej_zero/len(outputs_df))*100:.3f}%)")
    print(f"Mean % of observations rejected: {mean_pc_opt_rej:.3f}%")

    if cutoff_ARCSEC not in selected_values_to_plot:
        continue

    ax.hist(
        opt_parameter,
        bins = opt_bins,
        label = f'Cutoff: {cutoff_ARCSEC} arcsec',
        ec = isu.cmap_custom10[colormode](plot_idx%9),
        alpha = 1.0 if cutoff_ARCSEC in selected_values_to_highlight else 0.5,
        histtype = 'step',
        linestyle = isu.linestyles[plot_idx%2],
        linewidth = 3 if cutoff_ARCSEC in selected_values_to_highlight else 2,
        zorder = 100,
    )

    plot_idx += 1


xmin, xmax = ax.get_xlim()
ax.set_xlim((-0.1,22))
ax.set_yscale('log')

# Legend, labels
ax.legend(
    loc = 'best',
    markerscale = 1.5,
    fontsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax.set_ylabel("Count")
ax.set_xlabel("Percent of observations rejected per body")

# Plot output
if (pb == 'show'):
    plt.show(block=False)
    plt.pause(0.001)
elif (pb == 'save'):
    fig_name = f"hist_observations-rejected-by-cutoff"
    fig_name += "__darkmode.pdf" if colormode else ".pdf"
    print(fig_name)
    plt.savefig(plots_save_path + fig_name, format="pdf")
    plt.close()