"""
This script is for processing the results of large runs of est_w_mpc_large.py,
primarily for the mid-term presentation
"""
#%% Imports & Loading
### Imports & Loading

import os
import json
import time as tm
import numpy as np
import pandas as pd
import IS_utils as isu
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import tudatpy.constants as cnst
from tudatpy.data.mpc import BatchMPC
from tudatpy.astro.time_representation import DateTime

# Define dataset to load
# dataset_to_load = 'other'
# dataset_to_load = 'RK6'
# dataset_to_load = 'draft'
dataset_to_load = 'final'

# Inputs
if dataset_to_load == 'RK6':
    timestamp_folder_names = [
        '260422-235854_RK6/', # full single run, RK6, ∆t=1d
    ]
elif dataset_to_load == 'draft':
    timestamp_folder_names = [
        # Draft run: RK8/10, ∆t=1d/12h/3h, residual filtering cutoff = 5 arcsec
        '260517-045652_0-3/',
        '260517-051531_4/',
        '260517-053036_5-154/',
        '260517-213027_155-347/', # residual filtering cutoff = 4 arcsec
    ]
elif dataset_to_load == 'final':
    timestamp_folder_names = [
        # Final run: RK8/10, ∆t=6h/3h, residual filtering cutoff = 3 arcsec
        '260611-172545_0-4/',       #  0.5 hours,  0.4 GB   # cumulative
        '260611-185119_5-52/',      #  5.6 hours,  4.2 GB   #  6.1 hours,  4.6 GB
        '260612-005119_53-158/',    # 12.8 hours,  9.6 GB   # 19.0 hours, 14.2 GB
        '260612-214607_159-171/',   #  1.7 hours,  1.2 GB   # 20.6 hours, 15.4 GB
        '260613-020313_172-300/',   # 12.7 hours,  9.9 GB   # 33.3 hours, 25.3 GB
        '260614-032235_301-347/',   #  5.3 hours,  3.9 GB   # 38.3 hours, 29.1 GB
        '260614-132323_161_223/',   #  0.2 hours,  0.2 GB   # 38.5 hours, 29.2 GB
        '260616-182532_10_57/',     #  1.4 hours,  0.9 GB   # 40.2 hours, 30.1 GB
    ]
else:
    timestamp_folder_names = [
        # '260202-104814_0-57/',
        # '260202-113408_58-82/',
        # '260202-011257_83-183/',
        # '260202-121954_184-299/',
        # '260202-131126_300-308/',
        # '260202-132135_309-347/',
        #
        # # '260302-160041/'
        #
        # '260305-121018_prov_0-218/',
        # '260305-130313_prov_218-347/',
        #
        # '260318-233753_29outliers/',
        #
        # '260331-211327_0-4/',
        # '260331-213159_5-27/',
        # '260331-222221_28-46/',
        # '260331-231656_47-74/',
        # '260331-235450_75-96/',
        # '260401-004304_97-139/',
        # '260401-014722_140-179/',
        # '260401-023940_180-216/',
        # '260401-105841_217-260/',
        # '260401-114352_261-304/',
        # '260401-123328_305-344/',
        # '260401-133422_345-347/',
    ]

large_run_base_dir = "yarkovsky-est-w-mpc/_large-runs/"
# results_path = isu.plot_dir + large_run_base_dir # old runs
results_path = isu.output_data_dir + large_run_base_dir

# Load first set of results into a dataframe
outputs_df = pd.read_csv(
    results_path + timestamp_folder_names[0] + "_outputs.csv",
    index_col = 0
)
# Add a timestamp_dir column
outputs_df['timestamp_dir'] = [timestamp_folder_names[0]]*len(outputs_df)
# Load the rest of the results and add them to the same dataframe
for run_folder_name in timestamp_folder_names[1:]:
    run_df = pd.read_csv(
        results_path + run_folder_name + "_outputs.csv",
        index_col = 0
    )
    # Make sure 'flag' column is read as strings
    run_df = run_df.astype({"flag": "str"})
    run_df['timestamp_dir'] = [run_folder_name]*len(run_df)
    outputs_df = pd.concat([outputs_df, run_df], ignore_index=True)
# Drop duplicate bodies (keeping the last result of any body that was re-run)
outputs_df.drop_duplicates(
    subset = ['body'],
    keep = 'last',
    inplace = True,
)

# Load all error files into single dict
error_dict = {}
for idx, run_folder_name in enumerate(timestamp_folder_names):
    try:
        with open(results_path + run_folder_name + "_errors.json", 'r') as json_fp:
            run_error_dict = json.load(json_fp)[0]
        error_dict = error_dict | run_error_dict
    except FileNotFoundError as ex:
        print(f"WARNING: {type(ex).__name__} - {ex}")


#%% Processing 1. get dependent variable stats, check convergence
### Processing 1. get dependent variable stats, check convergence

# For results before 'final':
# I forgot to save certain interesting things about the Earth-target distance
# dependent variable directly to the outputs .csv, so here I will load the
# relevant data and calculate them manually. Also saving some SBDB data that I
# hadn't saved before

T_START_extra_stats = tm.perf_counter()
est_w_mpc_plot_dir = 'yarkovsky-est-w-mpc/_large-runs/'

# Initialise empty lists
convergence = np.zeros(len(outputs_df)) * np.nan
mean_abs_norm_res_est = np.zeros(len(outputs_df)) * np.nan
max_abs_norm_res_est = np.zeros(len(outputs_df)) * np.nan
A2_JPL_AU_per_DAYsq = np.zeros(len(outputs_df)) * np.nan
A2_unc_JPL_AU_per_DAYsq = np.zeros(len(outputs_df)) * np.nan
earth_MOID_AU = np.zeros(len(outputs_df)) * np.nan
RMS_norm_res_JPL = np.zeros(len(outputs_df)) * np.nan
n_obs_used_JPL = np.zeros(len(outputs_df)) * np.nan
n_dop_obs_used_JPL = np.zeros(len(outputs_df)) * np.nan
n_del_obs_used_JPL = np.zeros(len(outputs_df)) * np.nan
obs_arc_JPL_DAY = np.zeros(len(outputs_df)) * np.nan

R_RA_res_norm, p_RA_res_norm = np.zeros(len(outputs_df)) * np.nan, np.zeros(len(outputs_df)) * np.nan
R_DEC_res_norm, p_DEC_res_norm = np.zeros(len(outputs_df)) * np.nan, np.zeros(len(outputs_df)) * np.nan
R_res_norm, p_res_norm = np.zeros(len(outputs_df)) * np.nan, np.zeros(len(outputs_df)) * np.nan

close_pass_dict = {}
min_dist_Mercury_AU = np.zeros(len(outputs_df)) * np.nan
min_dist_Venus_AU = np.zeros(len(outputs_df)) * np.nan
min_dist_Earth_AU = np.zeros(len(outputs_df)) * np.nan
min_dist_Moon_AU = np.zeros(len(outputs_df)) * np.nan
min_dist_Mars_AU = np.zeros(len(outputs_df)) * np.nan
min_dist_Ceres_AU = np.zeros(len(outputs_df)) * np.nan
min_dist_Vesta_AU = np.zeros(len(outputs_df)) * np.nan
min_dist_Pallas_AU = np.zeros(len(outputs_df)) * np.nan

RMS_JPLH_ephem_pos_unc_mag_KM = np.zeros(len(outputs_df)) * np.nan
RMS_JPLH_ephem_unc_mag_vel = np.zeros(len(outputs_df)) * np.nan

if dataset_to_load != 'final':
    initial_earth_dist_AU = np.zeros(len(outputs_df)) * np.nan
    max_earth_dist_AU = np.zeros(len(outputs_df)) * np.nan
    min_earth_dist_AU = np.zeros(len(outputs_df)) * np.nan
    RMS_earth_dist_AU = np.zeros(len(outputs_df)) * np.nan

# 'c_' is short for 'current_'
for body_idx, (c_SPKID, c_timestamp_dir, c_body_dir) in enumerate(zip(
    outputs_df.SPKID,#[1:2],
    outputs_df.timestamp_dir,#[1:2],
    outputs_df.body_dir,#[1:2]
)):
    # Load data
    try:
        est_w_mpc_result_path = est_w_mpc_plot_dir + c_timestamp_dir + c_body_dir
    except TypeError as ex:
        continue
    output_data_path = isu.output_data_dir + est_w_mpc_result_path
    with open(output_data_path + "_inputs.json", 'r') as json_fp:
        input_dict = json.load(json_fp)[0]
    EO_data_path = output_data_path + "estimation_output/"
    EO_best_iteration = isu.load_with_pickle(EO_data_path, "best_iteration.p")
    final_residuals_normalised = isu.load_with_pickle(output_data_path, "final_residuals_normalised.p")
    close_pass_dict[c_SPKID] = isu.load_with_pickle(output_data_path, "distances_target_to_bodies_highlights_AU.p")
    body_ephem_unc_table = isu.load_with_pickle(isu.ephem_unc_dir, f"{c_SPKID}.p")
    propagation_start_J2000 = input_dict["propagation_start_J2000"]
    propagation_end_J2000 = input_dict["propagation_end_J2000"]

    # # SBDB query
    body_query = isu.query_SBDB(c_SPKID)

    # Save key data
    mean_abs_norm_res_est[body_idx] = np.mean(np.abs(final_residuals_normalised))
    max_abs_norm_res_est[body_idx] = np.max(np.abs(final_residuals_normalised))
    A2_JPL_AU_per_DAYsq[body_idx] = body_query['A2_AU_per_DAYsq']
    A2_unc_JPL_AU_per_DAYsq[body_idx] = body_query['A2_unc_AU_per_DAYsq']
    earth_MOID_AU[body_idx] = body_query['earth_MOID_AU']
    RMS_norm_res_JPL[body_idx] = body_query['RMS_norm_res']
    n_obs_used_JPL[body_idx] = body_query['n_obs_used']
    n_dop_obs_used_JPL[body_idx] = body_query['n_dop_obs_used']
    n_del_obs_used_JPL[body_idx] = body_query['n_del_obs_used']
    obs_arc_JPL_DAY[body_idx] = body_query['obs_arc_DAY']

    # Correlation of residuals over time
    batch_table_filtered = isu.load_with_pickle(output_data_path, "batch_table_filtered.p")
    obs_epochs = batch_table_filtered.epoch
    RA_residuals_normalised = final_residuals_normalised[::2]
    DEC_residuals_normalised = final_residuals_normalised[1::2]
    pearson_R_RA_res_norm = stats.pearsonr(obs_epochs, RA_residuals_normalised)
    pearson_R_DEC_res_norm = stats.pearsonr(obs_epochs, DEC_residuals_normalised)
    pearson_R_res_norm = stats.pearsonr(np.repeat(obs_epochs, 2), final_residuals_normalised)
    R_RA_res_norm[body_idx], p_RA_res_norm[body_idx] = pearson_R_RA_res_norm.statistic, pearson_R_RA_res_norm.pvalue
    R_DEC_res_norm[body_idx], p_DEC_res_norm[body_idx] = pearson_R_DEC_res_norm.statistic, pearson_R_DEC_res_norm.pvalue
    R_res_norm[body_idx], p_res_norm[body_idx] = pearson_R_res_norm.statistic, pearson_R_res_norm.pvalue

    # Closest planetary passes data
    min_dist_Mercury_AU[body_idx] = close_pass_dict[c_SPKID]['Mercury']['min_dist_AU']
    min_dist_Venus_AU[body_idx] = close_pass_dict[c_SPKID]['Venus']['min_dist_AU']
    min_dist_Earth_AU[body_idx] = close_pass_dict[c_SPKID]['Earth']['min_dist_AU']
    min_dist_Moon_AU[body_idx] = close_pass_dict[c_SPKID]['Moon']['min_dist_AU']
    min_dist_Mars_AU[body_idx] = close_pass_dict[c_SPKID]['Mars']['min_dist_AU']
    min_dist_Ceres_AU[body_idx] = close_pass_dict[c_SPKID]['Ceres']['min_dist_AU']
    min_dist_Vesta_AU[body_idx] = close_pass_dict[c_SPKID]['Vesta']['min_dist_AU']
    min_dist_Pallas_AU[body_idx] = close_pass_dict[c_SPKID]['Pallas']['min_dist_AU']

    # Convert time columns in JPLH ephemeris table
    body_ephem_unc_table["ephemeris_time"] = [
        DateTime.from_julian_day(jd)
        for jd in body_ephem_unc_table["datetime_jd"]
    ]
    body_ephem_unc_table["ephemeris_time_J2000"] = [
        dt.to_epoch()
        for dt in body_ephem_unc_table["ephemeris_time"]
    ]
    # Keep only the rows of the JPLH ephemeris table corresponding to the target
    # body's propagation period
    mask_after_window_start = np.where(
        body_ephem_unc_table["ephemeris_time_J2000"] > propagation_start_J2000, True, False
    )
    mask_before_window_end = np.where(
        body_ephem_unc_table["ephemeris_time_J2000"] < propagation_end_J2000, True, False
    )
    mask_within_obs_range = np.logical_and(mask_after_window_start, mask_before_window_end)
    body_ephem_unc_table = body_ephem_unc_table[mask_within_obs_range]
    # Get RMS of S_r^JPLH and S_v^JPLH
    ephem_pos_unc_KM = np.array([
            body_ephem_unc_table["x_s"],
            body_ephem_unc_table["y_s"],
            body_ephem_unc_table["z_s"],
    ], dtype=float).T / (3 * 1000) # uncertainties are in metres, and are 3sigma, so convert to 1sigma in km
    ephem_pos_unc_mag_KM = np.linalg.norm(ephem_pos_unc_KM, axis=1)
    RMS_JPLH_ephem_pos_unc_mag_KM[body_idx] = isu.rms(ephem_pos_unc_mag_KM)
    #
    ephem_vel_unc = np.array([
            body_ephem_unc_table["vx_s"],
            body_ephem_unc_table["vy_s"],
            body_ephem_unc_table["vz_s"],
    ], dtype=float).T / 3 # convert to 1sigma
    ephem_vel_unc_mag = np.linalg.norm(ephem_vel_unc, axis=1)
    RMS_JPLH_ephem_unc_mag_vel[body_idx] = isu.rms(ephem_vel_unc_mag)

    if dataset_to_load != 'final':
        # Extract Earth-target distance from best iteration's EstimationOutput
        # Path to SingleArcVariationalSimulationResults object
        iter_data_path = EO_data_path + f'simulation_results_per_iteration_{EO_best_iteration}/'
        # Path to SingleArcSimulationResults object
        dyn_result_data_path = iter_data_path + f'dynamics_results/'
        dependent_variable_history = isu.load_with_pickle(dyn_result_data_path, "dependent_variable_history.p")
        # Calculate distance magnitudes
        dependent_variables_array = np.array(list(dependent_variable_history.values()))
        position_of_target_relative_to_earth = dependent_variables_array[:, 0:3]
        target_earth_distance_AU = np.linalg.norm(position_of_target_relative_to_earth, axis=1) / cnst.ASTRONOMICAL_UNIT
        # Save key data
        max_earth_dist_AU[body_idx] = target_earth_distance_AU.max()
        min_earth_dist_AU[body_idx] = target_earth_distance_AU.min()
        RMS_earth_dist_AU[body_idx] = isu.rms(target_earth_distance_AU)
        initial_earth_dist_AU[body_idx] = target_earth_distance_AU[0]

    ## Convergence checking
    # Extract residual history, parameter history and formal errors from EstimationOutput
    with open(output_data_path + "_inputs.json", 'r') as json_fp:
        # Read and extract inputs from "_inputs.json"
        input_dict = json.load(json_fp)[0]
    n_iterations = input_dict["number_of_pod_iterations"]
    residual_history = isu.load_with_pickle(EO_data_path, "residual_history.p")
    residual_history_ARCSEC = np.rad2deg(residual_history) * 3600
    RMS_residual_history_ARCSEC = [isu.rms(residual_history_ARCSEC[:,iter_idx]) for iter_idx in range(n_iterations)]
    delta_RMS_residual_history_ARCSEC = np.diff(RMS_residual_history_ARCSEC) # change in RMS of residuals for each iteration after the first, relative to the previous iteration

    parameter_history = isu.load_with_pickle(EO_data_path, "parameter_history.p") # contains parameter vectors used for each iteration, plus an additional one for the 'next' iteration, which has not been performed
    formal_errors = isu.load_with_pickle(EO_data_path, "formal_errors.p") # formal errors (of parameters) of best iteration
    delta_parameter_history = np.diff(parameter_history, axis=1) # change in parameter vector for each iteration after the first, relative to the previous iteration
    delta_initial_position_history = np.linalg.norm(delta_parameter_history[0:3,:], axis=0)
    delta_initial_velocity_history = np.linalg.norm(delta_parameter_history[3:6,:], axis=0)
    delta_A2_history = delta_parameter_history[6,:]

    # Check if estimation converged by best iteration
    # Check 1: ∆RMS residual between 2nd last iter and last iter is less than 0.001 arcsec
    delta_RMS_residual_best_ARCSEC = delta_RMS_residual_history_ARCSEC[-1]
    residuals_convergence_check = np.abs(delta_RMS_residual_best_ARCSEC) < 0.001
    # Check 2: ∆parameter between best iter and next iter is less than 0.1%
    # of the formal error of that parameter
    delta_parameter_best = delta_parameter_history[:,EO_best_iteration]
    normalised_delta_parameters = delta_parameter_best / formal_errors
    parameters_convergence_check = np.abs(normalised_delta_parameters) < 0.001
    # Check 3: If the best iteration was the last iteration AND the absolute
    # change in initial position, velocity, or A2 (compared to the next
    # iteration) is greater than a certain value, then the estimation didn't get
    # to 'try out' this significant parameter change, so marked as not converged
    absolute_change_check = True
    if EO_best_iteration == n_iterations-1:
        if np.abs(delta_initial_position_history[EO_best_iteration]) > 1.0: # 1 meter
            absolute_change_check = False
            print(f"{body_query['longname']} ∆pos_best = {delta_initial_position_history[EO_best_iteration]}")
        if np.abs(delta_initial_velocity_history[EO_best_iteration]) > 1e-7: # 100 nanometer/s
            absolute_change_check = False
            print(f"{body_query['longname']} ∆vel_best = {delta_initial_velocity_history[EO_best_iteration]}")
        if np.abs(delta_A2_history[EO_best_iteration]) > 1e-17: # 10 attometer/s^2
            absolute_change_check = False
            print(f"{body_query['longname']} ∆A2_best = {delta_A2_history[EO_best_iteration]}")
    # If all checks passed, mark body's estimation as converged
    if residuals_convergence_check and parameters_convergence_check.all() and absolute_change_check:
        convergence[body_idx] = True
    else:
        convergence[body_idx] = False
        # print(f"{body_query['longname']} failed convergence check")

    # Print occasional progress updates
    if (body_idx % 17) == 0:
        pc_done = ((body_idx+1) / len(outputs_df)) * 100
        print(f"Progress: {body_idx+1}/{len(outputs_df)} ({pc_done:.1f}%)")


# Convert A2 and A2_unc from AU/day^2 to m/s^2
A2_JPL = A2_JPL_AU_per_DAYsq * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2)
A2_unc_JPL = A2_unc_JPL_AU_per_DAYsq * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2)
# Convert obs_arc from DAY to YEAR
obs_arc_JPL_YEAR = obs_arc_JPL_DAY / 365.2425

# Add columns to outputs_df
outputs_df['convergence'] = convergence
outputs_df['mean_abs_norm_res_est'] = mean_abs_norm_res_est
outputs_df['max_abs_norm_res_est'] = max_abs_norm_res_est
outputs_df['A2_JPL'] = A2_JPL
outputs_df['σ_A2_JPL'] = A2_unc_JPL
outputs_df['SNR_A2_JPL'] = np.abs(outputs_df.A2_JPL / outputs_df.σ_A2_JPL)
outputs_df['n_obs_used_postfilter'] = outputs_df.n_total_obs_used - outputs_df.n_obs_filtered_out
outputs_df['earth_MOID_AU'] = earth_MOID_AU
outputs_df['RMS_norm_res_JPL'] = RMS_norm_res_JPL
outputs_df['n_obs_used_JPL'] = n_obs_used_JPL
outputs_df['n_dop_obs_used_JPL'] = n_dop_obs_used_JPL
outputs_df['n_del_obs_used_JPL'] = n_del_obs_used_JPL
outputs_df['obs_arc_JPL_YEAR'] = obs_arc_JPL_YEAR

outputs_df['R_RA_res_norm'], outputs_df['p_RA_res_norm'] = R_RA_res_norm, p_RA_res_norm
outputs_df['R_DEC_res_norm'], outputs_df['p_DEC_res_norm'] = R_DEC_res_norm, p_DEC_res_norm
outputs_df['R_res_norm'], outputs_df['p_res_norm'] = R_res_norm, p_res_norm

outputs_df['min_dist_Mercury_AU'] = min_dist_Mercury_AU
outputs_df['min_dist_Venus_AU'] = min_dist_Venus_AU
outputs_df['min_dist_Earth_AU'] = min_dist_Earth_AU
outputs_df['min_dist_Moon_AU'] = min_dist_Moon_AU
outputs_df['min_dist_Mars_AU'] = min_dist_Mars_AU
outputs_df['min_dist_Ceres_AU'] = min_dist_Ceres_AU
outputs_df['min_dist_Vesta_AU'] = min_dist_Vesta_AU
outputs_df['min_dist_Pallas_AU'] = min_dist_Pallas_AU

outputs_df['RMS_JPLH_ephem_pos_unc_mag_KM'] = RMS_JPLH_ephem_pos_unc_mag_KM
outputs_df['RMS_JPLH_ephem_vel_unc_mag'] = RMS_JPLH_ephem_unc_mag_vel

if dataset_to_load != 'final':
    # Add columns to outputs_df
    outputs_df['max_earth_dist_AU'] = max_earth_dist_AU
    outputs_df['min_earth_dist_AU'] = min_earth_dist_AU
    outputs_df['RMS_earth_dist_AU'] = RMS_earth_dist_AU
    outputs_df['initial_earth_dist_AU'] = initial_earth_dist_AU

T_END_extra_stats = tm.perf_counter()
pc_done = ((body_idx+1) / len(outputs_df)) * 100
print(f"Progress: {body_idx+1}/{len(outputs_df)} ({pc_done:.1f}%)")
print(f"DONE: {T_END_extra_stats - T_START_extra_stats:.3f}s")

outputs_df['n_opt_obs_used_NEOCC'] = outputs_df.n_opt_obs_lit - outputs_df.n_opt_obs_rej_lit
outputs_df['n_rad_obs_used_NEOCC'] = outputs_df.n_rad_obs_lit - outputs_df.n_rad_obs_rej_lit
outputs_df['n_rad_obs_used_JPL'] = outputs_df.n_dop_obs_used_JPL + outputs_df.n_del_obs_used_JPL


#%% Processing 2. categorise results
### Processing 2. categorise results

if dataset_to_load == 'RK6':
    # Only use for RK6 results:
    successes = outputs_df.loc[outputs_df['flag'] == "1"]
    rejects = successes.loc[successes.SNR_est < 3.0]
    successes = successes.loc[successes.SNR_est > 3.0]
    divergers = outputs_df.loc[outputs_df['flag'] == "div"]
    with_data = pd.concat([successes, rejects, divergers]).sort_index()

if dataset_to_load == 'draft':
    # Count number of each flag in dataframe (RK8+ results)
    successes = outputs_df.loc[outputs_df['flag'] == "1"]
    low_SNR = outputs_df.loc[outputs_df['flag'] == "rej"]
    RMSNR_cutoff = 1.0
    high_RMS = successes.loc[successes.RMS_norm_res_est >= RMSNR_cutoff]
    successes = successes.loc[successes.RMS_norm_res_est < RMSNR_cutoff]
    rejects = pd.concat([low_SNR, high_RMS]).sort_index()
    divergers = outputs_df.loc[outputs_df['flag'] == "div"]
    with_data = pd.concat([successes, rejects, divergers]).sort_index()
    good_fits = pd.concat([successes, low_SNR]).sort_index()

if dataset_to_load == 'final':
    # Check if mean of normalised residuals is statistically higher than what
    # would be expected The absolute value of normalised residuals (|NR|) are
    # expected to follow a half-normal distribution, which has the following
    # expected value and variance:
    expected_NR = np.sqrt(2 / np.pi)
    variance_NR = 1 - (2 / np.pi)
    for idx, body_row in outputs_df.iterrows():
        n_samples = body_row.n_total_obs_used - body_row.n_obs_filtered_out
        # According to the Central Limit Theorem, with sufficiently large
        # n_samples, the mean of samples taken from the |NR| distribution should
        # follow a Gaussian distribution with mu = expected_NR and variance =
        # variance_NR / n_samples
        variance_CLT = variance_NR / n_samples
        sigma_CLT = np.sqrt(variance_CLT)
        # If the mean of normalised residuals is more than 3sigma larger than
        # the expected value, then there is only a 0.135% chance that this is
        # due to random chance. Therefore, reject any estimations where this is true
        mu_CLT_plus_3sigma_CLT = expected_NR + 3*sigma_CLT
        if body_row.mean_abs_norm_res_est > mu_CLT_plus_3sigma_CLT:
            print(body_row.body)
            outputs_df.at[idx, "flag"] = "rej2"

    low_SNR = outputs_df.loc[outputs_df['flag'] == "rej"]
    high_RMS = outputs_df.loc[outputs_df['flag'] == "rej2"]
    nonconverged = outputs_df.loc[outputs_df['convergence'] == 0]
    rejects = pd.concat([low_SNR, high_RMS, nonconverged]).sort_index()
    divergers = outputs_df.loc[outputs_df['flag'] == "div"]
    successes = outputs_df.loc[(outputs_df['flag'] == "1") & (outputs_df['convergence'] == 1)]
    with_data = pd.concat([successes, rejects, divergers]).sort_index()
    good_fits = pd.concat([successes, low_SNR]).sort_index()


errors = outputs_df.loc[outputs_df['flag'] == "error"]
skips = outputs_df.loc[outputs_df['flag'] == "skip"]
unnums = outputs_df.loc[outputs_df['flag'] == "unnum"]
no_data = pd.concat([errors, skips, unnums]).sort_index()

n_total = len(outputs_df)
n_successes = len(successes)
n_low_SNR = len(low_SNR)   # comment out for RK6 results
n_high_RMS = len(high_RMS) # comment out for RK6 results
n_nonconverged = len(nonconverged)
n_rejects = len(rejects)
n_divergers = len(divergers)
n_with_data = len(with_data)
n_errors = len(errors)
n_skips = len(skips)
n_unnums = len(unnums)
n_no_data = len(no_data)

print(f"Number of asteroids: {n_total}")
print(f"Number of With Data: {n_with_data} ({(n_with_data/n_total)*100:.4g}%)")
print(f"  - of which successes: {n_successes} ({(n_successes/n_total)*100:.4g}%)")
print(f"  - of which rejects: {n_rejects} ({(n_rejects/n_total)*100:.4g}%)")
print(f"  - of which divergers: {n_divergers} ({(n_divergers/n_total)*100:.4g}%)")
print(f"Number of No Data: {n_no_data} ({(n_no_data/n_total)*100:.4g}%)")
print(f"  - of which unnums: {n_unnums} ({(n_unnums/n_total)*100:.4g}%)")
print(f"  - of which errors: {n_errors} ({(n_errors/n_total)*100:.4g}%)")
print(f"  - of which skips: {n_skips} ({(n_skips/n_total)*100:.4g}%)")

print(f"\nRejects:")
print(f"  - of which rejected due to low SNR: {n_low_SNR} ({(n_low_SNR/n_total)*100:.4g}%)")    # comment out for RK6 results
print(f"  - of which rejected due to high RMS: {n_high_RMS} ({(n_high_RMS/n_total)*100:.4g}%)") # comment out for RK6 results
print(f"  - of which rejected due to non-convergence: {n_nonconverged} ({(n_nonconverged/n_total)*100:.4g}%)") # comment out for RK6 results

print(f"\n[All percentages are out of the total number of asteroids in the set ({n_total})]")

opposite_A2_signs = successes.loc[np.sign(successes['A2_est']) != np.sign(successes['A2_lit'])]
print(f"\nNumber of successes where est. and lit. A2 values have opposite sign: {len(opposite_A2_signs)}")


# REJECTS:
# Low SNR:
    # If the SNR of the estimation of A2 is less than 3.0, this estimation is
    # rejected. This is the same as what Fenucci et al do
# High RMS:
    # If the RMS of the normalised residuals is greater than 2.0, this
    # estimation is rejected. Assuming the weights applied to the observations
    # are accurate reflections of their error/standard deviation, the expected
    # normalised residual of any given observation is the mean of a half-normal
    # distribution with sigma=1, which is 0.797885. Only 4.28% of observations
    # can be expected to have residuals between 2 and 3, and only 0.27% can be
    # expected to have residuals larger than 3. If the RMS of all of normalised
    # residuals for a certain body is over 2.5 times the expected mean, then
    # probably something has gone wrong with this fit.

# DIVERGERS:
# UPDATE: this flag has been removed for most recent runs
# If the change in initial position from the first iteration to the best
# iteration is more than 10,000 km, the estimation is marked as diverged.

# ERRORS:
# SpiceSPKINSUFFDATA for body '101955' (Bennu) - issues with Bennu's ephemeris file (both SPK and JPLH)
# KeyError for body '4528 07' (fenucci index 252) - I think this should be
# '452807' # UPDATE: Fixed
# KeyError for body '4888 03' (fenucci index 252) - I think this should be
# '488803' # UPDATE: Fixed
# 8x RuntimeErrors for bodies: ['401885', '488453', '277810', '37655', '4769', '455415', '525484', '443837']
#     All have messages similar to the following:
#     Error computing observation of type AngularPosition with link ends transmitter: (e1885); receiver: (Earth, 644) at epoch 52613189.526638.
#     Original error: Error in light-time solution, computing light time with reference epoch 52613189.526638 (DateTime: 2001-09-01 10:46:29.526638388633728) at receiver.
#     Original error: Error in ephemeris, requesting state at epoch 52613189.526638.
#     Original error: Error when setting global state of e1885 from ephemeris.
#     Original error: Error in ephemeris, requesting state at epoch 52613189.526638.
#     Original error: Error in tabulated ephemeris.
#     Original error: Error in interpolator, requesting data point outside of boundaries, requested data at 52613189.526638 but limit values are 52617600.000000 and 720576000.000000

# SKIPS:
# UPDATE: now only 101955 Bennu is skipped because SPK/JPLH ephemerides are weird
# Some asteroids were manually skipped because they somehow caused the
# estimation process to get stuck or errors when calculating Correlation Matrix,
# causing my Jupyter kernel to crash
# I re-ran these and actually they diverge but they don't crash as long as I
# don't try to make a correlation matrix.
# Error message when I do include corrlation matrix and run in terminal (not Jupyter):
#     qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""
#     python: /home/ronanjmc/anaconda3/envs/tudat-space-v1.0-dev/include/eigen3/Eigen/src/Core/DenseBase.h:262:
#     void Eigen::DenseBase<Eigen::Block<Eigen::Matrix<double, -1,
#     -1>>>::resize(Index, Index) [Derived = Eigen::Block<Eigen::Matrix<double,
#     -1, -1>>]: Assertion `rows == this->rows() && cols == this->cols() &&
#     "DenseBase::resize() does not actually allow to resize."' failed. Aborted
#     (core dumped)

# UNNUMS:
# UPDATE: This is fixed now. Asteroids with only provisional designations can be
# run just fine
# I haven't updated to more recent version of Tudatpy and I think some parts of
# my code only currently work if a permanent designation number is available, so
# these have been skipped for now. I think after updating and making some small
# changes to the code, these should not be a problem.


#%% Looking at non-convergers

not_converged = with_data.loc[with_data.convergence == 0]
# 'c_' is short for 'current_'
for body_idx, (c_SPKID, c_timestamp_dir, c_body_dir) in enumerate(zip(
    not_converged.SPKID,#[1:2],
    not_converged.timestamp_dir,#[1:2],
    not_converged.body_dir,#[1:2]
)):
    try:
        est_w_mpc_result_path = est_w_mpc_plot_dir + c_timestamp_dir + c_body_dir
    except TypeError as ex:
        continue
    output_data_path = isu.output_data_dir + est_w_mpc_result_path
    EO_data_path = output_data_path + "estimation_output/"
    EO_best_iteration = isu.load_with_pickle(EO_data_path, "best_iteration.p")

    # SBDB query
    body_query = isu.query_SBDB(c_SPKID)

    # Extract residual history, parameter history and formal errors from EstimationOutput
    with open(output_data_path + "_inputs.json", 'r') as json_fp:
        # Read and extract inputs from "_inputs.json"
        input_dict = json.load(json_fp)[0]
    n_iterations = input_dict["number_of_pod_iterations"]
    residual_history = isu.load_with_pickle(EO_data_path, "residual_history.p")
    residual_history_ARCSEC = np.rad2deg(residual_history) * 3600
    RMS_residual_history_ARCSEC = [isu.rms(residual_history_ARCSEC[:,iter_idx]) for iter_idx in range(n_iterations)]
    delta_RMS_residual_history_ARCSEC = np.diff(RMS_residual_history_ARCSEC) # change in RMS of residuals for each iteration after the first, relative to the previous iteration

    parameter_history = isu.load_with_pickle(EO_data_path, "parameter_history.p") # contains parameter vectors used for each iteration, plus an additional one for the 'next' iteration, which has not been performed
    formal_errors = isu.load_with_pickle(EO_data_path, "formal_errors.p") # formal errors (of parameters) of best iteration
    delta_parameter_history = np.diff(parameter_history, axis=1) # change in parameter vector for each iteration after the first, relative to the previous iteration
    delta_initial_position_history = np.linalg.norm(delta_parameter_history[0:3,:], axis=0)
    delta_initial_velocity_history = np.linalg.norm(delta_parameter_history[3:6,:], axis=0)
    delta_A2_history = delta_parameter_history[6,:]

    # Check if estimation converged by best iteration
    # Check 1: ∆RMS residual between 2nd last iter and last iter is less than 0.001 arcsec
    delta_RMS_residual_best_ARCSEC = delta_RMS_residual_history_ARCSEC[-1]
    residuals_convergence_check = np.abs(delta_RMS_residual_best_ARCSEC) < 0.001
    # Check 2: ∆parameter between best iter and next iter is less than 1%
    # of the formal error of that parameter
    delta_parameter_best = delta_parameter_history[:,EO_best_iteration]
    normalised_delta_parameters = delta_parameter_best / formal_errors
    parameters_convergence_check = np.abs(normalised_delta_parameters) < 0.001


    if not residuals_convergence_check:
        print("\n\n")
        print(body_query['longname'])
        print(f"EO_best_iteration: {EO_best_iteration}")
        print(f"RMS_residual_history_ARCSEC: {RMS_residual_history_ARCSEC}")
        print(f"delta_RMS_residual_history_ARCSEC: {delta_RMS_residual_history_ARCSEC}")
        print(f"delta_RMS_residual_best_ARCSEC: {delta_RMS_residual_best_ARCSEC}")
        print(f"residuals_convergence_check: {residuals_convergence_check}")
        print(" ")

    if not parameters_convergence_check.all():
        print("\n\n")
        print(body_query['longname'])
        print(f"delta_initial_position_history: {delta_initial_position_history}")
        print(f"delta_initial_velocity_history: {delta_initial_velocity_history}")
        print(f"delta_A2_history: {delta_A2_history}")
        print(f"delta_parameter_best: {delta_parameter_best}")
        print(f"normalised_delta_parameters: {normalised_delta_parameters}")
        print(f"parameters_convergence_check: {parameters_convergence_check}")


#%% Common plotting settings
### Common plotting settings
#region Plot kwargs

colormode = 1
yark_results_plot_dir = f"yark-results-processing/{dataset_to_load}/"
plots_save_path = isu.plot_dir + yark_results_plot_dir

# Colour coding
# - full dataset: blue (0)
# - successes: green (2)
# - rejected due to low SNR: orange (1)
# - rejected due to high RMS: yellow (4)
# - rejected: all (3)
# - good fits: successes + low SNR: brown (6)
# - data from NEOCC: purple (9)
# - data from JPL SBDB: grey (7)

legend_kwargs = dict(
    markerscale = 1.5,
    fontsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
hist_contour_kwargs = dict(
    histtype = 'step',
    linewidth = 2,
    zorder = 2.5,
)
scatter_edgecolors = [isu.text_color[0], '#CCCCCC']
scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    marker = '.',
    s = 65,
    zorder = 2.5,
)
reject_scatter_kwargs = dict(
    linewidths = 1,
    marker = 'x',
    s = 20,
    zorder = 2.6,
)
errorbar_scatter_kwargs = dict(
    fmt = 'o',
    markersize = 2.5,
    alpha = 0.6 if colormode else 0.4,
    elinewidth = 1,
)
text_bbox = dict(
    facecolor = isu.legend_bg_color[colormode],
    edgecolor = isu.text_color[colormode],
    boxstyle = 'round',
    alpha = 0.4,
)
text_box_kwargs = dict(
    color = isu.text_color[colormode],
    fontsize = isu.tiny_font_size,
    family = 'monospace',
)
# k1 spans
k1_spans = [
    ( 0,  1),
    ( 1,  2),
    ( 2,  3),
    ( 3,  1e5),
]
k1_span_colors = [
    isu.cmap_custom10[colormode](2), # green
    isu.cmap_custom10[colormode](4), # yellow
    isu.cmap_custom10[colormode](1), # orange
    isu.cmap_custom10[colormode](3), # red
]

def output_plot(
    pb = 'show',
    colormode = 0,
    plots_save_path = plots_save_path,
    fig_name = 'figure',
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


#%% Body interestion info function
### Body interestion info function

def print_interesting_body_info(
    data_df,
    body_name,
    silent = False,
):
    bdf_query = isu.query_SBDB(body_name)
    bdf = data_df.loc[data_df.SPKID == int(bdf_query['SPKID'])].squeeze()

    if not silent:
        print(f"Body: {bdf.body}")
        print(f"SPKID: {bdf.SPKID}")
        print(f"a = {bdf.semimajoraxis_AU} AU")
        print(f"e = {bdf.eccentricity}")
        print(f"period = {bdf.orbitperiod_DAY} days")
        print(f"Integrator: {bdf.integrator}")
        print(f"Step size = {bdf.step_size_DAY} day")
        print(f"∆ observation arc (filtering): {bdf['∆obs_arc_due_to_filtering_YEAR']} year")
        print(f"Observation arc (post filter): {bdf.filtered_obs_arc_est_YEAR} year")
        print(f"Best iteration: {bdf.best_iteration}")
        print(f"Runtime: {bdf.runtime/60} min")
        print(f"Initial state change: {bdf.initial_state_change_KM} km")
        print(f"Body directory: {bdf.body_dir}")
        print(f"Timestamp directory: {bdf.timestamp_dir}")

        print(f"N obs used (Tudat) = {bdf.n_total_obs_used}")
        print(f"N obs filtered (Tudat) = {bdf.n_obs_filtered_out}")
        print(f"N obs opical used (NEOCC) = {bdf.n_opt_obs_used_NEOCC}")
        print(f"N obs radar used (NEOCC) = {bdf.n_rad_obs_used_NEOCC}")
        print(f"N obs radar used (JPL) = {bdf.n_rad_obs_used_JPL}")

        print(f"SNR A2 Tudat = {bdf.SNR_A2_est}")
        print(f"SNR A2 NEOCC = {bdf.SNR_lit}")
        print(f"SNR A2 JPL = {bdf.SNR_A2_JPL}")

        print(f"RMS O-C A2 Tudat (prefit) = {bdf.prefit_RMS_res_ARCSEC}")
        print(f"RMS O-C A2 Tudat (postfit) = {bdf.RMS_res_est}")

        print(f"RMSNR A2 Tudat = {bdf.RMS_norm_res_est}")
        print(f"RMSNR A2 NEOCC = {bdf.RMS_norm_res_lit}")
        print(f"RMSNR A2 JPL = {bdf.RMS_norm_res_JPL}")

        print(f"k1 NEOCC = {bdf.k1}")
        print(f"k2 NEOCC = {bdf.k2}")
        print(f"k1 JPL = {bdf.k1}")
        print(f"k2 JPL = {bdf.k2}")

        print(f"RMS FE_pos = {bdf.RMS_formal_error_pos_mag_KM} km")
        print(f"RMS TE_pos = {bdf.RMS_true_error_pos_mag_KM} km")
        print(f"RMS (FE/TE)_pos = {bdf.RMS_formal_error_pos_mag_KM / bdf.RMS_true_error_pos_mag_KM}")
        print(f"JPL RMS FE_pos = {bdf.RMS_JPLH_ephem_pos_unc_mag_KM} km")
        print(f"RMS FE_pos_Tudat / FE_pos_JPL = {bdf.RMS_formal_error_pos_mag_KM / bdf.RMS_JPLH_ephem_pos_unc_mag_KM}")

        print(f"RMS FE_vel = {bdf.RMS_formal_error_vel_mag_MM/1000} m/s")
        print(f"RMS TE_vel = {bdf.RMS_true_error_vel_mag_MM/1000} m/s")
        print(f"RMS (FE/TE)_vel = {bdf.RMS_formal_error_vel_mag_MM / bdf.RMS_true_error_vel_mag_MM}")



    return bdf

bdf = print_interesting_body_info(with_data, '2101', silent=False)


#%% 0. Colour testing
### 0. Colour testing
#region 0. Colour testing

import importlib
isu = importlib.reload(isu)

for cm in [0,1]:
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.67])*isu.half_width_INCH,
        facecolor = isu.fig_background[cm],
        layout = 'constrained',
    )
    ax = isu.set_default_2d_ax_settings(ax, cm)

    x = np.linspace(0,1,20)
    for idx in range(10):
        ax.plot(
            x,
            np.array([idx+1]*len(x)) + np.random.random(len(x)),
            label = str(idx),
            color = isu.cmap_custom10[cm](idx),
            linewidth = 2.0,
        )
    plt.show(block=False)
    plt.pause(0.001)

#%% H1. Histogram of runtimes
### H1. Histogram of runtimes
#region H1. Hist. runtimes

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_runtime"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
print(f"Total runtime all estimations: {sum(with_data.runtime)/3600:.2f} hours")

# Histograms
min_runtime = with_data.runtime.min()
max_runtime = with_data.runtime.max()
bins_linear = np.linspace(min_runtime, max_runtime, 20)
bins_log = np.logspace(
    np.log10(min_runtime),
    np.log10(max_runtime),
    20,
)
bins_for_plot = bins_linear
bins_for_plot = bins_log
ax.set_xscale('log')

ax.hist(
    successes.runtime,
    bins = bins_for_plot,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax.hist(
    low_SNR.runtime,
    bins = bins_for_plot,
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    ax.hist(
        high_RMS.runtime,
        bins = bins_for_plot,
        label = "Rej. res.",
        ec = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        **hist_contour_kwargs,
    )

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("Runtime [s]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H2. Histogram of initial pos. change
### H2. Histogram of initial pos. change
#region H2. Hist. Δinit. pos.

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_initial-position-change"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
nonzero_initial_position_changes_1 = successes.loc[successes.initial_state_change_KM > 0]
nonzero_initial_position_changes_rej = rejects.loc[rejects.initial_state_change_KM > 0]
min_position_change = min(
    nonzero_initial_position_changes_1.initial_state_change_KM.min(),
    nonzero_initial_position_changes_rej.initial_state_change_KM.min(),
    divergers.initial_state_change_KM.min(),
)
max_position_change = max(
    nonzero_initial_position_changes_1.initial_state_change_KM.max(),
    nonzero_initial_position_changes_rej.initial_state_change_KM.max(),
    divergers.initial_state_change_KM.max(),
)
logbins_combi = np.logspace(
    np.log10(min_position_change),
    np.log10(max_position_change),
    30,
)
ax.hist(
    nonzero_initial_position_changes_1.initial_state_change_KM,
    bins = logbins_combi,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
nonzero_initial_position_changes_rej_SNR = low_SNR.loc[low_SNR.initial_state_change_KM > 0]
ax.hist(
    nonzero_initial_position_changes_rej_SNR.initial_state_change_KM,
    bins = logbins_combi,
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    nonzero_initial_position_changes_rej_res = high_RMS.loc[high_RMS.initial_state_change_KM > 0]
    ax.hist(
        nonzero_initial_position_changes_rej_res.initial_state_change_KM,
        bins = logbins_combi,
        label = "Rej. res.",
        ec = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        **hist_contour_kwargs,
    )
# ax.hist(
#     nonzero_initial_position_changes_rej.initial_state_change_KM,
#     bins = logbins_combi,
#     label = "Rejected",
#     ec = isu.cmap_custom10[colormode](3),
#     **hist_contour_kwargs,
# )

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_xscale('log')
ax.set_ylabel("Count")
ax.set_xlabel("Initial position change magnitude [km]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H3. Histogram of SNR_A2_est
### H3. Histogram of SNR_A2_est
#region H3. Hist. SNR_A2_est

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_SNR_A2_est"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Vertical line at SNR rejection cutoff
SNR_cutoff = 3.0
ax.axvline(
    SNR_cutoff,
    label = f"SNR = {SNR_cutoff}",
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)

# Histograms
# min_SNR = min(successes.SNR_est.min(), rejects.SNR_est.min())
# max_SNR = max(successes.SNR_est.max(), rejects.SNR_est.max())
min_SNR = min(successes.SNR_A2_est.min(), rejects.SNR_A2_est.min())
max_SNR = max(successes.SNR_A2_est.max(), rejects.SNR_A2_est.max())
logbins_rejects = np.logspace(np.log10(min_SNR), np.log10(SNR_cutoff), 20)
logbins_successes = np.logspace(np.log10(SNR_cutoff), np.log10(max_SNR), 20)
ax.hist(
    low_SNR.SNR_A2_est,
    # rejects.SNR_est,
    bins = logbins_rejects,
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    ax.hist(
        high_RMS.SNR_A2_est,
        # rejects.SNR_est,
        bins = logbins_successes,
        label = "Rej. res.",
        linestyle = '--',
        ec = isu.cmap_custom10[colormode](4),
        **hist_contour_kwargs,
    )
# ax.hist(
#     rejects.SNR_A2_est,
#     # rejects.SNR_est,
#     bins = logbins_rejects,
#     label = "Rejected",
#     ec = isu.cmap_custom10[colormode](3),
#     **hist_contour_kwargs,
# )
ax.hist(
    successes.SNR_A2_est,
    # successes.SNR_est,
    bins = logbins_successes,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_xscale('log')
ax.set_ylabel("Count")
ax.set_xlabel(r"$SNR_{A_2}$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H4.1 Histogram of k1_NEOCC
### H4.1 Histogram of k1_NEOCC
#region H4.1 Hist. k1
# k1_NEOCC = | A2_NEOCC - A2_est | / σ_A2_est

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_k1_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# x lims and ticks
xlims = (0, 10)
binwidth = 0.25
bins = np.arange(xlims[0], xlims[1]+0.1, binwidth)
xticks = np.arange(xlims[0], xlims[1]+0.1, 1)
ax.set_xticks(xticks)
ax.set_xlim((xlims[0], xlims[1]))

# k spans
for i, (span, color) in enumerate(zip(k1_spans, k1_span_colors)):
    ax.axvspan(
        xmin = span[0],
        xmax = span[1],
        color = color,
        alpha = 0.20,
    )

# Histogram
ax.hist(
    successes.k1,
    # abs(successes['A2_lit+Xσ_A2_est']),
    bins = bins,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax.hist(
    low_SNR.k1,
    # abs(successes['A2_lit+Xσ_A2_est']),
    bins = bins,
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    ax.hist(
        high_RMS.k1,
        # abs(successes['A2_lit+Xσ_A2_est']),
        bins = bins,
        label = "Rej. res.",
        ec = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        **hist_contour_kwargs,
    )

# Half-normal distribution (folded normal distribution with mu=0)
hist_count = len(successes.k1)
# hist_count = len(successes['A2_lit+Xσ_A2_est'])
mu = 0
variance = 1
sigma = np.sqrt(variance)
normal_distr_xspan = np.linspace(mu + xlims[0]*sigma, mu + xlims[1]*sigma, 200)
plt.plot(
    normal_distr_xspan,
    stats.foldnorm.pdf(normal_distr_xspan, mu) * hist_count * binwidth,
    label = rf"Half-normal distr. ($\sigma={sigma:.0f}$)",
    alpha = 1,
    linewidth = 2,
    linestyle = '--',
    color = isu.text_color[colormode],
)

# Text box for excluded data
n_included = sum(successes.k1.between(xlims[0], xlims[1], inclusive='both'))
# n_included = sum(abs(successes['A2_lit+Xσ_A2_est']).between(xlims[0], xlims[1], inclusive='both'))
n_excluded = hist_count - n_included
text_box_note = f"N data points: {hist_count}"
if n_excluded > 0:
    text_box_note += f"\nN outside plot range: {n_excluded}"
# ax.text(
#     x = 0.03,
#     y = 0.97,
#     s = text_box_note,
#     horizontalalignment = 'left',
#     verticalalignment = 'top',
#     transform = ax.transAxes,
#     bbox = text_bbox,
#     **text_box_kwargs,
# )
print(text_box_note)

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel(r"$k_1^\text{NEOCC}$")
# ylims = ax.get_ylim()
# ax.set_ylim((-2,ylims[1]))
# ax.axhline(linewidth=1.5,color='k')
# ax.set_title(
#     r"$A2_{lit.} = A2_{est.} + k_1\cdot\sigma_{A2,est.}$",
# )

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H4.2 Histogram of k2_NEOCC
### H4.2 Histogram of k2_NEOCC
#region H4.2 Hist. k2
# k2_NEOCC = | A2_NEOCC - A2_est | / σ_A2_NEOCC

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_k2_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# x lims and ticks
xlims = (0, 10)
binwidth = 0.25
bins = np.arange(xlims[0], xlims[1]+0.1, binwidth)
xticks = np.arange(xlims[0], xlims[1]+0.1, 1)
ax.set_xticks(xticks)
ax.set_xlim((xlims[0], xlims[1]))

# k spans
for i, (span, color) in enumerate(zip(k1_spans, k1_span_colors)):
    ax.axvspan(
        xmin = span[0],
        xmax = span[1],
        color = color,
        alpha = 0.20,
    )

# Histogram
ax.hist(
    successes.k2,
    # abs(successes['A2_est+Xσ_A2_lit']),
    bins = bins,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax.hist(
    low_SNR.k2,
    # abs(successes['A2_lit+Xσ_A2_est']),
    bins = bins,
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    ax.hist(
        high_RMS.k2,
        # abs(successes['A2_lit+Xσ_A2_est']),
        bins = bins,
        label = "Rej. res.",
        ec = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        **hist_contour_kwargs,
    )

# Half-normal distribution (folded normal distribution with mu=0)
hist_count = len(successes.k2)
# hist_count = len(successes['A2_est+Xσ_A2_lit'])
mu = 0
variance = 1
sigma = np.sqrt(variance)
normal_distr_xspan = np.linspace(mu + xlims[0]*sigma, mu + xlims[1]*sigma, 200)
plt.plot(
    normal_distr_xspan,
    stats.foldnorm.pdf(normal_distr_xspan, mu) * hist_count * binwidth,
    label = rf"Half-normal distr. ($\sigma={sigma:.0f}$)",
    alpha = 1,
    linewidth = 2,
    linestyle = '--',
    color = isu.text_color[colormode],
)

# Text box for excluded data
n_included = sum(successes.k2.between(xlims[0], xlims[1], inclusive='both'))
# n_included = sum(abs(successes['A2_est+Xσ_A2_lit']).between(xlims[0], xlims[1], inclusive='both'))
n_excluded = hist_count - n_included
text_box_note = f"N data points: {hist_count}"
if n_excluded > 0:
    text_box_note += f"\nN outside plot range: {n_excluded}"
# ax.text(
#     x = 0.03,
#     y = 0.97,
#     s = text_box_note,
#     horizontalalignment = 'left',
#     verticalalignment = 'top',
#     transform = ax.transAxes,
#     bbox = text_bbox,
#     **text_box_kwargs,
# )
print(text_box_note)

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel(r"$k_2^\text{NEOCC}$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H4.3 Histogram subplots of k1_NEOCC, k2_NEOCC
### H4.3 Histogram subplots of k1_NEOCC, k2_NEOCC
#region H4.3 Hist. k1,k2
# k1_NEOCC = | A2_NEOCC - A2_est | / σ_A2_est
# k2_NEOCC = | A2_NEOCC - A2_est | / σ_A2_NEOCC

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_k1_k2_subplots_NEOCC"
# format = 'pdf'
format = 'png'

plot_rejects = True

fig, axs = plt.subplots(
    2, 1,
    figsize = np.array([1.0,0.80])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)

for ax in axs.flatten():
    ax = isu.set_default_2d_ax_settings(ax, colormode)

    # x lims and ticks
    xlims = (0, 10)
    xticks = np.arange(xlims[0], xlims[1]+0.1, 1)
    ax.set_xticks(xticks)
    ax.set_xlim((xlims[0], xlims[1]))

    # k spans
    for i, (span, color) in enumerate(zip(k1_spans, k1_span_colors)):
        ax.axvspan(
            xmin = span[0],
            xmax = span[1],
            color = color,
            alpha = 0.20,
        )

# Histograms
binwidth = 0.25
bins = np.arange(xlims[0], xlims[1]+0.1, binwidth)
axs[0].hist(
    successes.k1,
    # abs(successes['A2_lit+Xσ_A2_est']),
    bins = bins,
    label = "Successful",
    # label = r"Successful ($k_1$)",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
axs[1].hist(
    successes.k2,
    # abs(successes['A2_est+Xσ_A2_lit']),
    bins = bins,
    label = "Successful",
    # label = r"Successful ($k_2$)",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
if plot_rejects:
    axs[0].hist(
        rejects.k1,
        bins = bins,
        label = "Rejected",
        ec = isu.cmap_custom10[colormode](3),
        linestyle = '--',
        **hist_contour_kwargs,
    )
    axs[1].hist(
        rejects.k2,
        bins = bins,
        label = "Rejected",
        ec = isu.cmap_custom10[colormode](3),
        linestyle = '--',
        **hist_contour_kwargs,
    )
# Half-normal distribution (folded normal distribution with mu=0)
hist_count = len(successes)
# hist_count = len(successes['A2_est+Xσ_A2_lit'])
mu = 0
variance = 1
sigma = np.sqrt(variance)
normal_distr_xspan = np.linspace(mu + xlims[0]*sigma, mu + xlims[1]*sigma, 200)
for ax in axs.flatten():
    ax.plot(
        normal_distr_xspan,
        stats.foldnorm.pdf(normal_distr_xspan, mu) * hist_count * binwidth,
        label = rf"Half-normal distr. ($\sigma={sigma:.0f}$)",
        alpha = 1,
        linewidth = 2,
        linestyle = '--',
        color = isu.text_color[colormode],
    )

# Count excluded data for k1 (successes)
n_included = sum(successes.k1.between(xlims[0], xlims[1], inclusive='both'))
# n_included = sum(abs(successes['A2_lit+Xσ_A2_est']).between(xlims[0], xlims[1], inclusive='both'))
n_excluded = hist_count - n_included
print(f"N data points for successes: {hist_count}")
print(f"N outside plot range for k1: {n_excluded}")

# Count excluded data for k2 (successes)
n_included = sum(successes.k2.between(xlims[0], xlims[1], inclusive='both'))
# n_included = sum(abs(successes['A2_est+Xσ_A2_lit']).between(xlims[0], xlims[1], inclusive='both'))
n_excluded = hist_count - n_included
print(f"N outside plot range for k2: {n_excluded}")

if plot_rejects:
    hist_count = len(rejects)
    # Count excluded data for k1 (rejects)
    n_included = sum(rejects.k1.between(xlims[0], xlims[1], inclusive='both'))
    n_excluded = hist_count - n_included
    print(f"\nN data points for rejects: {hist_count}")
    print(f"N outside plot range for k1: {n_excluded}")

    # Count excluded data for k2 (rejects)
    n_included = sum(rejects.k2.between(xlims[0], xlims[1], inclusive='both'))
    n_excluded = hist_count - n_included
    print(f"N outside plot range for k2: {n_excluded}")

for ax in axs.flatten():
    # Legend, labels
    ax.legend(
        loc = 'best',
        **legend_kwargs,
    )
    ax.set_ylabel("Count")
axs[0].set_xlabel(r"$k_1^\text{NEOCC}$")
axs[1].set_xlabel(r"$k_2^\text{NEOCC}$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H4.4 Histogram subplots of k1_JPL, k2_JPL
### H4.4 Histogram subplots of k1_JPL, k2_JPL
#region H4.4 Hist. k1,k2
# k1_JPL = | A2_JPL - A2_est | / σ_A2_est
# k2_JPL = | A2_JPL - A2_est | / σ_A2_JPL

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_k1_k2_subplots_JPL"
# format = 'pdf'
format = 'png'

plot_rejects = True

# Calculate k1_JPL and k2_JPL
successes['k1_JPL'] = np.abs(successes.A2_JPL - successes.A2_est) / successes.σ_A2_est
successes['k2_JPL'] = np.abs(successes.A2_JPL - successes.A2_est) / successes.σ_A2_JPL
if plot_rejects:
    rejects['k1_JPL'] = np.abs(rejects.A2_JPL - rejects.A2_est) / rejects.σ_A2_est
    rejects['k2_JPL'] = np.abs(rejects.A2_JPL - rejects.A2_est) / rejects.σ_A2_JPL

fig, axs = plt.subplots(
    2, 1,
    figsize = np.array([1.0,0.80])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)

for ax in axs.flatten():
    ax = isu.set_default_2d_ax_settings(ax, colormode)

    # x lims and ticks
    xlims = (0, 10)
    # ax.set_ylim((-0.1, 100))
    xticks = np.arange(xlims[0], xlims[1]+0.1, 1)
    ax.set_xticks(xticks)
    ax.set_xlim((xlims[0], xlims[1]))

    # k spans
    for i, (span, color) in enumerate(zip(k1_spans, k1_span_colors)):
        ax.axvspan(
            xmin = span[0],
            xmax = span[1],
            color = color,
            alpha = 0.20,
        )

# Histograms
binwidth = 0.25
bins = np.arange(xlims[0], xlims[1]+0.1, binwidth)
axs[0].hist(
    successes.k1_JPL,
    # abs(successes['A2_lit+Xσ_A2_est']),
    bins = bins,
    label = "Successful",
    # label = r"Successful ($k_1$)",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
axs[1].hist(
    successes.k2_JPL,
    # abs(successes['A2_est+Xσ_A2_lit']),
    bins = bins,
    label = "Successful",
    # label = r"Successful ($k_2$)",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
if plot_rejects:
    axs[0].hist(
        rejects.k1_JPL,
        bins = bins,
        label = "Rejected",
        ec = isu.cmap_custom10[colormode](3),
        linestyle = '--',
        **hist_contour_kwargs,
    )
    axs[1].hist(
        rejects.k1_JPL,
        bins = bins,
        label = "Rejected",
        ec = isu.cmap_custom10[colormode](3),
        linestyle = '--',
        **hist_contour_kwargs,
    )

# Half-normal distribution (folded normal distribution with mu=0)
n_JPL_A2_data_missing = len(successes[np.isnan(successes.A2_JPL)])
print(f"N successful asteroids for which no JPL A2 value is available: {n_JPL_A2_data_missing}")
hist_count = len(successes) - n_JPL_A2_data_missing
# hist_count = len(successes['A2_est+Xσ_A2_lit'])
mu = 0
variance = 1
sigma = np.sqrt(variance)
normal_distr_xspan = np.linspace(mu + xlims[0]*sigma, mu + xlims[1]*sigma, 200)
for ax in axs.flatten():
    ax.plot(
        normal_distr_xspan,
        stats.foldnorm.pdf(normal_distr_xspan, mu) * hist_count * binwidth,
        label = rf"Half-normal distr. ($\sigma={sigma:.0f}$)",
        alpha = 1,
        linewidth = 2,
        linestyle = '--',
        color = isu.text_color[colormode],
    )

# Count excluded data for k1 (successes)
n_included = sum(successes.k1_JPL.between(xlims[0], xlims[1], inclusive='both'))
# n_included = sum(abs(successes['A2_lit+Xσ_A2_est']).between(xlims[0], xlims[1], inclusive='both'))
n_excluded = hist_count - n_included
print(f"N data points for successes: {hist_count}")
print(f"N outside plot range for k1: {n_excluded}")

# Count excluded data for k2 (successes)
n_included = sum(successes.k2_JPL.between(xlims[0], xlims[1], inclusive='both'))
# n_included = sum(abs(successes['A2_est+Xσ_A2_lit']).between(xlims[0], xlims[1], inclusive='both'))
n_excluded = hist_count - n_included
print(f"N outside plot range for k2: {n_excluded}")

if plot_rejects:
    n_JPL_A2_data_missing = len(rejects[np.isnan(rejects.A2_JPL)])
    print(f"\nN rejected asteroids for which no JPL A2 value is available: {n_JPL_A2_data_missing}")
    hist_count = len(rejects) - n_JPL_A2_data_missing
    # Count excluded data for k1 (rejects)
    n_included = sum(rejects.k1_JPL.between(xlims[0], xlims[1], inclusive='both'))
    n_excluded = hist_count - n_included
    print(f"N data points for rejects: {hist_count}")
    print(f"N outside plot range for k1: {n_excluded}")

    # Count excluded data for k2 (rejects)
    n_included = sum(rejects.k2_JPL.between(xlims[0], xlims[1], inclusive='both'))
    n_excluded = hist_count - n_included
    print(f"N outside plot range for k2: {n_excluded}")


for ax in axs.flatten():
    # Legend, labels
    ax.legend(
        loc = 'best',
        **legend_kwargs,
    )
    ax.set_ylabel("Count")
axs[0].set_xlabel(r"$k_1^\text{JPL}$")
axs[1].set_xlabel(r"$k_2^\text{JPL}$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H5. Histogram of Obervation Arc
### H5. Histogram of Obervation Arc
#region H5. Hist. Obervation Arc

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_obs_arc"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
min_arc = with_data.obs_arc_est_YEAR.min()
max_arc = with_data.obs_arc_est_YEAR.max()
bins_combi = np.linspace(min_arc, max_arc, 20)
ax.hist(
    with_data.obs_arc_est_YEAR,
    bins = bins_combi,
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)

ax.set_ylabel("Count")
ax.set_xlabel("Obervation arc [y]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H6. Histogram of Eccentricity
### H6. Histogram of Eccentricity
#region H6. Hist. e

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_eccentricity"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
bins_combi = np.arange(0, 1.01, 0.05)
ax.hist(
    with_data.eccentricity,
    bins = bins_combi,
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)

ax.set_ylabel("Count")
ax.set_xlabel("Eccentricity")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H7. Histogram of Semi-major axis
### H7. Histogram of Semi-major axis
#region H7. Hist. a

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_semimajoraxis"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
min_a = with_data.semimajoraxis_AU.min()
max_a = with_data.semimajoraxis_AU.max()
bins_combi = np.linspace(min_a, max_a, 20)
ax.hist(
    with_data.semimajoraxis_AU,
    bins = bins_combi,
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)

ax.set_ylabel("Count")
ax.set_xlabel("Semi-major axis [AU]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H8.1 Histogram of RMS_res_est
### H8.1 Histogram of RMS_res_est
#region H8.1 Hist. RMS_est

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_RMS_residuals_Tudat"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# # Histogram
# RMS_cutoff = 1.0
# parameter = good_fits.RMS_res_est
# # parameter = successes.RMS_est # only for RK6 results
# n_total = len(parameter)
# parameter = parameter.loc[parameter < RMS_cutoff] # filter out high values
# n_included = len(parameter)
# min_RMS = parameter.min()
# max_RMS = parameter.max()
# bins_combi = np.linspace(0, RMS_cutoff, 40)
# ax.hist(
#     parameter,
#     bins = bins_combi,
#     # label = 'Tudat (E-set)',
#     ec = isu.cmap_custom10[colormode](0),
#     # label = 'Tudat (G-set)',
#     # ec = isu.cmap_custom10[colormode](6),
#     **hist_contour_kwargs,
# )

# # Text box for excluded data
# n_excluded = n_total - n_included
# text_box_note = f"N data points: {n_total}"
# text_box_note += f"\nN outside plot range: {n_excluded}"
# print(text_box_note)


# Histogram (split by successful / rejected)
RMS_cutoff = 1.0
min_RMS = good_fits.RMS_res_est.min()
max_RMS = good_fits.RMS_res_est.max()
step = 0.05
bins_combi = np.arange(0, RMS_cutoff+step, step)
ax.hist(
    successes.RMS_res_est,
    bins = bins_combi,
    label = 'Successful',
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax.hist(
    low_SNR.RMS_res_est,
    bins = bins_combi,
    label = 'Rejected',
    ec = isu.cmap_custom10[colormode](3),
    linestyle = '--',
    **hist_contour_kwargs,
)


# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("RMS of residuals [arcsec]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H8.2a Histogram of RMS_norm_res_est
### H8.2a Histogram of RMS_norm_res_est
#region H8.2a Hist RMS_norm_est

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_RMS_normalised_residuals_Tudat"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

RMS_cutoff_plot = 1.2
# ax.set_xscale('log')
# ax.set_yscale('log')

# Vertical line at RMS rejection cutoff
RMS_cutoff = 2.0
if RMS_cutoff_plot > RMS_cutoff:
    ax.axvline(
        RMS_cutoff,
        label = f"RMS = {RMS_cutoff}",
        linewidth = 2,
        linestyle = 'dotted',
        color = isu.cmap_custom10[colormode](3),
        zorder = 2.6,
    )

# Histogram
# min_RMS = parameter.min()
# max_RMS = parameter.max()
# bins_combi = np.linspace(0, RMS_cutoff_plot, 40)
step = 0.05
bins_combi = np.arange(0, RMS_cutoff_plot+step, step)
ax.hist(
    successes.RMS_norm_res_est,
    bins = bins_combi,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax.hist(
    low_SNR.RMS_norm_res_est,
    bins = bins_combi,
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    linestyle = '--',
    **hist_contour_kwargs,
)
if RMS_cutoff_plot > RMS_cutoff:
    ax.hist(
        high_RMS.RMS_norm_res_est,
        bins = bins_combi,
        label = "Rej. res.",
        ec = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        **hist_contour_kwargs,
    )

# Count excluded data for successes
n_included = sum(successes.RMS_norm_res_est.between(0, RMS_cutoff_plot, inclusive='both'))
n_excluded = n_successes - n_included
print(f"N data points for successes: {n_successes}")
print(f"N outside plot range: {n_excluded}")

# Count excluded data for Rej SNR
n_included = sum(low_SNR.RMS_norm_res_est.between(0, RMS_cutoff_plot, inclusive='both'))
n_excluded = n_low_SNR - n_included
print(f"\nN data points for Rej. SNR: {n_low_SNR}")
print(f"N outside plot range: {n_excluded}")

# Count excluded data for Rej res.
n_included = sum(high_RMS.RMS_norm_res_est.between(0, RMS_cutoff_plot, inclusive='both'))
n_excluded = n_high_RMS - n_included
print(f"\nN data points for Rej. res.: {n_high_RMS}")
print(f"N outside plot range: {n_excluded}")

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("RMS of normalised residuals")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H8.2b Histogram of mean_norm_res_est
### H8.2b Histogram of mean_norm_res_est
#region H8.2b Hist mean_norm_est

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_mean_normalised_residuals_Tudat"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

RMS_cutoff_plot = 0.8
# ax.set_xscale('log')
# ax.set_yscale('log')

# Histogram
# min_RMS = parameter.min()
# max_RMS = parameter.max()
# bins_combi = np.linspace(0, RMS_cutoff_plot, 40)
step = 0.05
bins_combi = np.arange(0, RMS_cutoff_plot+step, step)
ax.hist(
    successes.mean_abs_norm_res_est,
    bins = bins_combi,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax.hist(
    low_SNR.mean_abs_norm_res_est,
    bins = bins_combi,
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    linestyle = '--',
    **hist_contour_kwargs,
)
if RMS_cutoff_plot > RMS_cutoff:
    ax.hist(
        high_RMS.mean_abs_norm_res_est,
        bins = bins_combi,
        label = "Rej. res.",
        ec = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        **hist_contour_kwargs,
    )

# Count excluded data for successes
n_included = sum(successes.mean_abs_norm_res_est.between(0, RMS_cutoff_plot, inclusive='both'))
n_excluded = n_successes - n_included
print(f"N data points for successes: {n_successes}")
print(f"N outside plot range: {n_excluded}")

# Count excluded data for Rej SNR
n_included = sum(low_SNR.mean_abs_norm_res_est.between(0, RMS_cutoff_plot, inclusive='both'))
n_excluded = n_low_SNR - n_included
print(f"\nN data points for Rej. SNR: {n_low_SNR}")
print(f"N outside plot range: {n_excluded}")

# Count excluded data for Rej res.
n_included = sum(high_RMS.mean_abs_norm_res_est.between(0, RMS_cutoff_plot, inclusive='both'))
n_excluded = n_high_RMS - n_included
print(f"\nN data points for Rej. res.: {n_high_RMS}")
print(f"N outside plot range: {n_excluded}")

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("Mean of abs. normalised residuals")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H8.3 Histogram of RMS_lit
### H8.3 Histogram of RMS_lit
#region H8.3 Hist. RMS_lit

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_RMS_normalised_residuals_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
parameter = with_data.RMS_norm_res_lit
min_RMS = parameter.min()
max_RMS = parameter.max()
# bins_combi = np.linspace(min_RMS, max_RMS, 20)
bins_combi = np.arange(0, 1.5, 0.05)

ax.hist(
    parameter,
    bins = bins_combi,
    label = "NEOCC",
    ec = isu.cmap_custom10[colormode](9),
    **hist_contour_kwargs,
)
ax.set_xlim((0,1.5))

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("RMS of normalised residuals")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H8.4 Histogram of RMS normalised residuals (est. and lit.)
### H8.4 Histogram of RMS normalised residuals (est. and lit.)
#region H8.4 Hist. RMS norm res (both)

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
# format = 'pdf'
format = 'png'

include_NEOCC = 1
include_JPL = 1
include_Tudat_rejects = 0
RMS_cutoff_plot = 1.2

fig_name = f"hist_RMS_normalised_residuals_Tudat"
if include_NEOCC:
    fig_name += "_NEOCC"
if include_JPL:
    fig_name += "_JPL"
if include_Tudat_rejects:
    fig_name += "_rejects"
fig_name += format

print(fig_name)

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.set_xlim((0,RMS_cutoff_plot))

# Gather data
# Tudat (est) -- good fits
RMS_norm_est_SUC = good_fits.RMS_norm_res_est
n_total_SUC = len(RMS_norm_est_SUC)
RMS_norm_est_SUC = RMS_norm_est_SUC.loc[RMS_norm_est_SUC < RMS_cutoff_plot] # filter out high values
n_included_SUC = len(RMS_norm_est_SUC)
n_excluded_SUC = n_total_SUC - n_included_SUC
mean_RMS_norm_est_SUC = np.mean(RMS_norm_est_SUC)
median_RMS_norm_est_SUC = np.median(RMS_norm_est_SUC)
print(f"TUDAT GOOD FITS")
text_box_note = f"N data points: {n_total_SUC}"
text_box_note += f"\nN outside plot range: {n_excluded_SUC}"
print(text_box_note)

if include_Tudat_rejects:
    # Tudat (est) -- high RMSNR rejects
    RMS_norm_est_REJ = high_RMS.RMS_norm_res_est
    n_total_REJ = len(RMS_norm_est_REJ)
    RMS_norm_est_REJ = RMS_norm_est_REJ.loc[RMS_norm_est_REJ < RMS_cutoff_plot] # filter out high values
    n_included_REJ = len(RMS_norm_est_REJ)
    n_excluded_REJ = n_total_REJ - n_included_REJ
    mean_RMS_norm_est_REJ = np.mean(RMS_norm_est_REJ)
    median_RMS_norm_est_REJ = np.median(RMS_norm_est_REJ)
    print(f"\nTUDAT HIGH RMSNR REJECTS")
    text_box_note = f"N data points: {n_total_REJ}"
    text_box_note += f"\nN outside plot range: {n_excluded_REJ}"
    print(text_box_note)


# Bins
bins_linear = np.arange(0, RMS_cutoff_plot+0.05, 0.05)

# Tudat histograms
ax.hist(
    RMS_norm_est_SUC,
    bins = bins_linear,
    label = "Tudat (E-set)", # actually the G-set if H-set is not empty
    ec = isu.cmap_custom10[colormode](0),
    alpha = 0.7,
    **hist_contour_kwargs,
)
ax.axvline(
    # mean_RMS_norm_est_SUC,
    # label = f"m = {mean_RMS_norm_est_SUC:.4f}",
    median_RMS_norm_est_SUC,
    label = f"m = {median_RMS_norm_est_SUC:.4f}",
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](0),
    zorder = 2.6,
)
if include_Tudat_rejects:
    ax.hist(
        RMS_norm_est_REJ,
        bins = bins_linear,
        label = "Tudat (H-set)",
        ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
        alpha = 0.7,
        linestyle = '--',
        **hist_contour_kwargs,
    )
    ax.axvline(
        # mean_RMS_norm_est_REJ,
        # label = f"m = {mean_RMS_norm_est_REJ:.4f}",
        median_RMS_norm_est_REJ,
        label = f"m = {median_RMS_norm_est_REJ:.4f}",
        linewidth = 2,
        linestyle = 'dotted',
        color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
        zorder = 2.6,
    )
# NEOCC histogram
if include_NEOCC:
    ax.hist(
        good_fits.RMS_norm_res_lit,
        bins = bins_linear,
        label = "NEOCC (E-set)", # actually the G-set if H-set is not empty
        ec = isu.cmap_custom10[colormode](9),
        alpha = 0.7,
        **hist_contour_kwargs,
    )
    mean_RMS_norm_est_lit = np.mean(good_fits.RMS_norm_res_lit)
    median_RMS_norm_est_lit = np.median(good_fits.RMS_norm_res_lit)
    ax.axvline(
        # mean_RMS_norm_est_lit,
        # label = f"m = {mean_RMS_norm_est_lit:.4f}",
        median_RMS_norm_est_lit,
        label = f"m = {median_RMS_norm_est_lit:.4f}",
        linewidth = 2,
        linestyle = 'dotted',
        color = isu.cmap_custom10[colormode](9),
        zorder = 2.6,
    )
# JPL histogram
if include_JPL:
    ax.hist(
        good_fits.RMS_norm_res_JPL,
        bins = bins_linear,
        label = "JPL (E-set)", # actually the G-set if H-set is not empty
        ec = isu.cmap_custom10[colormode](7),
        alpha = 0.7,
        **hist_contour_kwargs,
    )
    mean_RMS_norm_est_JPL = np.mean(good_fits.RMS_norm_res_JPL)
    median_RMS_norm_est_JPL = np.median(good_fits.RMS_norm_res_JPL)
    ax.axvline(
        # mean_RMS_norm_est_JPL,
        # label = f"m = {mean_RMS_norm_est_JPL:.4f}",
        median_RMS_norm_est_JPL,
        label = f"m = {median_RMS_norm_est_JPL:.4f}",
        linewidth = 2,
        linestyle = 'dotted',
        color = isu.cmap_custom10[colormode](7),
        zorder = 2.6,
    )

# Legend, labels
ax.legend(
    loc = 'best',
    markerscale = 1.5,
    fontsize = isu.tiny_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax.set_ylabel("Count")
ax.set_xlabel("RMS of normalised residuals")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H9. Bar chart of best iteration
### H9. Bar chart of best iteration
#region H9. Best iter.

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"bar_chart_best_iteration"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

best_iters = pd.Series(with_data.best_iteration, dtype=int)
species = list(set(best_iters))
species.sort()

rej_SNR_best_iters = pd.Series(low_SNR.best_iteration, dtype=int)
rej_res_best_iters = pd.Series(high_RMS.best_iteration, dtype=int)
successes_best_iters = pd.Series(successes.best_iteration, dtype=int)
rej_SNR_counts = np.zeros(len(species))
rej_res_counts = np.zeros(len(species))
successes_counts = np.zeros(len(species))
for idx, spec in enumerate(species):
    rej_SNR_counts[idx] = np.count_nonzero(rej_SNR_best_iters == spec)
    rej_res_counts[idx] = np.count_nonzero(rej_res_best_iters == spec)
    successes_counts[idx] = np.count_nonzero(successes_best_iters == spec)

if len(high_RMS) > 0:
    counts_dict = {
        "Rej. res.": rej_res_counts,
        "Rej. SNR": rej_SNR_counts,
        "Successful": successes_counts,
    }
    color_idx_dict = {
        "Rej. res.": 4,
        "Rej. SNR": 1,
        "Successful": 2,
    }
else:
    counts_dict = {
        "Rejected": rej_SNR_counts,
        "Successful": successes_counts,
    }
    color_idx_dict = {
        "Rejected": 3,
        "Successful": 2,
    }

tick_dict = {
    0: "1st",
    1: "2nd",
    2: "3rd",
    3: "4th",
}

bar_width = 0.6
bottom = np.zeros(len(species))
for label, counts in counts_dict.items():
    ax.bar(
        x = species,
        height = counts,
        bottom = bottom,
        width = bar_width,
        label = label,
        color = isu.cmap_custom10[colormode](color_idx_dict[label]),
        zorder = 2.5,
    )
    bottom += counts

# Legend, labels
ax.legend(
    loc = 'best',
    reverse = True,
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("Best iteration")
ax.set_xticks(
    ticks = species,
    labels = [tick_dict[spec] for spec in species],
    minor = False,
)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H10. Histogram of % rejected optical observations in literature
### H10. Histogram of % rejected optical observations in literature
#region H10. Hist rej obs lit

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_rejected_optical_observations"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Gather data
# Tudat (est) -- successes
opt_est_parameter_1 = (successes.n_obs_filtered_out / successes.n_total_obs_used) * 100
mean_pc_opt_rej = np.mean(opt_est_parameter_1)
opt_est_parameter_1_nz = opt_est_parameter_1.loc[opt_est_parameter_1 != 0]
n_opt_rej_zero = len(successes) - len(opt_est_parameter_1_nz)
print(f"TUDAT SUCCESSES")
print(f"N for which ZERO optical observations were rejected: {n_opt_rej_zero} ({(n_opt_rej_zero/len(successes))*100:.3f}%)")
print(f"Mean % of optical observations rejected: {mean_pc_opt_rej:.3f}%")

# Tudat (est) -- rejects
opt_est_parameter_rej = (rejects.n_obs_filtered_out / rejects.n_total_obs_used) * 100
mean_pc_opt_rej = np.mean(opt_est_parameter_rej)
opt_est_parameter_rej_nz = opt_est_parameter_rej.loc[opt_est_parameter_rej != 0]
n_opt_rej_zero = len(rejects) - len(opt_est_parameter_rej_nz)
print(f"\nTUDAT REJECTS")
print(f"N for which ZERO optical observations were rejected: {n_opt_rej_zero} ({(n_opt_rej_zero/len(rejects))*100:.3f}%)")
print(f"Mean % of optical observations rejected: {mean_pc_opt_rej:.3f}%")

# NEOCC (lit)
opt_lit_parameter = (with_data.n_opt_obs_rej_lit / with_data.n_opt_obs_lit) * 100
mean_pc_opt_rej = np.mean(opt_lit_parameter)
opt_lit_parameter_nz = opt_lit_parameter.loc[opt_lit_parameter != 0]
n_opt_rej_zero = len(with_data) - len(opt_lit_parameter_nz)
print(f"\nNEOCC")
print(f"N for which ZERO optical observations were rejected: {n_opt_rej_zero} ({(n_opt_rej_zero/len(with_data))*100:.3f}%)")
print(f"Mean % of optical observations rejected: {mean_pc_opt_rej:.3f}%")

# Bins
min_pc_opt_rej = min(opt_est_parameter_1.min(), opt_est_parameter_rej.min(), opt_lit_parameter.min())
max_pc_opt_rej = max(opt_est_parameter_1.max(), opt_est_parameter_rej.max(), opt_lit_parameter.max())
opt_bins_linear = np.arange(0, max_pc_opt_rej+0.5, 0.5)
opt_bins_log = np.logspace(
    np.log10(min_pc_opt_rej),
    np.log10(max_pc_opt_rej),
    20,
)
bins_to_plot = opt_bins_linear
# bins_to_plot = opt_bins_log
# ax.set_xscale('log')
# ax.set_yscale('log')

# Histogram: literature optical observations
ax.hist(
    opt_lit_parameter,
    bins = bins_to_plot,
    label = 'NEOCC',
    ec = isu.cmap_custom10[colormode](9),
    alpha = 0.7,
    **hist_contour_kwargs,
)

# Histograms: Tudat (ground optical observations)
ax.hist(
    opt_est_parameter_1,
    bins = bins_to_plot,
    label = 'Tudat (succ.)',
    ec = isu.cmap_custom10[colormode](2),
    alpha = 0.7,
    **hist_contour_kwargs,
)
ax.hist(
    opt_est_parameter_rej,
    bins = bins_to_plot,
    label = 'Tudat (rej.)',
    ec = isu.cmap_custom10[colormode](3),
    linestyle = '--',
    alpha = 0.7,
    **hist_contour_kwargs,
)


# # Histogram: literature radar observations
# rad_parameter = ((with_data.n_rad_obs_rej_lit / with_data.n_rad_obs_lit) * 100).dropna()
# n_with_radar = len(rad_parameter)
# mean_pc_rad_rej = np.mean(rad_parameter)
# rad_parameter = rad_parameter.loc[rad_parameter != 0]
# n_rad_rej_zero = n_with_radar - len(rad_parameter)
# print(f"N for which ZERO radar observations were rejected (lit.): {n_rad_rej_zero} ({(n_rad_rej_zero/n_with_radar)*100:.3f}%)")
# print(f"Mean % of radar observations rejected (lit.): {mean_pc_rad_rej:.3f}%")
# rad_bins = np.arange(0, max(rad_parameter)+0.5, 0.5)
# opt_log_bins = np.logspace(
#     np.log10(min(rad_parameter)),
#     np.log10(max(rad_parameter)),
#     20
# )
# ax.hist(
#     rad_parameter,
#     bins = opt_log_bins,
#     label = 'Radar',
#     ec = isu.cmap_custom10[colormode](5),
#     alpha = 0.7,
#     **hist_contour_kwargs,
# )


# ax.set_yscale('log')
# ax.set_xscale('log')
# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("% Rejected. Optical Observations")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H11. Histogram of target-Earth distance
### H11. Histogram of target-Earth distance
#region H11. Hist.

# This plot is only relevant if dataset_to_load != 'final'

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
parameter = with_data.min_earth_dist_AU
ax.set_xlabel("Min. target-Earth distance [AU]")
fig_name = f"hist_min_target-Earth_distance"
# format = 'pdf'
format = 'png'

# parameter = with_data.max_earth_dist_AU
# ax.set_xlabel("Max. target-Earth distance [AU]")
# fig_name = f"hist_max_target-Earth_distance"
# format = 'pdf'
format = 'png'

# parameter = with_data.earth_MOID_AU
# ax.set_xlabel("Earth MOID [AU]")
# fig_name = f"hist_earth_MOID"
# format = 'pdf'
format = 'png'

# parameter = with_data.min_earth_dist_AU / with_data.earth_MOID_AU
# ax.set_xlabel("Min. target-Earth distance / Earth MOID")
# ax.set_yscale("log")
# bins_combi = np.linspace(0, 10, 20)

# n_total = len(parameter)
# parameter = parameter.loc[parameter < RMS_cutoff] # filter out high values
# n_included = len(parameter)
min_dist = parameter.min()
max_dist = parameter.max()
bins_combi = np.linspace(0, max_dist, 20)
ax.hist(
    parameter,
    bins = bins_combi,
    # label = 'Tudat (successful)',
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)

# Text box for excluded data
# n_excluded = n_total - n_included
# text_box_note = f"N data points: {n_total}"
# if n_excluded > 0:
#     text_box_note += f"\nN outside plot range: {n_excluded}"
# ax.text(
#     x = 0.97,
#     y = 0.97,
#     s = text_box_note,
#     horizontalalignment = 'right',
#     verticalalignment = 'top',
#     transform = ax.transAxes,
#     bbox = text_bbox,
#     **text_box_kwargs,
# )
# print(text_box_note)

ax.set_ylabel("Count")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H12. Histogram of RMS TE pos.
### H12. Histogram of RMS TE pos.
#region H12. Hist. RMS_TE_p

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_RMS_true_position_error"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.set_xscale('log')

# Histogram
min_RMS = successes.RMS_true_error_pos_mag_KM.min()
max_RMS = successes.RMS_true_error_pos_mag_KM.max()
bins_combi = np.linspace(min_RMS, max_RMS, 30)
logbins_combi = np.logspace(
    np.log10(min_RMS),
    np.log10(max_RMS),
    30,
)
ax.hist(
    successes.RMS_true_error_pos_mag_KM,
    bins = logbins_combi,
    label = 'Successful',
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
# Histogram
min_RMS = rejects.RMS_true_error_pos_mag_KM.min()
max_RMS = rejects.RMS_true_error_pos_mag_KM.max()
bins_combi = np.linspace(min_RMS, max_RMS, 30)
logbins_combi = np.logspace(
    np.log10(min_RMS),
    np.log10(max_RMS),
    30,
)
ax.hist(
    rejects.RMS_true_error_pos_mag_KM,
    bins = logbins_combi,
    label = 'Rejected',
    ec = isu.cmap_custom10[colormode](3),
    linestyle = '--',
    **hist_contour_kwargs,
)

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("RMS true position error [km]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H13. Histogram of max(abs(NR)) / RMSNR
### H13. Histogram of max(abs(NR)) / RMSNR
#region H13. Hist. NR/RMSNR

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_max_NR_over_RMSNR"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram settings
max_NR_over_RMSNR_successes = successes.max_abs_norm_res_est / successes.RMS_norm_res_est
max_NR_over_RMSNR_low_SNR = low_SNR.max_abs_norm_res_est / low_SNR.RMS_norm_res_est
max_NR_over_RMSNR_high_RMS = high_RMS.max_abs_norm_res_est / high_RMS.RMS_norm_res_est

min_ratio = min(
    max_NR_over_RMSNR_successes.min(),
    max_NR_over_RMSNR_low_SNR.min(),
    max_NR_over_RMSNR_high_RMS.min(),
)
max_ratio = max(
    max_NR_over_RMSNR_successes.max(),
    max_NR_over_RMSNR_low_SNR.max(),
    max_NR_over_RMSNR_high_RMS.max(),
)
bins_combi = np.linspace(min_ratio, max_ratio, 30)
logbins_combi = np.logspace(
    np.log10(min_ratio),
    np.log10(max_ratio),
    30,
)

# Histograms
ax.hist(
    max_NR_over_RMSNR_successes,
    bins = bins_combi,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax.hist(
    max_NR_over_RMSNR_low_SNR,
    bins = bins_combi,
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(max_NR_over_RMSNR_high_RMS) > 0:
    ax.hist(
        max_NR_over_RMSNR_high_RMS,
        bins = bins_combi,
        label = "Rej. res.",
        linestyle = '--',
        ec = isu.cmap_custom10[colormode](4),
        **hist_contour_kwargs,
    )

# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel("max(|NR|) / RMSNR")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H14a. Histogram of FE_Tudat / FE_JPLH (position)
### H14a. Histogram of FE_Tudat / FE_JPLH (position)
#region H14a. Hist. FE/FE pos

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_FE-Tudat-JPLH_ratio_pos"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')

# Histogram
FE_ratio = with_data.RMS_formal_error_pos_mag_KM / with_data.RMS_JPLH_ephem_pos_unc_mag_KM
logbins = np.logspace(
    np.log10(FE_ratio.min()),
    np.log10(FE_ratio.max()),
    30
)
ax.hist(
    FE_ratio,
    bins = logbins,
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)
ax.set_xscale('log')

ax.set_ylabel("Count")
ax.set_xlabel(r"$FE_{RMS}^{Tudat} / FE_{RMS}^{JPLH}$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% H14b. Histogram of FE_Tudat / FE_JPLH (velocity)
### H14b. Histogram of FE_Tudat / FE_JPLH (velocity)
#region H14a. Hist. FE/FE vel

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_FE-Tudat-JPLH_ratio_vel"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')

# Histogram
FE_ratio = (with_data.RMS_formal_error_vel_mag_MM / 1000) / with_data.RMS_JPLH_ephem_vel_unc_mag
logbins = np.logspace(
    np.log10(FE_ratio.min()),
    np.log10(FE_ratio.max()),
    30
)
ax.hist(
    FE_ratio,
    bins = logbins,
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)
ax.set_xscale('log')

ax.set_ylabel("Count")
ax.set_xlabel(r"$FE_{RMS}^{Tudat} / FE_{RMS}^{JPLH}$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H15. Histogram of RMS prefit res / RMS postfit res (arcsec)
### H15. Histogram of RMS prefit res / RMS postfit res (arcsec)
#region H15. Hist. RMS pre/post

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_RMS_res_ARCSEC_pre_vs_post"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')

# Histogram
pre_post_RMS_ratio = with_data.prefit_RMS_res_ARCSEC / with_data.RMS_res_est
logbins = np.logspace(
    np.log10(pre_post_RMS_ratio.min()),
    np.log10(pre_post_RMS_ratio.max()),
    30
)
ax.hist(
    pre_post_RMS_ratio,
    bins = logbins,
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)
ax.set_xscale('log')

ax.set_ylabel("Count")
ax.set_xlabel(r"$RMS(O-C)_{pre}\ /\ RMS(O-C)_{post}$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H16a. Histogram of |A2_Tudat| / |A2_NEOCC|
### H16a. Histogram of |A2_Tudat| / |A2_NEOCC|
#region H16a. Hist. A2/A2

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_abs_A2_Tudat_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')

# Data
both_pos = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_lit) == 1)]
both_neg = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_lit) == -1)]
est_pos_lit_neg = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_lit) == -1)]
est_neg_lit_pos = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_lit) == 1)]

same_sign = pd.concat([both_pos, both_neg])
opposite_sign = pd.concat([est_pos_lit_neg, est_neg_lit_pos])
print(f"N data points with same sign: {len(same_sign)}")
print(f"N data points with opposite sign: {len(opposite_sign)}")

A2_ratio_same_sign = abs(same_sign.A2_est) / abs(same_sign.A2_lit)
A2_ratio_opp_sign = abs(opposite_sign.A2_est) / abs(opposite_sign.A2_lit)
min_A2_ratio = min(A2_ratio_same_sign.min(), A2_ratio_opp_sign.min())
max_A2_ratio = max(A2_ratio_same_sign.max(), A2_ratio_opp_sign.max())
# logbins = np.logspace(
#     np.log10(min_A2_ratio),
#     np.log10(max_A2_ratio),
#     30
# )
step = 0.05
bins_linear = np.arange(0.4, 1.75+step, step)

# Histogram
ax.hist(
    A2_ratio_same_sign,
    bins = bins_linear,
    ec = isu.cmap_custom10[colormode](2),
    label = r"$A_2^{Tudat} / A_2^{NEOCC} > 0$",
    **hist_contour_kwargs,
)
ax.hist(
    A2_ratio_opp_sign,
    bins = bins_linear,
    ec = isu.cmap_custom10[colormode](5),
    label = r"$A_2^{Tudat} / A_2^{NEOCC} < 0$",
    **hist_contour_kwargs,
)

# Legend, labels
plt.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel(r"$|\ A_2^{Tudat} /\ A_2^{NEOCC}|$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H16b. Histogram of |A2_Tudat| / |A2_JPL|
### H16b. Histogram of |A2_Tudat| / |A2_JPL|
#region H16b. Hist. A2/A2

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_abs_A2_Tudat_JPL"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')
# ax.set_xscale('log')

# Data
both_pos = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_JPL) == 1)]
both_neg = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_JPL) == -1)]
est_pos_lit_neg = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_JPL) == -1)]
est_neg_lit_pos = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_JPL) == 1)]

same_sign = pd.concat([both_pos, both_neg])
opposite_sign = pd.concat([est_pos_lit_neg, est_neg_lit_pos])
print(f"N data points with same sign: {len(same_sign)}")
print(f"N data points with opposite sign: {len(opposite_sign)}")

A2_ratio_same_sign = abs(same_sign.A2_est) / abs(same_sign.A2_JPL)
A2_ratio_opp_sign = abs(opposite_sign.A2_est) / abs(opposite_sign.A2_JPL)
min_A2_ratio = min(A2_ratio_same_sign.min(), A2_ratio_opp_sign.min())
max_A2_ratio = max(A2_ratio_same_sign.max(), A2_ratio_opp_sign.max())
# logbins = np.logspace(
#     np.log10(min_A2_ratio),
#     np.log10(max_A2_ratio),
#     30
# )
step = 0.05
bins_linear = np.arange(0.2, 1.85+step, step)

# Histogram
ax.hist(
    A2_ratio_same_sign,
    bins = bins_linear,
    ec = isu.cmap_custom10[colormode](2),
    label = r"$A_2^{Tudat} / A_2^{JPL} > 0$",
    **hist_contour_kwargs,
)
if len(A2_ratio_opp_sign) > 0:
    ax.hist(
        A2_ratio_opp_sign,
        bins = bins_linear,
        ec = isu.cmap_custom10[colormode](5),
        label = r"$A_2^{Tudat} / A_2^{JPL} < 0$",
        **hist_contour_kwargs,
    )

# ax.tick_params(
#     which = 'minor',
#     labelbottom = True,
#     color = isu.text_color[colormode],
#     labelsize = isu.tiny_font_size,
#     labelcolor = isu.text_color[colormode],
# )
# ax.xaxis.set_minor_formatter('{x:.1f}')


# Legend, labels
plt.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel(r"$|\ A_2^{Tudat} /\ A_2^{JPL}|$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H16c. Histogram of |A2_JPL| / |A2_NEOCC|
### H16c. Histogram of |A2_JPL| / |A2_NEOCC|
#region H16c. Hist. A2/A2

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_abs_A2_JPL_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')

# Data
both_pos = with_data.loc[(np.sign(with_data.A2_lit) == 1) & (np.sign(with_data.A2_JPL) == 1)]
both_neg = with_data.loc[(np.sign(with_data.A2_lit) == -1) & (np.sign(with_data.A2_JPL) == -1)]
est_pos_lit_neg = with_data.loc[(np.sign(with_data.A2_lit) == 1) & (np.sign(with_data.A2_JPL) == -1)]
est_neg_lit_pos = with_data.loc[(np.sign(with_data.A2_lit) == -1) & (np.sign(with_data.A2_JPL) == 1)]

same_sign = pd.concat([both_pos, both_neg])
opposite_sign = pd.concat([est_pos_lit_neg, est_neg_lit_pos])
print(f"N data points with same sign: {len(same_sign)}")
print(f"N data points with opposite sign: {len(opposite_sign)}")

A2_ratio_same_sign = abs(same_sign.A2_JPL) / abs(same_sign.A2_lit)
A2_ratio_opp_sign = abs(opposite_sign.A2_JPL) / abs(opposite_sign.A2_lit)
min_A2_ratio = min(A2_ratio_same_sign.min(), A2_ratio_opp_sign.min())
max_A2_ratio = max(A2_ratio_same_sign.max(), A2_ratio_opp_sign.max())
logbins = np.logspace(
    np.log10(min_A2_ratio),
    np.log10(max_A2_ratio),
    30
)
step = 0.05
bins_linear = np.arange(0.35, 3.90+step, step)

# Histogram
ax.hist(
    A2_ratio_same_sign,
    bins = bins_linear,
    ec = isu.cmap_custom10[colormode](0),
    label = r"$A_2^{JPL} / A_2^{NEOCC} > 0$",
    **hist_contour_kwargs,
)
ax.hist(
    A2_ratio_opp_sign,
    bins = bins_linear,
    ec = isu.cmap_custom10[colormode](5),
    label = r"$A_2^{JPL} / A_2^{NEOCC} < 0$",
    **hist_contour_kwargs,
)

# Legend, labels
plt.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel(r"$|\ A_2^{JPL} /\ A_2^{NEOCC}|$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H17a Histogram of (σ_A2_est / σ_A2_NEOCC)
### H17a Histogram of (σ_A2_est / σ_A2_NEOCC)
#region H17a Hist. σ_A2 ratio

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_A2_err_ratio_Tudat_NEOCC"
# format = 'pdf'
format = 'png'

ratio_A2_uncs_successes = successes.σ_A2_est / successes.σ_A2_lit
ratio_A2_uncs_rejects = rejects.σ_A2_est / rejects.σ_A2_lit

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')
ax.set_xscale('log')
ax.set_yscale('log')

min_ratio = min(ratio_A2_uncs_successes.min(), ratio_A2_uncs_rejects.min())
max_ratio = max(ratio_A2_uncs_successes.max(), ratio_A2_uncs_rejects.max())
logbins = np.logspace(
    np.log10(min_ratio),
    np.log10(max_ratio),
    30
)

# Histogram
# bins = np.arange(xlims[0], xlims[1]+0.1, binwidth)
ax.hist(
    ratio_A2_uncs_successes,
    bins = logbins,
    label = "Successful",
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.8,
    **hist_contour_kwargs,
)
ax.hist(
    ratio_A2_uncs_rejects,
    bins = logbins,
    label = "Rejected",
    color = isu.cmap_custom10[colormode](3),
    alpha = 0.8,
    linestyle = '--',
    **hist_contour_kwargs,
)

# Legend, labels
plt.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel(
    r"$σ_{A_2}^\text{Tudat} / σ_{A_2}^\text{NEOCC}$",
    fontsize = isu.medium_font_size,
)
# fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% H17b Histogram of (σ_A2_est / σ_A2_JPL)
### H17b Histogram of (σ_A2_est / σ_A2_JPL)
#region H17b Hist. σ_A2 ratio

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_A2_err_ratio_Tudat_JPL"
# format = 'pdf'
format = 'png'

ratio_A2_uncs_successes = successes.σ_A2_est / successes.σ_A2_JPL
ratio_A2_uncs_rejects = rejects.σ_A2_est / rejects.σ_A2_JPL

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')
ax.set_xscale('log')
# ax.set_yscale('log')

min_ratio = min(ratio_A2_uncs_successes.min(), ratio_A2_uncs_rejects.min())
max_ratio = max(ratio_A2_uncs_successes.max(), ratio_A2_uncs_rejects.max())
logbins = np.logspace(
    np.log10(min_ratio),
    np.log10(max_ratio),
    30
)

# Histogram
# bins = np.arange(xlims[0], xlims[1]+0.1, binwidth)
ax.hist(
    ratio_A2_uncs_successes,
    bins = logbins,
    label = "Successful",
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.8,
    **hist_contour_kwargs,
)
ax.hist(
    ratio_A2_uncs_rejects,
    bins = logbins,
    label = "Rejected",
    color = isu.cmap_custom10[colormode](3),
    alpha = 0.8,
    linestyle = '--',
    **hist_contour_kwargs,
)

# Legend, labels
plt.legend(
    loc = 'best',
    **legend_kwargs,
)
ax.set_ylabel("Count")
ax.set_xlabel(
    r"$σ_{A_2}^\text{Tudat} / σ_{A_2}^\text{JPL}$",
    fontsize = isu.medium_font_size,
)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% 3.1 Scatter plot SNR NEOCC vs. Tudat
### 3.1 Scatter plot SNR NEOCC vs. Tudat
#region 3.1 scatter SNR vs lit

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_SNR_A2_NEOCC_vs_SNR_A2_Tudat"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.75])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Vertical and horizontal lines at SNR rejection cutoff
SNR_cutoff = 3.0
ax.axvline(
    SNR_cutoff,
    label = f"SNR = {SNR_cutoff}",
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)
ax.axhline(
    SNR_cutoff,
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)


temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    alpha = 0.7,
)


# Successes - no radar
# successes_no_radar = successes.loc[successes.n_rad_obs_used_NEOCC == 0]
successes_no_radar = successes.loc[((successes.n_rad_obs_used_NEOCC == 0) | (np.isnan(successes.n_rad_obs_used_NEOCC)))]
successes_with_radar = successes.loc[successes.n_rad_obs_used_NEOCC > 0]
print(f"N successes without radar: {len(successes_no_radar)} ({(len(successes_no_radar) / len(successes))*100:.4f}%)")
print(f"N successes with radar: {len(successes_with_radar)} ({(len(successes_with_radar) / len(successes))*100:.4f}%)")
ax.scatter(
    successes_no_radar.SNR_A2_est,
    successes_no_radar.SNR_lit,
    color = isu.cmap_custom10[colormode](2),
    label = f'Successful ({len(successes_no_radar)})',
    marker = 'o',
    s = 20,
    **temp_scatter_kwargs,
)
ax.scatter(
    successes_with_radar.SNR_A2_est,
    successes_with_radar.SNR_lit,
    color = isu.cmap_custom10[colormode](2),
    label = f'Succ. (radar) ({len(successes_with_radar)})',
    marker = 'v',
    s = 25,
    **temp_scatter_kwargs,
)

# Low SNR
# low_SNR_no_radar = low_SNR.loc[low_SNR.n_rad_obs_used_NEOCC == 0]
low_SNR_no_radar = low_SNR.loc[((low_SNR.n_rad_obs_used_NEOCC == 0) | (np.isnan(low_SNR.n_rad_obs_used_NEOCC)))]
low_SNR_with_radar = low_SNR.loc[low_SNR.n_rad_obs_used_NEOCC > 0]
print(f"N low SNR without radar: {len(low_SNR_no_radar)} ({(len(low_SNR_no_radar) / len(low_SNR))*100:.4f}%)")
print(f"N low SNR without radar: {len(low_SNR_with_radar)} ({(len(low_SNR_with_radar) / len(low_SNR))*100:.4f}%)")
ax.scatter(
    low_SNR_no_radar.SNR_A2_est,
    low_SNR_no_radar.SNR_lit,
    color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = f'Rejected ({len(low_SNR_no_radar)})' if dataset_to_load == 'final' else 'Rej. SNR',
    marker = 'o',
    s = 25,
    **temp_scatter_kwargs,
)
ax.scatter(
    low_SNR_with_radar.SNR_A2_est,
    low_SNR_with_radar.SNR_lit,
    color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = f'Rej. (radar) ({len(low_SNR_with_radar)})' if dataset_to_load == 'final' else 'Rej. SNR (radar)',
    marker = 'v',
    s = 25,
    **temp_scatter_kwargs,
)

# High RMS
if len(high_RMS) > 0:
    ax.scatter(
        high_RMS.SNR_A2_est,
        high_RMS.SNR_lit,
        color = isu.cmap_custom10[colormode](4),
        label = 'Rej. res.',
        marker = 'D',
        s = 15,
        **temp_scatter_kwargs,
    )

# Diagonal line
xlims = ax.get_xlim()
ylims = ax.get_ylim()
min_SNR = max(xlims[0], ylims[0])
max_SNR = min(xlims[1], xlims[1])
ax.plot(
    np.linspace(min_SNR, max_SNR, 20),
    np.linspace(min_SNR, max_SNR, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    alpha = 0.7,
    zorder = 2.1,
)

# Axis settings
ax.set_xlabel(r"$SNR_{A_2}^{\text{Tudat}}$")
ax.set_ylabel(r"$SNR_{A_2}^{\text{NEOCC}}$")
ax.set_xscale('log')
ax.set_yscale('log')

# Legend, labels
ax.legend(
    loc = 'best',
    fontsize = isu.tiny_font_size,
    markerscale = 1.5,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 3.2 Scatter plot SNR JPL vs. Tudat
### 3.2 Scatter plot SNR JPL vs. Tudat
#region 3.2 scatter SNR vs JPL

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_SNR_A2_JPL_vs_SNR_A2_Tudat"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.75])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Vertical and horizontal lines at SNR rejection cutoff
SNR_cutoff = 3.0
ax.axvline(
    SNR_cutoff,
    label = f"SNR = {SNR_cutoff}",
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)
ax.axhline(
    SNR_cutoff,
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)


temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    alpha = 0.7,
)

# Successes - no radar
successes_no_radar = successes.loc[np.isnan(successes.n_rad_obs_used_JPL)]
successes_with_radar = successes.loc[successes.n_rad_obs_used_JPL > 0]
print(f"N successes without radar: {len(successes_no_radar)} ({(len(successes_no_radar) / len(successes))*100:.4f}%)")
print(f"N successes with radar: {len(successes_with_radar)} ({(len(successes_with_radar) / len(successes))*100:.4f}%)")
ax.scatter(
    successes_no_radar.SNR_A2_est,
    successes_no_radar.SNR_A2_JPL,
    color = isu.cmap_custom10[colormode](2),
    label = f'Successful ({len(successes_no_radar)})',
    marker = 'o',
    s = 20,
    **temp_scatter_kwargs,
)
ax.scatter(
    successes_with_radar.SNR_A2_est,
    successes_with_radar.SNR_A2_JPL,
    color = isu.cmap_custom10[colormode](2),
    label = f'Succ. (radar) ({len(successes_with_radar)})',
    marker = 'v',
    s = 25,
    **temp_scatter_kwargs,
)

# Low SNR
low_SNR_no_radar = low_SNR.loc[np.isnan(low_SNR.n_rad_obs_used_JPL)]
low_SNR_with_radar = low_SNR.loc[low_SNR.n_rad_obs_used_JPL > 0]
print(f"N low SNR without radar: {len(low_SNR_no_radar)} ({(len(low_SNR_no_radar) / len(low_SNR))*100:.4f}%)")
print(f"N low SNR without radar: {len(low_SNR_with_radar)} ({(len(low_SNR_with_radar) / len(low_SNR))*100:.4f}%)")
ax.scatter(
    low_SNR_no_radar.SNR_A2_est,
    low_SNR_no_radar.SNR_A2_JPL,
    color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = f'Rejected ({len(low_SNR_no_radar)})' if dataset_to_load == 'final' else 'Rej. SNR',
    marker = 'o',
    s = 25,
    **temp_scatter_kwargs,
)
ax.scatter(
    low_SNR_with_radar.SNR_A2_est,
    low_SNR_with_radar.SNR_A2_JPL,
    color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = f'Rej. (radar) ({len(low_SNR_with_radar)})' if dataset_to_load == 'final' else 'Rej. SNR (radar)',
    marker = 'v',
    s = 25,
    **temp_scatter_kwargs,
)

# High RMS
if len(high_RMS) > 0:
    ax.scatter(
        high_RMS.SNR_A2_est,
        high_RMS.SNR_A2_JPL,
        color = isu.cmap_custom10[colormode](4),
        label = 'Rej. res.',
        marker = 'D',
        s = 15,
        **temp_scatter_kwargs,
    )

# Diagonal line
xlims = ax.get_xlim()
ylims = ax.get_ylim()
min_SNR = max(xlims[0], ylims[0])
max_SNR = min(xlims[1], xlims[1])
ax.plot(
    np.linspace(min_SNR, max_SNR, 20),
    np.linspace(min_SNR, max_SNR, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    alpha = 0.7,
    zorder = 2.1,
)

# Axis settings
ax.set_xlabel(r"$SNR_{A_2}^{\text{Tudat}}$")
ax.set_ylabel(r"$SNR_{A_2}^{\text{JPL}}$")
ax.set_xscale('log')
ax.set_yscale('log')

# Legend, labels
ax.legend(
    loc = 'best',
    fontsize = isu.tiny_font_size,
    markerscale = 1.5,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% 3.3 Scatter plot SNR JPL vs. NEOCC
### 3.3 Scatter plot SNR JPL vs. NEOCC
#region 3.3 scatter SNR vs JPL,NEOCC

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_SNR_A2_JPL_vs_SNR_A2_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.75])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Vertical and horizontal lines at SNR rejection cutoff
SNR_cutoff = 3.0
ax.axvline(
    SNR_cutoff,
    label = f"SNR = {SNR_cutoff}",
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)
ax.axhline(
    SNR_cutoff,
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)


temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    alpha = 0.7,
)

# Successes - no radar
radar_used_by_JPL_only = with_data.loc[(with_data.n_rad_obs_used_JPL > 0) & ((with_data.n_rad_obs_used_NEOCC == 0) | (np.isnan(with_data.n_rad_obs_used_NEOCC)))]
radar_used_by_NEOCC_only = with_data.loc[(np.isnan(with_data.n_rad_obs_used_JPL)) & (with_data.n_rad_obs_used_NEOCC > 0)]
radar_used_by_both = with_data.loc[(with_data.n_rad_obs_used_JPL > 0) & (with_data.n_rad_obs_used_NEOCC > 0)]
radar_used_by_neither = with_data.loc[(np.isnan(with_data.n_rad_obs_used_JPL)) & ((with_data.n_rad_obs_used_NEOCC == 0) | (np.isnan(with_data.n_rad_obs_used_NEOCC)))]

print(f"N where radar used by JPL only: {len(radar_used_by_JPL_only)} ({(len(radar_used_by_JPL_only) / len(with_data))*100:.4f}%)")
print(f"N where radar used by NEOCC only: {len(radar_used_by_NEOCC_only)} ({(len(radar_used_by_NEOCC_only) / len(with_data))*100:.4f}%)")
print(f"N where radar used by both: {len(radar_used_by_both)} ({(len(radar_used_by_both) / len(with_data))*100:.4f}%)")
print(f"N where radar used by neither: {len(radar_used_by_neither)} ({(len(radar_used_by_neither) / len(with_data))*100:.4f}%)")

ax.scatter(
    radar_used_by_neither.SNR_lit,
    radar_used_by_neither.SNR_A2_JPL,
    color = isu.cmap_custom10[colormode](6),
    label = 'No radar',
    marker = 'o',
    s = 20,
    **temp_scatter_kwargs,
)
ax.scatter(
    radar_used_by_both.SNR_lit,
    radar_used_by_both.SNR_A2_JPL,
    color = isu.cmap_custom10[colormode](5),
    label = 'Both radar',
    marker = 'v',
    s = 25,
    **temp_scatter_kwargs,
)
ax.scatter(
    radar_used_by_JPL_only.SNR_lit,
    radar_used_by_JPL_only.SNR_A2_JPL,
    color = isu.cmap_custom10[colormode](7),
    label = 'JPL radar',
    marker = 's',
    s = 25,
    **temp_scatter_kwargs,
)
if len(radar_used_by_NEOCC_only) > 0:
    ax.scatter(
        radar_used_by_NEOCC_only.SNR_lit,
        radar_used_by_NEOCC_only.SNR_A2_JPL,
        color = isu.cmap_custom10[colormode](9),
        label = 'NEOCC radar',
        marker = 'X',
        s = 25,
        **temp_scatter_kwargs,
    )

# Diagonal line
xlims = ax.get_xlim()
ylims = ax.get_ylim()
min_SNR = max(xlims[0], ylims[0])
max_SNR = min(xlims[1], xlims[1])
ax.plot(
    np.linspace(min_SNR, max_SNR, 20),
    np.linspace(min_SNR, max_SNR, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    alpha = 0.7,
    zorder = 2.1,
)

# Axis settings
ax.set_xlabel(r"$SNR_{A_2}^{\text{NEOCC}}$")
ax.set_ylabel(r"$SNR_{A_2}^{\text{JPL}}$")
ax.set_xscale('log')
ax.set_yscale('log')

# Legend, labels
ax.legend(
    loc = 'best',
    fontsize = isu.tiny_font_size,
    markerscale = 1.5,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% 4.1 Scatter plot mean |NR| vs SNR_A2
### 4.1 Scatter plot mean |NR| vs SNR_A2
#region 4.1 scatter MANR vs SNR

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
# fig_name = f"scatter_RMSNR_vs_SNR_A2"
fig_name = f"scatter_MANR_vs_SNR_A2"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.75])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Vertical line at SNR rejection cutoff
SNR_cutoff = 3.0
ax.axvline(
    SNR_cutoff,
    label = f"SNR = {SNR_cutoff}",
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)

temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    alpha = 0.9,
)

# Successes
ax.scatter(
    successes.SNR_A2_est,
    # successes.RMS_norm_res_est,
    successes.mean_abs_norm_res_est,
    color = isu.cmap_custom10[colormode](2),
    label = 'Successful',
    marker = 'o',
    s = 20,
    **temp_scatter_kwargs,
)
# Low SNR
ax.scatter(
    low_SNR.SNR_A2_est,
    # low_SNR.RMS_norm_res_est,
    low_SNR.mean_abs_norm_res_est,
    color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    marker = 'X',
    s = 25,
    **temp_scatter_kwargs,
)
# High RMS
if len(high_RMS) > 0:
    ax.scatter(
        high_RMS.SNR_A2_est,
        # high_RMS.RMS_norm_res_est,
        high_RMS.mean_abs_norm_res_est,
        color = isu.cmap_custom10[colormode](4),
        label = 'Rej. res.',
        marker = 'D',
        s = 15,
        **temp_scatter_kwargs,
    )

# Axis settings
ax.set_xlabel(r"$SNR_{A_2}$")
ax.set_ylabel("Mean abs. norm. res.")
ax.set_xscale('log')
# Legend, labels
ax.legend(
    loc = 'best',
    **legend_kwargs,
)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 4.2 Scatter plot mean |NR| vs SNR_A2, w. histograms
### 4.2 Scatter plot mean |NR| vs SNR_A2, w. histograms
#region 4.2 MANR vs SNR

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_MANR_vs_SNR_A2_with-histograms"
# format = 'pdf'
format = 'png'

# Plot
fig = plt.figure(
    figsize = np.array([1.0,0.85])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
gs = gridspec.GridSpec(
    2, 2,
    figure = fig,
    width_ratios = [1, 2.5],
    height_ratios = [2, 1],
    hspace = 0.005,
    wspace = 0.005,
)
# Set up axes
ax_scat = fig.add_subplot(gs[0, 1])
ax_MANR_hist = fig.add_subplot(gs[0, 0], sharey=ax_scat)
ax_SNR_hist = fig.add_subplot(gs[1, 1], sharex=ax_scat)
ax_blank_L = fig.add_subplot(gs[1, 0])
ax_scat = isu.set_default_2d_ax_settings(ax_scat, colormode)
ax_MANR_hist = isu.set_default_2d_ax_settings(ax_MANR_hist, colormode)
ax_SNR_hist = isu.set_default_2d_ax_settings(ax_SNR_hist, colormode)

## Scatter plot
# Vertical line at SNR rejection cutoff
SNR_cutoff = 3.0
ax_scat.axvline(
    SNR_cutoff,
    label = f"SNR = {SNR_cutoff}",
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)

temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    alpha = 0.9,
)

# Successes
ax_scat.set_xscale('log')
ax_scat.scatter(
    successes.SNR_A2_est,
    # successes.RMS_norm_res_est,
    successes.mean_abs_norm_res_est,
    color = isu.cmap_custom10[colormode](2),
    label = 'Successful',
    marker = 'o',
    s = 20,
    **temp_scatter_kwargs,
)
# Low SNR
ax_scat.scatter(
    low_SNR.SNR_A2_est,
    # low_SNR.RMS_norm_res_est,
    low_SNR.mean_abs_norm_res_est,
    color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    marker = 'X',
    s = 25,
    **temp_scatter_kwargs,
)
# High RMS
if len(high_RMS) > 0:
    ax_scat.scatter(
        high_RMS.SNR_A2_est,
        # high_RMS.RMS_norm_res_est,
        high_RMS.mean_abs_norm_res_est,
        color = isu.cmap_custom10[colormode](4),
        label = 'Rej. res.',
        marker = 'D',
        s = 15,
        **temp_scatter_kwargs,
    )


## MANR Histogram
RMS_cutoff_plot = 0.75
step = 0.05
bins_combi = np.arange(0.1, RMS_cutoff_plot+step, step)
ax_MANR_hist.hist(
    successes.mean_abs_norm_res_est,
    bins = bins_combi,
    orientation = "horizontal",
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax_MANR_hist.hist(
    low_SNR.mean_abs_norm_res_est,
    bins = bins_combi,
    orientation = "horizontal",
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    ax_MANR_hist.hist(
        high_RMS.mean_abs_norm_res_est,
        bins = bins_combi,
        orientation = "horizontal",
        label = "Rej. res.",
        ec = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        **hist_contour_kwargs,
    )


## SNR Histogram
# Vertical line at SNR rejection cutoff
SNR_cutoff = 3.0
ax_SNR_hist.axvline(
    SNR_cutoff,
    label = f"SNR = {SNR_cutoff}",
    linewidth = 2,
    linestyle = 'dotted',
    color = isu.cmap_custom10[colormode](3),
    zorder = 2.6,
)
# Histogram curves
min_SNR = min(successes.SNR_A2_est.min(), rejects.SNR_A2_est.min())
max_SNR = max(successes.SNR_A2_est.max(), rejects.SNR_A2_est.max())
logbins_rejects = np.logspace(np.log10(min_SNR), np.log10(SNR_cutoff), 20)
logbins_successes = np.logspace(np.log10(SNR_cutoff), np.log10(max_SNR), 20)
ax_SNR_hist.hist(
    successes.SNR_A2_est,
    # successes.SNR_est,
    bins = logbins_successes,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax_SNR_hist.hist(
    low_SNR.SNR_A2_est,
    # rejects.SNR_est,
    bins = logbins_rejects,
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    ax_SNR_hist.hist(
        high_RMS.SNR_A2_est,
        # rejects.SNR_est,
        bins = logbins_successes,
        label = "Rej. res.",
        linestyle = '--',
        ec = isu.cmap_custom10[colormode](4),
        **hist_contour_kwargs,
    )

# Legend, axis settings, labels, etc.
temp_legend_kwargs = dict(
    markerscale = 1.5,
    fontsize = isu.tiny_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax_scat.legend(loc = 'best', **temp_legend_kwargs)
ax_scat.tick_params(labelbottom=False, labelleft=False)
#
ax_MANR_hist.legend(loc = 'best', **temp_legend_kwargs)
ax_MANR_hist.set_ylabel("Mean of abs. norm. res.")
ax_MANR_hist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#
ax_SNR_hist.legend(loc = 'upper left', **temp_legend_kwargs)
ax_SNR_hist.set_xlabel(r"Signal-to-noise ratio $SNR_{A_2}$")
ax_SNR_hist.tick_params(right=True, labelright=True, left=False, labelleft=False)
#
temp_text_kwargs = dict(
    s = "Count",
    color = isu.text_color[colormode],
    fontsize = isu.medium_font_size,
)
ax_blank_L.text(
    x = 0.4,
    y = 1.0,
    horizontalalignment = 'center',
    verticalalignment = 'top',
    transform = ax_blank_L.transAxes,
    **temp_text_kwargs,
)
ax_blank_L.text(
    x = 1.0,
    y = 0.4,
    horizontalalignment = 'right',
    verticalalignment = 'center',
    transform = ax_blank_L.transAxes,
    rotation = 90,
    rotation_mode = 'default',
    **temp_text_kwargs
)

# Spines, ticks, facecolor for blank axes
for ax in [ax_blank_L]:
    for spine in ax.spines.values(): # ax borders
        spine.set_edgecolor(isu.fig_background[colormode])
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.set_facecolor(isu.fig_background[colormode])
    ax.grid(visible=False)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 5.1 Scatter plot of A2_est vs A2_lit
### 5.1 Scatter plot of A2_est vs A2_lit
#region 5.1 scatter A2

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_A2_Tudat_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,1.0])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

lim = 2e-11
xlims = np.array([-lim, lim])
ylims = np.array([-lim, lim])
ax.set_xlim((xlims[0], xlims[1]))
ax.set_ylim((ylims[0], ylims[1]))
ax.axvline(linewidth=1, color=isu.text_color[colormode])
ax.axhline(linewidth=1, color=isu.text_color[colormode])
# ax.set_xscale('')

# Quantify how many points are excluded by lims
x_excluded = np.logical_not(successes.A2_lit.between(xlims[0], xlims[1], inclusive='both'))
y_excluded = np.logical_not(successes.A2_est.between(ylims[0], ylims[1], inclusive='both'))
n_excluded = sum(np.logical_or(x_excluded, y_excluded))
# Add text box
text_box_note = f"N data points: {len(successes.A2_lit)}"
if n_excluded > 0:
    text_box_note += f"\nN outside plot range: {n_excluded}"
# ax.text(
#     x = 0.03,
#     y = 0.97,
#     s = text_box_note,
#     horizontalalignment = 'left',
#     verticalalignment = 'top',
#     transform = ax.transAxes,
#     color = isu.text_color[colormode],
#     fontsize = isu.tiny_font_size,
#     family = 'monospace',
#     bbox = text_bbox,
# )
text_box_note += f"\nError bar size: 1σ"
print(text_box_note)

ax.errorbar(
    x = successes.A2_lit,
    y = successes.A2_est,
    xerr = successes.σ_A2_lit,
    yerr = successes.σ_A2_est,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](2),
    # ecolor = isu.cmap_custom10[colormode](1),
    **errorbar_scatter_kwargs,
)
# ax.errorbar(
#     x = rejects.A2_lit,
#     y = rejects.A2_est,
#     xerr = rejects.σ_A2_lit,
#     yerr = rejects.σ_A2_est,
#     label = "Rejected",
#     color = isu.cmap_custom10[colormode](1),
#     # ecolor = isu.cmap_custom10[colormode](1),
#     **errorbar_scatter_kwargs,
# )

# Dashed diagonal line
ax.plot(
    np.linspace(xlims[0], xlims[1], 100),
    np.linspace(xlims[0], xlims[1], 100),
    color = isu.text_color[colormode],
    linestyle = '--',
    linewidth = 2,
    label = r"$A2_{est.} =\ A2_{lit.}$",
    alpha = 0.7,
)

# Legend, labels
ax.legend(
    loc = 'lower right',
    **legend_kwargs,
)
ax.set_ylabel(r"$A_2$ , Tudat [$m/s^2$]")
ax.set_xlabel(r"$A_2$ , NEOCC [$m/s^2$]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 5.2a Scatter plot of |A2_est| vs |A2_NEOCC|
### 5.2a Scatter plot of |A2_est| vs |A2_NEOCC|
#region 5.2a scatter |A2|

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_abs_A2_Tudat_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.set_xscale('log')
ax.set_yscale('log')

print("Error bar size: 1σ")

# Scatter plots (2 colours) - successes
both_pos = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_lit) == 1)]
both_neg = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_lit) == -1)]
est_pos_lit_neg = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_lit) == -1)]
est_neg_lit_pos = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_lit) == 1)]

same_sign = pd.concat([both_pos, both_neg])
opposite_sign = pd.concat([est_pos_lit_neg, est_neg_lit_pos])
print(f"N data points with same sign: {len(same_sign)}")
print(f"N data points with opposite sign: {len(opposite_sign)}")

ax.errorbar(
    x = abs(same_sign.A2_lit),
    y = abs(same_sign.A2_est),
    xerr = same_sign.σ_A2_lit,
    yerr = same_sign.σ_A2_est,
    label = r"$A_2^{Tudat} / A_2^{NEOCC} > 0$",
    color = isu.cmap_custom10[colormode](2),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    **errorbar_scatter_kwargs,
)
ax.errorbar(
    x = abs(opposite_sign.A2_lit),
    y = abs(opposite_sign.A2_est),
    xerr = opposite_sign.σ_A2_lit,
    yerr = opposite_sign.σ_A2_est,
    label = r"$A_2^{Tudat} / A_2^{NEOCC} < 0$",
    color = isu.cmap_custom10[colormode](5),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    fmt = 's',
    markersize = 4.5,
    alpha = 0.4,
    elinewidth = 1,
)
xlims = ax.get_xlim()
ylims = ax.get_ylim()


# Rejects
ax.errorbar(
    x = abs(rejects.A2_lit),
    y = abs(rejects.A2_est),
    xerr = rejects.σ_A2_lit,
    yerr = rejects.σ_A2_est,
    label = "Rejected",
    color = isu.cmap_custom10[colormode](3),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    fmt = 'o',
    markersize = 2.5,
    alpha = 0.1,
    elinewidth = 1,
)
# # Scatter plots (2 colours) - rejects
# both_pos = rejects.loc[(np.sign(rejects.A2_est) == 1) & (np.sign(rejects.A2_lit) == 1)]
# both_neg = rejects.loc[(np.sign(rejects.A2_est) == -1) & (np.sign(rejects.A2_lit) == -1)]
# est_pos_lit_neg = rejects.loc[(np.sign(rejects.A2_est) == 1) & (np.sign(rejects.A2_lit) == -1)]
# est_neg_lit_pos = rejects.loc[(np.sign(rejects.A2_est) == -1) & (np.sign(rejects.A2_lit) == 1)]
# same_sign = pd.concat([both_pos, both_neg])
# # opposite_sign = pd.concat([est_pos_lit_neg, est_neg_lit_pos]) #!!! switch to this
# opposite_sign = est_pos_lit_neg
# ax.errorbar(
#     x = abs(same_sign.A2_lit),
#     y = abs(same_sign.A2_est),
#     xerr = same_sign.σ_A2_lit,
#     yerr = same_sign.σ_A2_est,
#     label = r"$A_2^{Tudat} / A_2^{NEOCC} > 0$",
#     color = isu.cmap_custom10[colormode](1),
#     markeredgecolor = scatter_edgecolors[colormode],
#     markeredgewidth = 0.5,
#     fmt = 'o',
#     markersize = 2.5,
#     alpha = 0.1,
#     elinewidth = 1,
# )
# ax.errorbar(
#     x = abs(opposite_sign.A2_lit),
#     y = abs(opposite_sign.A2_est),
#     xerr = opposite_sign.σ_A2_lit,
#     yerr = opposite_sign.σ_A2_est,
#     label = r"$A_2^{Tudat} / A_2^{NEOCC} < 0$",
#     color = isu.cmap_custom10[colormode](4),
#     markeredgecolor = scatter_edgecolors[colormode],
#     markeredgewidth = 0.5,
#     fmt = 's',
#     markersize = 4.5,
#     alpha = 0.1,
#     elinewidth = 1,
# )


# Dashed diagonal line
min_abs_A2 = min(abs(successes.A2_est).min(), abs(successes.A2_lit).min())
max_abs_A2 = min(abs(successes.A2_est).max(), abs(successes.A2_lit).max())
ax.plot(
    np.linspace(min_abs_A2, max_abs_A2, 20),
    np.linspace(min_abs_A2, max_abs_A2, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    label = r"$A_2^{Tudat} = A_2^{NEOCC}$",
)
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Legend, labels
leg = fig.legend(
    loc = 'outside center right',
    **legend_kwargs,
)
# for lh in leg.legend_handles:
#     # Make legend items full opacity
#     lh.set_alpha(1)

# Labels / titles
ax.set_ylabel(r"$|\ A_2|$, Tudat [$m/s^2$]")
ax.set_xlabel(r"$|\ A_2|$, NEOCC [$m/s^2$]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 5.2b Scatter plot of |A2_est| vs |A2_JPL|
### 5.2b Scatter plot of |A2_est| vs |A2_JPL|
#region 5.2b scatter |A2|

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_abs_A2_Tudat_JPL"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.set_xscale('log')
ax.set_yscale('log')

print("Error bar size: 1σ")

# Scatter plots (4 colours)
both_pos = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_JPL) == 1)]
both_neg = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_JPL) == -1)]
est_pos_JPL_neg = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_JPL) == -1)]
est_neg_JPL_pos = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_JPL) == 1)]

# Scatter plots (2 colours)
same_sign = pd.concat([both_pos, both_neg])
opposite_sign = pd.concat([est_pos_JPL_neg, est_neg_JPL_pos])
print(f"N data points with same sign: {len(same_sign)}")
print(f"N data points with opposite sign: {len(opposite_sign)}")

ax.errorbar(
    x = abs(same_sign.A2_JPL),
    y = abs(same_sign.A2_est),
    xerr = same_sign.σ_A2_JPL,
    yerr = same_sign.σ_A2_est,
    label = r"$A_2^{Tudat} / A_2^{JPL} > 0$",
    color = isu.cmap_custom10[colormode](2),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    **errorbar_scatter_kwargs,
)
ax.errorbar(
    x = abs(opposite_sign.A2_JPL),
    y = abs(opposite_sign.A2_est),
    xerr = opposite_sign.σ_A2_JPL,
    yerr = opposite_sign.σ_A2_est,
    label = r"$A_2^{Tudat} / A_2^{JPL} < 0$",
    color = isu.cmap_custom10[colormode](5),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    fmt = 's',
    markersize = 4.5,
    alpha = 0.4,
    elinewidth = 1,
)
xlims = ax.get_xlim()
ylims = ax.get_ylim()

# Rejects
ax.errorbar(
    x = abs(rejects.A2_JPL),
    y = abs(rejects.A2_est),
    xerr = rejects.σ_A2_JPL,
    yerr = rejects.σ_A2_est,
    label = "Rejected",
    color = isu.cmap_custom10[colormode](3),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    fmt = 'o',
    markersize = 2.5,
    alpha = 0.1,
    elinewidth = 1,
)

# Dashed diagonal line
min_abs_A2 = min(abs(successes.A2_est).min(), abs(successes.A2_JPL).min())
max_abs_A2 = min(abs(successes.A2_est).max(), abs(successes.A2_JPL).max())
ax.plot(
    np.linspace(min_abs_A2, max_abs_A2, 20),
    np.linspace(min_abs_A2, max_abs_A2, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    label = r"$A_2^{Tudat} = A_2^{JPL}$",
)
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Legend, labels
leg = fig.legend(
    loc = 'outside center right',
    **legend_kwargs,
)
for lh in leg.legend_handles:
    # Make legend items full opacity
    lh.set_alpha(1)

# Labels / titles
ax.set_ylabel(r"$|\ A_2|$, Tudat [$m/s^2$]")
ax.set_xlabel(r"$|\ A_2|$, JPL [$m/s^2$]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)
#%% 5.2c Scatter plots of |A2_est| vs |A2_NEOCC| and |A2_est| vs |A2_JPL|
### 5.2c Scatter plots of |A2_est| vs |A2_NEOCC| and |A2_est| vs |A2_JPL|
#region 5.2c Scatters |A2|

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_abs_A2_subplots_Tudat_NEOCC_JPL"
# format = 'pdf'
format = 'png'
include_rejects = False

fig, axs = plt.subplots(
    1, 2,
    figsize = np.array([1.0,0.50])*isu.full_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
print("Error bar size: 1σ")

for ax in axs.flatten():
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    ax.set_xscale('log')
    ax.set_yscale('log')

# Tudat vs. NEOCC (left subplot)
both_pos = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_lit) == 1)]
both_neg = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_lit) == -1)]
est_pos_lit_neg = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_lit) == -1)]
est_neg_lit_pos = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_lit) == 1)]

same_sign = pd.concat([both_pos, both_neg])
opposite_sign = pd.concat([est_pos_lit_neg, est_neg_lit_pos])
print(f"Tudat vs. NEOCC: N data points with same sign: {len(same_sign)}")
print(f"Tudat vs. NEOCC: N data points with opposite sign: {len(opposite_sign)}")

axs[0].errorbar(
    x = abs(same_sign.A2_lit),
    y = abs(same_sign.A2_est),
    xerr = same_sign.σ_A2_lit,
    yerr = same_sign.σ_A2_est,
    label = r"$A_2^{Tudat} / A_2^{lit.} > 0$",
    color = isu.cmap_custom10[colormode](2),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    **errorbar_scatter_kwargs,
)
axs[0].errorbar(
    x = abs(opposite_sign.A2_lit),
    y = abs(opposite_sign.A2_est),
    xerr = opposite_sign.σ_A2_lit,
    yerr = opposite_sign.σ_A2_est,
    label = r"$A_2^{Tudat} / A_2^{lit.} < 0$",
    color = isu.cmap_custom10[colormode](5),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.8,
    fmt = 's',
    markersize = 4.5,
    alpha = 0.6 if colormode else 0.4,
    elinewidth = 1,
)
xlims_NEOCC = axs[0].get_xlim()
ylims_NEOCC = axs[0].get_ylim()

# Rejects
if include_rejects:
    axs[0].errorbar(
        x = abs(rejects.A2_lit),
        y = abs(rejects.A2_est),
        xerr = rejects.σ_A2_lit,
        yerr = rejects.σ_A2_est,
        label = "Rejected",
        color = isu.cmap_custom10[colormode](3),
        markeredgecolor = scatter_edgecolors[colormode],
        markeredgewidth = 0.5,
        fmt = 'o',
        markersize = 2.5,
        alpha = 0.1,
        elinewidth = 1,
    )

# Dashed diagonal line
min_abs_A2 = min(abs(successes.A2_est).min(), abs(successes.A2_lit).min())
max_abs_A2 = min(abs(successes.A2_est).max(), abs(successes.A2_lit).max())
axs[0].plot(
    np.linspace(min_abs_A2, max_abs_A2, 20),
    np.linspace(min_abs_A2, max_abs_A2, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    label = r"$A_2^{Tudat} = A_2^{lit.}$",
)

# Labels / titles
axs[0].set_ylabel(r"$|\ A_2|$, Tudat [$m/s^2$]")
axs[0].set_xlabel(r"$|\ A_2|$, NEOCC [$m/s^2$]")



# Tudat vs. JPL (right subplot)
both_pos = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_JPL) == 1)]
both_neg = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_JPL) == -1)]
est_pos_JPL_neg = successes.loc[(np.sign(successes.A2_est) == 1) & (np.sign(successes.A2_JPL) == -1)]
est_neg_JPL_pos = successes.loc[(np.sign(successes.A2_est) == -1) & (np.sign(successes.A2_JPL) == 1)]

same_sign = pd.concat([both_pos, both_neg])
opposite_sign = pd.concat([est_pos_JPL_neg, est_neg_JPL_pos])
print(f"Tudat vs. JPL: N data points with same sign: {len(same_sign)}")
print(f"Tudat vs. JPL: N data points with opposite sign: {len(opposite_sign)}")

axs[1].errorbar(
    x = abs(same_sign.A2_JPL),
    y = abs(same_sign.A2_est),
    xerr = same_sign.σ_A2_JPL,
    yerr = same_sign.σ_A2_est,
    # label = r"$A_2^{Tudat} / A_2^{JPL} > 0$",
    color = isu.cmap_custom10[colormode](2),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    **errorbar_scatter_kwargs,
)
axs[1].errorbar(
    x = abs(opposite_sign.A2_JPL),
    y = abs(opposite_sign.A2_est),
    xerr = opposite_sign.σ_A2_JPL,
    yerr = opposite_sign.σ_A2_est,
    # label = r"$A_2^{Tudat} / A_2^{JPL} < 0$",
    color = isu.cmap_custom10[colormode](5),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.8,
    fmt = 's',
    markersize = 4.5,
    alpha = 0.4,
    elinewidth = 1,
)
xlims_JPL = axs[1].get_xlim()
ylims_JPL = axs[1].get_ylim()

# Rejects
if include_rejects:
    axs[1].errorbar(
        x = abs(rejects.A2_JPL),
        y = abs(rejects.A2_est),
        xerr = rejects.σ_A2_JPL,
        yerr = rejects.σ_A2_est,
        # label = "Rejected",
        color = isu.cmap_custom10[colormode](3),
        markeredgecolor = scatter_edgecolors[colormode],
        markeredgewidth = 0.5,
        fmt = 'o',
        markersize = 2.5,
        alpha = 0.1,
        elinewidth = 1,
    )

# Dashed diagonal line
min_abs_A2 = min(abs(successes.A2_est).min(), abs(successes.A2_JPL).min())
max_abs_A2 = min(abs(successes.A2_est).max(), abs(successes.A2_JPL).max())
axs[1].plot(
    np.linspace(min_abs_A2, max_abs_A2, 20),
    np.linspace(min_abs_A2, max_abs_A2, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    # label = r"$A_2^{Tudat} = A_2^{JPL}$",
)

# Labels / titles
axs[1].set_xlabel(r"$|\ A_2|$, JPL [$m/s^2$]")
axs[1].tick_params(labelleft=False)

for ax in axs.flatten():
    # Set same axis limits for both plots
    ax.set_xlim(
        min(xlims_NEOCC[0], xlims_JPL[0]),
        max(xlims_NEOCC[1], xlims_JPL[1]),
    )
    ax.set_ylim(
        min(ylims_NEOCC[0], ylims_JPL[0]),
        max(ylims_NEOCC[1], ylims_JPL[1]),
    )
    # Add ticks to top and right
    ax.tick_params(which='both', right = True, top = True)



# Legend, labels
leg = fig.legend(
    loc = 'outside upper center',
    ncol = 4,
    **legend_kwargs,
)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 5.2d Scatter plot of |A2_JPL| vs |A2_NEOCC|
### 5.2d Scatter plot of |A2_JPL| vs |A2_NEOCC|
#region 5.2d scatter |A2|

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_abs_A2_JPL_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.set_xscale('log')
ax.set_yscale('log')

print("Error bar size: 1σ")

# Scatter plots (4 colours)
both_pos = with_data.loc[(np.sign(with_data.A2_lit) == 1) & (np.sign(with_data.A2_JPL) == 1)]
both_neg = with_data.loc[(np.sign(with_data.A2_lit) == -1) & (np.sign(with_data.A2_JPL) == -1)]
NEOCC_pos_JPL_neg = with_data.loc[(np.sign(with_data.A2_lit) == 1) & (np.sign(with_data.A2_JPL) == -1)]
NEOCC_neg_JPL_pos = with_data.loc[(np.sign(with_data.A2_lit) == -1) & (np.sign(with_data.A2_JPL) == 1)]
# both_pos = successes.loc[(np.sign(successes.A2_lit) == 1) & (np.sign(successes.A2_JPL) == 1)]
# both_neg = successes.loc[(np.sign(successes.A2_lit) == -1) & (np.sign(successes.A2_JPL) == -1)]
# NEOCC_pos_JPL_neg = successes.loc[(np.sign(successes.A2_lit) == 1) & (np.sign(successes.A2_JPL) == -1)]
# NEOCC_neg_JPL_pos = successes.loc[(np.sign(successes.A2_lit) == -1) & (np.sign(successes.A2_JPL) == 1)]

# Scatter plots (2 colours)
same_sign = pd.concat([both_pos, both_neg])
opposite_sign = pd.concat([NEOCC_pos_JPL_neg, NEOCC_neg_JPL_pos])
print(f"N data points with same sign: {len(same_sign)}")
print(f"N data points with opposite sign: {len(opposite_sign)}")

ax.errorbar(
    x = abs(same_sign.A2_lit),
    y = abs(same_sign.A2_JPL),
    xerr = same_sign.σ_A2_lit,
    yerr = same_sign.σ_A2_JPL,
    label = r"$A_2^{JPL} / A_2^{NEOCC} > 0$",
    color = isu.cmap_custom10[colormode](0),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.5,
    **errorbar_scatter_kwargs,
)
ax.errorbar(
    x = abs(opposite_sign.A2_lit),
    y = abs(opposite_sign.A2_JPL),
    xerr = opposite_sign.σ_A2_lit,
    yerr = opposite_sign.σ_A2_JPL,
    label = r"$A_2^{JPL} / A_2^{NEOCC} < 0$",
    color = isu.cmap_custom10[colormode](5),
    markeredgecolor = scatter_edgecolors[colormode],
    markeredgewidth = 0.8,
    fmt = 's',
    markersize = 4.5,
    alpha = 0.4,
    elinewidth = 1,
)
xlims = ax.get_xlim()
ylims = ax.get_ylim()

# Dashed diagonal line
# min_abs_A2 = min(abs(successes.A2_est).min(), abs(successes.A2_JPL).min())
# max_abs_A2 = min(abs(successes.A2_est).max(), abs(successes.A2_JPL).max())
min_abs_A2 = min(abs(with_data.A2_est).min(), abs(with_data.A2_JPL).min())
max_abs_A2 = min(abs(with_data.A2_est).max(), abs(with_data.A2_JPL).max())
ax.plot(
    np.linspace(min_abs_A2, max_abs_A2, 20),
    np.linspace(min_abs_A2, max_abs_A2, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    label = r"$A_2^{JPL} = A_2^{NEOCC}$",
)
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Legend, labels
leg = fig.legend(
    loc = 'outside center right',
    **legend_kwargs,
)
for lh in leg.legend_handles:
    # Make legend items full opacity
    lh.set_alpha(1)

# Labels / titles
ax.set_xlabel(r"$|\ A_2|$, NEOCC [$m/s^2$]")
ax.set_ylabel(r"$|\ A_2|$, JPL [$m/s^2$]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 5.4 Scatter plot of A2_est vs A2_lit, split by k_lim
### 5.4 Scatter plot of A2_est vs A2_lit, split by k_lim
#region 5.4 scatter A2 split

# # Plot output
# colormode = 1
# pb = 'show'
# # pb = 'save'
# fig_name = f""
# # format = 'pdf'
# format = 'png'

# # Plot settings
# k_lim = 2
# plot_good_successes = 1 # estimations for which |k1| < k_lim
# plot_bad_successes = 1 # estimations for which |k1| > k_lim
# plot_error_bars = 0
# error_bar_size = 2 # in units of sigma (use int only)
# plot_acceptable_range = 1 # pink area
# acceptable_range_factor = 2.5

# fig, ax = plt.subplots(
#     1, 1,
#     figsize = (6, 6),
#     facecolor = isu.fig_background[colormode],
# )
# ax = isu.set_default_2d_ax_settings(ax, colormode)
# lim = 2e-11
# xlims = np.array([-lim, lim])
# ylims = np.array([-lim, lim])
# ax.set_xlim((xlims[0], xlims[1]))
# ax.set_ylim((ylims[0], ylims[1]))

# ax.axvline(linewidth=1.5, color=isu.text_color[colormode])
# ax.axhline(linewidth=1.5, color=isu.text_color[colormode])
# # ax.set_xscale('')

# # Split successful estimations into those with |k1| < k_lim and those with |k1| > k_lim
# successes['within_k_lim'] = np.abs(successes['A2_lit+Xσ_A2_est']) <= np.ones(len(successes))*k_lim
# good_successes = successes.loc[np.abs(successes['A2_lit+Xσ_A2_est']) <= k_lim]
# bad_successes = successes.loc[np.abs(successes['A2_lit+Xσ_A2_est']) > k_lim]

# # Quantify how many points are excluded by lims
# x_good_excluded = np.logical_not(good_successes.A2_lit.between(xlims[0], xlims[1], inclusive='both'))
# y_good_excluded = np.logical_not(good_successes.A2_est.between(ylims[0], ylims[1], inclusive='both'))
# n_good_excluded = sum(np.logical_or(x_good_excluded, y_good_excluded))
# x_bad_excluded = np.logical_not(bad_successes.A2_lit.between(xlims[0], xlims[1], inclusive='both'))
# y_bad_excluded = np.logical_not(bad_successes.A2_est.between(ylims[0], ylims[1], inclusive='both'))
# n_bad_excluded = sum(np.logical_or(x_bad_excluded, y_bad_excluded))
# n_excluded = n_good_excluded + n_bad_excluded

# # Determine "outliers" by the points which fall outside the "acceptable range"
# successes['A2_est_over_A2_lit'] = successes.A2_est / successes.A2_lit
# outlier_successes = successes.loc[(successes['A2_est_over_A2_lit'] < (1/acceptable_range_factor)) | (successes['A2_est_over_A2_lit'] > acceptable_range_factor)]

# # Print data info
# print(f"k_lim: {k_lim}")
# print(f"N data points total: {len(successes.A2_lit)}")
# if plot_good_successes:
#     print(f"N with |k_1|<{k_lim}: {len(good_successes.A2_lit)} (of which {n_good_excluded} outside plot range)")
# if plot_bad_successes:
#     print(f"N with |k_1|>{k_lim}: {len(bad_successes.A2_lit)} (of which {n_bad_excluded} outside plot range)")
# if plot_error_bars:
#     print(f"Error bar size: {error_bar_size}σ")
# if plot_acceptable_range:
#     print(f"m in legend: m ∈ {{{1}/{acceptable_range_factor}, {acceptable_range_factor}}}")
#     print(f"N total inside pink range: {len(successes) - len(outlier_successes)}")
#     print(f"N total outside pink range: {len(outlier_successes)}")

# # Scatter plots
# # |k_1| < k_lim (good successes)
# if plot_good_successes:
#     # with error bars
#     if plot_error_bars:
#         ax.errorbar(
#             x = good_successes.A2_lit,
#             y = good_successes.A2_est,
#             xerr = good_successes.σ_A2_lit*error_bar_size,
#             yerr = good_successes.σ_A2_est*error_bar_size,
#             fmt = 'o',
#             markersize = 3,
#             label = rf"Successful, $|k_1|<{k_lim}$",
#             alpha = 0.7,
#             color = isu.cmap_custom10[colormode](0),
#             ecolor = isu.cmap_custom10[colormode](0),
#             elinewidth = 1.2,
#         )
#     # without error bars
#     else:
#         ax.scatter(
#             good_successes.A2_lit,
#             good_successes.A2_est,
#             alpha = 0.7,
#             label = rf"Successful, $|k_1|$<{k_lim}",
#             color = isu.cmap_custom10[colormode](0),
#             marker = '.',
#         )
# # |k_1| > k_lim (bad successes)
# if plot_bad_successes:
#     # with error bars
#     if plot_error_bars:
#         ax.errorbar(
#             x = bad_successes.A2_lit,
#             y = bad_successes.A2_est,
#             xerr = bad_successes.σ_A2_lit*error_bar_size,
#             yerr = bad_successes.σ_A2_est*error_bar_size,
#             fmt = 'o',
#             markersize = 3,
#             label = rf"Successful, $|k_1|>{k_lim}$",
#             alpha = 0.7,
#             color = isu.cmap_custom10[colormode](1),
#             ecolor = isu.cmap_custom10[colormode](1),
#             elinewidth = 1.2,
#         )
#     # without error bars
#     else:
#         ax.scatter(
#             bad_successes.A2_lit,
#             bad_successes.A2_est,
#             alpha = 0.7,
#             label = rf"Successful, $|k_1|$>{k_lim}",
#             color = isu.cmap_custom10[colormode](1),
#             marker = '.',
#         )

# # A2_est = A2_lit line
# reference_line_x = np.linspace(xlims[0], xlims[1], 100)
# ax.plot(
#     reference_line_x,
#     reference_line_x*1,
#     color = isu.cmap_custom10[colormode](2),
#     linestyle = '--',
#     label = r"$A2_{est.} =\ A2_{lit.}$",
#     alpha = 0.7,
#     linewidth = 2,
# )
# if plot_acceptable_range:
#     ax.plot(
#         reference_line_x,
#         reference_line_x * acceptable_range_factor,
#         color = isu.cmap_custom10[colormode](4),
#         linestyle = '--',
#         # label = rf"$A2_{{est.}} =\ {acceptable_range_factor} \cdot A2_{{lit.}}$",
#         # label = rf"$A2_{{est.}} =\ m \cdot A2_{{lit.}}, m \in [\frac{{1}}{{{acceptable_range_factor}}}, {acceptable_range_factor}]$",
#         label = rf"$A2_{{est.}} =\ m \cdot A2_{{lit.}}$",
#         alpha = 0.7,
#         linewidth = 2,
#     )
#     ax.plot(
#         reference_line_x,
#         reference_line_x * (1/acceptable_range_factor),
#         color = isu.cmap_custom10[colormode](4),
#         linestyle = '--',
#         # label = rf"$A2_{{est.}} =\ \frac{{1}}{{{acceptable_range_factor}}} \cdot A2_{{lit.}}$",
#         alpha = 0.7,
#         linewidth = 2,
#     )
#     ax.fill_between(
#         reference_line_x,
#         reference_line_x * acceptable_range_factor,
#         reference_line_x * (1/acceptable_range_factor),
#         color = isu.cmap_custom10[colormode](4),
#         # label = r"$A2_{est.} =\ A2_{lit.}$",
#         alpha = 0.3,
#     )

# # Labels / titles
# ax.legend(
#     loc = 'best',
#     markerscale = 1.2,
#     fontsize = isu.small_font_size,
#     labelcolor = isu.text_color[colormode],
#     facecolor = isu.legend_bg_color[colormode],
#     framealpha = 0.6,
# )
# ax.set_ylabel(r"$A_2$ , Tudat [$m/s^2$]")
# ax.set_xlabel(r"$A_2$ , NEOCC [$m/s^2$]")
# fig.suptitle(
#     r"$A2_{est.}$ vs. $A2_{lit.}$",
#     # weight = 'bold',
#     fontsize = isu.big_font_size,
# )
# # ax.set_title(
# #     # r"$A2_{lit.} = A2_{est.} + k\cdot\sigma_{A2,est.}$",
# # )
# fig.set_tight_layout(True)

# # Plot output
# output_plot(
#     pb = pb,
#     colormode = colormode,
#     plots_save_path = plots_save_path,
#     fig_name = fig_name,
#     format = format,
# )

#%% 6.1 Scatter plot of σ_A2_est vs σ_A2_NEOCC
### 6.1 Scatter plot of σ_A2_est vs σ_A2_NEOCC
#region 6. scatter σ_A2

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_A2_unc_Tudat_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.75])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.set_xscale('log')
ax.set_yscale('log')

# Histogram
ax.scatter(
    successes.σ_A2_lit,
    successes.σ_A2_est,
    label = "Successful",
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.7,
    marker = '.',
    s = 65,
    edgecolors = scatter_edgecolors[colormode],
    linewidths = 0.5,
    zorder = 2.5,
)
xlims = ax.get_xlim()
ylims = ax.get_ylim()

ax.scatter(
    rejects.σ_A2_lit,
    rejects.σ_A2_est,
    label = "Rejected",
    color = isu.cmap_custom10[colormode](3),
    alpha = 0.5,
    marker = 'x',
    s = 20,
    linewidths = 1,
    zorder = 2.6,
)

# Dashed diagonal line
min_sigA2 = min(successes.σ_A2_est.min(), successes.σ_A2_lit.min())
max_sigA2 = min(successes.σ_A2_est.max(), successes.σ_A2_lit.max())
ax.plot(
    np.linspace(min_sigA2, max_sigA2, 20),
    np.linspace(min_sigA2, max_sigA2, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    linewidth = 2,
    # label = r"$σ_{A_2^{Tudat}} = σ_{A_2^{NEOCC}}$",
)
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Legend, labels
ax.legend(
    loc = 'best',
    markerscale = 1.2,
    fontsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax.set_ylabel(r"$σ_{A_2^{Tudat}}$ [$m/s^2$]")
ax.set_xlabel(r"$σ_{A_2^{NEOCC}}$ [$m/s^2$]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 6.2 Scatter plot of σ_A2_est vs σ_A2_JPL
### 6.2 Scatter plot of σ_A2_est vs σ_A2_JPL
#region 6. scatter σ_A2

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_A2_unc_Tudat_JPL"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.75])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.set_xscale('log')
ax.set_yscale('log')

# Histogram
ax.scatter(
    successes.σ_A2_JPL,
    successes.σ_A2_est,
    label = "Successful",
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.7,
    marker = '.',
    s = 65,
    edgecolors = scatter_edgecolors[colormode],
    linewidths = 0.5,
    zorder = 2.5,
)
xlims = ax.get_xlim()
ylims = ax.get_ylim()

ax.scatter(
    rejects.σ_A2_JPL,
    rejects.σ_A2_est,
    label = "Rejected",
    color = isu.cmap_custom10[colormode](3),
    alpha = 0.5,
    marker = 'x',
    s = 20,
    linewidths = 1,
    zorder = 2.5,
)

# Dashed diagonal line
min_sigA2 = min(successes.σ_A2_est.min(), successes.σ_A2_lit.min())
max_sigA2 = min(successes.σ_A2_est.max(), successes.σ_A2_lit.max())
ax.plot(
    np.linspace(min_sigA2, max_sigA2, 20),
    np.linspace(min_sigA2, max_sigA2, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    linewidth = 2,
    # label = r"$σ_{A_2^{Tudat}} = σ_{A_2^{JPL}}$",
)
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Legend, labels
ax.legend(
    loc = 'best',
    markerscale = 1.2,
    fontsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax.set_ylabel(r"$σ_{A_2^{Tudat}}$ [$m/s^2$]")
ax.set_xlabel(r"$σ_{A_2^{JPL}}$ [$m/s^2$]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 7.1 Scatter plot of k1 vs Observation arc
### 7.1 Scatter plot of k1 vs Observation arc
#region 7.1 scatter k1 vs arc

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_k1_NEOCC_obs_arc"
# fig_name = f"scatter_k2_NEOCC_obs_arc"
# fig_name = f"scatter_k1_JPL_obs_arc"
# fig_name = f"scatter_k1_JPL_obs_arc"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
ax.scatter(
    successes.obs_arc_est_YEAR,
    np.abs(successes.k1),
    # np.abs(successes.k2),
    # np.abs(successes.k1_JPL),
    # np.abs(successes.k2_JPL),
    # np.abs(successes['A2_lit+Xσ_A2_est']), # RK6 only
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.7,
    **scatter_kwargs,
)
ax.set_ylabel(r"$k_1^\text{NEOCC}$")
# ax.set_ylabel(r"$k_2^\text{NEOCC}$")
# ax.set_ylabel(r"$k_1^\text{JPL}$")
# ax.set_ylabel(r"$k_2^\text{JPL}$")


# Axis settings
ax.set_yscale('log')
xlims = (0, 100)
ymin, ymax = ax.get_ylim()
ax.set_xlim((xlims[0], xlims[1]))
ax.set_ylim((ymin, ymax))

# k spans
for i, (span, color) in enumerate(zip(k1_spans, k1_span_colors)):
    ax.axhspan(
        ymin = span[0],
        ymax = span[1],
        color = color,
        alpha = 0.25,
    )

# Quantify how many points are excluded by lims
x_excluded = np.logical_not(successes.obs_arc_est_YEAR.between(xlims[0], xlims[1], inclusive='both'))
y_excluded = np.logical_not(np.abs(successes.k1).between(ymin, ymax, inclusive='both'))
# y_excluded = np.logical_not(np.abs(successes['A2_lit+Xσ_A2_est']).between(ylims[0], ylims[1], inclusive='both'))
n_excluded = sum(np.logical_or(x_excluded, y_excluded))
# Add text box
text_box_note = f"N data points: {len(successes.obs_arc_est_YEAR)}"
text_box_note += f"\nN outside plot range: {n_excluded}"
text_box_note += f"\nN where k1=0: {sum(successes.k1 == 0)}"
# text_box_note += f"\nN where k1=0: {sum(successes['A2_lit+Xσ_A2_est'] == 0)}"
print(text_box_note)

# Labels / titles
ax.set_xlabel("Observation arc [year]")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)



#%% 7.2 Scatter plot of k1 vs Eccentricity
### 7.2 Scatter plot of k1 vs Eccentricity
#region 7.2 scatter k1 vs e

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_k1_NEOCC_eccentricity"
# fig_name = f"scatter_k2_NEOCC_eccentricity"
# fig_name = f"scatter_k1_JPL_eccentricity"
# fig_name = f"scatter_k1_JPL_eccentricity"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
ax.scatter(
    successes.eccentricity,
    np.abs(successes.k1),
    # np.abs(successes.k2),
    # np.abs(successes.k1_JPL),
    # np.abs(successes.k2_JPL),
    # np.abs(successes['A2_lit+Xσ_A2_est']),
    # label = "Right Ascension",
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.7,
    **scatter_kwargs,
)
ax.set_ylabel(r"$k_1^\text{NEOCC}$")
# ax.set_ylabel(r"$k_2^\text{NEOCC}$")
# ax.set_ylabel(r"$k_1^\text{JPL}$")
# ax.set_ylabel(r"$k_2^\text{JPL}$")

# Axis settings
ax.set_yscale('log')
xlims = (0, 1)
ymin, ymax = ax.get_ylim()
ax.set_xlim((xlims[0], xlims[1]))
ax.set_ylim((ymin, ymax))

# k spans
for i, (span, color) in enumerate(zip(k1_spans, k1_span_colors)):
    ax.axhspan(
        ymin = span[0],
        ymax = span[1],
        color = color,
        alpha = 0.25,
    )

# Quantify how many points are excluded by lims
x_excluded = np.logical_not(successes.eccentricity.between(xlims[0], xlims[1], inclusive='both'))
y_excluded = np.logical_not(np.abs(successes.k1).between(ymin, ymax, inclusive='both'))
# y_excluded = np.logical_not(np.abs(successes['A2_lit+Xσ_A2_est']).between(ylims[0], ylims[1], inclusive='both'))
n_excluded = sum(np.logical_or(x_excluded, y_excluded))
# Add text box
text_box_note = f"N data points: {len(successes.eccentricity)}"
text_box_note += f"\nN outside plot range: {n_excluded}"
text_box_note += f"\nN where k1=0: {sum(successes.k1 == 0)}"
# text_box_note += f"\nN where k1=0: {sum(successes['A2_lit+Xσ_A2_est'] == 0)}"
print(text_box_note)

# Labels / titles
ax.set_xlabel(r"Eccentricity $e$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 7.3 Scatter plot of k1 vs Semi-major axis
### 7.3 Scatter plot of k1 vs Semi-major axis
#region 7.3 scatter k1 vs a

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_k1_NEOCC_semimajoraxis"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
ax.scatter(
    successes.semimajoraxis_AU,
    np.abs(successes.k1),
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.7,
    **scatter_kwargs,
)
ax.set_ylabel(r"$k_1^\text{NEOCC}$")

# Axis settings
ax.set_yscale('log')
_, xmax = ax.get_xlim()
ax.set_xlim((0.5, xmax))
ymin, ymax = ax.get_ylim()
ax.set_ylim((ymin, ymax))

# k spans
for i, (span, color) in enumerate(zip(k1_spans, k1_span_colors)):
    ax.axhspan(
        ymin = span[0],
        ymax = span[1],
        color = color,
        alpha = 0.25,
    )

# Labels / titles
ax.set_xlabel(r"Semi-major axis $a$")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 7.4 Scatter plot of k1 vs RMSNR
### 7.4 Scatter plot of k1 vs RMSNR
#region 7.4 scatter k1 vs RMSNR

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_k1_NEOCC_RMSNR"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)

# Histogram
ax.scatter(
    successes.RMS_norm_res_est,
    np.abs(successes.k1),
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.7,
    **scatter_kwargs,
)
ax.set_ylabel(r"$k_1^\text{NEOCC}$")

# Axis settings
ax.set_yscale('log')
xmin, xmax = ax.get_xlim()
ax.set_xlim((xmin, xmax))
ymin, ymax = ax.get_ylim()
ax.set_ylim((ymin, ymax))

# k spans
for i, (span, color) in enumerate(zip(k1_spans, k1_span_colors)):
    ax.axhspan(
        ymin = span[0],
        ymax = span[1],
        color = color,
        alpha = 0.25,
    )

# Labels / titles
ax.set_xlabel("RMSNR")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 10.1 Scatter plot of FE_RMS vs TE_RMS (position)
### 10.1 Scatter plot of FE_RMS vs TE_RMS (position)
#region 10.1 scat RMS ratio p.
# RMS of formal error vs RMS of true error (position magnitude error)

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f""
# format = 'pdf'
format = 'png'

TE_RMSs_pos_KM = with_data.RMS_true_error_pos_mag_KM
FE_RMSs_pos_KM = with_data.RMS_formal_error_pos_mag_KM
# TE_RMSs_pos_KM = successes.RMS_true_error_pos_mag_KM
# FE_RMSs_pos_KM = successes.RMS_formal_error_pos_mag_KM
set_lims = True

fig, ax = plt.subplots(
    1, 1,
    figsize = (6, 6),
    facecolor = isu.fig_background[colormode],
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(linewidth=1.5, color=isu.text_color[colormode])
ax.axhline(linewidth=1.5, color=isu.text_color[colormode])

if set_lims:
    # Axis limits
    upper_lim = 1e5
    lower_lim = 0.3
    xlims = np.array([lower_lim, upper_lim])
    ylims = np.array([lower_lim*10, upper_lim/5])
    ax.set_xlim((xlims[0], xlims[1]))
    ax.set_ylim((ylims[0], ylims[1]))

    # Quantify how many points are excluded by lims
    x_excluded = np.logical_not(with_data.RMS_true_error_pos_mag_KM.between(xlims[0], xlims[1], inclusive='both'))
    y_excluded = np.logical_not(with_data.RMS_formal_error_pos_mag_KM.between(ylims[0], ylims[1], inclusive='both'))
    # x_excluded = np.logical_not(successes.RMS_true_error_pos_mag_KM.between(xlims[0], xlims[1], inclusive='both'))
    # y_excluded = np.logical_not(successes.RMS_formal_error_pos_mag_KM.between(ylims[0], ylims[1], inclusive='both'))
    n_excluded = sum(np.logical_or(x_excluded, y_excluded))
    # Add text box
    text_box_note = f"N data points: {len(TE_RMSs_pos_KM)}"
    if n_excluded > 0:
        text_box_note += f"\nN outside plot range: {n_excluded}"
    ax.text(
        x = 0.03,
        y = 0.97,
        s = text_box_note,
        horizontalalignment = 'left',
        verticalalignment = 'top',
        transform = ax.transAxes,
        color = isu.text_color[colormode],
        fontsize = isu.tiny_font_size,
        family = 'monospace',
        bbox = text_bbox,
    )

# Scatter plot
ax.scatter(
    TE_RMSs_pos_KM,
    FE_RMSs_pos_KM,
    alpha = 0.7,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](0),
    marker = '.',
)
ax.plot(
    np.linspace(0.1, 1e13, 100),
    np.linspace(0.1, 1e13, 100),
    color = isu.cmap_custom10[colormode](4),
    linestyle = '--',
    label = r"$FE_{RMS} =\ TE_{RMS}$",
    alpha = 0.7,
)
ax.legend(
    loc = 'best',
    markerscale = 1.2,
    fontsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax.set_xscale('log')
ax.set_yscale('log')

# Labels / titles
ax.set_xlabel(r"$TE_{RMS}$ [km]")
ax.set_ylabel(r"$FE_{RMS}$ [km]")
fig.suptitle(
    r"$FE_{RMS}$ vs $TE_{RMS}$ (position error)",
    # weight = 'bold',
    fontsize = isu.big_font_size,
)
fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)



#%% 10.2 Scatter plot of FE_RMS vs TE_RMS (velocity)
### 10.2 Scatter plot of FE_RMS vs TE_RMS (velocity)
#region 10.2 scat RMS ratio v.
# RMS of formal error vs RMS of true error (velocity magnitude error)

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f""
# format = 'pdf'
format = 'png'

TE_RMSs_vel = np.array(with_data.RMS_true_error_vel_mag_MM.to_list()) / 1_000
FE_RMSs_vel = np.array(with_data.RMS_formal_error_vel_mag_MM.to_list()) / 1_000
# TE_RMSs_vel = np.array(successes.RMS_true_error_vel_mag_MM.to_list()) / 1_000
# FE_RMSs_vel = np.array(successes.RMS_formal_error_vel_mag_MM.to_list()) / 1_000
# TE_RMSs_vel = np.array(successes.RMS_true_error_vel_mag_KM.to_list()) * 1_000
# FE_RMSs_vel = np.array(successes.RMS_formal_error_vel_mag_KM.to_list()) * 1_000
set_lims = True

fig, ax = plt.subplots(
    1, 1,
    figsize = (6, 6),
    facecolor = isu.fig_background[colormode],
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(linewidth=1.5, color=isu.text_color[colormode])
ax.axhline(linewidth=1.5, color=isu.text_color[colormode])

if set_lims:
    # Axis limits
    upper_lim = 20
    lower_lim = 3e-5
    xlims = np.array([lower_lim, upper_lim])
    ylims = np.array([lower_lim*10, upper_lim/10])
    ax.set_xlim((xlims[0], xlims[1]))
    ax.set_ylim((ylims[0], ylims[1]))

    # Quantify how many points are excluded by lims
    x_excluded = np.logical_not(with_data.RMS_true_error_vel_mag_MM.between(xlims[0] * 1_000, xlims[1] * 1_000, inclusive='both'))
    y_excluded = np.logical_not(with_data.RMS_formal_error_vel_mag_MM.between(ylims[0] * 1_000, ylims[1] * 1_000, inclusive='both'))
    # x_excluded = np.logical_not(successes.RMS_true_error_vel_mag_MM.between(xlims[0] * 1_000, xlims[1] * 1_000, inclusive='both'))
    # y_excluded = np.logical_not(successes.RMS_formal_error_vel_mag_MM.between(ylims[0] * 1_000, ylims[1] * 1_000, inclusive='both'))
    # x_excluded = np.logical_not(successes.RMS_true_error_vel_mag_KM.between(xlims[0] / 1_000, xlims[1] / 1_000, inclusive='both'))
    # y_excluded = np.logical_not(successes.RMS_formal_error_vel_mag_KM.between(ylims[0] / 1_000, ylims[1] / 1_000, inclusive='both'))
    n_excluded = sum(np.logical_or(x_excluded, y_excluded))
    # Add text box
    text_box_note = f"N data points: {len(TE_RMSs_vel)}"
    if n_excluded > 0:
        text_box_note += f"\nN outside plot range: {n_excluded}"
    ax.text(
        x = 0.03,
        y = 0.97,
        s = text_box_note,
        horizontalalignment = 'left',
        verticalalignment = 'top',
        transform = ax.transAxes,
        color = isu.text_color[colormode],
        fontsize = isu.tiny_font_size,
        family = 'monospace',
        bbox = text_bbox,
    )

# Scatter plots
ax.scatter(
    TE_RMSs_vel,
    FE_RMSs_vel,
    alpha = 0.7,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](0),
    marker = '.',
)
ax.plot(
    np.linspace(1e-5, 1e11, 100),
    np.linspace(1e-5, 1e11, 100),
    color = isu.cmap_custom10[colormode](4),
    linestyle = '--',
    label = r"$FE_{RMS} =\ TE_{RMS}$",
    alpha = 0.7,
)
ax.legend(
    loc = 'best',
    markerscale = 1.2,
    fontsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax.set_xscale('log')
ax.set_yscale('log')

# Labels / titles
ax.set_xlabel(r"$TE_{RMS}$ [m/s]")
ax.set_ylabel(r"$FE_{RMS}$ [m/s]")
fig.suptitle(
    r"$FE_{RMS}$ vs $TE_{RMS}$ (velocity error)",
    # weight = 'bold',
    fontsize = isu.big_font_size,
)
fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)



#%% 10.3 Scatter plots of FE_RMS vs TE_RMS (position and velocity)
### 10.3 Scatter plots of FE_RMS vs TE_RMS (position and velocity)
#region 10.3 scat RMS p.+v.
# Subplots
    # RMS of formal error vs RMS of true error (position magnitude error)
    # RMS of formal error vs RMS of true error (velocity magnitude error)

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_RMS_FE_vs_RMS_TE_pos_vel_subplots"
# format = 'pdf'
format = 'png'

TE_RMSs_pos_KM = with_data.RMS_true_error_pos_mag_KM
FE_RMSs_pos_KM = with_data.RMS_formal_error_pos_mag_KM
TE_RMSs_vel = with_data.RMS_true_error_vel_mag_MM / 1_000
FE_RMSs_vel = with_data.RMS_formal_error_vel_mag_MM / 1_000
# TE_RMSs_pos_KM = successes.RMS_true_error_pos_mag_KM
# FE_RMSs_pos_KM = successes.RMS_formal_error_pos_mag_KM
# TE_RMSs_vel = successes.RMS_true_error_vel_mag_MM / 1_000
# FE_RMSs_vel = successes.RMS_formal_error_vel_mag_MM / 1_000
# TE_RMSs_vel = successes.RMS_true_error_vel_mag_KM * 1_000
# FE_RMSs_vel = successes.RMS_formal_error_vel_mag_KM * 1_000
set_lims = False

fig, axs = plt.subplots(
    1, 2,
    figsize = np.array([1.0,0.67])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
for ax in axs.flatten():
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    ax.axvline(linewidth=1.5, color=isu.text_color[colormode])
    ax.axhline(linewidth=1.5, color=isu.text_color[colormode])

if set_lims:
    ## Position Error subplot
    # Axis limits
    upper_lim = 1e5
    lower_lim = 0.3
    xlims = np.array([lower_lim, upper_lim])
    ylims = np.array([lower_lim*10, upper_lim/5])
    axs[0].set_xlim((xlims[0], xlims[1]))
    axs[0].set_ylim((ylims[0], ylims[1]))

    # Quantify how many points are excluded by lims
    x_excluded = np.logical_not(with_data.RMS_true_error_pos_mag_KM.between(xlims[0], xlims[1], inclusive='both'))
    y_excluded = np.logical_not(with_data.RMS_formal_error_pos_mag_KM.between(ylims[0], ylims[1], inclusive='both'))
    # x_excluded = np.logical_not(successes.RMS_true_error_pos_mag_KM.between(xlims[0], xlims[1], inclusive='both'))
    # y_excluded = np.logical_not(successes.RMS_formal_error_pos_mag_KM.between(ylims[0], ylims[1], inclusive='both'))
    n_excluded = sum(np.logical_or(x_excluded, y_excluded))
    # Add text box
    text_box_note = f"N data points: {len(TE_RMSs_pos_KM)}"
    # if n_excluded > 0:
    text_box_note += f"\nN outside plot range: {n_excluded}"
    # axs[0].text(
    #     x = 0.03,
    #     y = 0.97,
    #     s = text_box_note,
    #     horizontalalignment = 'left',
    #     verticalalignment = 'top',
    #     transform = axs[0].transAxes,
    #     color = isu.text_color[colormode],
    #     fontsize = isu.tiny_font_size,
    #     family = 'monospace',
    #     bbox = text_bbox,
    # )
    print(f"POSITION:")
    print(text_box_note)

    ## Velocity Error subplot
    # Axis limits
    upper_lim = 20
    lower_lim = 3e-5
    xlims = np.array([lower_lim, upper_lim])
    ylims = np.array([lower_lim*10, upper_lim/10])
    axs[1].set_xlim((xlims[0], xlims[1]))
    axs[1].set_ylim((ylims[0], ylims[1]))

    # Quantify how many points are excluded by lims
    x_excluded = np.logical_not(with_data.RMS_true_error_vel_mag_MM.between(xlims[0] * 1_000, xlims[1] * 1_000, inclusive='both'))
    y_excluded = np.logical_not(with_data.RMS_formal_error_vel_mag_MM.between(ylims[0] * 1_000, ylims[1] * 1_000, inclusive='both'))
    # x_excluded = np.logical_not(successes.RMS_true_error_vel_mag_MM.between(xlims[0] * 1_000, xlims[1] * 1_000, inclusive='both'))
    # y_excluded = np.logical_not(successes.RMS_formal_error_vel_mag_MM.between(ylims[0] * 1_000, ylims[1] * 1_000, inclusive='both'))
    # x_excluded = np.logical_not(successes.RMS_true_error_vel_mag_KM.between(xlims[0] / 1_000, xlims[1] / 1_000, inclusive='both'))
    # y_excluded = np.logical_not(successes.RMS_formal_error_vel_mag_KM.between(ylims[0] / 1_000, ylims[1] / 1_000, inclusive='both'))
    n_excluded = sum(np.logical_or(x_excluded, y_excluded))
    # Add text box
    text_box_note = f"N data points: {len(TE_RMSs_vel)}"
    # if n_excluded > 0:
    text_box_note += f"\nN outside plot range: {n_excluded}"
    # axs[1].text(
    #     x = 0.03,
    #     y = 0.97,
    #     s = text_box_note,
    #     horizontalalignment = 'left',
    #     verticalalignment = 'top',
    #     transform = axs[1].transAxes,
    #     color = isu.text_color[colormode],
    #     fontsize = isu.tiny_font_size,
    #     family = 'monospace',
    #     bbox = text_bbox,
    # )
    print(f"\nVELOCITY:")
    print(text_box_note)

## Scatter plots
# Position Error subplot
axs[0].scatter(
    TE_RMSs_pos_KM,
    FE_RMSs_pos_KM,
    color = isu.cmap_custom10[colormode](0),
    alpha = 0.7,
    **scatter_kwargs,
)
# Diagonal line
min_pos_err = max(TE_RMSs_pos_KM.min(), FE_RMSs_pos_KM.min())
max_pos_err = min(TE_RMSs_pos_KM.max(), FE_RMSs_pos_KM.max())
axs[0].plot(
    np.linspace(min_pos_err, max_pos_err, 20),
    np.linspace(min_pos_err, max_pos_err, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    # label = r"$FE_{RMS} =\ TE_{RMS}$",
    alpha = 0.7,
    zorder = 2.1,
)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
# axs[0].legend(
#     loc = 'best',
#     markerscale = 1.2,
#     fontsize = isu.small_font_size,
#     labelcolor = isu.text_color[colormode],
#     facecolor = isu.legend_bg_color[colormode],
# )

# Velocity Error subplot
axs[1].scatter(
    TE_RMSs_vel,
    FE_RMSs_vel,
    color = isu.cmap_custom10[colormode](0),
    alpha = 0.7,
    **scatter_kwargs,
)

# Diagonal line
min_vel_err = max(TE_RMSs_vel.min(), FE_RMSs_vel.min())
max_vel_err = min(TE_RMSs_vel.max(), FE_RMSs_vel.max())
axs[1].plot(
    np.linspace(min_vel_err, max_vel_err, 20),
    np.linspace(min_vel_err, max_vel_err, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    # label = r"$FE_{RMS} =\ TE_{RMS}$",
    alpha = 0.7,
    zorder = 2.1,
)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
# axs[1].legend(
#     loc = 'best',
#     **legend_kwargs,
# )

## Labels / titles
# Position Error subplot
axs[0].set_xlabel(r"$TE_{RMS}$ [km]")
axs[0].set_ylabel(r"$FE_{RMS}$ [km]")
# Velocity Error subplot
axs[1].set_xlabel(r"$TE_{RMS}$ [m/s]")
axs[1].set_ylabel(r"$FE_{RMS}$ [m/s]")

## Fig title, subplot titles
axs[0].set_title("Position Error", fontsize=isu.medium_font_size)
axs[1].set_title("Velocity Error", fontsize=isu.medium_font_size)

fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% 10.4a Scatter plots of FE_RMS vs TE_RMS (position) with histograms
### 10.4a Scatter plots of FE_RMS vs TE_RMS (position) with histograms
#region 10.4a scat RMS p. wh

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_RMS_FE_vs_RMS_TE_pos_with-histograms"
# format = 'pdf'
format = 'png'

TE_RMSs_pos_KM = with_data.RMS_true_error_pos_mag_KM
FE_RMSs_pos_KM = with_data.RMS_formal_error_pos_mag_KM

fig = plt.figure(
    figsize = np.array([1.0,0.85])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
gs = gridspec.GridSpec(
    2, 2,
    figure = fig,
    width_ratios = [1, 3],
    height_ratios = [3, 1],
    hspace = 0.05,
    wspace = 0.05,
)

# Set up axes
ax_scat_pos = fig.add_subplot(gs[0, 1])
ax_FE_pos = fig.add_subplot(gs[0, 0], sharey=ax_scat_pos)
ax_TE_pos = fig.add_subplot(gs[1, 1], sharex=ax_scat_pos)
for ax in [ax_scat_pos,ax_FE_pos,ax_TE_pos]:
    ax = isu.set_default_2d_ax_settings(ax, colormode)
ax_blank_L = fig.add_subplot(gs[1, 0])

## Scatter plot
ax_scat_pos.scatter(
    TE_RMSs_pos_KM,
    FE_RMSs_pos_KM,
    color = isu.cmap_custom10[colormode](0),
    alpha = 0.7,
    **scatter_kwargs,
)
# Diagonal line
min_pos_err = max(TE_RMSs_pos_KM.min(), FE_RMSs_pos_KM.min())
max_pos_err = min(TE_RMSs_pos_KM.max(), FE_RMSs_pos_KM.max())
ax_scat_pos.plot(
    np.linspace(min_pos_err, max_pos_err, 20),
    np.linspace(min_pos_err, max_pos_err, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    alpha = 0.9,
    zorder = 2.1,
)
ax_scat_pos.set_xscale('log')
ax_scat_pos.set_yscale('log')
ax_scat_pos.margins(0.02)

# FE histogram
ymin, ymax = ax_scat_pos.get_ylim()
logbins = np.logspace(
    np.log10(ymin),
    np.log10(ymax),
    30,
)
ax_FE_pos.hist(
    FE_RMSs_pos_KM,
    bins = logbins,
    orientation = "horizontal",
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)

# TE histogram
xmin, xmax = ax_scat_pos.get_xlim()
logbins = np.logspace(
    np.log10(xmin),
    np.log10(xmax),
    30,
)
ax_TE_pos.hist(
    TE_RMSs_pos_KM,
    bins = logbins,
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)


## Labels / titles
ax_scat_pos.tick_params(labelbottom=False, labelleft=False)
ax_scat_pos.set_title("Position Error", fontsize=isu.medium_font_size)
ax_FE_pos.set_ylabel(r"$FE_{RMS}$ [km]")
ax_FE_pos.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
ax_TE_pos.set_xlabel(r"$TE_{RMS}$ [km]")
ax_TE_pos.tick_params(right=True, labelright=True, left=True, labelleft=False)
#
temp_text_kwargs = dict(
    s = "Count",
    color = isu.text_color[colormode],
    fontsize = isu.small_font_size,
)
for ax in [ax_blank_L]:
    ax.text(
        x = 0.4,
        y = 1.0,
        horizontalalignment = 'center',
        verticalalignment = 'top',
        transform = ax.transAxes,
        **temp_text_kwargs,
    )
    ax.text(
        x = 1.0,
        y = 0.4,
        horizontalalignment = 'right',
        verticalalignment = 'center',
        transform = ax.transAxes,
        rotation = 90,
        rotation_mode = 'default',
        **temp_text_kwargs
    )
    for spine in ax.spines.values(): # ax borders
        spine.set_edgecolor(isu.fig_background[colormode])
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.set_facecolor(isu.fig_background[colormode])
        ax.grid(visible=False)


# fig.set_tight_layout(True)
# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 10.4b Scatter plots of FE_RMS vs TE_RMS (velocity) with histograms
### 10.4b Scatter plots of FE_RMS vs TE_RMS (velocity) with histograms
#region 10.4b scat RMS v. wh

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_RMS_FE_vs_RMS_TE_vel_with-histograms"
# format = 'pdf'
format = 'png'

TE_RMSs_vel = with_data.RMS_true_error_vel_mag_MM / 1_000
FE_RMSs_vel = with_data.RMS_formal_error_vel_mag_MM / 1_000

fig = plt.figure(
    figsize = np.array([1.0,0.85])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
gs = gridspec.GridSpec(
    2, 2,
    figure = fig,
    width_ratios = [1, 3],
    height_ratios = [3, 1],
    hspace = 0.05,
    wspace = 0.05,
)

# Set up axes
ax_scat_vel = fig.add_subplot(gs[0, 1])
ax_FE_vel = fig.add_subplot(gs[0, 0], sharey=ax_scat_vel)
ax_TE_vel = fig.add_subplot(gs[1, 1], sharex=ax_scat_vel)
for ax in [ax_scat_vel,ax_FE_vel,ax_TE_vel]:
    ax = isu.set_default_2d_ax_settings(ax, colormode)
ax_blank_L = fig.add_subplot(gs[1, 0])


# Scatter plot
ax_scat_vel.scatter(
    TE_RMSs_vel,
    FE_RMSs_vel,
    color = isu.cmap_custom10[colormode](0),
    alpha = 0.7,
    **scatter_kwargs,
)
# Diagonal line
min_vel_err = max(TE_RMSs_vel.min(), FE_RMSs_vel.min())
max_vel_err = min(TE_RMSs_vel.max(), FE_RMSs_vel.max())
ax_scat_vel.plot(
    np.linspace(min_vel_err, max_vel_err, 20),
    np.linspace(min_vel_err, max_vel_err, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    alpha = 0.9,
    zorder = 2.1,
)
ax_scat_vel.set_xscale('log')
ax_scat_vel.set_yscale('log')
ax_scat_vel.margins(0.02)

# FE histogram
ymin, ymax = ax_scat_vel.get_ylim()
logbins = np.logspace(
    np.log10(ymin),
    np.log10(ymax),
    30,
)
ax_FE_vel.hist(
    FE_RMSs_vel,
    bins = logbins,
    orientation = "horizontal",
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)

# TE histogram
xmin, xmax = ax_scat_vel.get_xlim()
logbins = np.logspace(
    np.log10(xmin),
    np.log10(xmax),
    30,
)
ax_TE_vel.hist(
    TE_RMSs_vel,
    bins = logbins,
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)

## Labels / titles
ax_scat_vel.tick_params(labelbottom=False, labelleft=False)
ax_scat_vel.set_title("Velocity Error", fontsize=isu.medium_font_size)
ax_FE_vel.set_ylabel(r"$FE_{RMS}$ [m/s]")
ax_FE_vel.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
ax_TE_vel.set_xlabel(r"$TE_{RMS}$ [m/s]")
ax_TE_vel.tick_params(right=True, labelright=True, left=True, labelleft=False)
#
temp_text_kwargs = dict(
    s = "Count",
    color = isu.text_color[colormode],
    fontsize = isu.small_font_size,
)
for ax in [ax_blank_L]:
    ax.text(
        x = 0.4,
        y = 1.0,
        horizontalalignment = 'center',
        verticalalignment = 'top',
        transform = ax.transAxes,
        **temp_text_kwargs,
    )
    ax.text(
        x = 1.0,
        y = 0.4,
        horizontalalignment = 'right',
        verticalalignment = 'center',
        transform = ax.transAxes,
        rotation = 90,
        rotation_mode = 'default',
        **temp_text_kwargs
    )
    for spine in ax.spines.values(): # ax borders
        spine.set_edgecolor(isu.fig_background[colormode])
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.set_facecolor(isu.fig_background[colormode])
        ax.grid(visible=False)


# fig.set_tight_layout(True)
# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)






#%% 11.1 Scatter plot of Ratio of RMSs (position) vs Ratio of A2_uncs
### 11.1 Scatter plot of Ratio of RMSs (position) vs Ratio of A2_uncs
#region 11.1 scat Rat.v.Rat pos
# Ratio 1: RMS of formal error / RMS of true error (position magnitude error)
# Ratio 2: σ_A2_est / σ_A2_lit

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f""
# format = 'pdf'
format = 'png'
set_lims = True

TE_RMSs_pos_KM = successes.RMS_true_error_pos_mag_KM
# TE_RMSs_pos_KM = successes.RMS_JPLH_ephem_pos_unc_mag_KM # formal error of JPL
FE_RMSs_pos_KM = successes.RMS_formal_error_pos_mag_KM
ratio_RMSs_pos_err = FE_RMSs_pos_KM / TE_RMSs_pos_KM
# ratio_A2s = successes.A2_est / successes.A2_lit
ratio_A2_uncs = successes.σ_A2_est / successes.σ_A2_lit

fig, ax = plt.subplots(
    1, 1,
    figsize = (6, 6),
    facecolor = isu.fig_background[colormode],
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
# ax.axvline(linewidth=1.5, color=isu.text_color[colormode])
# ax.axhline(linewidth=1.5, color=isu.text_color[colormode])

if set_lims:
    # Axis limits
    xlims = np.array([0, 6])
    ylims = np.array([0, 8])
    # xlims = np.array([10,200])
    # ylims = np.array([-1, 8])
    ax.set_xlim((xlims[0], xlims[1]))
    ax.set_ylim((ylims[0], ylims[1]))

    # Quantify how many points are excluded by lims
    x_excluded = np.logical_not(ratio_A2_uncs.between(xlims[0], xlims[1], inclusive='both'))
    y_excluded = np.logical_not(ratio_RMSs_pos_err.between(ylims[0], ylims[1], inclusive='both'))
    n_excluded = sum(np.logical_or(x_excluded, y_excluded))
    # Add text box
    text_box_note = f"N data points: {len(TE_RMSs_pos_KM)}"
    if n_excluded > 0:
        text_box_note += f"\nN outside plot range: {n_excluded}"
    ax.text(
        x = 0.03,
        y = 0.97,
        s = text_box_note,
        horizontalalignment = 'left',
        verticalalignment = 'top',
        transform = ax.transAxes,
        color = isu.text_color[colormode],
        fontsize = isu.tiny_font_size,
        family = 'monospace',
        bbox = text_bbox,
    )

# Scatter plots
ax.scatter(
    ratio_A2_uncs,
    ratio_RMSs_pos_err,
    alpha = 0.7,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](2),
    **scatter_kwargs,
)
# ax.plot(
#     np.linspace(1e-5, 1e11, 100),
#     np.linspace(1e-5, 1e11, 100),
#     color = isu.cmap_custom10[colormode](4),
#     linestyle = '--',
#     # label = r"$FE_{RMS} =\ TE_{RMS}$",
#     alpha = 0.7,
# )
# ax.legend(
#     loc = 'best',
#     markerscale = 1.2,
#     fontsize = isu.small_font_size,
#     labelcolor = isu.text_color[colormode],
#     facecolor = isu.legend_bg_color[colormode],
# )
# ax.set_xscale('log')
# ax.set_yscale('log')

# Labels / titles
ax.set_xlabel(
    r"$σ_{A2,est.} / σ_{A2,lit.}$",
    fontsize = isu.medium_font_size,
)
ax.set_ylabel(
    r"$(FE_{RMS} / TE_{RMS})_{pos.}$",
    fontsize = isu.medium_font_size,
)
fig.suptitle(
    r"RMS ratio (position) vs $σ_{A2}$ ratio",
    # weight = 'bold',
    fontsize = isu.big_font_size,
)
fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% 11.2 Scatter plot of Ratio of RMSs (velocity) vs Ratio of A2_uncs
### 11.2 Scatter plot of Ratio of RMSs (velocity) vs Ratio of A2_uncs
#region 11.2 scat Rat.v.Rat vel
# Ratio 1: RMS of formal error / RMS of true error (velocity magnitude error)
# Ratio 2: σ_A2_est / σ_A2_lit

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f""
# format = 'pdf'
format = 'png'
set_lims = False

TE_RMSs_vel = np.array(successes.RMS_true_error_vel_mag_MM.to_list()) / 1_000
FE_RMSs_vel = np.array(successes.RMS_formal_error_vel_mag_MM.to_list()) / 1_000
# TE_RMSs_vel = np.array(successes.RMS_true_error_vel_mag_KM.to_list()) * 1_000
# FE_RMSs_vel = np.array(successes.RMS_formal_error_vel_mag_KM.to_list()) * 1_000
ratio_RMSs_vel_err = FE_RMSs_vel / TE_RMSs_vel
# ratio_A2s = successes.A2_est / successes.A2_lit
ratio_A2_uncs = successes.σ_A2_est / successes.σ_A2_lit

fig, ax = plt.subplots(
    1, 1,
    figsize = (6, 6),
    facecolor = isu.fig_background[colormode],
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
# ax.axvline(linewidth=1.5, color=isu.text_color[colormode])
# ax.axhline(linewidth=1.5, color=isu.text_color[colormode])

if set_lims:
    # Axis limits
    # xlims = np.array([-4, 6])
    # ylims = np.array([-0.1, 6])
    xlims = np.array([0, 6])
    ylims = np.array([0, 8])
    ax.set_xlim((xlims[0], xlims[1]))
    ax.set_ylim((ylims[0], ylims[1]))

    # Quantify how many points are excluded by lims
    x_excluded = np.logical_not(ratio_A2_uncs.between(xlims[0], xlims[1], inclusive='both'))
    y_excluded = np.logical_not(ratio_RMSs_pos_err.between(ylims[0], ylims[1], inclusive='both'))
    n_excluded = sum(np.logical_or(x_excluded, y_excluded))
    # Add text box
    text_box_note = f"N data points: {len(TE_RMSs_pos_KM)}"
    if n_excluded > 0:
        text_box_note += f"\nN outside plot range: {n_excluded}"
    ax.text(
        x = 0.03,
        y = 0.97,
        s = text_box_note,
        horizontalalignment = 'left',
        verticalalignment = 'top',
        transform = ax.transAxes,
        color = isu.text_color[colormode],
        fontsize = isu.tiny_font_size,
        family = 'monospace',
        bbox = text_bbox,
    )

# Scatter plots
ax.scatter(
    ratio_A2_uncs,
    ratio_RMSs_vel_err,
    alpha = 0.7,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](0),
    marker = '.',
)
if set_lims:
    ax.plot(
        np.linspace(1e-5, 1e11, 100),
        np.linspace(1e-5, 1e11, 100),
        color = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        label = r"$FE_{RMS} =\ TE_{RMS}$",
        alpha = 0.7,
    )
# ax.legend(
#     loc = 'best',
#     markerscale = 1.2,
#     fontsize = isu.small_font_size,
#     labelcolor = isu.text_color[colormode],
#     facecolor = isu.legend_bg_color[colormode],
# )
# ax.set_xscale('log')
# ax.set_yscale('log')

# Labels / titles
ax.set_xlabel(
    r"$σ_{A2,est.} / σ_{A2,lit.}$",
    fontsize = isu.medium_font_size,
)
ax.set_ylabel(
    r"$(FE_{RMS} / TE_{RMS})_{vel.}$",
    fontsize = isu.medium_font_size,
)
fig.suptitle(
    r"RMS ratio (velocity) vs $σ_{A2}$ ratio",
    # weight = 'bold',
    fontsize = isu.big_font_size,
)
fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)



#%% 11.3 Scatter plots of Ratio of RMSs (position and velocity) vs Ratio of A2_uncs
### 11.3 Scatter plots of Ratio of RMSs (position and velocity) vs Ratio of A2_uncs
#region 11.3 scat rat.v.Rat p+v
# Subplots
    # Position: (RMS of formal error / RMS of true error) vs (σ_A2_est / σ_A2_lit)
    # Velocity: (RMS of formal error / RMS of true error) vs (σ_A2_est / σ_A2_lit)

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_RMS_err_ratio_vs_A2_err_ratio_pos_vel_subplots"
# format = 'pdf'
format = 'png'

set_lims = True

fig, axs = plt.subplots(
    1, 2,
    figsize = np.array([1.0,0.67])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
for ax in axs.flatten():
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    # ax.axvline(linewidth=1.5, color=isu.text_color[colormode])
    # ax.axhline(linewidth=1.5, color=isu.text_color[colormode])

if set_lims:
    ## Position Error subplot
    # Axis limits
    # upper_lim = 1e5
    # lower_lim =
    # xlims = np.array([lower_lim, upper_lim])
    # ylims = np.array([lower_lim, upper_lim])
    xlims = np.array([0, 5])
    ylims = np.array([0, 10])
    axs[0].set_xlim((xlims[0], xlims[1]))
    axs[0].set_ylim((ylims[0], ylims[1]))

    # Quantify how many points are excluded by lims
    x_excluded = np.logical_not(ratio_A2_uncs.between(xlims[0], xlims[1], inclusive='both'))
    y_excluded = np.logical_not(ratio_RMSs_pos_err.between(ylims[0], ylims[1], inclusive='both'))
    n_excluded = sum(np.logical_or(x_excluded, y_excluded))
    # Add text box
    text_box_note = f"N data points: {len(TE_RMSs_pos_KM)}"
    # if n_excluded > 0:
    text_box_note += f"\nN outside plot range: {n_excluded}"
    # axs[0].text(
    #     x = 0.03,
    #     y = 0.97,
    #     s = text_box_note,
    #     horizontalalignment = 'left',
    #     verticalalignment = 'top',
    #     transform = axs[0].transAxes,
    #     color = isu.text_color[colormode],
    #     fontsize = isu.tiny_font_size,
    #     family = 'monospace',
    #     bbox = text_bbox,
    # )
    print(f"POSITION:")
    print(text_box_note)

    ## Velocity Error subplot
    # Axis limits
    # upper_lim = 1e5
    # lower_lim =
    # xlims = np.array([lower_lim, upper_lim])
    # ylims = np.array([lower_lim, upper_lim])
    xlims = np.array([0, 5])
    ylims = np.array([0, 10])
    axs[1].set_xlim((xlims[0], xlims[1]))
    axs[1].set_ylim((ylims[0], ylims[1]))

    # Quantify how many points are excluded by lims
    x_excluded = np.logical_not(ratio_A2_uncs.between(xlims[0], xlims[1], inclusive='both'))
    y_excluded = np.logical_not(ratio_RMSs_pos_err.between(ylims[0], ylims[1], inclusive='both'))
    n_excluded = sum(np.logical_or(x_excluded, y_excluded))
    # Add text box
    text_box_note = f"N data points: {len(TE_RMSs_pos_KM)}"
    # if n_excluded > 0:
    text_box_note += f"\nN outside plot range: {n_excluded}"
    # axs[1].text(
    #     x = 0.03,
    #     y = 0.97,
    #     s = text_box_note,
    #     horizontalalignment = 'left',
    #     verticalalignment = 'top',
    #     transform = axs[1].transAxes,
    #     color = isu.text_color[colormode],
    #     fontsize = isu.tiny_font_size,
    #     family = 'monospace',
    #     bbox = text_bbox,
    # )
    print(f"\nVELOCITY:")
    print(text_box_note)

## Scatter plots
# Position Error subplot
axs[0].scatter(
    ratio_A2_uncs,
    ratio_RMSs_pos_err,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](2),
    alpha = 0.7,
    **scatter_kwargs,
)
# Diagonal line
min_p = min(xlims[0], ylims[0])
max_p = min(xlims[1], xlims[1])
axs[0].plot(
    np.linspace(min_p, max_p, 20),
    np.linspace(min_p, max_p, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    # label = r"$(FE_{RMS}/TE_{RMS})_{pos.} = σ_{A_2}^\text{Tudat} / σ_{A_2}^\text{NEOCC}$",
    alpha = 0.7,
    zorder = 2.1,
)
# axs[0].legend(
#     loc = 'best',
#     markerscale = 1.2,
#     fontsize = isu.small_font_size,
#     labelcolor = isu.text_color[colormode],
#     facecolor = isu.legend_bg_color[colormode],
# )
# axs[0].set_xscale('log')
# axs[0].set_yscale('log')

# Velocity Error subplot
axs[1].scatter(
    ratio_A2_uncs,
    ratio_RMSs_vel_err,
    alpha = 0.7,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](2),
    **scatter_kwargs,
)
# Diagonal line
axs[1].plot(
    np.linspace(min_p, max_p, 20),
    np.linspace(min_p, max_p, 20),
    color = isu.text_color[colormode],
    linestyle = '--',
    # label = r"$(FE_{RMS}/TE_{RMS})_{vel.} = σ_{A2,est.} / σ_{A2,lit.}$",
    alpha = 0.7,
    zorder = 2.1,
)
# axs[1].legend(
#     loc = 'best',
#     markerscale = 1.2,
#     fontsize = isu.small_font_size,
#     labelcolor = isu.text_color[colormode],
#     facecolor = isu.legend_bg_color[colormode],
# )
# axs[1].set_xscale('log')
# axs[1].set_yscale('log')


## Labels / titles
# Position Error subplot
axs[0].set_xlabel(
    r"$σ_{A_2}^\text{Tudat} / σ_{A_2}^\text{NEOCC}$",
    fontsize = isu.medium_font_size,
)
axs[0].set_ylabel(
    r"$(FE_{RMS} / TE_{RMS})_{pos.}$",
    fontsize = isu.medium_font_size,
)
# Velocity Error subplot
axs[1].set_xlabel(
    r"$σ_{A_2}^\text{Tudat} / σ_{A_2}^\text{NEOCC}$",
    fontsize = isu.medium_font_size,
)
axs[1].set_ylabel(
    r"$(FE_{RMS} / TE_{RMS})_{vel.}$",
    fontsize = isu.medium_font_size,
)

## Fig title, subplot titles
axs[0].set_title("Position Error")
axs[1].set_title("Velocity Error")

fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)





#%% 12.2 Histogram of Ratio of RMSs (position)
### 12.2 Histogram of Ratio of RMSs (position)
#region 12.2 Hist. RMS rat. p

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_RMS_err_ratio_pos"
# format = 'pdf'
format = 'png'

TE_RMSs_pos_KM = with_data.RMS_true_error_pos_mag_KM
FE_RMSs_pos_KM = with_data.RMS_formal_error_pos_mag_KM
# TE_RMSs_pos_KM = successes.RMS_true_error_pos_mag_KM
# FE_RMSs_pos_KM = successes.RMS_formal_error_pos_mag_KM
ratio_RMSs_pos_err = FE_RMSs_pos_KM / TE_RMSs_pos_KM

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(linewidth=1, color=isu.text_color[colormode])
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')
ax.set_xscale('log')

# Histogram
xlims = (0,20)
binwidth = 0.25
bins = np.arange(xlims[0], xlims[1]+0.1, binwidth)
logbins = np.logspace(
    np.log10(min(ratio_RMSs_pos_err)),
    np.log10(xlims[1]),
    30,
)
ax.hist(
    ratio_RMSs_pos_err,
    bins = logbins,
    # bins = bins,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](0),
    alpha = 0.8,
    **hist_contour_kwargs,
)

# Text box for excluded data
hist_count = len(ratio_RMSs_pos_err)
n_included = sum(ratio_RMSs_pos_err.between(xlims[0], xlims[1], inclusive='both'))
n_excluded = hist_count - n_included
text_box_note = f"N data points: {hist_count}"
# if n_excluded > 0:
text_box_note += f"\nN outside plot range: {n_excluded}"
print(text_box_note)

# Labels, title
ax.set_ylabel("Count")
ax.set_xlabel(
    r"$(FE_{RMS} / TE_{RMS})_{pos.}$",
    fontsize = isu.medium_font_size,
)
fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)



#%% 12.3 Histogram of Ratio of RMSs (velocity)
### 12.3 Histogram of Ratio of RMSs (velocity)
#region 12.3 Hist. RMS rat. v

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_RMS_err_ratio_vel"
# format = 'pdf'
format = 'png'

TE_RMSs_vel = with_data.RMS_true_error_vel_mag_MM / 1_000
FE_RMSs_vel = with_data.RMS_formal_error_vel_mag_MM / 1_000
# TE_RMSs_vel = successes.RMS_true_error_vel_mag_MM / 1_000
# FE_RMSs_vel = successes.RMS_formal_error_vel_mag_MM / 1_000
# TE_RMSs_vel = successes.RMS_true_error_vel_mag_KM * 1_000
# FE_RMSs_vel = successes.RMS_formal_error_vel_mag_KM * 1_000
ratio_RMSs_vel_err = FE_RMSs_vel / TE_RMSs_vel

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
# ax.axvline(linewidth=1, color=isu.text_color[colormode])
ax.axvline(1, color=isu.text_color[colormode], linewidth=1, linestyle='--')
ax.set_xscale('log')

# Histogram
xlims = (0,20)
binwidth = 0.25
bins = np.arange(xlims[0], xlims[1]+0.1, binwidth)
logbins = np.logspace(
    np.log10(min(ratio_RMSs_vel_err)),
    np.log10(xlims[1]),
    30,
)
ax.hist(
    ratio_RMSs_vel_err,
    bins = logbins,
    # label = "Successful",
    color = isu.cmap_custom10[colormode](0),
    alpha = 0.8,
    **hist_contour_kwargs,
)

# Text box for excluded data
hist_count = len(ratio_RMSs_vel_err)
n_included = sum(ratio_RMSs_vel_err.between(xlims[0], xlims[1], inclusive='both'))
n_excluded = hist_count - n_included
text_box_note = f"N data points: {hist_count}"
# if n_excluded > 0:
text_box_note += f"\nN outside plot range: {n_excluded}"
print(text_box_note)

# Labels, title
ax.set_ylabel("Count")
ax.set_xlabel(r"$(FE_{RMS} / TE_{RMS})_{vel.}$")
fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% 12.4 Histograms of FE/TE (position and velocity)
### 12.4 Histograms of FE/TE (position and velocity)
#region 12.4 Hists FE/TE

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"hist_RMS-FE-TE-ratio_pos_vel"
# format = 'pdf'
format = 'png'

FE_RMSs_pos_KM = with_data.RMS_formal_error_pos_mag_KM
TE_RMSs_pos_KM = with_data.RMS_true_error_pos_mag_KM
ratio_RMSs_pos_err = FE_RMSs_pos_KM / TE_RMSs_pos_KM

TE_RMSs_vel = with_data.RMS_true_error_vel_mag_MM / 1_000
FE_RMSs_vel = with_data.RMS_formal_error_vel_mag_MM / 1_000
ratio_RMSs_vel_err = FE_RMSs_vel / TE_RMSs_vel

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(linewidth=1, color=isu.text_color[colormode])
ax.axvline(1, color=isu.text_color[colormode], linestyle='--')
ax.set_xscale('log')

# Histograms
min_x = min(ratio_RMSs_pos_err.min(), ratio_RMSs_vel_err.min())
max_x = max(ratio_RMSs_pos_err.max(), ratio_RMSs_vel_err.max())
xlims = (min_x,max_x)
logbins = np.logspace(
    np.log10(min_x),
    np.log10(max_x),
    30,
)
ax.hist(
    ratio_RMSs_pos_err,
    bins = logbins,
    label = "Position",
    color = isu.cmap_custom10[colormode](0),
    alpha = 0.8,
    **hist_contour_kwargs,
)
ax.hist(
    ratio_RMSs_vel_err,
    bins = logbins,
    label = "Velocity",
    color = isu.cmap_custom10[colormode](1),
    alpha = 0.8,
    linestyle = '--',
    **hist_contour_kwargs,
)

# # Text box for excluded data
# hist_count = len(ratio_RMSs_pos_err)
# n_included = sum(ratio_RMSs_pos_err.between(xlims[0], xlims[1], inclusive='both'))
# n_excluded = hist_count - n_included
# text_box_note = f"N data points: {hist_count}"
# # if n_excluded > 0:
# text_box_note += f"\nN outside plot range: {n_excluded}"
# print(text_box_note)

ax.legend(
    loc = 'upper left',
    **legend_kwargs,
)

# Labels, title
ax.set_ylabel("Count")
ax.set_xlabel(
    r"$FE_{RMS} / TE_{RMS}$",
    fontsize = isu.medium_font_size,
)
# fig.set_tight_layout(True)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% 14.1a Scatter subplots of e, a, coloured by V_a, V_p
### 14.1a Scatter subplots of e, a, coloured by V_a, V_p
#region 14.1a Scatter e, a V

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_e-vs-a_v-a_v-p"
# format = 'pdf'
format = 'png'

sun_GM = 1.327124400420322e+20 # from SPICE
velocity_at_peri_KM = np.array([
    np.sqrt( sun_GM * (1 / (a_AU*cnst.ASTRONOMICAL_UNIT)) * ((1+e)/(1-e)) )
    for (a_AU, e) in zip(with_data.semimajoraxis_AU, with_data.eccentricity)
]) / 1000
velocity_at_apo_KM = np.array([
    np.sqrt( sun_GM * (1 / (a_AU*cnst.ASTRONOMICAL_UNIT)) * ((1-e)/(1+e)) )
    for (a_AU, e) in zip(with_data.semimajoraxis_AU, with_data.eccentricity)
]) / 1000
# max_v = max(velocity_at_peri_KM.max(), velocity_at_apo_KM.max())
# min_v = min(velocity_at_peri_KM.min(), velocity_at_apo_KM.min())

# Plot
fig, axs = plt.subplots(
    1, 2,
    figsize = (0.99*isu.textwidth_INCH, 0.40*isu.textwidth_INCH),
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)

# Scatter plot for V_a / velocity at aphelion
sc_Va = axs[0].scatter(
    with_data.semimajoraxis_AU,
    with_data.eccentricity,
    c = velocity_at_apo_KM,
    cmap = 'jet',
    **scatter_kwargs,
)
# colorbar
cb_Va = fig.colorbar(sc_Va, ax=axs[0])#, label=")
cb_Va.set_label(
    label = r"$v_a$ [km/s]",
    # label = r"Velocity at  $V_a$ [km/s]",
    fontsize = isu.medium_font_size,
    color = isu.text_color[colormode],
)
cb_Va.outline.set_edgecolor(isu.text_color[colormode])
cb_Va.ax.tick_params(
    color = isu.text_color[colormode],
    labelsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
)

# Scatter plot for V_p / velocity at perihelion
sc_Vp = axs[1].scatter(
    with_data.semimajoraxis_AU,
    with_data.eccentricity,
    c = velocity_at_peri_KM,
    cmap = 'jet',
    **scatter_kwargs,
)
# Colorbar
cb_Vp = fig.colorbar(sc_Vp, ax=axs[1])#, label=")
cb_Vp.set_label(
    label = r"$v_p$ [km/s]",
    # label = r"Velocity at perihelion $V_p$ [km/s]",
    fontsize = isu.medium_font_size,
    color = isu.text_color[colormode],
)
cb_Vp.outline.set_edgecolor(isu.text_color[colormode])
cb_Vp.ax.tick_params(
    color = isu.text_color[colormode],
    labelsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
)

# Axis settings, labels
for ax in axs:
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    ax.set_ylim((0, 0.93))
    ax.set_yticks(np.arange(0,0.95,0.1))
    _, xmax = ax.get_xlim()
    ax.set_xlim((0.5, xmax))
    ax.set_xlabel(r"Semi-major axis $a$ [AU]")
    ax.set_ylabel(r"Eccentricity $e$")

# Subplot titles
axs[0].set_title(
    r"Velocity at Aphelion $v_a$",
    fontsize = isu.medium_font_size,
)
axs[1].set_title(
    r"Velocity at Perihelion $v_p$",
    fontsize = isu.medium_font_size,
)


# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

""" OLD VERSION: full (e,a) space
sun_GM = 1.327124400420322e+20 # from SPICE
a_range_AU = np.linspace(0.5,3,101,endpoint=True)
e_range = np.linspace(0.95,0,96,endpoint=True)
a_AU, e = np.meshgrid(a_range_AU, e_range)
velocity_at_peri = np.sqrt( sun_GM * (1 / (a_AU*cnst.ASTRONOMICAL_UNIT)) * ((1+e)/(1-e)) )
velocity_at_apo = np.sqrt( sun_GM * (1 / (a_AU*cnst.ASTRONOMICAL_UNIT)) * ((1-e)/(1+e)) )
velocity_at_peri_KM = velocity_at_peri / 1000
velocity_at_apo_KM = velocity_at_apo / 1000

# Plot
fig, ax = plt.subplots(
    1, 1,
    figsize=(9, 7),
    facecolor = isu.fig_background[colormode],
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
im = ax.imshow(
    velocity_at_peri_KM,
    cmap = 'plasma',
    norm = 'log',
    extent = [a_range_AU.min(), a_range_AU.max(), e_range.min(), e_range.max()]
)
ax.grid()
ax.set_xlabel("a [AU]")
ax.set_ylabel("e [-]")
# colorbar
cb = plt.colorbar(im)
cb.outline.set_edgecolor(isu.text_color[colormode])
cb.ax.tick_params(
    color = isu.text_color[colormode],
    labelsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
)
"""

#%% 14.1b Scatter subplots of e, a, coloured by V_p, a_p
### 14.1b Scatter subplots of e, a, coloured by V_p, a_p
#region 14.1b Scatter e, a, V, acc.

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_e-vs-a_v-p_a-p"
# format = 'pdf'
format = 'png'

sun_GM = 1.327124400420322e+20 # from SPICE
velocity_at_peri_KM = np.array([
    np.sqrt( sun_GM * (1 / (a_AU*cnst.ASTRONOMICAL_UNIT)) * ((1+e)/(1-e)) )
    for (a_AU, e) in zip(with_data.semimajoraxis_AU, with_data.eccentricity)
]) / 1000
acceleration_at_peri = np.array([
    sun_GM / ( (a_AU*cnst.ASTRONOMICAL_UNIT)**2 * (1-e)**2 )
    for (a_AU, e) in zip(with_data.semimajoraxis_AU, with_data.eccentricity)
])
# max_v = max(velocity_at_peri_KM.max(), velocity_at_apo_KM.max())
# min_v = min(velocity_at_peri_KM.min(), velocity_at_apo_KM.min())

# Plot
fig, axs = plt.subplots(
    1, 2,
    figsize = (0.99*isu.textwidth_INCH, 0.40*isu.textwidth_INCH),
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)

import matplotlib.colors as mplcolors

# Scatter plot for V_p / velocity at perihelion
sc_Vp = axs[0].scatter(
    with_data.semimajoraxis_AU,
    with_data.eccentricity,
    c = velocity_at_peri_KM,
    norm = mplcolors.LogNorm(),
    cmap = 'jet',
    **scatter_kwargs,
)
# Colorbar
cb_Vp = fig.colorbar(sc_Vp, ax=axs[0])#, label=")
cb_Vp.set_label(
    label = r"$v_p$ [km/s]",
    # label = r"Velocity at perihelion $V_p$ [km/s]",
    fontsize = isu.medium_font_size,
    color = isu.text_color[colormode],
)
cb_Vp.outline.set_edgecolor(isu.text_color[colormode])
cb_Vp.ax.tick_params(
    which = 'major',
    color = isu.text_color[colormode],
    labelsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
)
cb_Vp.ax.tick_params(
    which = 'minor',
    color = isu.text_color[colormode],
    labelsize = isu.tiny_font_size,
    labelcolor = isu.text_color[colormode],
)


# Scatter plot for a_p / acceleration at perihelion
sc_ap = axs[1].scatter(
    with_data.semimajoraxis_AU,
    with_data.eccentricity,
    c = acceleration_at_peri,
    norm = mplcolors.LogNorm(),
    cmap = 'jet',
    **scatter_kwargs,
)
# colorbar
cb_ap = fig.colorbar(sc_ap, ax=axs[1])#, label=")
cb_ap.set_label(
    label = r"$\ddot{r}_p$ [m/$s^2$]",
    # label = r"Acceleration at perihelion $\ddot{r}_p$ [m/s2]",
    fontsize = isu.medium_font_size,
    color = isu.text_color[colormode],
)
cb_ap.outline.set_edgecolor(isu.text_color[colormode])
cb_ap.ax.tick_params(
    which = 'major',
    color = isu.text_color[colormode],
    labelsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
)
cb_ap.ax.tick_params(
    which = 'minor',
    color = isu.text_color[colormode],
    labelsize = isu.tiny_font_size,
    labelcolor = isu.text_color[colormode],
)


# Axis settings, labels
for ax in axs:
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    ax.set_ylim((0, 0.93))
    ax.set_yticks(np.arange(0,0.95,0.1))
    _, xmax = ax.get_xlim()
    ax.set_xlim((0.5, xmax))
    ax.set_xlabel(r"Semi-major axis $a$ [AU]")
    ax.set_ylabel(r"Eccentricity $e$")

# Subplot titles
axs[0].set_title(
    r"Velocity at Perihelion $v_p$",
    fontsize = isu.medium_font_size,
)
axs[1].set_title(
    r"Acceleration at Perihelion $\ddot{r}_p$",
    fontsize = isu.medium_font_size,
)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 14.2 Scatter plot of e, a, coloured by observation arc
### 14.2 Scatter plot of e, a, coloured by observation arc
#region 14.2 Scatter e,a arc

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
plot_only_integrator_analysis_set = True
fig_name = f"scatter_e-vs-a_w-arc_w-hists"
fig_name += "__selected-set" if plot_only_integrator_analysis_set else ""
# format = 'pdf'
format = 'png'

integrator_analysis_bodies = [
    #               # ARC [y]   e [-]       a [AU]
    "595116",       # 23.96     0.0281      1.3107    # DONE
    "2016 CO246",   #  4.02     0.1262      0.9948    # DONE
    "499998",       #  6.30     0.2166      1.3159    # DONE - 24 mins
    "2063",         # 48.14     0.3494      1.0780    # DONE - 184 mins
    "2018 GY",      #  4.04     0.4063      0.7667    # DONE
    "1221",         # 92.95     0.4346      1.9198    # DONE - 130 mins
    "480808",       # 29.02     0.5262      0.6709    # DONE
    "4179",         # 92.11     0.6247      2.5430    # DONE - 127 mins
    "152664",       # 31.57     0.7140      2.5202    # DONE - 86 mins
    "2017 DN109",   #  6.20     0.8268      1.1958    # DONE
    "1566",         # 76.15     0.8270      1.0780    # DONE
    "1995 CR",      # 31.04     0.8684      0.9069    # DONE - 85 mins
    "137924",       # 29.00     0.8950      0.8764    # DONE - 111 mins
    "152563",       # 71.15     0.2717      0.9079
    "7350",         # 59.74     0.3912      1.3560
    "7336",         # 42.24     0.4820      2.3055
    "8014",         # 35.61     0.4560      1.7476
    "2009 JR2",     # 14.50     0.1790      1.2498
    "513126",       # 19.52     0.5826      1.7859
    "177016",       # 21.92     0.5822      1.1601
    "364136",       # 20.09     0.7549      0.6766
    "2101",         # 89.43     0.7641      1.8740
]

ia_data_set = pd.DataFrame()
for ia_body in integrator_analysis_bodies:
    perm_des = with_data.loc[with_data.designation_number == ia_body]
    prov_des = with_data.loc[with_data.provisional_designation == ia_body]
    ia_data_set = pd.concat([ia_data_set, perm_des, prov_des])

if not plot_only_integrator_analysis_set:
    data_set = outputs_df
else:
    data_set = ia_data_set
    print(f"N in integrator analysis set: {len(ia_data_set)}")

# Plot
fig = plt.figure(
    figsize = np.array([1.0,0.67])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
gs = gridspec.GridSpec(
    2, 3,
    figure = fig,
    width_ratios = [1, 5, 1],
    height_ratios = [4, 1],
    hspace = 0.005,
    wspace = 0.005,
)
# Set up scatter plots
ax_scat = fig.add_subplot(gs[0, 1])
ax_e_hist = fig.add_subplot(gs[0, 0], sharey=ax_scat)
ax_a_hist = fig.add_subplot(gs[1, 1], sharex=ax_scat)
ax_blank_L = fig.add_subplot(gs[1, 0])
ax_blank_R = fig.add_subplot(gs[1, 2])
ax_scat = isu.set_default_2d_ax_settings(ax_scat, colormode)
ax_e_hist = isu.set_default_2d_ax_settings(ax_e_hist, colormode)
ax_a_hist = isu.set_default_2d_ax_settings(ax_a_hist, colormode)

# Scatter plot
PHAs = data_set.loc[data_set.PHA_flag == True]
non_PHAs = data_set.loc[data_set.PHA_flag == False]
print(f"N PHAs: {len(PHAs)}")
print(f"N Non-PHAs: {len(non_PHAs)}")
# min_arc = data_set.obs_arc_est_YEAR.min()
# max_arc = data_set.obs_arc_est_YEAR.max()
min_arc = 0.0
max_arc = 100.0
temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    vmin = min_arc,
    vmax = max_arc,
    cmap = 'jet',
    s = 50 if plot_only_integrator_analysis_set else 25,
)
sc = ax_scat.scatter(
    PHAs.semimajoraxis_AU,
    PHAs.eccentricity,
    c = PHAs.obs_arc_est_YEAR,
    label = 'PHAs',
    marker = 'v',
    **temp_scatter_kwargs,
)
ax_scat.scatter(
    non_PHAs.semimajoraxis_AU,
    non_PHAs.eccentricity,
    c = non_PHAs.obs_arc_est_YEAR,
    label = 'Non-PHAs',
    marker = 'o',
    **temp_scatter_kwargs,
)
if plot_only_integrator_analysis_set:
    ax_scat.scatter(
        with_data.semimajoraxis_AU,
        with_data.eccentricity,
        label = 'Not selected',
        marker = '.',
        s = 3,
        color = isu.text_color[colormode],
        alpha = 0.3,
        zorder = 2.4,
    )

# colorbar for scatter plot
cb = fig.colorbar(sc, ax=ax_scat)#, label=")
cb.outline.set_edgecolor(isu.text_color[colormode])
cb.ax.tick_params(
    color = isu.text_color[colormode],
    labelsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
    # Move ticks and tick labels to the left side
    right = True, labelright = False,
    left = False, labelleft = False,
)

# Histograms
bins_e = np.arange(0, 1.01, 0.05)
ax_e_hist.hist(
    data_set.eccentricity,
    bins = bins_e,
    orientation = "horizontal",
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)
bins_a = np.arange(0.60, 2.70, 0.10)
ax_a_hist.hist(
    data_set.semimajoraxis_AU,
    bins = bins_a,
    # orientation = "horizontal",
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)
# Observation arc histogram
ax_arc_hist = fig.add_subplot(gs[0, 2], sharey=cb.ax)
ax_arc_hist = isu.set_default_2d_ax_settings(ax_arc_hist, colormode)
bins_arc = np.arange(0.0, 101.0, 5.0)
ax_arc_hist.hist(
    data_set.obs_arc_est_YEAR,
    bins = bins_arc,
    orientation = "horizontal",
    ec = isu.cmap_custom10[colormode](0),
    **hist_contour_kwargs,
)


# Legend, axis settings, labels, etc.
temp_legend_kwargs = dict(
    markerscale = 1.0 if plot_only_integrator_analysis_set else 1.5,
    fontsize = isu.tiny_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax_scat.legend(loc = 'best', **temp_legend_kwargs)
ax_scat.set_ylim((0, 0.93))
ax_scat.set_yticks(np.arange(0,0.95,0.1))
_, xmax = ax_scat.get_xlim()
ax_scat.set_xlim((0.5, xmax))
ax_scat.tick_params(labelbottom=False, labelleft=False)
#
ax_e_hist.set_ylabel(r"Eccentricity $e$")
ax_e_hist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#
ax_a_hist.set_xlabel(r"Semi-major axis $a$ [AU]")
ax_a_hist.tick_params(right=True, labelright=True, left=False, labelleft=False)
#
ax_arc_hist.set_ylabel(r"Observation arc [y]")
ax_arc_hist.yaxis.set_label_position("right")
ax_arc_hist.set_xlim(ax_arc_hist.get_xlim()[::-1])
ax_arc_hist.tick_params(
    left = False, labelleft = False,
    right = True, labelright = True,
    top = True, labeltop = True,
    bottom = False, labelbottom = False,
)
ax_arc_hist.set_ylim((0, 100))
#
temp_text_kwargs = dict(
    s = "Count",
    color = isu.text_color[colormode],
    fontsize = isu.small_font_size,
)
ax_blank_L.text(
    x = 0.4,
    y = 1.0,
    horizontalalignment = 'center',
    verticalalignment = 'top',
    transform = ax_blank_L.transAxes,
    **temp_text_kwargs,
)
ax_blank_L.text(
    x = 1.0,
    y = 0.4,
    horizontalalignment = 'right',
    verticalalignment = 'center',
    transform = ax_blank_L.transAxes,
    rotation = 90,
    rotation_mode = 'default',
    **temp_text_kwargs
)
ax_blank_R.text(
    x = 0.5,
    y = 1.0,
    horizontalalignment = 'center',
    verticalalignment = 'top',
    transform = ax_blank_R.transAxes,
    **temp_text_kwargs,
)
# Spines, ticks, facecolor for blank axes
for ax in [ax_blank_L, ax_blank_R]:
    for spine in ax.spines.values(): # ax borders
        spine.set_edgecolor(isu.fig_background[colormode])
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.set_facecolor(isu.fig_background[colormode])
    ax.grid(visible=False)
if plot_only_integrator_analysis_set:
    # for ax in [ax_e_hist, ax_a_hist, ax_arc_hist]:
    ax_e_hist.set_xlim((0, 3.4))
    ax_a_hist.set_ylim((0, 3.4))
    ax_arc_hist.set_xlim((3.4, 0))

    ia_hist_ticks = np.arange(0, 3.4, 3)
    ax_e_hist.set_xticks(ia_hist_ticks)
    ax_a_hist.set_yticks(ia_hist_ticks)
    ax_arc_hist.set_xticks(ia_hist_ticks)


# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 14.3 Scatter plot of e, a, coloured by 1/rej
### 14.3 Scatter plot of e, a, coloured by 1/rej
#region 14.3 Scatter e, a rej

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_e_a_success_reject"
# format = 'pdf'
format = 'png'

# Plot
fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.67])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)

# Scatter plot
temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    alpha = 0.9,
)
ax.scatter(
    successes.semimajoraxis_AU,
    successes.eccentricity,
    color = isu.cmap_custom10[colormode](2),
    label = 'Successful',
    marker = 'o',
    s = 20,
    **temp_scatter_kwargs,
)
ax.scatter(
    low_SNR.semimajoraxis_AU,
    low_SNR.eccentricity,
    color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    marker = 'X',
    s = 25,
    **temp_scatter_kwargs,
)
if len(high_RMS) > 0:
    ax.scatter(
        high_RMS.semimajoraxis_AU,
        high_RMS.eccentricity,
        color = isu.cmap_custom10[colormode](4),
        label = 'Rej. res.',
        marker = 'D',
        s = 15,
        **temp_scatter_kwargs,
    )

# Legend, labels, axis settings,
leg = ax.legend(
    loc = 'best',
    **legend_kwargs,
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.set_ylim((0, 0.93))
ax.set_yticks(np.arange(0,0.95,0.1))
_, xmax = ax.get_xlim()
ax.set_xlim((0.5, xmax))
ax.set_xlabel(r"Semi-major axis $a$ [AU]")
ax.set_ylabel(r"Eccentricity $e$")


# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 14.4 Scatter e, a, rej, w. histograms
### 14.4 Scatter e, a, rej, w. histograms
#region 14.4 Scat e, a rej hist

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"scatter_e_a_success_reject_with-histograms"
# format = 'pdf'
format = 'png'

# Plot
fig = plt.figure(
    figsize = np.array([1.0,0.75])*isu.twothirds_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
gs = gridspec.GridSpec(
    2, 2,
    figure = fig,
    width_ratios = [1, 3],
    height_ratios = [2, 1],
    hspace = 0.005,
    wspace = 0.005,
)
# Set up axes
ax_scat = fig.add_subplot(gs[0, 1])
ax_e_hist = fig.add_subplot(gs[0, 0], sharey=ax_scat)
ax_a_hist = fig.add_subplot(gs[1, 1], sharex=ax_scat)
ax_blank_L = fig.add_subplot(gs[1, 0])
ax_scat = isu.set_default_2d_ax_settings(ax_scat, colormode)
ax_e_hist = isu.set_default_2d_ax_settings(ax_e_hist, colormode)
ax_a_hist = isu.set_default_2d_ax_settings(ax_a_hist, colormode)

## Scatter plot
# Scatter plot
temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    alpha = 0.9,
)
ax_scat.scatter(
    successes.semimajoraxis_AU,
    successes.eccentricity,
    color = isu.cmap_custom10[colormode](2),
    label = 'Successful',
    marker = 'o',
    s = 20,
    **temp_scatter_kwargs,
)
ax_scat.scatter(
    low_SNR.semimajoraxis_AU,
    low_SNR.eccentricity,
    color = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    marker = 'X',
    s = 25,
    **temp_scatter_kwargs,
)
if len(high_RMS) > 0:
    ax_scat.scatter(
        high_RMS.semimajoraxis_AU,
        high_RMS.eccentricity,
        color = isu.cmap_custom10[colormode](4),
        label = 'Rej. res.',
        marker = 'D',
        s = 15,
        **temp_scatter_kwargs,
    )
_, xmax = ax_scat.get_xlim()

## Eccentricity Histogram
step = 0.05
bins_combi = np.arange(0.0, 0.95+step, step)
ax_e_hist.hist(
    successes.eccentricity,
    bins = bins_combi,
    orientation = "horizontal",
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax_e_hist.hist(
    low_SNR.eccentricity,
    bins = bins_combi,
    orientation = "horizontal",
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    ax_e_hist.hist(
        high_RMS.eccentricity,
        bins = bins_combi,
        orientation = "horizontal",
        label = "Rej. res.",
        ec = isu.cmap_custom10[colormode](4),
        linestyle = '--',
        **hist_contour_kwargs,
    )


## Semi-Major Axis Histogram
step = 0.1
bins_combi = bins_a = np.arange(0.60, 2.60+step, step)
ax_a_hist.hist(
    successes.semimajoraxis_AU,
    bins = bins_combi,
    label = "Successful",
    ec = isu.cmap_custom10[colormode](2),
    **hist_contour_kwargs,
)
ax_a_hist.hist(
    low_SNR.semimajoraxis_AU,
    bins = bins_combi,
    label = 'Rejected' if dataset_to_load == 'final' else 'Rej. SNR',
    ec = isu.cmap_custom10[colormode](3) if dataset_to_load == 'final' else isu.cmap_custom10[colormode](1),
    linestyle = '--',
    **hist_contour_kwargs,
)
if len(high_RMS) > 0:
    ax_a_hist.hist(
        high_RMS.semimajoraxis_AU,
        bins = bins_combi,
        label = "Rej. res.",
        linestyle = '--',
        ec = isu.cmap_custom10[colormode](4),
        **hist_contour_kwargs,
    )


## Legend, axis settings, labels, etc.
temp_legend_kwargs = dict(
    markerscale = 1.5,
    fontsize = isu.tiny_font_size,
    labelcolor = isu.text_color[colormode],
    facecolor = isu.legend_bg_color[colormode],
)
ax_scat.legend(loc = 'best', **temp_legend_kwargs)
ax_scat.set_ylim((0, 0.93))
ax_scat.set_yticks(np.arange(0,0.95,0.1))
ax_scat.set_xlim((0.5, xmax))
ax_scat.tick_params(labelbottom=False, labelleft=False)
#
# ax_e_hist.legend(loc = 'best', **temp_legend_kwargs)
ax_e_hist.set_ylabel(r"Eccentricity $e$")
ax_e_hist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#
ax_a_hist.legend(loc = 'best', **temp_legend_kwargs)
ax_a_hist.set_xlabel(r"Semi-major axis $a$ [AU]")
ax_a_hist.tick_params(right=True, labelright=True, left=False, labelleft=False)
#
temp_text_kwargs = dict(
    s = "Count",
    color = isu.text_color[colormode],
    fontsize = isu.medium_font_size,
)
ax_blank_L.text(
    x = 0.4,
    y = 1.0,
    horizontalalignment = 'center',
    verticalalignment = 'top',
    transform = ax_blank_L.transAxes,
    **temp_text_kwargs,
)
ax_blank_L.text(
    x = 1.0,
    y = 0.4,
    horizontalalignment = 'right',
    verticalalignment = 'center',
    transform = ax_blank_L.transAxes,
    rotation = 90,
    rotation_mode = 'default',
    **temp_text_kwargs
)

# Spines, ticks, facecolor for blank axes
for ax in [ax_blank_L]:
    for spine in ax.spines.values(): # ax borders
        spine.set_edgecolor(isu.fig_background[colormode])
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.set_facecolor(isu.fig_background[colormode])
    ax.grid(visible=False)

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)

#%% 15.1 Scatter plot of RMS_norm_res_est vs. min_earth_dist_AU
### 15.1 Scatter plot of RMS_norm_res_est vs. min_earth_dist_AU
#region 15.1 scat res vs. dist

# This plot is only relevant if dataset_to_load != 'final'

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
# fig_name = f"scatter_RMS_norm_res_vs_min_target-Earth_dist__k1_NEOCC"
fig_name = f"scatter_RMS_norm_res_vs_min_target-Earth_dist__k2_NEOCC"
# fig_name = f"scatter_RMS_norm_res_vs_initial_target-Earth_dist__k1_NEOCC"
# fig_name = f"scatter_RMS_norm_res_vs_initial_target-Earth_dist__k2_NEOCC"
# format = 'pdf'
format = 'png'

fig, ax = plt.subplots(
    1, 1,
    figsize = np.array([1.0,0.60])*isu.half_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = isu.set_default_2d_ax_settings(ax, colormode)
ax.axvline(linewidth=1, color=isu.text_color[colormode])

min_arc = 0.0
max_arc = 10.0
temp_scatter_kwargs = dict(
    linewidths = 0.5,
    edgecolors = scatter_edgecolors[colormode],
    zorder = 2.5,
    vmin = min_arc,
    vmax = max_arc,
    cmap = 'jet',
    alpha = 0.7,
    s = 60,
)
sc = ax.scatter(
    # x = successes.initial_earth_dist_AU,
    x = successes.min_earth_dist_AU,
    y = successes.RMS_norm_res_est,
    # c = successes.k1,
    c = successes.k2,
    marker = '.',
    **temp_scatter_kwargs,
)
# colorbar
cb = fig.colorbar(sc, ax=ax)#, label=")
cb.set_label(
    # label = r"$k_1^\text{NEOCC}$",
    label = r"$k_2^\text{NEOCC}$",
    fontsize = isu.medium_font_size,
    color = isu.text_color[colormode],
)
cb.outline.set_edgecolor(isu.text_color[colormode])
cb.ax.tick_params(
    color = isu.text_color[colormode],
    labelsize = isu.small_font_size,
    labelcolor = isu.text_color[colormode],
)

# # Legend, labels
# leg = fig.legend(
#     loc = 'outside center right',
#     **legend_kwargs,
# )
# for lh in leg.legend_handles:
#     # Make legend items full opacity
#     lh.set_alpha(1)

# Labels / titles
# ax.set_xlabel("Initial target-Earth distance [AU]")
ax.set_xlabel("Min. target-Earth distance [AU]")
ax.set_ylabel("RMS normalised residuals")

# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
)


#%% 15.2 Scatter plots of RMSNR_est vs. min dist planets AU
### 15.2 Scatter plot of RMSNR_est vs. min dist planets AU
#region 15.2 scat res vs. dist

# Plot output
# colormode = 1
pb = 'show'
# pb = 'save'

# format = 'pdf'
format = 'png'

def RMSNR_vs_min_planet_dist(
    x, y,
    x_label, y_label,
    fig_name,
    z = None, z_label = None,
    pb = 'show',
    xlims = None,
    format = 'pdf',
):

    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.60])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = 'constrained',
    )
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale('log')

    min_arc = 0.0
    max_arc = 10.0
    temp_scatter_kwargs = dict(
        linewidths = 0.5,
        edgecolors = scatter_edgecolors[colormode],
        zorder = 2.5,
        vmin = min_arc,
        vmax = max_arc,
        cmap = 'jet',
        alpha = 0.7,
        s = 60,
    )
    sc = ax.scatter(
        x = x,
        y = y,
        c = z,
        marker = '.',
        **temp_scatter_kwargs,
    )
    # colorbar
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(
        label = z_label,
        fontsize = isu.medium_font_size,
        color = isu.text_color[colormode],
    )
    cb.outline.set_edgecolor(isu.text_color[colormode])
    cb.ax.tick_params(
        color = isu.text_color[colormode],
        labelsize = isu.small_font_size,
        labelcolor = isu.text_color[colormode],
    )

    # Plot output
    output_plot(
        pb = pb,
        colormode = colormode,
        plots_save_path = plots_save_path,
        fig_name = fig_name,
        format = format,
    )

relative_position_bodies = ["Mercury","Venus","Earth","Moon","Mars","Ceres","Vesta","Pallas"]
for relative_position_body in relative_position_bodies:
    RMSNR_vs_min_planet_dist(
        x = successes[f'min_dist_{relative_position_body}_AU'],
        y = successes.RMS_norm_res_est,
        x_label = f"Min. target-{relative_position_body} distance [AU]",
        y_label = "RMSNR",
        fig_name = f"scatter_RMSNR_est_vs_min_dist_{relative_position_body}__k1_NEOCC",
        z = successes.k1,
        z_label = r"$k_1^\text{NEOCC}$",
        pb = pb,
        xlims = None,
        format = format,
    )

#%% 16. Scatter plot of RMS_norm_res_est vs. various
### 16. Scatter plot of RMS_norm_res_est vs. various
#region 16 scat res vs. various

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
# format = 'pdf'
format = 'png'

def scatter_simple(x, y, color_idx=2, label=None,
                   x_rej=None, y_rej=None, color_idx_rej=3, label_rej=None,
                   x_label=None, y_label=None,
                   fig_name=None, pb='show',
                   diagonal=False, kspans=False, xlims=None,
                   x_log=None, y_log=None,
                   h_line=None, v_line=None,
                   format='pdf',
):

    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.60])*isu.half_width_INCH,
        facecolor = isu.fig_background[colormode],
        layout = 'constrained',
    )
    ax = isu.set_default_2d_ax_settings(ax, colormode)

    ax.scatter(
        x, y,
        color = isu.cmap_custom10[colormode](color_idx),
        alpha = 0.6,
        label = label,
        **scatter_kwargs,
    )
    if not (isinstance(x_rej, type(None)) or isinstance(y_rej, type(None)) or isinstance(color_idx_rej, type(None))):
        ax.scatter(
            x_rej, y_rej,
            color = isu.cmap_custom10[colormode](color_idx_rej),
            alpha = 0.6,
            label = label_rej,
            **reject_scatter_kwargs,
        )

    if (label or label_rej):
        # Legend, labels
        ax.legend(
            loc = 'best',
            **legend_kwargs,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if diagonal:
        min_x = max(min(x), min(y))
        min_y = min(max(x), max(y))
        ax.plot(
            np.linspace(min_x, min_y, 20),
            np.linspace(min_x, min_y, 20),
            color = isu.text_color[colormode],
            linestyle = '--',
            alpha = 0.7,
        )

    if kspans:
        xlims = ax.get_xlim()
        for span, color in zip(k1_spans, k1_span_colors):
            ax.axvspan(
                xmin = span[0],
                xmax = span[1],
                color = color,
                alpha = 0.20,
            )
        ax.set_xlim((0, xlims[1]))

    if xlims:
        ax.set_xlim(xlims)
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    if h_line:
        ax.axhline(h_line, color=isu.text_color[colormode], linewidth=1, linestyle='--')
    if v_line:
        ax.axvline(v_line, color=isu.text_color[colormode], linewidth=1, linestyle='--')

    # Plot output
    output_plot(
        pb = pb,
        colormode = colormode,
        plots_save_path = plots_save_path,
        fig_name = fig_name,
        format = format,
    )

# 1a: RMSNR Tudat vs. arc
scatter_simple(
    x = with_data.obs_arc_est_YEAR,
    y = with_data.RMS_norm_res_est,
    color_idx = 0,
    x_label = "Observation arc [year]",
    y_label = "RMS norm. res. (Tudat)",
    fig_name = "scatter_RMSNR_vs_obs_arc",
    pb = pb,
    format = format,
)
# 1b: SNR Tudat vs. arc
scatter_simple(
    x = with_data.obs_arc_est_YEAR,
    y = with_data.SNR_A2_est,
    color_idx = 0,
    x_label = "Observation arc [year]",
    y_label = r"$SNR_{A_2}$",
    y_log = True,
    h_line = 3,
    fig_name = "scatter_SNR_vs_obs_arc",
    pb = pb,
    format = format,
)

# 2: RMSNR Tudat vs. eccentricity
scatter_simple(
    x = with_data.eccentricity,
    y = with_data.RMS_norm_res_est,
    color_idx = 0,
    x_label = r"Eccentricity $e$",
    y_label = "RMS norm. res. (Tudat)",
    fig_name = "scatter_RMSNR_vs_eccentricity",
    xlims = (0,1),
    pb = pb,
    format = format,
)
# 2b: SNR Tudat vs. eccentricity
scatter_simple(
    x = with_data.eccentricity,
    y = with_data.SNR_A2_est,
    color_idx = 0,
    x_label = r"Eccentricity $e$",
    y_label = r"$SNR_{A_2}$",
    fig_name = "scatter_SNR_vs_eccentricity",
    y_log = True,
    h_line = 3,
    xlims = (0,1),
    pb = pb,
    format = format,
)

# 3a: RMSNR Tudat vs. semi major axis
scatter_simple(
    x = with_data.semimajoraxis_AU,
    y = with_data.RMS_norm_res_est,
    color_idx = 0,
    x_label = r"Semi-major axis $a$ [AU]",
    y_label = "RMS norm. res. (Tudat)",
    fig_name = "scatter_RMSNR_vs_semimajoraxis",
    xlims = (0.6,2.6),
    pb = pb,
    format = format,
)
# 3b: SNR Tudat vs. semi major axis
scatter_simple(
    x = with_data.semimajoraxis_AU,
    y = with_data.SNR_A2_est,
    color_idx = 0,
    x_label = r"Semi-major axis $a$ [AU]",
    y_label = r"$SNR_{A_2}$",
    fig_name = "scatter_RMSNR_vs_semimajoraxis",
    y_log = True,
    h_line = 3,
    xlims = (0.6,2.6),
    pb = pb,
    format = format,
)

# 4a: RMSNR Tudat vs. k1_NEOCC
scatter_simple(
    x = successes.k1,
    y = successes.RMS_norm_res_est,
    # label = 'Successful',
    # x_rej = low_SNR.k1,
    # y_rej = low_SNR.RMS_norm_res_est,
    # label_rej = 'Rejected',
    x_label = r"$k_1^\text{NEOCC}$",
    y_label = "RMS norm. res. (Tudat)",
    fig_name = "scatter_RMSNR_vs_k1_NEOCC",
    kspans = True,
    pb = pb,
    format = format,
)
# 4b: RMSNR Tudat vs. k2_NEOCC
scatter_simple(
    x = successes.k2,
    y = successes.RMS_norm_res_est,
    # label = 'Successful',
    # x_rej = low_SNR.k2,
    # y_rej = low_SNR.RMS_norm_res_est,
    # label_rej = 'Rejected',
    x_label = r"$k_2^\text{NEOCC}$",
    y_label = "RMS norm. res. (Tudat)",
    fig_name = "scatter_RMSNR_vs_k2_NEOCC",
    kspans = True,
    # xlims = (0,1),
    pb = pb,
    format = format,
)

# 5a: RMSNR NEOCC vs RMSNR Tudat
scatter_simple(
    x = with_data.RMS_norm_res_est,
    y = with_data.RMS_norm_res_lit,
    color_idx = 0,
    x_label = "RMSNR Tudat",
    y_label = "RMSNR NEOCC",
    fig_name = "scatter_RMSNR_NEOCC_vs_Tudat",
    diagonal = True,
    pb = pb,
    format = format,
)
# 5b: RMSNR JPL vs RMSNR Tudat
scatter_simple(
    x = with_data.RMS_norm_res_est,
    y = with_data.RMS_norm_res_JPL,
    color_idx = 0,
    x_label = "RMSNR Tudat",
    y_label = "RMSNR JPL",
    fig_name = "scatter_RMSNR_JPL_vs_Tudat",
    diagonal = True,
    pb = pb,
    format = format,
)
# 5c: RMSNR JPL vs RMSNR NEOCC
scatter_simple(
    x = with_data.RMS_norm_res_lit,
    y = with_data.RMS_norm_res_JPL,
    color_idx = 0,
    x_label = "RMSNR NEOCC",
    y_label = "RMSNR JPL",
    fig_name = "scatter_RMSNR_JPL_vs_NEOCC",
    diagonal = True,
    pb = pb,
    format = format,
)

# 6: FE/TE (pos) vs. various
TE_RMSs_pos_KM = with_data.RMS_true_error_pos_mag_KM
FE_RMSs_pos_KM = with_data.RMS_formal_error_pos_mag_KM
ratio_RMSs_pos_err = FE_RMSs_pos_KM / TE_RMSs_pos_KM
# Eccentricity
scatter_simple(
    x = with_data.eccentricity,
    y = ratio_RMSs_pos_err,
    color_idx = 0,
    x_label = r"Eccentricity $e$",
    y_label = r"$(FE_{RMS} / TE_{RMS})_{pos.}$",
    fig_name = "scatter_FE-TE-pos_vs_eccentricity",
    y_log = True,
    h_line = 1,
    pb = pb,
    format = format,
)
# Semi-major axis
scatter_simple(
    x = with_data.semimajoraxis_AU,
    y = ratio_RMSs_pos_err,
    color_idx = 0,
    x_label = "Semi-major axis a [AU]",
    y_label = r"$(FE_{RMS} / TE_{RMS})_{pos.}$",
    fig_name = "scatter_FE-TE-pos_vs_semimajoraxis",
    y_log = True,
    h_line = 1,
    pb = pb,
    format = format,
)
# RMSNR
scatter_simple(
    x = with_data.RMS_norm_res_est,
    y = ratio_RMSs_pos_err,
    color_idx = 0,
    x_label = "RMS norm. res. (Tudat)",
    y_label = r"$(FE_{RMS} / TE_{RMS})_{pos.}$",
    fig_name = "scatter_FE-TE-pos_vs_RMSNR",
    y_log = True,
    h_line = 1,
    pb = pb,
    format = format,
)

# 7a: FE Tudat vs. FE JPLH (position)
scatter_simple(
    x = with_data.RMS_JPLH_ephem_pos_unc_mag_KM,
    y = with_data.RMS_formal_error_pos_mag_KM,
    color_idx = 0,
    x_label = r"$FE_{RMS}^{JPLH}$ [km]",
    y_label = r"$FE_{RMS}^{Tudat}$ [km]",
    fig_name = "scatter_FE-Tudat_vs_FE-JPLH_pos",
    x_log = True,
    y_log = True,
    diagonal = True,
    # xlims = (1,1e4),
    pb = pb,
    format = format,
)
# 7b: FE Tudat vs. FE JPLH (velocity)
scatter_simple(
    x = with_data.RMS_JPLH_ephem_vel_unc_mag,
    y = with_data.RMS_formal_error_vel_mag_MM / 1000,
    color_idx = 0,
    x_label = r"$FE_{RMS}^{JPLH} [m/s]$",
    y_label = r"$FE_{RMS}^{Tudat} [m/s]$",
    fig_name = "scatter_FE-Tudat_vs_FE-JPLH_pos",
    x_log = True,
    y_log = True,
    diagonal = True,
    # xlims = (1e-3, 1e0),
    pb = pb,
    format = format,
)

#%% 17. 3d plot of all 348 asteroid orbits
### 17. 3d plot of all 348 asteroid orbits
#region 3d orbits

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'

# 1. Create a dummy DataFrame matching your structure
# a: semi-major axis (AU), e: eccentricity, i: inclination (deg)
# Om: longitude of ascending node (deg), w: argument of perihelion (deg)
data = {
    'asteroid': ['Ceres', 'Pallas', 'Vesta'],
    'a': [2.767, 2.772, 2.361],
    'e': [0.078, 0.231, 0.089],
    'i': [10.59, 34.84, 7.14],
    'Om': [80.33, 173.08, 103.81],
    'w': [73.59, 310.04, 151.20]
}
df = pd.DataFrame(data)

def get_orbit_coordinates(a, e, i, Om, w, num_pts=200):
    """Calculates the 3D Cartesian coordinates of an orbit."""
    # Convert angles from degrees to radians
    i_rad = np.radians(i)
    Om_rad = np.radians(Om)
    w_rad = np.radians(w)

    # Generate true anomaly (theta) around the ellipse
    theta = np.linspace(0, 2 * np.pi, num_pts)

    # Calculate radius r for each theta using the polar equation of an ellipse
    r = a * (1 - e**2) / (1 + e * np.cos(theta))

    # Coordinates in the orbital plane (z is 0)
    x_orbital = r * np.cos(theta)
    y_orbital = r * np.sin(theta)
    z_orbital = np.zeros_like(theta)
    coords_orbital = np.vstack((x_orbital, y_orbital, z_orbital))

    # Rotation matrices
    # 1. Rotate by argument of perihelion (w) around Z-axis
    # 2. Rotate by inclination (i) around X-axis
    # 3. Rotate by longitude of ascending node (Om) around Z-axis
    cos_w, sin_w = np.cos(w_rad), np.sin(w_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
    cos_Om, sin_Om = np.cos(Om_rad), np.sin(Om_rad)

    R_w = np.array([[cos_w, -sin_w, 0], [sin_w, cos_w, 0], [0, 0, 1]])
    R_i = np.array([[1, 0, 0], [0, cos_i, -sin_i], [0, sin_i, cos_i]])
    R_Om = np.array([[cos_Om, -sin_Om, 0], [sin_Om, cos_Om, 0], [0, 0, 1]])

    # Combined transformation matrix
    R = R_Om @ R_i @ R_w

    # Transform all orbital plane coordinates to reference frame
    coords_reference = R @ coords_orbital

    return coords_reference[0], coords_reference[1], coords_reference[2]

# 2. Set up the 3D Matplotlib figure
fig = plt.figure(
    figsize = np.array([1.0,0.60])*isu.full_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = fig.add_subplot(111, projection='3d')

# ax.grid(visible=False)
# ax.set_facecolor(isu.fig_background[colormode])
# for spine in ax.spines.values(): # ax borders
#     spine.set_edgecolor(isu.fig_background[colormode])
# ax.tick_params(
#     axis = 'both',
#     which = 'major',
#     labelsize = isu.small_font_size,
#     labelcolor = isu.fig_background[colormode],
#     color = isu.fig_background[colormode],
# )
# ax.set_ylabel(
#     None,
#     fontsize = isu.medium_font_size,
#     color = isu.fig_background[colormode],
# )
# ax.set_xlabel(
#     None,
#     fontsize = isu.medium_font_size,
#     color = isu.fig_background[colormode],
# )
# ax.set_title( # subtitle below main title / subplot title
#     None,
#     color = isu.text_color[colormode],
# )


# Plot the Sun at the center (focal point)
ax.scatter(
    0, 0, 0,
    color = 'gold',
    s = 150,
    label = 'Sun',
    # edgecolor = 'orange',
)

# 3. Iterate over the DataFrame rows and plot each asteroid
for idx, row in df.iterrows():
    x, y, z = get_orbit_coordinates(row['a'], row['e'], row['i'], row['Om'], row['w'])

    # Plot the orbital line
    line, = ax.plot(
        x, y, z,
        label = row['asteroid'],
        linewidth = 2,
    )

    # Plot current position (or perihelion as an anchor point)
    # ax.scatter(x[0], y[0], z[0], color=line.get_color(), s=40)

# 4. Final plot styling
# ax.set_title('3D Asteroid Orbital Ellipses', fontsize=14, fontweight='bold')
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')

# Equalize axis scaling to avoid visual distortion of the ellipses
max_range = df['a'].max() * 1.2
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend(loc='upper left')

plt.show()

#%% 17. 3d plot of all 348 asteroid orbits
### 17. 3d plot of all 348 asteroid orbits
#region 3d orbits

# Plot output
colormode = 1
pb = 'show'
# pb = 'save'
fig_name = f"3d_orbit_all_348_asteroids"
# format = 'pdf'
format = 'png'

def get_orbit_coordinates(a, e, i, Om, w, theta=None, num_pts=200):
    """Calculates the 3D Cartesian coordinates of an orbit."""
    # Convert angles from degrees to radians
    i_rad, Om_rad, w_rad = np.radians([i, Om, w])

    if theta == None:
        # Generate true anomaly (theta) around the ellipse
        theta = np.linspace(0, 2 * np.pi, num_pts)
    else:
        # If a specific value has been given, convert it to radians
        theta = np.deg2rad(theta)

    # Calculate radius r for each theta using the polar equation of an ellipse
    r = a * (1 - e**2) / (1 + e * np.cos(theta))

    # Coordinates in the orbital plane (z is 0)
    coords_orbital = np.vstack((
        r * np.cos(theta),
        r * np.sin(theta),
        np.zeros_like(theta),
    ))

    # Rotation matrices
    # 1. Rotate by argument of perihelion (w) around Z-axis
    # 2. Rotate by inclination (i) around X-axis
    # 3. Rotate by longitude of ascending node (Om) around Z-axis
    cos_w, sin_w = np.cos(w_rad), np.sin(w_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
    cos_Om, sin_Om = np.cos(Om_rad), np.sin(Om_rad)

    R_w = np.array([
        [cos_w, -sin_w, 0],
        [sin_w, cos_w, 0],
        [0, 0, 1],
    ])
    R_i = np.array([
        [1, 0, 0],
        [0, cos_i, -sin_i],
        [0, sin_i, cos_i],
    ])
    R_Om = np.array([
        [cos_Om, -sin_Om, 0],
        [sin_Om, cos_Om, 0],
        [0, 0, 1],
    ])

    # Combined transformation matrix
    R = R_Om @ R_i @ R_w

    # Transform all orbital plane coordinates to reference frame
    coords_reference = R @ coords_orbital

    return coords_reference[0], coords_reference[1], coords_reference[2]

# Planet DataFrame
# https://www.met.reading.ac.uk/~ross/Astronomy/Planets.html
planet_data = {
    'planet':   ['Mercury', 'Venus',   'Earth',   'Mars',    'Jupiter', 'Saturn',  'Uranus',   'Neptune',  'Pluto'],
    'a':        [0.38709893,0.72333199,1.00000011,1.52366231,5.20336301,9.53707032,19.19126393,30.06896348,39.48168677],
    'e':        [0.20563069,0.00677323,0.01671022,0.09341233,0.04839266,0.05415060,0.04716771, 0.00858587, 0.24880766 ],
    'i':        [7.00487,   3.39471,   0.00005,   1.85061,   1.30530,   2.48446,   0.76986,    1.76917,    17.14175   ],
    'Om':       [48.33167,  76.68069,  -11.26064, 49.57854,  100.55615, 113.71504, 74.22988,   131.72169,  110.30347  ],
    'w':        [77.45645,  131.53298, 102.94719, 336.04084, 14.75385,  92.43194,  170.96424,  44.97135,   224.06676  ],
    'ma':       [252.25084, 181.97973, 100.46435, 355.45332, 34.40438,  49.94432,  313.23218,  304.88003,  238.92881  ],
    'color_idx':[7,         6,         0,         3,         1,         4,         8,          9,           5],
}
planet_data = pd.DataFrame(planet_data)

# 1. Set up the 3D Matplotlib figure
fig = plt.figure(
    figsize = np.array([1.0,0.7])*3*isu.full_width_INCH,
    facecolor = isu.fig_background[colormode],
    layout = 'constrained',
)
ax = fig.add_subplot(111, projection='3d', facecolor=isu.fig_background[colormode])
# Minimalist Override: Remove all background elements, grids, ticks, and frames
ax.set_axis_off()

# 2. Plot the Sun at the center
ax.scatter(
    0, 0, 0,
    color = isu.cmap_custom10[colormode](4),
    s = 150,
    # zorder=5,
)

# 4. Iterate over the DataFrame rows and plot each Planets
for idx, row in planet_data[0:4].iterrows():
    # Plot orbital path
    x, y, z = get_orbit_coordinates(
        row['a'],
        row['e'],
        row['i'],
        row['Om'],
        row['w'],
    )
    line, = ax.plot(
        x, y, z,
        linewidth = 3.5,
        label = row['planet'],
        color = isu.cmap_custom10[colormode](row['color_idx']),
        alpha = 0.8,
    )

    # Plot current position (or perihelion as an anchor point)
    # ax.scatter(x[0], y[0], z[0], color=line.get_color(), s=40)
    x, y, z = get_orbit_coordinates(
        row['a'], row['e'], row['i'], row['Om'], row['w'], row['ma'],
    )
    ax.scatter(
        x[0], y[0], z[0],
        color=line.get_color(),
        s = 100,
        alpha = 0.8,
    )

# 3. Iterate over the DataFrame rows and plot each asteroid
for idx, row in outputs_df.iterrows():
    body = isu.query_SBDB(row['SPKID'])

    # Plot orbital path
    x, y, z = get_orbit_coordinates(
        body['semimajoraxis_AU'],
        body['eccentricity'],
        body['inclination_DEG'],
        body['RA_ascendingnode_DEG'],
        body['argument_periapsis_DEG'],
    )
    line, = ax.plot(
        x, y, z,
        linewidth = 1.5,
        label = body['longname'],
        color = isu.cmap_custom10[colormode](7),
        alpha = 0.15,
    )

    # # Plot position
    # # ax.scatter(x[0], y[0], z[0], color=line.get_color(), s=40)
    # x, y, z = get_orbit_coordinates(
    #     body['semimajoraxis_AU'],
    #     body['eccentricity'],
    #     body['inclination_DEG'],
    #     body['RA_ascendingnode_DEG'],
    #     body['argument_periapsis_DEG'],
    #     body['mean_anomaly_DEG'],
    # )
    # ax.scatter(
    #     x[0], y[0], z[0],
    #     color = isu.cmap_custom10[colormode](7),
    #     # color=line.get_color(),
    #     s = 20,
    #     alpha = 0.3,
    # )

# 5. Equalize axis scaling to avoid visual distortion
max_range = df['a'].max() * 0.6
max_range = df['a'].max() * 1.0
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)


# Plot output
output_plot(
    pb = pb,
    colormode = colormode,
    plots_save_path = plots_save_path,
    fig_name = fig_name,
    format = format,
    dpi = 300,
)


#%% 18. QR codes
### 18. QR codes
#region QR codes

import qrcode as qr


url = 'https://github.com/ronanjmc/MScThesis_RonanMcIntyre_Public'
fig_name = "qr_github.png"

img = qr.make(url)
# # type(img)  # qrcode.image.pil.PilImage
img.save(plots_save_path + "PNGs/" + fig_name)


url = 'https://resolver.tudelft.nl/uuid:56640d62-64f6-4041-bf46-9ffe02394a1b'
fig_name = "qr_thesis.png"

img = qr.make(url)
# # type(img)  # qrcode.image.pil.PilImage
img.save(plots_save_path + "PNGs/" + fig_name)

# qrc = qr.QRCode(
#     version=1,
#     error_correction=qr.constants.ERROR_CORRECT_L,
#     box_size=10,
#     border=4,
# )
# qrc.add_data(url)
# qrc.make(fit=True)
# # img = qrc.make_image(fill_color=(0, 30, 49), back_color=(207,197,139))
# img.save("mn142-qr.png")

