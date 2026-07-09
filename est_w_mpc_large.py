""" est_w_mpc_large.py:
File name is an abbreviation for 'Estimation with Minor Planet Centre (MPC) data'.
est_w_mpc.py is a script that performs an initial state estimation for a given
body using astrometric observations from the MPC. The Yarkovsky effect (A2
parameter) can also be included in this estimation. Different estimation setups
(e.g. with/without debiasing, with/without weighting) can be compared.

This script is a conversion of est_w_mpc.py into a function so that it can be
called many times to do a large run of estimations in one go. The comparison of
different estimation setups is not available in this script.

The basic structure of this script is based on the "Improved state estimation
with MPC" example* on the Tudat Space website, and several of the plots in this
script is directly from that example, with edits for style/formatting.
* https://docs.tudat.space/en/latest/examples/tudatpy-examples/estimation/improved_estimation_with_mpc.html#Comparison-Round-2:-Weighting,-Star-Catalog-Corrections-and-Satellite-data

"""

#%%%%%%# A. Imports
######## A. Imports
#region A. Imports

import time as tm
T_START_imports = tm.perf_counter()
print("Importing packages and loading SPICE kernels...")

# Directory business
import os
local_path = os.path.dirname(__file__)
os.chdir(local_path) # ensure that the cwd is the subfolder where this file is

# Tudat imports
from tudatpy.dynamics import (
    environment_setup,
    propagation_setup,
    parameters_setup,
)
from tudatpy.estimation import (
    observations,
    observations_setup,
    estimation_analysis,
    observable_models_setup,
)
import tudatpy.constants as cnst
from tudatpy.interface import spice
from tudatpy.astro import frame_conversion
from tudatpy.astro.time_representation import DateTime
from tudatpy.data.mpc import BatchMPC
from tudatpy.data.horizons import HorizonsQuery

# Other Python imports
import json
import numpy as np
import pandas as pd
import datetime as dttm

# Local imports
import IS_utils as isu
import estimation_plots as est_plots

# Load SPICE kernels, other data
from tudatpy.exceptions.spice_exceptions import SpiceNOSUCHFILE, SpiceSPKINSUFFDATA
spice.clear_kernels()
spice.load_standard_kernels()
# Yarkovsky data
Fenucci_etal2024_tabB1 = isu.load_Fenucci_etal2024_tabB1()
# Greenberg_et_al_2020_tab = isu.load_Greenberg_et_al_2020_tab()

T_END_imports = tm.perf_counter()
print(f"Done ({T_END_imports - T_START_imports:.3f}s) \n")


#%%%%%%######################   B. Function   ##################################
#############################   B. Function   ##################################
#region B. Function

def estimation_with_MPC(
    ## Target body
    body_to_propagate: str,                         # accepts designation number, common name, or provisional designation of asteroid
    ## Observations
    obs_range_start: list[int] = [1900, 1, 1],      # start of time range to consider for astrometric observations of body_to_propagate. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.
    obs_range_end: list[int] = [2026, 7, 1],        # end of time range to consider for astrometric observations of body_to_propagate. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.
    spacecraft_to_use: list[str] = [],              # DEFAULT RECOMMENDED! list of spacecraft whose observations to use
    observatories_to_exclude_MPC_codes: list[str] = [], # list of ground observatories to filter out from observation set
    #region 1. Function inputs
    ## Dynamic model
    dynamic_model_def: str = "LB+LMBA+AB+Yark",     # define the acceleration model (start with "LB", add "+EIH", "+LMBA", "+CP", "+AB", "+Yark" as desired). "+CP" NOT RECOMMENDED!
    n_LMBAs: int = 16,                              # number of MBAs whose gravitation is to be included in dynamics (takes the n_LMBAs most massive MBAs). Only relevant if 'LMBA' is included in dynamic_model_def
    n_MBAs_CP_search: int = 388,                    # number of MBAs to include when searching for close passes (takes the n_MBAs_CP_search most massive MBAs after excluding the n_LMBAs most massive MBAs). Maximum: 396 - n_LMBAs (as of 11 Jun 2025)
    close_pass_threshold_AU: float = 0.5,           # distance threshold that defines what is considered a close pass [AU]
    close_pass_influence_threshold: float = 0.01,   # close pass displacement threshold s [metres]; s = 0.5*a_max*t^2; a_max is max grav. acc. due to close pass; t is propagation duration. close passers with less influence than this are filtered out
    ## Integrator
    integrator: str = None,                        # fixed step integration method. "Euler" or "RK[X]" (Runge-Kutta method of order X, with X = [1,2,3,4,5,6,7,8,9,10,12,14])
    step_size_DAY: float = None,                     # duration of propagation timestep in days
    ## Estimation
    number_of_pod_iterations: int = 4,              # number of iterations of the estimation process
    use_catalog_cor: bool = True,                   # whether to use offset star catalog biases as described in: "Star catalog position and proper motion corrections in asteroid astrometry II: The Gaia era" by Eggl et al. (2018)
    use_obs_weights: bool = True,                   # whether to weigh observations using the weighting scheme presented in "Statistical analysis of astrometric errors for the most productive asteroid surveys" by Veres et al. (2017)
    residual_filter_cutoff_ARCSEC: float = 5.0,     # residuals in the prefit that are larger than this cutoff (in arcsec) will be filtered out and not used in the estimation
    ## Frames
    central_body: str = "Sun",
    global_frame_origin: str = "SSB",
    global_frame_orientation: str = "J2000",
    ## Plotting & Printing
    verbose: bool = True,                           # choose True to include additional supplementary print statements
    quiet: bool = False,                            # choose True to suppress all print statements
    colormode: int = 0,                             # 0 for lightmode, 1 for darkmode when showing plots (both colormodes are saved when plot_behaviour is 'save' anyway)
    plot_behaviour: str = 'save',                   # 'save' to save all plots as PDFs. 'show' to display them. 'both' to save and display. 'showboth' to display plots in both light and darkmode
    est_w_mpc_plot_dir: str = "yarkovsky-est-w-mpc/_large-runs/", # base directory for saving run outputs (plots and data)
    plots_in_RSW: bool = True,                            # If false, plots will show components in XYZ/Cartesian frame
    comparison_reference: str = 'spk'               # 'spk' or 'horizons', affects which data source is used as reference in position error plots
    ## Inputs from file
    # input_JSON_file: str = None,                     # use input variables from a previously saved JSON file. Supercedes any other specified input parameters
    ## Global variables used in this function:
    # - Fenucci_etal2024_tabB1
    # - Greenberg_et_al_2020_tab
    # - spacecraft_codes
    # - bodies_to_skip
    # - timestamp_str
    # - timestamp_dir
):


    ######################## 2. Input processing ###############################
    #region 2. Input processing

    ## Printing setup
    if verbose and quiet:
        raise ValueError("Verbose mode and quiet mode have both been selected. Choose one or neither.")
    # Function that prints args if quiet=False and does nothing if quiet=True
    print_default = print if not quiet else lambda *a, **k: None
    # Function that prints args if verbose=True and does nothing if verbose=False
    print_verbose = print if verbose else lambda *a, **k: None

    ## Use input varleiabs from a previously saved JSON file
    # if input_JSON_file:
    #     read_path = isu.plot_dir + est_w_mpc_plot_dir + ...
    #     with open(read_path + "_inputs.json", 'r') as json_fp:
    #         input_read = json.load(json_fp)[0]
    #     for i in input_read:
    #     globals()[i] = input_read[i]

    ## Get body data from JPL SBDB and print selected info
    body = isu.query_SBDB(body_to_propagate, print_body_info=verbose)

    if (body_to_propagate in bodies_to_skip):
        print(f"WARNING: Skipping asteroid {body['longname']}.")
        flag = "skip"
        output_data = {
            'body': body['longname'],
            'flag': flag,
            # # BODY DATA
            'designation_number': body['designation_number'],
            'common_name': body['common_name'],
            'provisional_designation': body['provisional_designation'],
            'SPKID': body['SPKID'],
            'NEO_flag': body['NEO_flag'],
            'PHA_flag': body['PHA_flag'],
            'orbitperiod_DAY': body['orbitperiod_DAY'],
            'semimajoraxis_AU': body['semimajoraxis_AU'],
            'eccentricity': body['eccentricity'],
            'diameter_KM': body['diameter_KM'],
            'diameter_unc_KM': body['diameter_unc_KM'],
            'diameter_ref': body['diameter_ref'],
            'earth_MOID_AU': body['earth_MOID_AU'],
            # # JPL SBDB ORBIT FIT DATA
            'RMS_norm_res_JPL': body['RMS_norm_res'],
            'n_obs_used_JPL': body['n_obs_used'],
            'n_dop_obs_used_JPL': body['n_dop_obs_used'],
            'n_del_obs_used_JPL': body['n_del_obs_used'],
            'obs_arc_JPL_YEAR': body['obs_arc_DAY'] / 365.2425,
            # 'A2_JPL': body['A2_AU_per_DAYsq'] * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2),
            # 'A2_unc_JPL': body['A2_unc_AU_per_DAYsq'] * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2),
        }
        return output_data, flag

    ## Integrator and ∆t selection
    if (integrator is None) and (step_size_DAY is None):
        integrator, step_size_DAY = isu.define_integrator_and_step_size_DAY_from_eccentricity(
            e = body['eccentricity'],
        )
        print(f"e = {body['eccentricity']:.3f} | Integrator: {integrator} | ∆t={step_size_DAY*24} h")
    elif (integrator is None) or (step_size_DAY is None):
        raise ValueError("Either integrator and step_size_DAY should both be None or they should both be defined.")
    coefficient_set, order_to_use = isu.process_integrator_input(
        integrator, print_integrator_info=verbose
    )

    ## Observation range time variables
    obs_range_start_J2000 = DateTime(*obs_range_start).epoch() # in seconds since J2000
    obs_range_end_J2000 = DateTime(*obs_range_end).epoch() # in seconds since J2000

    ## Spacecraft observatories
    use_SC_data = True if spacecraft_to_use else False
    spacecraft_to_use_MPC_codes = [isu.spacecraft_codes[name][0] for name in spacecraft_to_use]

    ## Process dynamic model definition
    EIH_needed = True if "EIH" in dynamic_model_def else False
    LMBAs_needed = True if "LMBA" in dynamic_model_def else False
    n_LMBAs = 0 if LMBAs_needed == False else n_LMBAs # set n_LMBAs to 0 if not needed
    CPs_needed = True if "CP" in dynamic_model_def else False
    ABs_needed = True if "AB" in dynamic_model_def else False
    Yark_needed = True if "Yark" in dynamic_model_def else False

    ## Yarkovsky parameter
    if Yark_needed:
        A2_target_body, A2_target_body_unc, Fenucci_body_data, yark_source = \
            isu.get_Yarkovsky_parameter(body, verbose)

    ## Plotting behaviour
    plot_behaviour_list, _ = isu.define_plot_behaviour_lists(
        plot_behaviour, colormode,
    )

    ## Create dictionary of all relevant inputs, to be saved to a file later
    input_dict = {
        'body_to_propagate': body_to_propagate,
        # Observations
        'obs_range_start': obs_range_start,
        'obs_range_end': obs_range_end,
        'spacecraft_to_use': spacecraft_to_use,
        'observatories_to_exclude_MPC_codes': observatories_to_exclude_MPC_codes,
        # Dynamic model
        'dynamic_model_def': dynamic_model_def,
        'n_LMBAs': n_LMBAs,
        'n_MBAs_CP_search': n_MBAs_CP_search,
        'close_pass_threshold_AU': close_pass_threshold_AU,
        'close_pass_influence_threshold': close_pass_influence_threshold,
        # Integrator
        'integrator': integrator,
        'step_size_DAY': step_size_DAY,
        # Estimation
        'number_of_pod_iterations': number_of_pod_iterations,
        'use_SC_data': use_SC_data,
        'use_catalog_cor': use_catalog_cor,
        'use_obs_weights': use_obs_weights,
        'residual_filter_cutoff_ARCSEC': residual_filter_cutoff_ARCSEC,
        # Frames
        'central_body': central_body,
        'global_frame_origin': global_frame_origin,
        'global_frame_orientation': global_frame_orientation,
        # Plotting & printing
        'verbose': verbose,
        'quiet': quiet,
        'colormode': colormode,
        'plot_behaviour': plot_behaviour,
        'est_w_mpc_plot_dir': est_w_mpc_plot_dir,
        'plots_in_RSW': plots_in_RSW,
        'comparison_reference': comparison_reference,
    }


    ##################### 3. Retrieve MPC observations #########################
    #region 3.Retrieve MPC

    ## Retrieve observations
    T_START_MPC = tm.perf_counter()
    print_verbose("\nMPC OBSERVATIONS:")
    print_verbose(f"Retrieving MPC observations for {body['longname']} (SPKID: {body['SPKID']}) "
                  f"in range {obs_range_start[0]}-{obs_range_start[1]}-{obs_range_start[2]} "
                  f"to {obs_range_end[0]}-{obs_range_end[1]}-{obs_range_end[2]}...")
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
    # Show batch summary
    if verbose:
        batch.summary()
    if spacecraft_to_use:
        print_verbose("Sumary of space telescopes in batch:")
        print_verbose(batch.observatories_table(only_space_telescopes=True))
    T_END_MPC = tm.perf_counter()
    print_verbose(f"MPC observations retrieved ({T_END_MPC - T_START_MPC:.3f}s)\n")

    ## Filter observations
    # Filter out problematic observatories, if any
    batch.filter(
        observatories_exclude = observatories_to_exclude_MPC_codes,
    )
    n_ground_obs_dropped = n_obs_available - len(batch.table)
    # Filter out unneeded spacecraft observations
    spacecraft_in_batch = batch.observatories_table(only_space_telescopes=True).Name.to_list()
    spacecraft_to_exclude = list(
        set(spacecraft_in_batch) - ( set(spacecraft_to_use) & set(spacecraft_in_batch) )
    )
    spacecraft_to_exclude_MPC_codes = [
        isu.spacecraft_codes[name][0] for name in spacecraft_to_exclude
    ]
    batch.filter(
        observatories_exclude = spacecraft_to_exclude_MPC_codes,
    )
    # Observation numbers
    n_space_obs_dropped = n_obs_available - n_ground_obs_dropped - len(batch.table)
    n_total_obs_used = len(batch.table)
    n_space_obs_used = len(batch.table.query("observatory == @spacecraft_to_use_MPC_codes"))
    n_ground_obs_used = n_total_obs_used - n_space_obs_used

    ## Time variables
    first_obs_J2000 = batch.epoch_start
    last_obs_J2000 = batch.epoch_end
    propagation_start_DateTime, propagation_start_J2000 = \
        isu.set_propagation_start_from_first_observation_epoch(first_obs_J2000)
    propagation_end_DateTime, propagation_end_J2000 = \
        isu.set_propagation_end_from_last_observation_epoch(last_obs_J2000)
    input_dict['propagation_start_J2000'] = propagation_start_J2000
    input_dict['propagation_end_J2000'] = propagation_end_J2000
    # Other time variables
    obs_arc_YEAR = (last_obs_J2000 - first_obs_J2000) / cnst.JULIAN_YEAR
    propagation_duration_SEC = propagation_end_J2000 - propagation_start_J2000
    propagation_duration_YEAR = propagation_duration_SEC / cnst.JULIAN_YEAR
    time_buffer = 2 * 31 * cnst.JULIAN_DAY  # 2 month time buffer to avoid interpolation errors

    ## Print some info
    print(f"Propagation start and end dates: "
        f"{propagation_start_DateTime.year}-{propagation_start_DateTime.month}-{propagation_start_DateTime.day} "
        f"to {propagation_end_DateTime.year}-{propagation_end_DateTime.month}-{propagation_end_DateTime.day} "
        f"({propagation_duration_YEAR:.4g} years)")
    print_verbose(f"Observation arc: {obs_arc_YEAR:.4g} years")
    print_verbose(f"N observations available: {n_obs_available}")
    print_verbose(f"N ground observations excluded: {n_ground_obs_dropped} "
                  f"(from {observatories_to_exclude_MPC_codes})")
    print_verbose(f"N space observations excluded: {n_space_obs_dropped} "
                  f"(from {spacecraft_to_exclude})")
    print_verbose(f"N observations to be used: {n_total_obs_used} "
                  f"({n_ground_obs_used} ground + {n_space_obs_used} space)\n")



    ################# 4.1 Environment setup (define bodies) ####################
    #region 4.1 Env. bodies

    print_verbose("\nENVIRONMENT SET-UP:")
    ## Bodies to be retrieved through SPICE
    bodies_SPICE = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Moon",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    ]

    ## SiMDA bodies
    # Search for close passes between target body and other asteroids (checks those
    # listed in SiMDA file) during propagation period
    if CPs_needed:
        close_passers = isu.search_for_close_passes_with_Kepler_elements(
            target_body_name = body_to_propagate,
            n_MBAs_search = n_MBAs_CP_search,
            n_MBAs_already_included = n_LMBAs,
            search_period_initial_time = propagation_start_DateTime,
            n_orbits = propagation_duration_SEC / (body['orbitperiod_DAY'] * cnst.JULIAN_DAY),
            close_pass_threshold_AU = close_pass_threshold_AU,
            timestep_DAY = step_size_DAY * 5.0,
            verbose = verbose,
        )
    else:
        print_verbose("Skipping close pass search.")
        close_passers = {}

    # SiMDA bodies: largest main-belt asteroids (LMBAs) and close passers (CPs)
    target_int = int(body['designation_number']) \
        if body['designation_number'] != '[unnumbered]' else 0
    bodies_SiMDA_filtered = (
        pd.read_csv(isu.SiMDA_file)
        .query("DYN != 'COM' & DYN != 'TNO'") # *
        .assign(NUM=lambda x: np.int32(x.NUM))
        .query("NUM != @target_int") # remove propagated body, if present
        .sort_values(by='MASS', ascending=False) # sort bodies by mass, descending order
        .loc[:, ["NUM", "DESIGNATION", "DIAM", "DYN", "MASS"]]
    )
    # * filter out comets and trans-Neptunian objects (COM, TNO). Remaining
    #   types are AMO, APO, MBA, NEA, OMB (see https://pdssbn.astro.umd.edu/data_other/objclass.shtml)
    bodies_SiMDA_largest = bodies_SiMDA_filtered.head(n_LMBAs) # take n_LMBAs most massive
    bodies_SiMDA_closepasses = bodies_SiMDA_filtered[
        bodies_SiMDA_filtered['NUM'].isin(close_passers)
    ] # take any close passers

    # Filter close passers that will have minimal influence on target body
    if CPs_needed:
        for idx, cp in bodies_SiMDA_closepasses.iterrows():
            # Assume maximum acceleration applies throughout propagation, use suvat
            # to calculate a quasi "displacement" due to this acceleration. Only
            # keep a CP body if this displacement is greater than close_pass_influence_threshold
            max_acc_due_to_cp = (cp['MASS'] * cnst.GRAVITATIONAL_CONSTANT) / \
                ((close_passers[cp['NUM']] * cnst.ASTRONOMICAL_UNIT)**2)
            displacement_due_to_cp = 0.5 * max_acc_due_to_cp * (propagation_duration_SEC**2)
            # print_verbose(f"{cp['NUM']:<{8}} {max_acc_due_to_cp*(10**15):.5} {'fm/s2':<{8}} {displacement_due_to_cp:.5} m")
            if displacement_due_to_cp < close_pass_influence_threshold:
                bodies_SiMDA_closepasses = bodies_SiMDA_closepasses.drop(idx)
        # Limit to 30 influential close passers (max amount that fits on a plot)
        bodies_SiMDA_closepasses = bodies_SiMDA_closepasses.head(30)
        print_verbose(f"Of {len(close_passers)} close passers found, adding "
                      f"{len(bodies_SiMDA_closepasses)} to dynamics.")

    bodies_SiMDA_included = pd.concat(
        [bodies_SiMDA_largest, bodies_SiMDA_closepasses]
    ).drop_duplicates().reset_index(drop=True)
    largest_MBAs = bodies_SiMDA_largest.NUM.to_list()
    closepass_MBAs = bodies_SiMDA_closepasses.NUM.to_list()
    all_included_MBAs = bodies_SiMDA_included.NUM.to_list()
    all_included_MBAs_masses = bodies_SiMDA_included.MASS.to_list()

    # Get SBDB body data for all SiMDA bodies (nice to have common name, etc. for plotting)
    included_MBAs_data_dict = {}
    for designation_number in all_included_MBAs:
        body_data = isu.query_SBDB(str(designation_number))
        included_MBAs_data_dict[designation_number] = body_data

    print_verbose(f"Number of additional bodies from SiMDA: "
                  f"{len(bodies_SiMDA_included)} ({len(bodies_SiMDA_largest)} largest + "
                  f"{len(bodies_SiMDA_closepasses)} close passers) \n")


    ## Additional bodies
    if ABs_needed:
        bodies_additional_JPL = [("999", "Pluto")] # format for each body is (JPL Horizons id, common name)
        bodies_additional_masses_KG = [1.46180119e22] # Mass of Pluto system. Source: Table 8 of Brozović et al. 2024 (https://iopscience.iop.org/article/10.3847/1538-3881/ad39f0#ajad39f0t8)
    else:
        bodies_additional_JPL = []
        bodies_additional_masses_KG = []



    ############### 4.1 Environment setup (get ephemerides) ####################
    #region 4.2 Env. ephem.

    T_START_get_ephemerides = tm.perf_counter()
    print_verbose("Loading ephemerides...")
    step_size_MIN = (step_size_DAY * cnst.JULIAN_DAY) / 60
    timestep_horizons_MIN = step_size_MIN * 5.0 # Longer time step for non-target body HorizonsQueries
    JPLHQ_location = central_body

    # Ephemerides for space telescopes
    print_verbose("...for space telescopes...")
    SC_ephemerides = {}
    for sc in spacecraft_to_use:
        query = HorizonsQuery(
            query_id = isu.spacecraft_codes[sc][1],
            location = f"@{JPLHQ_location}",
            epoch_start = propagation_start_J2000 - time_buffer,
            epoch_end = propagation_start_J2000 + time_buffer,
            epoch_step = f"{int(timestep_horizons_MIN)}m", # Time step in minutes so that step_size_DAY doesn't need to be an integer
            extended_query = True, # extended query allows for more data to be retrieved.
        )
        SC_ephemerides[sc] = query.create_ephemeris_tabulated(
            frame_origin = global_frame_origin,
            frame_orientation = global_frame_orientation,
        )

    # Ephemerides for SiMDA asteroids (largest + close passers)
    print_verbose("...for SiMDA bodies (LMBAs + CPs)...")
    MBA_ephemerides = {}
    for code in all_included_MBAs:
        # Try to load ephemeris from SPK file
        try:
            spice.load_kernel(isu.SPK_file_dir + included_MBAs_data_dict[code]['SPKID'] + ".bsp" )
            print_verbose(f"    SPK file of {included_MBAs_data_dict[code]['longname']} loaded. "
                          f"(SPKID: {included_MBAs_data_dict[code]['SPKID']})")
            MBA_ephemerides[code] = environment_setup.ephemeris.direct_spice(
                global_frame_origin,
                global_frame_orientation,
                included_MBAs_data_dict[code]['SPKID']
            )
        # If no SPK file, use HorizonsQuery
        except (SpiceNOSUCHFILE) as ex:
            print_verbose(f"    Error when loading SPK file for "
                  f"{included_MBAs_data_dict[code]['longname']} "
                  f"(SPKID: {included_MBAs_data_dict[code]['SPKID']}). "
                  f"Using JPL HorizonsQuery instead.")
            query = HorizonsQuery(
                query_id = f"{code};",
                location = f"@{JPLHQ_location}",
                epoch_start = propagation_start_J2000 - time_buffer,
                epoch_end = propagation_end_J2000 + time_buffer,
                epoch_step = f"{int(timestep_horizons_MIN)}m",
                extended_query = True,
            )
            MBA_ephemerides[code] = query.create_ephemeris_tabulated(
                frame_origin = JPLHQ_location,
                frame_orientation = global_frame_orientation,
            )

    # Ephemerides for additional bodies
    print_verbose("...for additional bodies...")
    additional_ephemerides = {}
    for code in bodies_additional_JPL:
        query = HorizonsQuery(
            query_id = f"{code[0]}",
            location = f"@{JPLHQ_location}",
            epoch_start = propagation_start_J2000 - time_buffer,
            epoch_end = propagation_end_J2000 + time_buffer,
            epoch_step = f"{int(timestep_horizons_MIN)}m",
            extended_query = True,
        )
        additional_ephemerides[code[1]] = query.create_ephemeris_tabulated(
            frame_origin = JPLHQ_location,
            frame_orientation = global_frame_orientation,
        )
    bodies_additional = list(additional_ephemerides.keys())

    # Ephemeris for target body (used for estimation and plotting))
    print_verbose(f"...for target body {body['longname']}...")
    # Try to load ephemeris from SPK file
    try:
        if not os.path.exists(isu.SPK_file_dir + body['SPKID'] + ".bsp"):
            # If no SPK file for this body exists already, then generate it
            spkid = isu.generate_spk_file(
                spkid = body['SPKID'],
                spk_file_path = isu.SPK_file_dir,
                spk_file_name = body['SPKID'],
            )
        spice.load_kernel(isu.SPK_file_dir + body['SPKID'] + ".bsp")
        print_verbose(f"SPK file of {body['longname']} loaded. (SPKID: {body['SPKID']})")
        target_body_ephemeris = environment_setup.ephemeris.direct_spice(
            global_frame_origin, global_frame_orientation, body['SPKID']
        )
        target_body_ephemeris = environment_setup.ephemeris.tabulated_from_existing(
            target_body_ephemeris,
            start_time = propagation_start_J2000 - time_buffer,
            end_time = propagation_end_J2000 + time_buffer,
            time_step = step_size_DAY * cnst.JULIAN_DAY,
        )
    # If no SPK file, use HorizonsQuery
    except SpiceNOSUCHFILE as err:
    # except:
        # print(f"Unexpected {err=}, {type(err)=}")
        print_verbose(f"Error when loading SPK file for {body['longname']} "
                      f"(SPKID: {body['SPKID']}). \nUsing JPL HorizonsQuery instead.")
        target_ephemeris_query = HorizonsQuery(
            query_id = f"{target_int};",
            location = f"@{JPLHQ_location}",
            epoch_start = propagation_start_J2000 - time_buffer,
            epoch_end = propagation_end_J2000 + time_buffer,
            epoch_step = f"{int(step_size_MIN)}m", # smaller time step for target body
            extended_query = True,
        )
        target_body_ephemeris = target_ephemeris_query.create_ephemeris_tabulated(
            frame_origin = JPLHQ_location,
            frame_orientation = global_frame_orientation,
        )

    T_END_get_ephemerides = tm.perf_counter()
    print_verbose(f"Ephemerides loaded ({T_END_get_ephemerides - T_START_get_ephemerides:.3f}s) \n")



    ###################### 4.3 Ephemeris uncertainty ###########################
    #region 4.3 Ephem. unc.

    # Get ephemeris uncertainty information (from local file, if available,
    # otherwise by retrieving via astroquery)
    try:
        body_ephem_unc_table = isu.load_with_pickle(isu.ephem_unc_dir, f"{body['SPKID']}.p")

    except FileNotFoundError:
        _ = isu.retrieve_and_save_ephemeris_uncertainty(
            spkid = body['SPKID'],
            file_path = isu.ephem_unc_dir,
            file_name = spkid,
        )
        body_ephem_unc_table = isu.load_with_pickle(isu.ephem_unc_dir, f"{body['SPKID']}.p")

    # Make columns with epoch times using DateTime object and J2000 format
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



    ########### 4.4 Environment setup (create system of bodies) ################
    #region 4.4 Env. system

    # Add SPICE bodies
    body_settings = environment_setup.get_default_body_settings(
        bodies_SPICE, global_frame_origin, global_frame_orientation
    )
    # Add spacecraft and their ephemerides to body settings
    for name in spacecraft_to_use:
        body_settings.add_empty_settings(name)
        body_settings.get(name).ephemeris_settings = SC_ephemerides[name]
    # Add SiMDA asteroids (largest + close passers) to body settings (ephemerides and gravity fields)
    for code, mass in zip(all_included_MBAs, all_included_MBAs_masses):
        body_settings.add_empty_settings(included_MBAs_data_dict[code]['SPKID'])
        body_settings.get(included_MBAs_data_dict[code]['SPKID']).ephemeris_settings = MBA_ephemerides[code]
        body_settings.get(included_MBAs_data_dict[code]['SPKID']).gravity_field_settings = (
            environment_setup.gravity_field.central(mass * cnst.GRAVITATIONAL_CONSTANT)
        )
    # Add additional bodies to body settings (ephemerides and gravity fields)
    for code, mass in zip(bodies_additional_JPL, bodies_additional_masses_KG):
        body_settings.add_empty_settings(str(code[1]))
        body_settings.get((code[1])).ephemeris_settings = additional_ephemerides[code[1]]
        body_settings.get((code[1])).gravity_field_settings = (
            environment_setup.gravity_field.central(mass * cnst.GRAVITATIONAL_CONSTANT)
        )
    # Add target body with name = SPKID
    body_settings.add_empty_settings(body['SPKID'])
    body_settings.get(body['SPKID']).gravity_field_settings = (
            environment_setup.gravity_field.central(0 * cnst.GRAVITATIONAL_CONSTANT)
    ) # ^define gravity field but set mass to zero (needed for EIH)
    body_settings.get(body['SPKID']).ephemeris_settings = target_body_ephemeris

    # Use ephemerides and masses for Jupiter, Saturn, Mars system barycentres
    body_settings.get( "Jupiter" ).ephemeris_settings = environment_setup.ephemeris.direct_spice(
        global_frame_origin, global_frame_orientation, "Jupiter_Barycenter" )
    body_settings.get( "Jupiter" ).gravity_field_settings = environment_setup.gravity_field.central(
        spice.get_body_gravitational_parameter( "Jupiter_Barycenter") )
    body_settings.get( "Saturn" ).ephemeris_settings = environment_setup.ephemeris.direct_spice(
        global_frame_origin, global_frame_orientation, "Saturn_Barycenter" )
    body_settings.get( "Saturn" ).gravity_field_settings = environment_setup.gravity_field.central(
        spice.get_body_gravitational_parameter( "Saturn_Barycenter") )
    body_settings.get( "Mars" ).ephemeris_settings = environment_setup.ephemeris.direct_spice(
        global_frame_origin, global_frame_orientation, "Mars_Barycenter" )
    body_settings.get( "Mars" ).gravity_field_settings = environment_setup.gravity_field.central(
        spice.get_body_gravitational_parameter( "Mars_Barycenter") )

    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies_to_propagate = [body['SPKID']]
    central_bodies = [central_body]

    # Get initial state of target body
    try:
        initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name = body['SPKID'],
            observer_body_name = central_body,
            reference_frame_name = global_frame_orientation,
            aberration_corrections = "NONE",
            ephemeris_time = propagation_start_J2000,
        )
    except SpiceSPKINSUFFDATA as err:
        print("Getting target body initial state estimate via HorizonsQuery, not SPK file")
        target_query = HorizonsQuery(
            query_id = f"{body['designation_number']};",
            location = f"@{central_body}",
            epoch_list = [propagation_start_J2000]
        )
        initial_state = target_query.cartesian(frame_orientation = global_frame_orientation)[0][1:]



    ##################### 5. Acceleration settings #############################
    #region 5.Acc. settings

    # Large body accelerations (LB) (Sun + 8 planets + Moon)
    accelerations_large_bodies = isu.define_large_body_accelerations(EIH_needed)

    # Largest MBAs body accelerations
    accelerations_LMBAs = {
        included_MBAs_data_dict[code]['SPKID']: \
            [propagation_setup.acceleration.point_mass_gravity()]
            for code in largest_MBAs
    }
    # Close passers body accelerations
    accelerations_CPs = {
        included_MBAs_data_dict[code]['SPKID']: \
            [propagation_setup.acceleration.point_mass_gravity()]
            for code in closepass_MBAs
    }
    # Additional body accelerations
    accelerations_ABs = {
        str(name): [propagation_setup.acceleration.point_mass_gravity()]
        for name in bodies_additional
    }
    # Yarkovsky accelerations
    if Yark_needed:
        sun_accelerations_list = [propagation_setup.acceleration.yarkovsky(A2_target_body)] \
            + accelerations_large_bodies["Sun"]
        accelerations_yarkovsky = {
            "Sun": sun_accelerations_list,
        }
        accelerations_large_bodies = accelerations_large_bodies | accelerations_yarkovsky

    accelerations_all = accelerations_large_bodies | accelerations_LMBAs | \
        accelerations_CPs | accelerations_ABs



    ###################### 6. Propagation settings #############################
    #region 6.Prop. settings

    # Integrator settings
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_step = step_size_DAY * cnst.JULIAN_DAY,
        coefficient_set = coefficient_set,
        order_to_use = order_to_use,
    )

    relative_position_bodies = {
        # Name in SystemOfBodies : [common/readable name for plots, color_idx, linestyle]
        "Mercury":  ["Mercury", 7, 'solid'], # grey
        "Venus":    ["Venus",   1, 'solid'], # orange
        "Earth":    ["Earth",   0, 'solid'], # blue
        "Moon":     ["Moon",    8, 'dashed'], # cyan
        "Mars":     ["Mars",    3, 'solid'], # red
        "20000001": ["Ceres",   3, 'dashed'], # green
        "20000004": ["Vesta",   6, 'dashed'], # brown
        "20000002": ["Pallas",  9, 'dashed'], # purple
    }

    # Dependent variables to save
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position(body['SPKID'], rel_pos_bod)
        for rel_pos_bod in relative_position_bodies
    ]


    ####################### 7. Perform estimation ##############################
    #region 7. Perform est.

    def perform_estimation(
        bodies,
        use_spacecraft_data: bool,
        apply_star_catalog_debias: bool,
        apply_weighting_scheme: bool,
    ):

        # The satellites are present in the integration of all setups, the
        # included_spacecraft parameter in to_tudat() dictates whether a
        # satellite's observations are used
        if use_spacecraft_data:
            included_spacecraft = {
                isu.spacecraft_codes[name][0]: name for name in spacecraft_to_use
            }
        else:
            included_spacecraft = None

        # The observation collection is created with BatchMPC.to_tudat().
        # Internally, to_tudat() links a space telescope's observatory code to
        # the spacecraft's dynamics
        batch_temp = batch.copy()

        # As of 14 June 2026, the observatory with code X71 (OASMiS, Sao Miguel
        # da Serra) is missing from the table that is returned from
        # astroquery.mpc.get_observatory_codes(), which is when defining the
        # observation links. Because of this, X71 is not defined in the
        # observation_settings_list, despite X71's observations being in the
        # ObservationCollection. This results in an error when calling
        # observations.compute_residuals_and_dependent_variables(). To avoid
        # this, X71's observations are filtered out here. What is filtered is 4
        # observations from 2 June 2026 and 5 observations from 4 June 2026.
        batch_temp.filter(observatories_exclude = ['X71'])

        observation_collection = batch_temp.to_tudat(
            bodies = bodies,
            included_satellites = included_spacecraft,
            apply_star_catalog_debias = apply_star_catalog_debias,
            apply_weights_VFCC17 = apply_weighting_scheme,
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

        #region 7.1 Filter obs
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
        prefiltered_obs_epochs_J2000 = observation_collection.concatenated_times[::2]
        # Make a dataframe of simulated residuals, sorted chronologically
        simulated_residuals_ARCSEC = pd.DataFrame(dict(
            epoch_J2000 = prefiltered_obs_epochs_J2000,
            simulated_residual_RA = prefiltered_residuals_ARCSEC[::2],
            simulated_residual_DEC = prefiltered_residuals_ARCSEC[1::2],
        )).sort_values(by='epoch_J2000').reset_index(drop=True)
        # Add simulated residuals to the DataFrame batch_temp_table
        batch_temp_table = batch_temp.table.reset_index(drop=True)
        batch_temp_table['simulated_residual_RA_ARCSEC'] = \
            simulated_residuals_ARCSEC.simulated_residual_RA
        batch_temp_table['simulated_residual_DEC_ARCSEC'] = \
            simulated_residuals_ARCSEC.simulated_residual_DEC

        # Filter outliers from ObservationCollection object
        residual_filter_cutoff_RAD = np.deg2rad(residual_filter_cutoff_ARCSEC / 3600.0)
        residual_filter = observations.observations_processing.observation_filter(
            observations.observations_processing.ObservationFilterType.residual_filtering,
            residual_filter_cutoff_RAD,
        )
        observation_collection.filter_observations(residual_filter)
        observation_collection.remove_empty_observation_sets()
        if len(observation_collection.get_concatenated_observations()) == 0:
            raise RuntimeError("All observations filtered out")
        postfiltered_residuals_RAD = observation_collection.get_concatenated_residuals()

        # Make one table of only outliers and one without outliers
        filter_mask = (
            np.abs(batch_temp_table.simulated_residual_RA_ARCSEC) > residual_filter_cutoff_ARCSEC) \
            | (np.abs(batch_temp_table.simulated_residual_DEC_ARCSEC) > residual_filter_cutoff_ARCSEC
        )
        batch_table_outliers = batch_temp_table.loc[filter_mask] # contains ONLY high-residual observations that were filtered out
        batch_table_filtered = batch_temp_table.loc[~filter_mask] # excludes the high-residual observations that were filtered out

        # Check if epoch of first/last observations have changed due to filtering
        nonlocal first_obs_J2000, last_obs_J2000, propagation_start_J2000, propagation_end_J2000
        change_in_first_obs_epoch_DAY = \
            (min(observation_collection.concatenated_times) - first_obs_J2000) / cnst.JULIAN_DAY
        change_in_last_obs_epoch_DAY = \
            (last_obs_J2000 - max(observation_collection.concatenated_times)) / cnst.JULIAN_DAY
        first_obs_J2000 = min(observation_collection.concatenated_times)
        last_obs_J2000 = max(observation_collection.concatenated_times)
        if change_in_first_obs_epoch_DAY > 1.0:
            # Change start time
            new_propagation_start_DateTime, propagation_start_J2000 = \
                isu.set_propagation_start_from_first_observation_epoch(first_obs_J2000)
            input_dict['propagation_start_J2000'] = propagation_start_J2000
            print(f"Propagation start time changed from "
                f"{propagation_start_DateTime.year}-{propagation_start_DateTime.month}-{propagation_start_DateTime.day} "
                f"to {new_propagation_start_DateTime.year}-{new_propagation_start_DateTime.month}-{new_propagation_start_DateTime.day} "
                f"due to observation filtering")
            # New initial state
            nonlocal initial_state
            initial_state = spice.get_body_cartesian_state_at_epoch(
                target_body_name = body['SPKID'],
                observer_body_name = central_body,
                reference_frame_name = global_frame_orientation,
                aberration_corrections = "NONE",
                ephemeris_time = propagation_start_J2000,
            )
        if change_in_last_obs_epoch_DAY > 1.0:
            # Change end time
            new_propagation_end_DateTime, propagation_end_J2000 = \
                isu.set_propagation_end_from_last_observation_epoch(last_obs_J2000)
            input_dict['propagation_end_J2000'] = propagation_end_J2000
            print(f"Propagation end time changed from "
                f"{propagation_end_DateTime.year}-{propagation_end_DateTime.month}-{propagation_end_DateTime.day} "
                f"to {new_propagation_end_DateTime.year}-{new_propagation_end_DateTime.month}-{new_propagation_end_DateTime.day} "
                f"due to observation filtering")

        # If start OR end time has changed
        if (change_in_first_obs_epoch_DAY > 1.0) or (change_in_last_obs_epoch_DAY > 1.0):
            # Update ephemeris uncertainty table to match new propagation period
            nonlocal body_ephem_unc_table
            mask_after_window_start = np.where(
                body_ephem_unc_table["ephemeris_time_J2000"] > propagation_start_J2000, True, False
            )
            mask_before_window_end = np.where(
                body_ephem_unc_table["ephemeris_time_J2000"] < propagation_end_J2000, True, False
            )
            mask_within_obs_range = np.logical_and(mask_after_window_start, mask_before_window_end)
            body_ephem_unc_table = body_ephem_unc_table[mask_within_obs_range]

        # Filter observations from batch_temp
        # Note: this will not filter all outlier observations from the batch.
        # Only those at the start/end
        batch_temp.filter(
            epoch_start = first_obs_J2000,
            epoch_end = last_obs_J2000,
        )

        #region 7.2 Est. setup
        # Termination settings
        termination_settings = propagation_setup.propagator.time_termination(
            termination_time = propagation_end_J2000,
            terminate_exactly_on_final_condition = True
        )

        # Set up the accelerations settings for each body, in this case only the target body
        acceleration_settings = {}
        for body_to_propagate in bodies_to_propagate:
            acceleration_settings[str(body_to_propagate)] = accelerations_all
        # Create the acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )

        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies = central_bodies,
            acceleration_models = acceleration_models,
            bodies_to_integrate = bodies_to_propagate,
            initial_states = initial_state,
            initial_time = propagation_start_J2000,
            integrator_settings = integrator_settings,
            termination_settings = termination_settings,
            output_variables = dependent_variables_to_save,
        )

        # Set up parameter settings to propagate the state transition matrix
        parameters_to_estimate_settings = parameters_setup.initial_states(
            propagator_settings, bodies
        )
        parameters_to_estimate_settings.append(
            parameters_setup.yarkovsky_parameter(body['SPKID'], "Sun")
        )
        # Create the parameters that will be estimated
        parameters_to_estimate = parameters_setup.create_parameter_set(
            parameters_to_estimate_settings, bodies, propagator_settings
        )
        original_parameter_vector = parameters_to_estimate.parameter_vector

        # Set up the estimator
        estimator = estimation_analysis.Estimator(
            bodies = bodies,
            estimated_parameters = parameters_to_estimate,
            observation_settings = observation_settings_list,
            propagator_settings = propagator_settings,
            integrate_on_creation = True,
        )

        # Provide the observation collection as input and set number of iterations
        estimation_input = estimation_analysis.EstimationInput(
            observations_and_times = observation_collection,
            convergence_checker = estimation_analysis.estimation_convergence_checker(
                maximum_iterations = number_of_pod_iterations,
            ),
        )

        # to_tudat() applies weights to a set of observations between an observatory
        # and the target. The method below tells Tudat to use the weights applied to
        # these sets. This step is required when setting weights through the
        # BatchMPC class
        if apply_weighting_scheme:
            estimation_input.set_weights_from_observation_collection()

        # Set methodological options
        estimation_input.define_estimation_settings(
            reintegrate_variational_equations = True,
            save_state_history_per_iteration = True,
        )

        #region 7.3 Est.
        # Perform the estimation
        print_verbose('Performing the estimation...')
        estimation_output = estimator.perform_estimation(estimation_input)
        print_verbose('Done. \n')

        # Print initial state change information
        initial_state_updated = parameters_to_estimate.parameter_vector
        initial_state_difference_vector = initial_state_updated - original_parameter_vector
        initial_position_difference_mag = np.linalg.norm(initial_state_difference_vector[0:3])
        print_default(f'Original initial state: {original_parameter_vector}')
        print_default(f'Updated initial state: {initial_state_updated} \n')
        print_default(f'Change in initial state: {initial_position_difference_mag/1000:_.3f} km')

        # Store the following outputs for plotting and analysis
        return estimation_output, batch_temp, observation_collection, estimator, \
            prefiltered_residuals_RAD, postfiltered_residuals_RAD, \
            batch_table_outliers, batch_table_filtered


    # Perform estimation
    T_START_estimation = tm.perf_counter()
    print_default(f"\nPERFORMING ESTIMATION FOR: {body['longname']}")

    estimation_output, batch, observation_collection, estimator, \
        prefiltered_residuals_RAD, postfiltered_residuals_RAD, \
        batch_table_outliers, batch_table_filtered = \
            perform_estimation(
                bodies,
                use_spacecraft_data = use_SC_data,
                apply_star_catalog_debias = use_catalog_cor,
                apply_weighting_scheme = use_obs_weights,
    )

    #region 7.4 Extract
    # Extract some important data from EstimationOutput object
    best_iteration_idx = estimation_output.best_iteration
    simulator_object = estimation_output.simulation_results_per_iteration[best_iteration_idx]
    state_history = simulator_object.dynamics_results.state_history
    output_times = np.array(sorted(list(state_history.keys())))
    #
    original_parameter_vector = estimation_output.parameter_history[:,0]
    final_parameter_vector = estimation_output.final_parameters
    initial_state_difference_vector = final_parameter_vector - original_parameter_vector
    initial_position_difference_mag = np.linalg.norm(initial_state_difference_vector[0:3])
    #
    residual_history = estimation_output.residual_history
    prefit_residuals_ARCSEC = np.rad2deg(np.array(residual_history[:, 0])) * 3600
    final_residuals_ARCSEC = np.rad2deg(np.array(residual_history[:, best_iteration_idx])) * 3600
    # Get some key info from the saved dependent variables
    dependent_variable_history = simulator_object.dynamics_results.dependent_variable_history
    dependent_variable_array = np.array(list(dependent_variable_history.values()))
    position_of_target_relative_to_bodies = {
        relative_position_bodies[rel_pos_bod][0]: dependent_variable_array[:, (3*idx):(3*idx + 3)]
        for idx, rel_pos_bod in enumerate(relative_position_bodies)
    }
    distances_target_to_bodies_highlights_AU = {}
    for bod_name, rel_pos in position_of_target_relative_to_bodies.items():
        target_body_distance_AU = np.linalg.norm(rel_pos, axis=1) / cnst.ASTRONOMICAL_UNIT
        # Save min, max, initial, and RMS of distance between target and each selected body
        distances_target_to_bodies_highlights_AU[bod_name] = dict(
            min_dist_AU = target_body_distance_AU.min(),
            max_dist_AU = target_body_distance_AU.max(),
            initial_dist_AU = target_body_distance_AU[0],
            RMS_dist_AU = isu.rms(target_body_distance_AU),
        )

    # Propagate the covariancees to get formal errors in RSW frame
    propagated_covariances = estimation_analysis.propagate_covariance_from_analysis_objects(
        analysis_output = estimation_output,
        state_transition_interface = estimator.state_transition_interface,
        output_times = output_times,
    )
    # Iterate through the time steps
    formal_errors_RSW_KM = np.zeros([len(output_times), 6])
    for i, time in enumerate(output_times):
        state_est = state_history[time]
        rot_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(state_est)
        full_rot_matrix = np.block([[rot_matrix, np.zeros([3,3])], [np.zeros([3,3]), rot_matrix]])
        cov = np.matrix(propagated_covariances[time])
        converted_formal_errors_matrix = full_rot_matrix @ cov @ full_rot_matrix.T
        formal_errors = np.sqrt(np.diagonal(converted_formal_errors_matrix))
        formal_errors_RSW_KM[i, :] = formal_errors / 1000
    # Propagate formal errors in Cartesian frame (since we don't have to do a frame
    # rotation, this is handier than propagating the covariancees and extracting the
    # diagonals)
    propagated_formal_errors = estimation_analysis.propagate_formal_errors(
        initial_covariance = estimation_output.covariance,
        state_transition_interface = estimator.state_transition_interface,
        output_times = output_times,
    )
    # Get formal errors as array in units of kilometers
    formal_errors_XYZ_KM = np.array(list(propagated_formal_errors.values())) / 1000

    T_END_estimation = tm.perf_counter()
    runtime_estimation = T_END_estimation - T_START_estimation
    print_default(f"\nEstimation done ({runtime_estimation:.3f}s) \n")



    ################### 8.1 Save inputs & outputs ##############################
    #region 8.1 Save in/outputs

    body_dir = f"{body['filing_name_long']}/"

    if 'save' in plot_behaviour_list:
        # Directory in which to save output data and plots
        body_output_data_save_path = isu.output_data_dir + est_w_mpc_plot_dir + timestamp_dir + body_dir
        plots_save_path = isu.plot_dir + est_w_mpc_plot_dir + timestamp_dir + body_dir
        if not os.path.exists(body_output_data_save_path):
            os.makedirs(body_output_data_save_path)
        if not os.path.exists(plots_save_path):
            os.makedirs(plots_save_path)

        ## Save inputs that went into this run to JSON file
        # To output data directory
        with open(body_output_data_save_path + "_inputs.json", "w") as f:
            json.dump([input_dict], f, indent=4)
        # And also to plot directory
        with open(plots_save_path + "_inputs.json", "w") as f:
            json.dump([input_dict], f, indent=4)

        ## Save outputs from this run using pickle
        isu.save_estimation_output_attributes(estimation_output, body_output_data_save_path)
        isu.save_batch_attributes(batch, body_output_data_save_path)
        isu.save_with_pickle(relative_position_bodies, body_output_data_save_path, "relative_position_bodies.p")
        isu.save_with_pickle(distances_target_to_bodies_highlights_AU, body_output_data_save_path, "distances_target_to_bodies_highlights_AU.p")
        isu.save_with_pickle(formal_errors_XYZ_KM, body_output_data_save_path, "formal_errors_XYZ_KM.p")
        isu.save_with_pickle(formal_errors_RSW_KM, body_output_data_save_path, "formal_errors_RSW_KM.p")
        isu.save_with_pickle(body, body_output_data_save_path, "body.p")
        isu.save_with_pickle(prefiltered_residuals_RAD, body_output_data_save_path, "prefiltered_residuals_RAD.p")
        isu.save_with_pickle(postfiltered_residuals_RAD, body_output_data_save_path, "postfiltered_residuals_RAD.p")
        isu.save_with_pickle(final_residuals_normalised, body_output_data_save_path, "final_residuals_normalised.p")
        isu.save_with_pickle(batch_table_outliers, body_output_data_save_path, "batch_table_outliers.p")
        isu.save_with_pickle(batch_table_filtered, body_output_data_save_path, "batch_table_filtered.p")

        print_verbose(f"Output data for {body['longname']} estimation saved to: \n{body_output_data_save_path}")


    ###################### 8.2 Post-loop prints ################################
    #region 8.2 Prints

    print_default("ESTIMATION RESULTS:")
    print_verbose(f"Formal errors (rounded to 6 sig. fig.):")
    formal_errs_rounded = [float(f"{fe:.6g}") for fe in list(estimation_output.formal_errors)]
    print_verbose(formal_errs_rounded)
    print_verbose("\n")

    print_default(f"""A2 Literature: (from Fenucci et al. 2024 table B1)
    A2 parameter            : {A2_target_body:.6g} m/s^2
    A2 uncertainty          : {A2_target_body_unc:.6g} m/s^2
    Signal-to-noise ratio   : {Fenucci_body_data['SNR'].item():.4g}
    RMS normalised res.     : {Fenucci_body_data['RMS'].item():.4g}
    Observation arc         : {Fenucci_body_data['deltat'].item():.4g} y \n""")

    A2_est = final_parameter_vector[6]
    fe_A2 = estimation_output.formal_errors[6]
    percent_change_from_ref = ((A2_est - A2_target_body)/A2_target_body)*100
    change_prefix_1 = "+" if percent_change_from_ref > 0 else ""
    k1 = abs(A2_target_body - A2_est) / fe_A2
    k2 = abs(A2_target_body - A2_est) / A2_target_body_unc

    # Post-fit residuals
    RMS_res_est_ARCSEC = isu.rms(final_residuals_ARCSEC)
    assumed_obs_err_RAD = np.sqrt(np.reciprocal(observation_collection.concatenated_weights))
    assumed_obs_err_ARCSEC = np.rad2deg(assumed_obs_err_RAD) * 3600
    final_residuals_normalised = final_residuals_ARCSEC / assumed_obs_err_ARCSEC
    RMS_normalised_res_est = isu.rms(final_residuals_normalised)
    mean_normalised_res_est = np.mean(final_residuals_normalised)
    mean_abs_normalised_res_est = np.mean(np.abs(final_residuals_normalised))

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

    # RMS of formal error
    formal_error_pos_mag_KM = np.linalg.norm(formal_errors_XYZ_KM[:, :3], axis=1)
    formal_error_vel_mag_KM = np.linalg.norm(formal_errors_XYZ_KM[:, 3:], axis=1)
    RMS_formal_error_pos_mag_KM = isu.rms(formal_error_pos_mag_KM)
    RMS_formal_error_vel_mag_MM = isu.rms(formal_error_vel_mag_KM) * 1_000_000

    reference_states = est_plots.get_reference_states(
        comparison_reference = comparison_reference,
        body = body,
        central_body = central_body,
        global_frame_orientation = global_frame_orientation,
        epoch_list = output_times,
    )
    state_history_array = np.array(list(state_history.values()))
    true_error_KM = (reference_states - state_history_array) / 1000
    true_error_pos_mag_KM = np.linalg.norm(true_error_KM[:, :3], axis=1)
    true_error_vel_mag_KM = np.linalg.norm(true_error_KM[:, 3:], axis=1)
    RMS_true_error_pos_mag_KM = isu.rms(true_error_pos_mag_KM)
    RMS_true_error_vel_mag_MM = isu.rms(true_error_vel_mag_KM) * 1_000_000

    print_default(f"""This Estimation:
    Best iteration (idx)    : {best_iteration_idx}
    A2 parameter            : {A2_est:.6g} m/s^2
    A2 formal error         : {fe_A2:.6g} m/s^2
    A2 SNR                  : {abs(A2_est/fe_A2):.4g}
    A2 mag. vs. to lit.     : {change_prefix_1}{percent_change_from_ref:.3g}% mag.
    k_1 [-]                 : {k1:.3g}
    k_2 [-]                 : {k2:.3g}

    Observation arc         : {filtered_obs_arc_YEAR:.2f} y
    ∆arc due to filter      : {change_in_obs_arc_due_to_filtering_YEAR:.3g} y

    Post-Fit:
        RMS residuals       : {RMS_res_est_ARCSEC:.3f} arcsec
        RMS normalised res. : {RMS_normalised_res_est:.3f}
    Other:
        Prefilter RMS res.  : {RMS_prefiltered_residuals_ARCSEC:.3f} arcsec
        Postfilter RMS res. : {RMS_postfiltered_residuals_ARCSEC:.3f} arcsec
        Pre-fit RMS res.    : {isu.rms(prefit_residuals_ARCSEC):.3f} arcsec
        N (%) obs. filtered : {n_obs_filtered_out} ({percent_obs_filtered_out:.2f}%)

    RMS formal error (pos.) : {RMS_formal_error_pos_mag_KM:.4g} km
    RMS true error (pos.)   : {RMS_true_error_pos_mag_KM:.4g} km
    FE_RMS/TE_RMS (pos.)    : {RMS_formal_error_pos_mag_KM / RMS_true_error_pos_mag_KM:.4g}

    RMS formal error (vel.) : {RMS_formal_error_vel_mag_MM:.4g} mm/s
    RMS true error (vel.)   : {RMS_true_error_vel_mag_MM:.4g} mm/s
    FE_RMS/TE_RMS (vel.)    : {RMS_formal_error_vel_mag_MM / RMS_true_error_vel_mag_MM:.4g} \n""")



    ######################## 9. Generate plots #################################
    #region 9. Plots

    # T_START_plots = tm.perf_counter()
    # print('Generating plots...')
    # est_plots.generate_estimation_plots(
    #     est_w_mpc_result_path = est_w_mpc_plot_dir + timestamp_dir + body_dir
    # )
    # T_END_plots = tm.perf_counter()
    # print_default(f"Done ({T_END_plots - T_START_plots:.3f}s) \n")



    ############################ X. Save outputs ###############################
    #region X.Outputs

    if abs(A2_est/fe_A2) < 3.0:
        flag = 'rej'
    else:
        flag = '1'

    if 'save' in plot_behaviour_list:
        output_data = {
            'body': body['longname'],
            'flag': flag,
            # # OBSERVATIONS
            'obs_start': str(obs_range_start),
            'obs_end': str(obs_range_end),
            'spacecraft_used': str(spacecraft_to_use),
            'observatories_excluded': str(observatories_to_exclude_MPC_codes),
            # # DYNAMIC MODEL
            'dynamic_model': dynamic_model_def,
            'n_LMBAs': n_LMBAs,
            # # INTEGRATOR
            'integrator': integrator,
            'step_size_DAY': step_size_DAY,
            'prop_start': str([propagation_start_DateTime.year, propagation_start_DateTime.month, propagation_start_DateTime.day]),
            'prop_end': str([propagation_end_DateTime.year, propagation_end_DateTime.month, propagation_end_DateTime.day]),
            'prop_duration_YEAR': propagation_duration_YEAR,
            # # ESTIMATION
            'n_iterations': number_of_pod_iterations,
            'use_SC_data': use_SC_data,
            'use_catalog_cor': use_catalog_cor,
            'use_obs_weights': use_obs_weights,
            'residual_filter_cutoff_ARCSEC': residual_filter_cutoff_ARCSEC,
            # # FRAMES
            'central_body': central_body,
            'global_frame_origin': global_frame_origin,
            'global_frame_orientation': global_frame_orientation,
            # # OBSERVATIONS USED
            'n_obs_available': n_obs_available,
            'n_ground_obs_dropped': n_ground_obs_dropped,
            'n_space_obs_dropped': n_space_obs_dropped,
            'n_ground_obs_used': n_ground_obs_used,
            'n_space_obs_used': n_space_obs_used,
            'n_total_obs_used': n_total_obs_used,
            'n_obs_filtered_out': n_obs_filtered_out,
            # # LITERATURE A2
            'yark_source': yark_source,
            'A2_lit': A2_target_body,
            'σ_A2_lit': A2_target_body_unc,
            'SNR_lit': Fenucci_body_data['SNR'].item(),
            'RMS_norm_res_lit': Fenucci_body_data['RMS'].item(),
            'obs_arc_lit_YEAR': Fenucci_body_data['deltat'].item(),
            'n_opt_obs_lit': Fenucci_body_data['NOptObs'].item(),
            'n_opt_obs_rej_lit': Fenucci_body_data['NRejOpt'].item(),
            'n_rad_obs_lit': Fenucci_body_data['NRadObs'].item(),
            'n_rad_obs_rej_lit': Fenucci_body_data['NRejRad'].item(),
            'n_opt_old_lit': Fenucci_body_data['NOptOld'].item(),
            # # ESTIMATED A2
            'best_iteration': int(estimation_output.best_iteration),
            'exc_during_inv': estimation_output.exception_during_inversion,
            'exc_during_prop': estimation_output.exception_during_propagation,
            'A2_est': A2_est,
            'σ_A2_est': fe_A2,
            'SNR_A2_est': abs(A2_est/fe_A2),
            'RMS_res_est': RMS_res_est_ARCSEC,
            'RMS_norm_res_est': RMS_normalised_res_est,
            'mean_norm_res_est': mean_normalised_res_est,
            'mean_abs_norm_res_est': mean_abs_normalised_res_est,
            '%_mag_diff': percent_change_from_ref,
            'k1': k1, # k1 = | A2_lit - A2_est | / σ_A2_est
            'k2': k2, # k2 = | A2_lit - A2_est | / σ_A2_lit
            'obs_arc_est_YEAR': obs_arc_YEAR,
            'filtered_obs_arc_est_YEAR': filtered_obs_arc_YEAR,
            '∆obs_arc_due_to_filtering_YEAR': change_in_obs_arc_due_to_filtering_YEAR,
            # # OTHER
            'formal_errs_rounded': str(formal_errs_rounded),
            'RMS_formal_error_pos_mag_KM': RMS_formal_error_pos_mag_KM,
            'RMS_formal_error_vel_mag_MM': RMS_formal_error_vel_mag_MM,
            'RMS_true_error_pos_mag_KM': RMS_true_error_pos_mag_KM,
            'RMS_true_error_vel_mag_MM': RMS_true_error_vel_mag_MM,
            'initial_state_change_KM': initial_position_difference_mag/1000,
            'prefilter_RMS_res_ARCSEC': RMS_prefiltered_residuals_ARCSEC,
            'postfilter_RMS_res_ARCSEC': RMS_postfiltered_residuals_ARCSEC,
            'prefit_RMS_res_ARCSEC': isu.rms(prefit_residuals_ARCSEC),
            'runtime': runtime_estimation,
            # # PLOTS
            'run_base_dir': est_w_mpc_plot_dir,
            'body_dir': body_dir,
            # # BODY DATA
            'designation_number': body['designation_number'],
            'common_name': body['common_name'],
            'provisional_designation': body['provisional_designation'],
            'SPKID': body['SPKID'],
            'NEO_flag': body['NEO_flag'],
            'PHA_flag': body['PHA_flag'],
            'orbitperiod_DAY': body['orbitperiod_DAY'],
            'semimajoraxis_AU': body['semimajoraxis_AU'],
            'eccentricity': body['eccentricity'],
            'diameter_KM': body['diameter_KM'],
            'diameter_unc_KM': body['diameter_unc_KM'],
            'diameter_ref': body['diameter_ref'],
            # # JPL SBDB ORBIT FIT DATA
            'RMS_norm_res_JPL': body['RMS_norm_res'],
            'n_obs_used_JPL': body['n_obs_used'],
            'n_dop_obs_used_JPL': body['n_dop_obs_used'],
            'n_del_obs_used_JPL': body['n_del_obs_used'],
            'obs_arc_JPL_YEAR': body['obs_arc_DAY'] / 365.2425,
            'A2_JPL': body['A2_AU_per_DAYsq'] * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2),
            'A2_unc_JPL': body['A2_unc_AU_per_DAYsq'] * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2),
        }
        # ADD TO OUTPUTS:

        return output_data, flag

#%%%%%%# C. Inputs
######## C. Inputs
#region C. Inputs

body_list = Fenucci_etal2024_tabB1.Asteroid.to_list()
start_idx = 0
end_idx = len(body_list)
# end_idx = 5
body_list = body_list[start_idx:end_idx]

body_list = ['152563', '85953']

large_run_base_dir = "yarkovsky-est-w-mpc/_large-runs/"
plot_behaviour = 'save'
# plot_behaviour = 'both'

observatories_to_exclude_dict = {}
bodies_to_skip = [
    # Will not work with my methods:
    '101955', # Bennu (SPK file not available)
]

# flags:
# 1     : run was successful
# rej.  : SNR_A2 < 3.0
# div   : potential divergence in estimation
# unnum : skipped because target body had no permanent designation number
# num   : skipped because target body had permanent designation number
# error : some error occurred
# skip  : target body is on skip list


#%%%%%%# D. Execute
######## D. Execute
#region D. Execute
import importlib
isu = importlib.reload(isu)

if __name__ == "__main__":
    T_START_run = tm.perf_counter()
    # Timestamp string to uniquely identify this run
    timestamp_str = dttm.datetime.now().strftime("%y%m%d-%H%M%S") # e.g. formats 28 November 2025, 1:57:01pm as '251128-135701'
    timestamp_dir = f"{timestamp_str}/"

    # Directory for saving outputs
    run_output_data_save_path = isu.output_data_dir + large_run_base_dir + timestamp_dir
    if not os.path.exists(run_output_data_save_path):
        os.makedirs(run_output_data_save_path)

    # Perform estimation for each body in body_list
    error_dict = {}
    skipped_bodies = []
    for body_idx, body_to_propagate in enumerate(body_list):
        body_timestamp = dttm.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        T_START_body = tm.perf_counter()
        print(f"\n\n\n### ESTIMATION FOR: {body_to_propagate} | {body_idx+1}/{len(body_list)} | idx: {start_idx+body_idx} | {body_timestamp} ###")

        if body_to_propagate in observatories_to_exclude_dict.keys():
            observatories_to_exclude_MPC_codes = observatories_to_exclude_dict[body_to_propagate]
        else:
            observatories_to_exclude_MPC_codes = []

        try:
            output_data, flag = estimation_with_MPC(
                ## Target body
                body_to_propagate = body_to_propagate,
                ## Observations
                obs_range_start = [1900, 1, 1],
                obs_range_end = [2026, 7, 1],
                spacecraft_to_use = [],
                observatories_to_exclude_MPC_codes = observatories_to_exclude_MPC_codes,
                ## Dynamic model
                dynamic_model_def = "LB+LMBA+AB+Yark",
                n_LMBAs = 16,
                n_MBAs_CP_search = 388,
                close_pass_threshold_AU = 0.5,
                close_pass_influence_threshold = 0.01,
                ## Integrator
                integrator = None,
                step_size_DAY = None,
                ## Estimation
                number_of_pod_iterations = 6, #!!! change back to 4
                use_catalog_cor = True,
                use_obs_weights = True,
                residual_filter_cutoff_ARCSEC = 3.0,
                ## Frames
                central_body = "Sun",
                global_frame_origin = "SSB",
                global_frame_orientation = "J2000",
                ## Plotting & Printing
                verbose = False,
                quiet = False,
                colormode = 0,
                plot_behaviour = plot_behaviour,
                est_w_mpc_plot_dir = large_run_base_dir,
                plots_in_RSW = True,
                comparison_reference = 'spk',
                ## Inputs from file
                # input_JSON_file =
            )
        except Exception as ex:
            print(f"WARNING: An error occurred: {type(ex).__name__} - {ex}")
            body = isu.query_SBDB(body_to_propagate)
            flag = "error"
            output_data = {
                'body': body['longname'],
                'flag': flag,
                # # BODY DATA
                'designation_number': body['designation_number'],
                'common_name': body['common_name'],
                'provisional_designation': body['provisional_designation'],
                'SPKID': body['SPKID'],
                'NEO_flag': body['NEO_flag'],
                'PHA_flag': body['PHA_flag'],
                'orbitperiod_DAY': body['orbitperiod_DAY'],
                'semimajoraxis_AU': body['semimajoraxis_AU'],
                'eccentricity': body['eccentricity'],
                'diameter_KM': body['diameter_KM'],
                'diameter_unc_KM': body['diameter_unc_KM'],
                'diameter_ref': body['diameter_ref'],
                'earth_MOID_AU': body['earth_MOID_AU'],
            }

            # Load existing error dict
            if error_dict:
                with open(run_output_data_save_path + "_errors.json", 'r') as json_fp:
                    error_dict = json.load(json_fp)[0]
            # Add current body name & error info
            error_dict[body_to_propagate] = [type(ex).__name__, str(ex)]
            # Save dict as JSON file
            with open(run_output_data_save_path + "_errors.json", "w") as f:
                json.dump([error_dict], f, indent=4)

        # Convert function output to dataframe
        body_output_df = pd.DataFrame(data=output_data, index=[body_idx])

        if body_idx != 0:
            # Read existing output file
            outputs_df = pd.read_csv(
                run_output_data_save_path + "_outputs.csv",
                index_col = 0
            )
            # Add new data
            outputs_df = pd.concat([outputs_df, body_output_df])
        else:
            outputs_df = body_output_df

        # Save to file
        outputs_df.to_csv(run_output_data_save_path + "_outputs.csv")

        body_timestamp = dttm.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        T_END_body = tm.perf_counter()
        print(f"### OUTCOME FOR {body_to_propagate}: {flag} | {body_idx+1}/{len(body_list)} | idx: {start_idx+body_idx} | {body_timestamp} | {(T_END_body - T_START_body)/60:.2f} min ###")
        cumulative_time = T_END_body - T_START_run
        average_time_per_body = cumulative_time / (body_idx+1)
        bodies_remaining = len(body_list) - (body_idx + 1)
        predicted_time_remaining = bodies_remaining * average_time_per_body
        print(f"### Cumulative time: {cumulative_time/60:.1f} min")
        print(f"### Predicted time remaining: {predicted_time_remaining/60:.1f} min")


    print(f"\n\n\nLARGE RUN COMPLETE | {len(body_list)} bodies | Timestamp: {timestamp_str} | {(T_END_body - T_START_run)/60:.3f} mins")

