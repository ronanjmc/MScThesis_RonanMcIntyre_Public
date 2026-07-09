""" acceleration_scenario_comparison.py:
This script propagates an orbit under a set of different acceleration scenarios,
and plots the results. Several different sets of acceleration scenarios are
defined (see section 5.2 for details). Estimation (which uses JPL Horizons
ephemeris 3D positional data as "observations" can be otionally included.
"""
""" Descriptions of Plots:
- P1. Position difference w.r.t. JPL Horizons: 
    For the baseline scenario and all other scenarios, plots the magnitude of
    the position difference between the target body's propagated state and the
    target body's state as taken from the JPL Horizons ephemeris over the full
    propagation period. 
- P2. Position difference w.r.t. baseline scenario:
    For all non-baseline scenarios, plots the magnitude of the position
    difference between the target body's baseline and non-baseline propagated
    states over the full propagation period. Skipped for scenario comparison Set
    0 (baseline only) 
- P3. Error difference (scenarios vs. baseline)
    "Error" is approximated as the difference between the target body's
    propagated state and the target body's state as taken from the JPL Horizons
    ephemeris, i.e. the ephemeris is taken to be the "true" value. This error is
    calculated for all epochs of the baseline scenario and all other scenarios.
    The magnitude of the difference between the positional error in the baseline
    scenario and the positional error in each of the non-baseline scenarios is
    what is plotted. Skipped for scenario comparison Set 0 (baseline only) 
- P4. Observation residuals
    Only plotted for Set 0 (baseline only) with use_estimation=True. Plots the
    observation residuals, i.e. the differences between the propgated state and
    the "observations" (3D positions from JPL Horizons ephemeris). Plots the
    magnitude of the poisition difference vector (same as in Plot P1.) but also
    plots the components of this vector in inertial, RSW, or TNW coordinates.

When plot_behaviour = 'show', the plots will display in the terminal where the
code is being run. The colormode input determines whether these are shown in
lightmode or darkmode.
    When plot_behaviour = 'save', lightmode versions of all relevant plots will
be saved to the folder:
    baseline-definition > plots >
    acceleration-scenario-comparison[-with-estimation] > [body-folder]
where [body-folder] corresponds to the target body and is formatted as
[designation_number]-[common_name], e.g. 2062-Aten. Darkmode versions of all
relevant plots are also saved in the 'darkmode' folder inside the target body
folder. 
"""

#%% 0. Inputs
### 0. Inputs

body_to_propagate = "Aten"     # str : accepts designation number, common name, or provisional designation of asteroid
integrator = "RK6"              # str : fixed step integration method. "Euler" or "RK[X]" (Runge-Kutta method of order X, with X = [1,2,3,4,5,6,7,8,9,10,12,14])
step_size_DAY = 1.0               # float : duration of propagation timestep in days
propagation_initial_time = [1985, 1, 1] # list : epoch of beginning of propagation. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.
n_orbits_to_propagate = 13.5     # float : sets duration of propagation, in terms of number of complete orbits around the Sun for the target body
n_LMBAs = 8                      # number of MBAs whose gravitation is to be included in dynamics (takes the n_LMBAs most massive MBAs). Only applies for scenarios including LMBAs
use_estimation = True          # bool : choose False to perform a simple propagation or True to use JPL Horizons ephemeris positional data as "observations", and to estimate the initial state of the target body 
verbose = False                 # bool : choose True to include additional supplementary print statements
quiet = False                   # bool : choose True to suppress all print statements

# Acceleration scenario comparison
scenario_comparison_set = 1        # see comments in cell 5 for full explanation of comparison sets
baseline_components_label = "LB"    # define the baseline accelation scenario (start with "LB", add "+EIH", "+LMBA", "+CP", "+AB" as desired)
# baseline_components_label = "LB+LMBA+AB" # LB+EIH+LMBA+CP+AB
# baseline_components_label = "LB+LMBA+CP+AB" # LB+EIH+LMBA+CP+AB

# Close pass search (only relevant for scenarios including close passers)
n_MBAs_CP_search = 396 - n_LMBAs    # number of MBAs to include when searching for close passes (takes the n_MBAs_CP_search most massive MBAs after excluding the n_LMBAs most massive MBAs). Maximum: 396 - n_LMBAs (as of 11 Jun 2025)
close_pass_threshold_AU = 0.5   # distance threshold that defines what is considered a close pass [AU] 
close_pass_influence_threshold = 0.01 # close pass displacement threshold s [metres]; s = 0.5*a_max*t^2; a_max is max grav. acc. due to close pass; t is propagation duration. close passers with less influence than this are filtered out

# Yarkovsky inputs (only relevant for scenarios including Yarkovsky acceleration)
estimate_A2 = False                 # choose True to include Yarkovsky parameter A2 as a parameter to estimate

# Plotting
colormode = 0                # bool : 0 for lightmode, 1 for darkmode when showing plots (both colormodes are saved when plot_behaviour is 'save' anyway)
plot_behaviour = 'save'
plot_behaviour = 'show'
# plot_behaviour = 'both'
scenario_comparison_plot_dir = "acceleration-scenario-comparison-with-estimation/" if use_estimation else "acceleration-scenario-comparison/"


#%% 1. Imports 
### 1. Imports 

# Printing setup
if verbose and quiet:
    raise ValueError("Verbose mode and quiet mode have both been selected. Choose one or neither.")
# Function that prints args if quiet=False and does nothing if quiet=True
print_default = print if not quiet else lambda *a, **k: None
# Function that prints args if verbose=True and does nothing if verbose=False
print_verbose = print if verbose else lambda *a, **k: None

import time as tm
T_START_imports = tm.time()
print_default("Importing packages and loading SPICE kernels...")

# Directory business
import os
local_path = os.path.dirname(__file__)
os.chdir(local_path) # ensure that the cwd is the subfolder where this file is

# Tudat imports
from tudatpy.interface import spice
from tudatpy.dynamics import simulator, environment_setup, propagation_setup
from tudatpy.estimation import estimation_analysis
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation.observations_setup import observations_simulation_settings, observations_wrapper
from tudatpy.dynamics import parameters_setup
import tudatpy.constants as cnst
from tudatpy.astro import frame_conversion 
from tudatpy.astro.time_representation import DateTime
# JPL SBDB, Horizons interfaces
from tudatpy.data.horizons import HorizonsQuery

# Other Python imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import itertools
import datetime as dttm
import IS_utils as isu

# Load SPICE kernels, other data
spice.clear_kernels()
spice.load_standard_kernels()
# !!! automatically Yarkovsky data from Literature > Yarkovsky_data > .dat files
body_A2s_AU_per_DAYsq = {
    # A compilation of asteroid data (including A2 values) and their sources can
    # be found in my "Body data" Google sheets document. All values in this dict
    # are in units of [AU/day^2]
    
    # [num] [name]  # Source 1                  # Source 2
    # "body":       [(value1, uncertainty1),    (value2, uncertainty2), ... , ]
    
    # 2062 Aten     # Fenucci et al. (2024)     # Farnocchia et al. (2013)
    "Aten":         [(-14.06e-15, 1.69e-15),    (-15.41e-15, 2.45e-15)],
    
    # 1566 Icarus   # Fenucci et al. (2024)
    "Icarus":       [(-3.89e-15, 0.34e-15)],
    
    # 101955 Bennu  # Fenucci et al. (2024)     # Farnocchia et al. (2013)
    "Bennu":        [(-45.26e-15, 0.24e-15),    (-45.49e-15, 0.23e-15)],
    
    # 2340 Hathor   # Fenucci et al. (2024)
    "Hathor":       [(-29.99e-15, 0.53e-15)],
    
    # 99942 Apophis # Fenucci et al. (2024)
    "Apophis":      [(-28.89e-15, 0.24e-15)],
    
    # 152563 (1992 BF) # Fenucci et al. (2024)
    "152563":       [(-26.34e-15, 0.92e-15)],
    
    # 524522 Zoozve # Fenucci et al. (2024)
    "Zoozve":       [(-59.84e-15, 1.05e-15)],
    
    # 612197 (2000 WP19) # Fenucci et al. (2024) # this one has a positive A2 value
    "612197":       [(191.e-15, 8.84e-15)],
} 

T_END_imports = tm.time()
print_default(f"Done ({T_END_imports - T_START_imports:.3f}s) \n")

#%% 2. Input processing
### 2. Input processing

## Integrator selection
coefficient_set, order_to_use = isu.process_integrator_input(integrator, print_integrator_info=verbose)

## Get body data from JPL SBDB and print selected info
body = isu.query_SBDB(body_to_propagate, print_body_info=verbose)
# spice.load_kernel(isu.SPK_file_dir + body['filing_name_short'] + ".bsp" )

## Process baseline acceleration scenario input
LMBAs_needed = True if "LMBA" in baseline_components_label else False
CPs_needed = True if "CP" in baseline_components_label else False
ABs_needed = True if "AB" in baseline_components_label else False
# !!! Add option for Yarkovsky in BL? Yark_needed = True if "Yark" in baseline_components_label else False

# Set flags if they are needed for the chosen scenario comparison set
LMBAs_needed = True if scenario_comparison_set in [1,4] else LMBAs_needed
CPs_needed = True if scenario_comparison_set in [2,4] else CPs_needed
ABs_needed = True if scenario_comparison_set in [3,4] else ABs_needed

# For comparison sets that don't involve LMBAs, set n_LMBAs to 0 to skip some parts of this script
n_LMBAs = 0 if LMBAs_needed == False else n_LMBAs
    
# Get Yarkovsky parameter and convert to SI [m/s^2] (if available, otherwise
# choose default value). Chooses first A2 value in list of dict.
if scenario_comparison_set in [6]:
    try:
        A2_target_body_AU_per_DAYsq = body_A2s_AU_per_DAYsq[body_to_propagate][0][0]
        A2_target_body = A2_target_body_AU_per_DAYsq * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2) 
    except KeyError:
        print_default(f"No A2 value found in body_A2s_AU_per_DAYsq dictionary for '{body_to_propagate}'. Using default A2 value of -1e-13 [m/s^2]")
        A2_target_body = -1e-13 

#%% 3. Constants
### 3. Constants

T_START_setup = tm.time()
print_default("Beginning set up...")

# Propagation options
propagation_duration_SEC = n_orbits_to_propagate * (body['orbitperiod_DAY'] * cnst.JULIAN_DAY)
propagation_initial_datetime = DateTime(*propagation_initial_time)
propagation_initial_time_J2000 = propagation_initial_datetime.epoch() # in seconds since J2000
propagation_final_time_J2000 = propagation_initial_time_J2000 + propagation_duration_SEC

# Propagation settings
central_body = "Sun"
global_frame_origin = "SSB"
global_frame_orientation = "J2000"

#%% 4.1 Environment setup (define bodies)
### 4.1 Environment setup (define bodies)

## Define Set of Bodies
# List of bodies to be retrieved through SPICE
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

# Search for close passes between target body and other asteroids during
# propagation period
if CPs_needed:
    close_passers = isu.search_for_close_passes_with_Kepler_elements(
        target_body_name = body_to_propagate,
        n_MBAs_search = n_MBAs_CP_search,
        n_MBAs_already_included = n_LMBAs,
        search_period_initial_time = propagation_initial_datetime,
        n_orbits = n_orbits_to_propagate,
        close_pass_threshold_AU = close_pass_threshold_AU,
        timestep_DAY = step_size_DAY * 5.0,
        verbose = verbose,
    )
else:
    print_default("Skipping close pass search.")
    close_passers = {}

# Main-belt asteroids: masses come from SiMDA, ephemerides from JPL Horizons
target_int = int(body['designation_number'])
bodies_SiMDA_filtered = (
    pd.read_csv(isu.SiMDA_file)
    .assign(NUM=lambda x: np.int32(x.NUM)) # ! this produces a warning because the NUM value is empty for some rows 
    .query("DYN != 'COM' & DYN != 'TNO'") # filter out comets and trans-Neptunian objects (COM, TNO)
    .query("NUM != @target_int") # remove propagated body, if present
    .sort_values(by='MASS', ascending=False) # sort bodies by mass, descending order
    .loc[:, ["NUM", "DESIGNATION", "DIAM", "DYN", "MASS"]]
)
bodies_SiMDA_largest = bodies_SiMDA_filtered.head(n_LMBAs) # take n_LMBAs most massive
bodies_SiMDA_closepasses = bodies_SiMDA_filtered[bodies_SiMDA_filtered['NUM'].isin(close_passers)] # take any close passers

# Filter close passers that will have minimal influence on target body
if CPs_needed:
    for idx, cp in bodies_SiMDA_closepasses.iterrows():
        max_acc_due_to_cp = (cp['MASS'] * cnst.GRAVITATIONAL_CONSTANT) / ((close_passers[cp['NUM']] * cnst.ASTRONOMICAL_UNIT)**2)
        displacement_due_to_cp = 0.5 * max_acc_due_to_cp * (propagation_duration_SEC**2) # this is a max/upper bound because it assumes the maximum acceleration over the whole propagation period
        # print_verbose(f"{cp['NUM']:<{8}} {max_acc_due_to_cp*(10**15):.5} {'fm/s2':<{8}} {displacement_due_to_cp:.5} m")
        if displacement_due_to_cp < close_pass_influence_threshold:
            bodies_SiMDA_closepasses = bodies_SiMDA_closepasses.drop(idx)
    bodies_SiMDA_closepasses = bodies_SiMDA_closepasses.head(30) # Limit to 30 influential close passers (max amount that fits on a plot)
    print_default(f"Of {len(close_passers)} close passers found, adding {len(bodies_SiMDA_closepasses)} to dynamics.")

bodies_SiMDA_included = pd.concat([bodies_SiMDA_largest, bodies_SiMDA_closepasses]).drop_duplicates().reset_index(drop=True)
largest_MBAs = bodies_SiMDA_largest.NUM.to_list()
closepass_MBAs = bodies_SiMDA_closepasses.NUM.to_list()
all_included_MBAs = bodies_SiMDA_included.NUM.to_list()
all_included_MBAs_masses = bodies_SiMDA_included.MASS.to_list()

# Get SBDB body data for all SiMDA bodies (nice to have common name, etc. for plotting)
included_MBAs_data_dict = {}
if len(all_included_MBAs) > 0:
    for designation_number in all_included_MBAs:
        body_data = isu.query_SBDB(str(designation_number))
        included_MBAs_data_dict[designation_number] = body_data

print_default(f"\nNumber of additional bodies from SiMDA: {len(bodies_SiMDA_included)} ({len(bodies_SiMDA_largest)} largest + {len(bodies_SiMDA_closepasses)} close passers) \n")


# Additional bodies: 
if ABs_needed:
    bodies_additional_JPL = [("999", "Pluto")] # format for each body is (JPL Horizons id, common name)
    bodies_additional_masses_KG = [1.3025e22] # !!! source?
else:
    bodies_additional_JPL = []
    bodies_additional_masses_KG = []

#%% 4.2 Environment setup (get JPL Horizons data)
### 4.2 Environment setup (get JPL Horizons data)

## Retrieve Necessary Body Ephemerides from JPL Horizons
T_START_JPL_queries = tm.time()   
print_default("Retrieving ephemerides from JPL Horizons...")         
timestep_horizons_DAY = step_size_DAY * 5
interpolator_buffer_DAY = (step_size_DAY + timestep_horizons_DAY) * 2 # Extra buffer needed to ensure interpolator doesn't request epoch outside specified range
time_buffer_DAY = 2 * 31 # to avoid interpolation errors (I think)

JPLHQ_location = central_body
# Ephemerides for SiMDA asteroids (largest + close passers)
print_verbose("...for SiMDA bodies (LMBAs + CPs)...")
MBA_ephemerides = {}
for code in all_included_MBAs:
    query = HorizonsQuery(
        query_id = f"{code};",
        location = f"@{JPLHQ_location}",
        epoch_start = propagation_initial_time_J2000 - (time_buffer_DAY + interpolator_buffer_DAY)*cnst.JULIAN_DAY, 
        epoch_end = propagation_final_time_J2000 + (time_buffer_DAY + interpolator_buffer_DAY)*cnst.JULIAN_DAY,
        epoch_step = f"{int(timestep_horizons_DAY)}d",
        extended_query = True,
    )
    MBA_ephemerides[code] = query.create_ephemeris_tabulated(
        frame_origin = JPLHQ_location,
        frame_orientation = global_frame_orientation,
    )

# Ephemeris for additional bodies
print_verbose("...for additional bodies...")
additional_ephemerides = {}
for code in bodies_additional_JPL:
    query = HorizonsQuery(
        query_id = f"{code[0]}",
        location = f"@{JPLHQ_location}",
        epoch_start = propagation_initial_time_J2000 - (time_buffer_DAY + interpolator_buffer_DAY)*cnst.JULIAN_DAY, 
        epoch_end = propagation_final_time_J2000 + (time_buffer_DAY + interpolator_buffer_DAY)*cnst.JULIAN_DAY,
        epoch_step = f"{int(timestep_horizons_DAY)}d",
        extended_query = True,
    )
    additional_ephemerides[code[1]] = query.create_ephemeris_tabulated(
        frame_origin = JPLHQ_location,
        frame_orientation = global_frame_orientation,
    )
bodies_additional = list(additional_ephemerides.keys())

# Retrieve target body initial state from JPL Horizons
print_verbose(f"...for target body {body['longname']}...")
target_query = HorizonsQuery(
    query_id = f"{body['designation_number']};",
    location = f"@{central_body}",
    epoch_list = [propagation_initial_time_J2000]
)
initial_state = target_query.cartesian(frame_orientation = global_frame_orientation)[0][1:]

# Retrieve target body entire ephemeris from JPL Horizons (this is used for estimation and/or when plotting)
target_ephemeris_query = HorizonsQuery(
    query_id = f"{body['designation_number']};",
    location = f"@{JPLHQ_location}",
    epoch_start = propagation_initial_time_J2000, # - (time_buffer_d + interpolator_buffer_d)*cnst.JULIAN_DAY, 
    epoch_end = propagation_final_time_J2000 + step_size_DAY*cnst.JULIAN_DAY, # + (time_buffer_d + interpolator_buffer_d)*cnst.JULIAN_DAY,
    epoch_step = f"{int(step_size_DAY)}d", # smaller time step for target body
    extended_query = True,
)
target_body_ephemeris = target_ephemeris_query.create_ephemeris_tabulated(
    frame_origin = JPLHQ_location,
    frame_orientation = global_frame_orientation,
)

T_END_JPL_queries = tm.time()
print_default(f"JPL Horizons queries complete ({T_END_JPL_queries - T_START_JPL_queries:.3f}s) \n")


#%% 4.3 Environment setup (create system of bodies)
### 4.3 Environment setup (create system of bodies)

# Add SPICE bodies
body_settings = environment_setup.get_default_body_settings(
    bodies_SPICE, global_frame_origin, global_frame_orientation
)
# Add SiMDA asteroids (largest + close passers) to body settings (ephemerides and gravity fields)
for code, mass in zip(all_included_MBAs, all_included_MBAs_masses):
    body_settings.add_empty_settings(str(code))
    body_settings.get(str(code)).ephemeris_settings = MBA_ephemerides[code]
    body_settings.get(str(code)).gravity_field_settings = (
        environment_setup.gravity_field.central(mass * cnst.GRAVITATIONAL_CONSTANT)
    )
# Add additional bodies to body settings (ephemerides and gravity fields)
for code, mass in zip(bodies_additional_JPL, bodies_additional_masses_KG):
    body_settings.add_empty_settings(str(code[1]))
    body_settings.get((code[1])).ephemeris_settings = additional_ephemerides[code[1]]
    body_settings.get((code[1])).gravity_field_settings = (
        environment_setup.gravity_field.central(mass * cnst.GRAVITATIONAL_CONSTANT)
    )
# Add target body, define gravity field but set mass to zero (needed for EIH)
body_settings.add_empty_settings(body['designation_number'])
body_settings.get(body['designation_number']).gravity_field_settings = ( 
        environment_setup.gravity_field.central(1e6 * cnst.GRAVITATIONAL_CONSTANT)
)
if use_estimation:
    # Apply tabulated ephemeris settings for target body 
    body_settings.get(body['designation_number']).ephemeris_settings = target_body_ephemeris

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
bodies_to_propagate = [body['designation_number']]
central_bodies = [central_body] 

#%% 5.1 Acceleration settings
### 5.1 Acceleration settings

# Large body accelerations (LB) (Sun + 8 planets + Moon)
accelerations_large_bodies = {
    "Sun": [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
        # propagation_setup.acceleration.einstein_infeld_hofmann(),
    ],
    "Mercury": [propagation_setup.acceleration.point_mass_gravity()],
    "Venus": [propagation_setup.acceleration.point_mass_gravity()],
    "Earth": [propagation_setup.acceleration.point_mass_gravity()],
    "Moon": [propagation_setup.acceleration.point_mass_gravity()],
    "Mars": [propagation_setup.acceleration.point_mass_gravity()],
    "Jupiter": [propagation_setup.acceleration.point_mass_gravity()],
    "Saturn": [propagation_setup.acceleration.point_mass_gravity()],
    "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
    "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
}
# Large body accelerations with Einstein-Infeld-Hoffman (LB+EIH) (Sun + 8 planets + Moon + EIH)
accelerations_large_bodies_EIH = {
    "Sun": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Mercury": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Venus": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Earth": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Moon": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Mars": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Jupiter": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Saturn": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Uranus": [propagation_setup.acceleration.einstein_infeld_hofmann()],
    "Neptune": [propagation_setup.acceleration.einstein_infeld_hofmann()],
}
# Largest SiMDA body accelerations
accelerations_LMBAs = {
    str(code): [propagation_setup.acceleration.point_mass_gravity()] for code in largest_MBAs
}
# Close passers SiMDA body accelerations
accelerations_CPs = {
    str(code): [propagation_setup.acceleration.point_mass_gravity()] for code in closepass_MBAs
}
# Additional body accelerations
accelerations_ABs = {
    str(name): [propagation_setup.acceleration.point_mass_gravity()] for name in bodies_additional
}
# Other accelerations
accelerations_other = {
    # "Sun": [],
}
# Yarkovsky acclerations are defined in cell 5.2

#%% 5.2 Acceleration scenarios 
### 5.2 Acceleration scenarios 
""" DEFINITIONS OF SCENARIO SETS + KEY FOR ABBREVIATIONS
Bodies + Relativistic effects in Literature:
Micheli et al. 2018: Sun + 8 planets + Moon, Pluto, 16 largest MBAs. Relativistic effects (unspecified) 
Fenucci et al. 2024: Same bodies as above. EIH for Sun + 8 planets + Moon
Chesley et al. 2014: Sun + 8 planets + Moon, Pluto, 16 largest MBAs + 9 selected MBAs. 
                     Relativistic effects Sun + 8 planets + Moon (presumably EIH).
                     Earth oblateness term included when Bennu is within 1 AU.

## ACCELERATION SCENARIOS (baseline +/- other accelerations):
Baseline: LB
Set 0: Baseline only
Set 1: Largest MBAs, one-by-one (1b1) (1+ n_LMBAs scenarios)
Set 2: Close passers 1b1 (1 + len(close_passers) scenarios)
Set 3: Additional bodies 1b1 (1 + len(bodies_additional) scenarios)
Set 4: Group combinations (8 scenarios)
  - LB 
  - LB + (SiMDA largest bodies)
  - LB                            + (close passers)
  - LB                                                + (additional bodies)
  - LB + (SiMDA largest bodies)   + (close passers)
  - LB + (SiMDA largest bodies)                       + (additional bodies)
  - LB                            + (close passers)   + (additional bodies)
  - LB + (SiMDA largest bodies)   + (close passers)   + (additional bodies)
Set 5: LB vs. LB+EIH 
Set 6: BL vs. BL+Yarkovsky (literature) (vs. BL+Yarkovsky (estimated))

## KEY
  - BL:     Baseline
  - LB:     Large nodies
  - LMBAs:  Largest MBAs
  - CP:     Close passers
  - th:     Threshold
  - AB:     Additional bodies
  - BCG:    Body group combinations
  - EIH:    Einstein-Infeld-Hoffmann
  - 1b1:    one-by-one 
  - ASC:    Acceleration scenario comparison (= Acc. scen. comparison)
  - Yark.:  Yarkovsky (effect/acceleration)
"""

# Define baseline scenario using baseline_components_label input
if "EIH" in baseline_components_label:
    accelerations_baseline = accelerations_large_bodies_EIH
else:
    accelerations_baseline = accelerations_large_bodies
if "LMBA" in baseline_components_label:
    accelerations_baseline = accelerations_baseline | accelerations_LMBAs
if "CP" in baseline_components_label:
    accelerations_baseline = accelerations_baseline | accelerations_CPs
if "AB" in baseline_components_label:
    accelerations_baseline = accelerations_baseline | accelerations_ABs
# Add option for Yarkovsky in BL?

baseline_label_abbrv = 'BL'                                 # BL is short for 'Baseline'
scenario_figname_prefix = f"Set{scenario_comparison_set}-"  # this is for the file names of the plots
plot_title = f"Acc. Scen. Comp. w/ Est.: " if use_estimation else f"Acc. Scen. Comparison: "
acceleration_scenario_list = [accelerations_baseline]
acceleration_scenario_labels = [f'Baseline: {baseline_components_label}']

# Set 0
if scenario_comparison_set == 0:
    plot_title_addition = "w/ Est." if use_estimation else ""
    plot_title = f"Baseline Scenario ({baseline_components_label}) {plot_title_addition}"
    scenario_figname_prefix += f"BL-only-{baseline_components_label}"

# Sets 1-3 (One-by-one body addition scenarios)
elif scenario_comparison_set in [1,2,3]:
    # Set 1
    if scenario_comparison_set == 1:
        body_list = largest_MBAs
        plot_title += f"Largest MBAs ({len(largest_MBAs)}) 1b1"
        scenario_figname_prefix += f"LMBA_n-{n_LMBAs}"

    # Set 2
    if scenario_comparison_set == 2:
        body_list = closepass_MBAs
        plot_title += f"Close Passers (<{close_pass_threshold_AU} AU) 1b1"
        scenario_figname_prefix += f"CP_th-{close_pass_threshold_AU}AU"

    # Set 3
    if scenario_comparison_set == 3:
        body_list = bodies_additional
        plot_title += f"Additional Bodies ({len(bodies_additional)}) 1b1"
        AB_str = str(bodies_additional).replace('[','').replace(']','').replace(', ','-').replace("'","")
        scenario_figname_prefix += f"AB_{AB_str}"

    # Same for Sets 1-3        
    for code in body_list:
        scenario_name = f"{baseline_label_abbrv} + "
        accelerations_scenario_specific = {}
        # Add point mass gravity for current MBA
        accelerations_scenario_specific[str(code)] = [propagation_setup.acceleration.point_mass_gravity()]
        # Scenario label (for printing, plots, etc.)
        if scenario_comparison_set == 1:
            scenario_name += f"{included_MBAs_data_dict[code]['labelname']}" 
        if scenario_comparison_set == 2:
            close_pass_addition = f" ({close_passers[code]:.3f} AU)"
            scenario_name += f"{included_MBAs_data_dict[code]['labelname']}{close_pass_addition}" 
        if scenario_comparison_set == 3:
            scenario_name += f"{code}" 
        acceleration_scenario_labels.append(scenario_name)
        # Append to acceleration scenario list
        acceleration_scenario_list.append(accelerations_baseline | accelerations_scenario_specific)

# Set 4
elif scenario_comparison_set == 4:
    plot_title += "Body Group Combinations"
    scenario_figname_prefix += f"BCG"

    body_groups = [bodies_additional, closepass_MBAs, largest_MBAs]
    body_group_labels = [
        f'AB ({len(bodies_additional)}x)', 
        f'CP ({len(closepass_MBAs)}x)', 
        f'LMBA ({len(largest_MBAs)}x)'
    ]
    # List of all binary combinations with length = number of body groups (excluding all-zero combination)
    combinations = list(itertools.product([0, 1], repeat=len(body_groups)))[1:]
    combinations = sorted(combinations, key=sum) # sort by number of included groups
    
    for combo in combinations:
        scenario_name = f"{baseline_label_abbrv}"
        accelerations_scenario_specific = {}
        if combo[2]:
            accelerations_scenario_specific = accelerations_scenario_specific | accelerations_LMBAs
        if combo[1]:
            accelerations_scenario_specific = accelerations_scenario_specific | accelerations_CPs
        if combo[0]:
            accelerations_scenario_specific = accelerations_scenario_specific | accelerations_ABs
        
        # Define label/name for this scenario
        scenario_name += f"{combo[2]*(' + ' + body_group_labels[2])}{combo[1]*(' + ' + body_group_labels[1])}{combo[0]*(' + ' + body_group_labels[0])}"
        acceleration_scenario_labels.append(scenario_name)
        # Append to acceleration scenario list
        acceleration_scenario_list.append(accelerations_baseline | accelerations_scenario_specific)

# Set 5
elif scenario_comparison_set == 5:
    plot_title += "EIH"
    scenario_figname_prefix += f"EIH"
    scenario_name = f"{baseline_label_abbrv} + EIH"
    accelerations_scenario_specific = accelerations_large_bodies_EIH
    acceleration_scenario_labels.append(scenario_name)
    acceleration_scenario_list.append(accelerations_scenario_specific)

# Set 6
elif scenario_comparison_set == 6:
    plot_title += "Yarkovsky"
    scenario_figname_prefix += f"Yark"
    
    # Yarkovsky acceleration on the target body
    scenario_name = f"{baseline_label_abbrv} + Yark. (lit.)"
    sun_accelerations_list = [propagation_setup.acceleration.yarkovsky(A2_target_body)] + accelerations_baseline["Sun"]
    accelerations_scenario_specific = {
        "Sun": sun_accelerations_list,
    }
    acceleration_scenario_labels.append(scenario_name)
    acceleration_scenario_list.append(accelerations_baseline | accelerations_scenario_specific)
    
    if use_estimation and estimate_A2:
        scenario_name = f"{baseline_label_abbrv} + Yark. (est.)"
        acceleration_scenario_labels.append(scenario_name)
        acceleration_scenario_list.append(accelerations_baseline | accelerations_scenario_specific)
    
# Note: the acceleration settings depend on the acceleration scenario.
# Therefore, the acceleration settings are defined anew for each iteration of
# the propagation/estimation loop

#%% 6. Propagation settings
### 6. Propagation settings

# Integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
    initial_time = propagation_initial_time_J2000,
    initial_time_step = step_size_DAY * cnst.JULIAN_DAY,
    coefficient_set = coefficient_set,
    order_to_use = order_to_use,
)

# Termination settings    
termination_settings = propagation_setup.propagator.time_termination(propagation_final_time_J2000)

# Dependent variables to save
dependent_variables_to_save = [
    propagation_setup.dependent_variable.relative_position("Jupiter", "Sun"),
    # propagation_setup.dependent_variable.relative_velocity("Jupiter", "Sun"),
    # propagation_setup.dependent_variable.relative_position("Saturn", "Sun"),
    # propagation_setup.dependent_variable.relative_velocity("Saturn", "Sun"),
    # propagation_setup.dependent_variable.keplerian_state(body['designation_number'], "Sun"),
    # propagation_setup.dependent_variable.single_acceleration(
    #     propagation_setup.acceleration.yarkovsky_acceleration_type, str(body['designation_number']), central_body, 
    # ),
]

# Note: the propagator settings depend on the acceleration scenario. Therefore,
# the propagator settings are defined anew for each iteration of the
# propagation/estimation loop

#%% 7. Estimation setup
### 7. Estimation setup

if not use_estimation:
    print_default("Skipping estimation setup. \n")
else:
    # Create link ends
    link_ends_body = dict()
    link_ends_body[links.LinkEndType.observed_body] = links.body_origin_link_end_id(body['designation_number'])
    link_definition_body = links.LinkDefinition(link_ends_body)

    # Define observation model settings
    position_observation_settings = [model_settings.cartesian_position(link_definition_body)]

    # Define observation simulation settings
    # Epochs at which the ephemeris will be checked
    observation_times = np.arange(
        propagation_initial_time_J2000, 
        propagation_final_time_J2000, 
        5.0 * step_size_DAY * cnst.JULIAN_DAY
    )
    # Create the observation simulation settings
    observation_simulation_settings = [observations_simulation_settings.tabulated_simulation_settings(
        model_settings.position_observable_type,
        link_definition_body,
        observation_times,
        reference_link_end_type = links.LinkEndType.observed_body
    )]

    # Simulate state of target body based on ephemeris
    # Create observation simulators
    ephemeris_observation_simulators = observations_simulation_settings.create_observation_simulators(
        position_observation_settings, bodies)
    # Get ephemeris states as ObservationCollection
    ephemeris_body_states = observations_wrapper.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies
    )

T_END_setup = tm.time()
print_default(f"Setup complete ({T_END_setup - T_START_setup:.3f}s) \n")


#%% 8.1 Loop over scenarios and perform propagation / estimation
### 8.1 Loop over scenarios and perform propagation / estimation

T_START_all_scenarios = tm.time()
# Print info about this run 
scenario_print_insert = "ESTIMATION" if use_estimation else "PROPAGATION"
if not quiet:
    pre_loop_print = f"{scenario_print_insert} using Set {scenario_comparison_set} with inputs:"
    pre_loop_print += f"\n  n_LMBAs:                            {n_LMBAs}" if LMBAs_needed else ""
    pre_loop_print += f"\n  close_pass_threshold_AU:            {close_pass_threshold_AU} AU" if CPs_needed else ""
    pre_loop_print += f"\n  close_pass_influence_threshold:     {close_pass_influence_threshold} m" if CPs_needed else ""
    pre_loop_print += f"\n  AB bodies:                          {bodies_additional}" if ABs_needed else ""
    pre_loop_print += f"\n  Target body:                        {body['longname']}"
    pre_loop_print += f"\n  Integrator:                         {integrator}"
    pre_loop_print += f"\n  Time step size:                     {step_size_DAY} days"
    pre_loop_print += f"\n  Start time:                         {propagation_initial_time}"
    pre_loop_print += f"\n  Number of orbits:                   {n_orbits_to_propagate}"
    pre_loop_print += f"\n  Baseline acceleration scenario:     {baseline_components_label}"
    pre_loop_print += f"\n  Central body:                       {central_body}"
    pre_loop_print += f"\n  Global frame origin:                {global_frame_origin}"
    pre_loop_print += f"\n  Global frame orientation:           {global_frame_orientation}\n"
    # pre_loop_print += f": {}"
    print(pre_loop_print)
    
scenario_state_histories = []
dependent_variable_histories = []
scenario_residuals = []
scenario_initial_state_changes = []
function_evaluations = np.zeros(len(acceleration_scenario_list))
runtime_scenarios = np.zeros(len(acceleration_scenario_list))

scenario_print_insert = "ESTIMATION" if use_estimation else "PROPAGATION"
for i, acceleration_scenario in enumerate(acceleration_scenario_list):
    T_START_scenario = tm.time()
    print_default(f"=============== {scenario_print_insert} FOR SCENARIO {i+1} / {len(acceleration_scenario_list)}: ===============")
    
    # Create acceleration settings and models
    acceleration_settings = {bodies_to_propagate[0]: acceleration_scenario}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies
    )
    # Propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies = central_bodies,
        acceleration_models = acceleration_models,
        bodies_to_integrate = bodies_to_propagate,
        initial_states = initial_state ,
        initial_time = propagation_initial_time_J2000,
        integrator_settings = integrator_settings,
        termination_settings = termination_settings,
        output_variables = dependent_variables_to_save,
    )
    # if verbose:
    #     console_print_settings = propagator_settings.print_settings
    #     console_print_settings.print_propagation_clock_time = True
    #     console_print_settings.print_number_of_function_evaluations = True
    #     console_print_settings.print_state_indices = True if i == 0 else False
    #     console_print_settings.print_termination_reason = True if i == 0 else False
    #     # console_print_settings.results_print_frequency_in_steps = 20

    ## PROPAGATION
    if not use_estimation:
        dynamics_simulator = simulator.create_dynamics_simulator(
            bodies, 
            propagator_settings,
        )
        
        # Save data from current acceleration scenario
        state_history = dynamics_simulator.state_history
        scenario_state_histories.append(state_history)
        dependent_variable_history = dynamics_simulator.dependent_variable_history
        dependent_variable_histories.append(dependent_variable_history)
        n_function_evaluations = max(dynamics_simulator.cumulative_number_of_function_evaluations.values())
        function_evaluations[i] = n_function_evaluations
        
    ## ESTIMATION
    if use_estimation:
        # Define estimable parameters
        parameters_to_estimate_settings = parameters_setup.initial_states(propagator_settings, bodies)
        if estimate_A2 and i==2: # !!! this is kinda hacky
            parameters_to_estimate_settings.append(parameters_setup.yarkovsky_parameter(body['designation_number'], "Sun"))
        parameters_to_estimate = parameters_setup.create_parameter_set(parameters_to_estimate_settings, bodies, propagator_settings)
        original_parameter_vector = parameters_to_estimate.parameter_vector

        # Perform the estimation
        estimator = estimation_analysis.Estimator(
            bodies, 
            parameters_to_estimate,
            position_observation_settings, 
            propagator_settings
        )
        estimation_input = estimation_analysis.EstimationInput(ephemeris_body_states) # Create input object for the estimation
        estimation_input.define_estimation_settings(save_state_history_per_iteration=True) # Set methodological options
        print_verbose('Performing the estimation...')
        estimation_output = estimator.perform_estimation(estimation_input)
        print_verbose('Done. \n')

        # Print initial state change information
        initial_state_updated = parameters_to_estimate.parameter_vector
        print_default(f'Original initial state: {original_parameter_vector}')
        print_default(f'Updated initial state: {initial_state_updated} \n')

        initial_state_difference_vector = initial_state_updated - original_parameter_vector
        initial_position_difference_mag = np.linalg.norm(initial_state_difference_vector[0:3])
        print_default(f'Change in initial state: {round(initial_position_difference_mag,3):_} m')
        
        # Save data from current acceleration scenario
        simulator_object = estimation_output.simulation_results_per_iteration[-1]
        state_history = simulator_object.dynamics_results.state_history
        scenario_state_histories.append(state_history)
        scenario_residuals.append(isu.rms(estimation_output.final_residuals))
        scenario_initial_state_changes.append(initial_position_difference_mag)
    
    T_END_scenario = tm.time()
    runtime_scenario = T_END_scenario - T_START_scenario
    print_default(f"\nScenario {i+1} done ({runtime_scenario:.3f}s) \n")
    runtime_scenarios[i] = runtime_scenario

#%% 8.2 End of loop prints 
### 8.2 End of loop prints 

T_END_all_scenarios = tm.time()
print_default(f"ALL SCENARIOS DONE ({T_END_all_scenarios - T_START_all_scenarios:.3f}s) \n")
if use_estimation:
    print_default(f"Residuals: {scenario_residuals}")



#%% P1. Plot position difference w.r.t. JPL Horizons 
""" P1. Plot position difference w.r.t. JPL Horizons: 
For the baseline scenario and all other scenarios, plots magnitude of the
position difference between the target body's propagated state and the target
body's state as taken from the JPL Horizons ephemeris over the full propagation
period.
"""

# colormode = 0
# plot_behaviour = 'show'
# plot_behaviour = 'save'
# plot_behaviour = 'both'

timestamp_str = dttm.datetime.now().strftime("%y%m%d-%H%M%S") # e.g. formats 28 November 2025, 1:57:01pm as '251128-135701'
if plot_behaviour == 'save':
    colormodes = [0,1]
elif plot_behaviour == 'both':
    colormodes = [colormode,0,1]
else:
    colormodes = [colormode]

# The ephemeris of the target is wrt the global_frame_origin
target_ephemeris_tabulated = target_ephemeris_query.cartesian(frame_orientation = global_frame_orientation)
epochs = target_ephemeris_tabulated[:,0]
target_ephemeris_tabulated = target_ephemeris_tabulated[:,1:]

# # The propagated state vector elements are wrt CB, whilst the body states from
# # ephemerides are wrt GFO. If CB != GFO, a correction must be made before they
# # can be compared.
# if central_body != global_frame_origin:
#     # Below only works for central_body = "Sun" and global_frame_origin = "SSB"
#     if not (central_body=='Sun' and global_frame_origin=='SSB'):
#         raise ValueError("Plotting only implemented for central body==global frame origin OR for central_body=='Sun' and global_frame_origin=='SSB'.")
    
#     # If CB==Sun and GFO==SSB, the target body ephemeris must be adjusted to be
#     # wrt the Sun before we can compare it with the propagated state. 
#     # If JPL HorizonsQuery was used to get the target body ephemeris, the
#     # adjustment at each epoch is the vector from the JPL Horizons SSB to the
#     # Sun.
#     print_default("Getting central body ephemeris from JPL Horizons...")
#     T_START_central_body_query = tm.time()
#     central_body_ephemeris_query = HorizonsQuery(
#         query_id = f"{central_body}",
#         location = f"@{global_frame_origin}",
#         epoch_list = epochs,
#         extended_query = True,
#     )
#     T_END_central_body_query = tm.time()
#     print_default(f"Done ({T_END_central_body_query - T_START_central_body_query:.3f}s)")
#     print_default("Tabulating central body ephemeris...")
#     central_body_ephemeris_tabulated = central_body_ephemeris_query.cartesian(frame_orientation = global_frame_orientation)
#     central_body_ephemeris_tabulated = central_body_ephemeris_tabulated[:,1:]
#     T_END_central_tabulation = tm.time()
#     print_default(f"Done ({T_END_central_tabulation - T_END_central_body_query:.3f}s)")
    
#     # Convert target_ephemeris_tabulated to be wrt central_body instead
#     target_ephemeris_tabulated = target_ephemeris_tabulated - central_body_ephemeris_tabulated

# If the target body ephemeris was loaded via SPICE, the adjustment at each
# epoch is the vector from the SPICE SSB to the Sun. sdfsdfsdfs'lsdf;lkmdfd   dssdsdsdsdsdddd
# # Get position of central_body wrt global_frame_origin at all epochs
# central_body_ephemeris = bodies.get("Sun").ephemeris
# central_body_ephemeris_tabulated = np.zeros(np.shape(target_ephemeris_tabulated))
# for i, epoch in enumerate(epochs):
#     central_body_ephemeris_tabulated[i,:] = central_body_ephemeris.cartesian_state(epoch)



# Latex string for making (e.g.) 'Body: 2062 Aten' part of plot subtitles bold
title_tex_body = f"$\\bf{{Body: }}$"
for i, namepart in enumerate(body['labelname'].split(" ")):
    title_tex_body += " "*i + r"$\bf{{{val}}}$".format(val=namepart)
        
for k, colormode in enumerate(colormodes):
    # Plot setup
    fig, ax = plt.subplots(
        1, 1, 
        figsize = (9,7),
        facecolor = isu.fig_background[colormode]
    )
    ax.axhline(linewidth=1.5, color=isu.text_color[colormode]) # make zero-line more obvious
    epochs_plot = (np.array(list(scenario_state_histories[0].keys())) / cnst.JULIAN_YEAR) + 2000 # approximate years for plot ticks

    # Loop through scenarios, plot position difference of each scenario vs. JPL Horizons
    for i, scenario_label in enumerate(acceleration_scenario_labels):
        scenario_state_history = np.array(list(scenario_state_histories[i].values()))
        state_difference_wrt_JPL = scenario_state_history - target_ephemeris_tabulated
        position_difference_wrt_JPL_mag = np.linalg.norm(state_difference_wrt_JPL[:,0:3], axis=1)
        
        ax.plot(
            epochs_plot, 
            position_difference_wrt_JPL_mag / 1000, # in km 
            label = scenario_label,
            color = isu.cmap_custom10[colormode]((i-1)%10) if i != 0 else isu.text_color[colormode],
            linestyle = isu.linestyles[(i-1)//10] if i != 0 else isu.linestyles[0],
            linewidth = 2.5 if i != 0 else 4.0,
            alpha = 0.8,
        )

    # Shrink axis to fit legend at side
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.60, box.height * 0.97])
    ax.legend(
        loc = 'center left', 
        bbox_to_anchor = (1.01, 0.5),
        markerscale = 1.2,
        fontsize = isu.small_font_size,
        labelcolor = isu.text_color[colormode],
        facecolor = isu.legend_bg_color[colormode],
    )

    # Grid, tick params, label settings, some colormode stuff
    ax = isu.set_default_2d_ax_settings(ax, colormode)
    # ax.set_yscale('log')

    # Labels and (sub)titles
    ax.set_ylabel(f"Position difference [km]")
    ax.set_xlabel("Year")
    plt.suptitle(
        plot_title, 
        fontsize = isu.big_font_size, 
        weight = 'bold',
    )
    step_size_unit_str = isu.convert_step_size_DAY_to_str_with_unit(step_size_DAY)
    ax.set_title(
        f"Position difference w.r.t. JPL Horizons \n {title_tex_body},  {integrator},  Δt={step_size_unit_str},  $\\text{{n}}_{{\\text{{orbits}}}}$={n_orbits_to_propagate}",
        # f"Position difference w.r.t. JPL Horizons \n Body: {body['longname']},  Integrator: {integrator}, Δt = {step_size_unit_str}", 
        fontsize = isu.medium_font_size,
    )
    # fig.set_tight_layout(True)
    if (plot_behaviour == 'show') or ((plot_behaviour == 'both') and (k==0)):
        plt.show(block=False)
        plt.pause(0.001)
    elif (plot_behaviour == 'save') or ((plot_behaviour == 'both') and (k>0)):
        body_dir = f"{body['filing_name_long']}/"
        darkmode_dir = f"{'darkmode/'*colormode}"
        save_path = isu.plot_dir + scenario_comparison_plot_dir + body_dir + darkmode_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_name = f"{timestamp_str}__{scenario_figname_prefix}_wrt-JPL.pdf"
        plt.savefig(save_path + fig_name, format="pdf")
        plt.close()
        print(f"ASC{' w/ estimation'*use_estimation}: position difference w.r.t JPL vs. time {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")



#%% P2. Plot position difference w.r.t. baseline scenario
""" P2. Plot position difference w.r.t. baseline scenario
For all non-baseline scenarios, plots the magnitude of the position difference
between the target body's baseline and non-baseline propagated states over the
full propagation period. Skipped for scenario comparison Set 0 (baseline only)
"""

if scenario_comparison_set == 0:
    # This plot is skipped for Set 0, since it's just the baseline scenario 
    print_default("Skipping plot P2 (position difference w.r.t. baseline scenario)")

else:
    for k, colormode in enumerate(colormodes):
        # Plot setup
        fig, ax = plt.subplots(
            1, 1, 
            figsize = (10,7),
            facecolor = isu.fig_background[colormode]
        )
        epochs_plot = (np.array(list(scenario_state_histories[0].keys())) / cnst.JULIAN_YEAR) + 2000 # approximate years for plot ticks
        baseline_state_history = np.array(list(scenario_state_histories[0].values()))

        # Loop through scenarios, plot position difference of each scenario vs. baseline
        for j, scenario_label in reversed(list(enumerate(acceleration_scenario_labels[1:]))):
            i = j+1 # offset index by 1 because baseline scenario is first in list and is not plotted
            scenario_state_history = np.array(list(scenario_state_histories[i].values()))
            state_difference_wrt_baseline = scenario_state_history - baseline_state_history
            position_difference_wrt_baseline_mag = np.linalg.norm(state_difference_wrt_baseline[:,0:3], axis=1)
            
            ax.plot(
                epochs_plot, 
                position_difference_wrt_baseline_mag, # in m
                label = scenario_label,
                color = isu.cmap_custom10[colormode](j%10),
                linestyle = isu.linestyles[j//10],
                linewidth = 2.5,
                alpha = 0.7, 
            )

        # Shrink axis to fit legend at side
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.60, box.height * 0.97])
        ax.legend(
            loc = 'center left', 
            bbox_to_anchor = (1.01, 0.5),
            markerscale = 1.2,
            fontsize = isu.small_font_size,
            labelcolor = isu.text_color[colormode],
            facecolor = isu.legend_bg_color[colormode],
            reverse = True,
        )

        # Grid, tick params, label settings, some colormode stuff
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        ax.set_yscale('log')
        
        # Labels and (sub)titles
        ax.set_ylabel(f"Position difference [m]")
        ax.set_xlabel("Year")
        plt.suptitle(
            plot_title, 
            fontsize = isu.big_font_size, 
            weight='bold',
        )
        step_size_unit_str = isu.convert_step_size_DAY_to_str_with_unit(step_size_DAY)
        ax.set_title(
            f"Position difference w.r.t. BL ({acceleration_scenario_labels[0]}) \n {title_tex_body},  {integrator},  Δt={step_size_unit_str},  $\\text{{n}}_{{\\text{{orbits}}}}$={n_orbits_to_propagate}",
            fontsize = isu.medium_font_size,
        )
        # fig.set_tight_layout(True)

        if (plot_behaviour == 'show') or ((plot_behaviour == 'both') and (k==0)):
            plt.show(block=False)
            plt.pause(0.001)
        elif (plot_behaviour == 'save') or ((plot_behaviour == 'both') and (k>0)):
            body_dir = f"{body['filing_name_long']}/"
            darkmode_dir = f"{'darkmode/'*colormode}"
            save_path = isu.plot_dir + scenario_comparison_plot_dir + body_dir + darkmode_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig_name = f"{timestamp_str}__{scenario_figname_prefix}_wrt-BL.pdf"
            plt.savefig(save_path + fig_name, format="pdf")
            plt.close()
            print(f"ASC{' w/ estimation'*use_estimation}: position difference w.r.t baseline vs. time {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")


#%% P3. Plot error difference (scenarios vs. baseline)
""" P3. Plot error difference (scenarios vs. baseline)
"Error" is approximated as the difference between the target body's propagated
state and the target body's state as taken from the JPL Horizons ephemeris, i.e.
the ephemeris is taken to be the "true" value. This error is calculated for all
epochs of the baseline scenario and all other scenarios. The magnitude of the
difference between the positional error in the baseline scenario and the
positional error in each of the non-baseline scenarios is what is plotted.
Skipped for scenario comparison Set 0 (baseline only) 
"""

# if central_body != global_frame_origin:
#     raise ValueError("Plotting only implemented for central body = global frame origin.")


if scenario_comparison_set == 0:
    # This plot is skipped for Set 0, since it's just the baseline scenario 
    print_default("Skipping plot P3 (error difference compared to baseline)")
else:

    # Baseline scenario position difference wrt JPL Horizons (BL error)
    baseline_state_history = np.array(list(scenario_state_histories[0].values()))
    baseline_state_difference_wrt_JPL = baseline_state_history - target_ephemeris_tabulated
    baseline_position_difference_wrt_JPL_mag = np.linalg.norm(baseline_state_difference_wrt_JPL[:,0:3], axis=1)

    for k, colormode in enumerate(colormodes):
        # Plot setup
        fig, ax = plt.subplots(
            1, 1, 
            figsize = (9,7),
            facecolor = isu.fig_background[colormode]
        )
        ax.axhline(linewidth=1.5, color=isu.text_color[colormode]) # make zero-line more obvious

        # Loop through scenarios, plot error difference of each scenario vs. baseline
        # (error calculated as position difference wrt JPL Horizons)
        for j, scenario_label in reversed(list(enumerate(acceleration_scenario_labels[1:]))):
            i = j+1 # offset index by 1 because baseline scenario is first in list and is not plotted
            scenario_state_history = np.array(list(scenario_state_histories[i].values()))
            scenario_state_difference_wrt_JPL = scenario_state_history - target_ephemeris_tabulated
            scenario_position_difference_wrt_JPL_mag = np.linalg.norm(scenario_state_difference_wrt_JPL[:,0:3], axis=1)
            scenario_vs_baseline_error_difference = scenario_position_difference_wrt_JPL_mag - baseline_position_difference_wrt_JPL_mag
            
            ax.plot(
                epochs_plot, 
                scenario_vs_baseline_error_difference, # in m 
                label = scenario_label,
                color = isu.cmap_custom10[colormode](j%10),
                linestyle = isu.linestyles[j//10],
                linewidth = 2.5,
                alpha = 0.7, 
            )

        # Shrink axis to fit legend at side
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.60, box.height * 0.97])
        ax.legend(
            loc = 'center left', 
            bbox_to_anchor = (1.01, 0.5),
            markerscale = 1.2,
            fontsize = isu.small_font_size,
            labelcolor = isu.text_color[colormode],
            facecolor = isu.legend_bg_color[colormode],
            reverse = True,
        )

        # Grid, tick params, label settings, some colormode stuff
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        # ax.set_yscale('log')
        
        # Labels and (sub)titles
        ax.set_ylabel(f"Error difference [m]")
        ax.set_xlabel("Year")
        plt.suptitle(
            plot_title, 
            fontsize = isu.big_font_size, 
            weight='bold',
        )
        step_size_unit_str = isu.convert_step_size_DAY_to_str_with_unit(step_size_DAY)
        ax.set_title(
            f"Error difference compared to BL ({acceleration_scenario_labels[0]}) \n {title_tex_body},  {integrator},  Δt={step_size_unit_str},  $\\text{{n}}_{{\\text{{orbits}}}}$={n_orbits_to_propagate}",
            fontsize = isu.medium_font_size
        )
        # fig.set_tight_layout(True)

        if (plot_behaviour == 'show') or ((plot_behaviour == 'both') and (k==0)):
            plt.show(block=False)
            plt.pause(0.001)
        elif (plot_behaviour == 'save') or ((plot_behaviour == 'both') and (k>0)):
            darkmode_dir = f"{'darkmode/'*colormode}"
            save_path = isu.plot_dir + scenario_comparison_plot_dir + body_dir + darkmode_dir
            fig_name = f"{timestamp_str}__{scenario_figname_prefix}_err-diff.pdf"
            plt.savefig(save_path + fig_name, format="pdf")
            plt.close()
            print(f"ASC{' w/ estimation'*use_estimation}: error difference compared to baseline {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")


#%% P4. Plot observation residuals
""" P4. Plot observation residuals
Only plotted for Set 0 (baseline only) with use_estimation=True. Plots the
observation residuals, i.e. the differences between the propgated state and the
"observations" (3D positions from JPL Horizons ephemeris). Plots the magnitude
of the poisition difference vector (same as in Plot P1.) but also plots the
components of this vector in inertial, RSW, or TNW coordinates.
"""

reference_frame = 'inertial'
reference_frame = 'RSW'
# reference_frame = 'TNW'

if not (use_estimation and (scenario_comparison_set == 0)):
    # This plot is skipped if estimation is off and for all sets except Set 0
    print_default("Skipping plot P4 (observation residual components)")

else:    
    for k, colormode in enumerate(colormodes):
        # Loop through all epochs and record state difference between JPL H.
        # ephemeris (interpolated) and propagated state of target body
        epoch_list = list(state_history.keys())
        observation_residuals = np.reshape(estimation_output.final_residuals, (-1,3))

        # Change to different reference frame if input dictates
        position_difference = np.zeros((len(observation_times), 3))
        if reference_frame == 'RSW':
            print_default("Converting reference frame...")
            label_list = ['ΔR', 'ΔS', 'ΔW', '|Δr|']
            for i, epoch in enumerate(observation_times):
                rotation_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(state_history[epoch])
                position_difference[i,:] = rotation_matrix @ observation_residuals[i,0:3]
            print_default("Done.")
            
        elif reference_frame == 'TNW':
            print_default("Converting reference frame...")
            label_list = ['ΔT', 'ΔN', 'ΔW', '|Δr|']
            for i, epoch in enumerate(observation_times):
                rotation_matrix = frame_conversion.inertial_to_tnw_rotation_matrix(state_history[epoch])
                position_difference[i,:] = rotation_matrix @ observation_residuals[i,0:3]
            print_default("Done.")
            
        elif reference_frame == 'inertial':
            label_list = ['Δx', 'Δy', 'Δz', '|Δr|']
            position_difference = observation_residuals[i,0:3]

        # Plot
        fig, ax = plt.subplots(
            1, 1, 
            figsize = (9,7),
            facecolor = isu.fig_background[colormode]
        )
        ax.axhline(linewidth=1.5, color=isu.text_color[colormode]) # make zero-line more obvious    
        observation_times_plot = (observation_times / cnst.JULIAN_YEAR) + 2000 # approximate years for plot ticks

        j = 3
        ax.plot(
            observation_times_plot, 
            np.linalg.norm(position_difference, axis=1) / 1000, # in km 
            label = label_list[j],
            color = isu.text_color[colormode],
            linestyle = isu.linestyles[1],
            linewidth = 4,
            alpha = 0.8
        )
        for axis in range(3):
            ax.plot(
                observation_times_plot, 
                position_difference[:,axis] / 1000, # in km 
                label = label_list[axis],
                color = isu.cmap_custom10[colormode](axis%10),
                linestyle = isu.linestyles[axis//10],
                linewidth = 2.5,
                alpha = 0.8
            )
        ax.legend(
            loc = 'best', 
            # bbox_to_anchor = (1.01, 0.5),
            markerscale = 1.2,
            fontsize = isu.small_font_size,
            labelcolor = isu.text_color[colormode],
            facecolor = isu.legend_bg_color[colormode],
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.97])

        # Grid, tick params, label settings, some colormode stuff
        ax = isu.set_default_2d_ax_settings(ax, colormode)
        # Labels and (sub)titles
        ax.set_ylabel(f"Residuals [km]")
        ax.set_xlabel("Year")
        plt.suptitle(
            plot_title,
            fontsize = isu.big_font_size, 
            weight = 'bold',
        )
        step_size_unit_str = isu.convert_step_size_DAY_to_str_with_unit(step_size_DAY)
        ax.set_title(
            f"Observation residuals \n {title_tex_body},  {integrator},  Δt={step_size_unit_str},  $\\text{{n}}_{{\\text{{orbits}}}}$={n_orbits_to_propagate}",
            fontsize = isu.medium_font_size,
        )
        # fig.set_tight_layout(True)
        
        if (plot_behaviour == 'show') or ((plot_behaviour == 'both') and (k==0)):
            plt.show(block=False)
            plt.pause(0.001)
        elif (plot_behaviour == 'save') or ((plot_behaviour == 'both') and (k>0)):
            body_dir = f"{body['filing_name_long']}/"
            darkmode_dir = f"{'darkmode/'*colormode}"
            save_path = isu.plot_dir + scenario_comparison_plot_dir + body_dir + darkmode_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig_name = f"{timestamp_str}__{scenario_figname_prefix}_obs-resids.pdf"
            plt.savefig(save_path + fig_name, format="pdf")
            plt.close()
            print(f"ASC{' w/ estimation'*use_estimation}: observation residuals vs. time {'darkmode '*colormode}figure saved: \n{save_path + fig_name}")

#%% Print results of A2 estimation
### Print results of A2 estimation

if scenario_comparison_set in [6]:
    # print_default(f"\nRESULTS OF A2 ESTIMATION:")
    # print_default(f"Literature A2 value:    {A2_target_body:5g} m/s^2")
    # print_default(f"Estimated A2 value:     {initial_state_updated[6]:5g} m/s^2")
    # print_default(f"A2 change:              {( (initial_state_updated[6]-A2_target_body) / A2_target_body)*100:4g} %")
    # print_default(f"Residual change:        {( (scenario_residuals[1]-scenario_residuals[0]) / scenario_residuals[0])*100:4g} %")
    print_default(body['longname'])
    print_default(integrator)
    print_default(step_size_DAY)
    print_default(propagation_initial_time)
    print_default(n_orbits_to_propagate)
    print_default(propagation_duration_SEC / cnst.JULIAN_YEAR)
    print_default(n_LMBAs) if LMBAs_needed else print_default("-")
    print_default(use_estimation)
    print_default(" ")
    print_default(scenario_comparison_set)
    print_default(baseline_components_label)
    print_default(" ")
    print_default(n_MBAs_CP_search) if CPs_needed else print_default("-")
    print_default(close_pass_threshold_AU) if CPs_needed else print_default("-")
    print_default(close_pass_influence_threshold) if CPs_needed else print_default("-")
    print_default(" ")
    print_default(estimate_A2)
    print_default(" ")
    print_default(" ") # A2
    print_default(A2_target_body)
    print_default(initial_state_updated[6])
    print_default(((initial_state_updated[6]-A2_target_body) / A2_target_body))
    print_default(" ")
    print_default(" ") # RESIDUALS
    print_default(scenario_residuals[0])
    print_default(scenario_residuals[1])
    print_default(((scenario_residuals[1]-scenario_residuals[0]) / scenario_residuals[0]))
    print_default(scenario_residuals[2])
    print_default(((scenario_residuals[2]-scenario_residuals[0]) / scenario_residuals[0]))
    print_default(" ")
    print_default(" ") # RUNTIMES
    print_default(runtime_scenarios[0])
    print_default(runtime_scenarios[1])
    print_default(((runtime_scenarios[1]-runtime_scenarios[0]) / runtime_scenarios[0]))
    print_default(runtime_scenarios[2])
    print_default(((runtime_scenarios[2]-runtime_scenarios[0]) / runtime_scenarios[0]))
    print_default(" ")
    print_default(" ") # INITIAL STATE CHANGE (magnitude of position change)
    print_default(scenario_initial_state_changes[0])
    print_default(scenario_initial_state_changes[1])
    print_default(((scenario_initial_state_changes[1]-scenario_initial_state_changes[0]) / scenario_initial_state_changes[0]))
    print_default(scenario_initial_state_changes[2])
    print_default(((scenario_initial_state_changes[2]-scenario_initial_state_changes[0]) / scenario_initial_state_changes[0]))
    print_default(" ")
    print_default(" ") # PLOTS
    print_default(scenario_comparison_plot_dir)
    print_default(body_dir)
    print_default(timestamp_str)
    
    