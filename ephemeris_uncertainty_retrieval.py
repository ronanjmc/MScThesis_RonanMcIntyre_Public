"""
This script can be used to retrieve ephemeris uncertainty data for a set of
bodies using astroquery's HorizonsClass and the extensions to this class found
in horizons_extended.py (compiled by Lars Hinüber (@larshinueber), based on code
by @megargayu on GitHub, see https://github.com/astropy/astroquery/pull/3273)
"""
#%%
import time as tm
T_START_overall = tm.perf_counter()
import IS_utils as isu
import pandas as pd
import numpy as np 
from tudatpy.astro.time_representation import DateTime
    

# Inputs 
ephem_unc_file_dir = "data/ephemeris_uncertainty/"
verbose = True 
start_time = '1901-01-01'
stop_time = '2030-01-01'
time_step = '10d' # integer followed by d (for day), y (for year), m (for minute), or s (for second)

# Function that prints args if verbose=True and does nothing if verbose=False
print_verbose = print if verbose else lambda *a, **k: None

# Get list of Fenucci bodies
Fenucci_etal2024_tabB1 = (
    pd.read_fwf(
        isu.yark_data_dir + "Fenucci-et-al-2024_tableB1.dat", 
        colspecs = 'infer',
        names = [           # From Fenucci-et-al-2024_readme.txt
            'Asteroid',     # ---       Name of the asteroid
            'H',            # mag       Absolute magnitude
            'RMS',          # ---       RMS of normalized residuals, 7D OD
            'RMS6D',        # ---       RMS of normalized residuals, 6D OD
            'A2',           # au/d2     Transversal acceleration component
            'e_A2',         # au/d2     Error in A2
            'da/dt',        # au/Myr    Semi-major axis drift
            'e_da/dt',      # au/Myr    Error in da/dt
            'max(da/dt)',   # au/Myr    Maximum da/dt from Monte Carlo model
            'SNR',          # ---       Signal-to-noise of A2 detection
            'FAccept',      # ---       Flag for acceptance of the detection [1=accept, Rej.=reject]
            'NOptObs',      # ---       Number of optical observations
            'NRejOpt',      # ---       Number of rejected optical observations in 7D OD
            'NRej6D',       # ---       Number of rejected optical observations in 6D OD
            'NRadObs',      # ---       Number of radar observations
            'NRejRad',      # ---       Number of rejected radar obs in 7D OD
            'NRejRad6D',    # ---       Number of rejected radar obs in 6D OD
            'NOptOld',      # ---       Number of old observations
            'Dlow',         # m         15th percentile of diameter
            'Dmed',         # m         50th percentile of diameter
            'Dhigh',        # m         85th percentile of diameter
            'ModFlag',      # ---       Flag for model used in Monte Carlo [0/1]
            'Prot',         # h         Rotation period of the asteroid [-1=unknown]
            'Tax',          # ---       Taxonomic complex of the asteroid
            'deltat'        # yr        Length of observational arc
        ]
    ).query("FAccept != 'Rej.'") # filter out all rejected detections
    .sort_values(by='SNR', ascending=False) # sort by signal-to-noise ratio
)
body_list = Fenucci_etal2024_tabB1['Asteroid'].to_list()
body_list = body_list[30:]

# Generate files
unsuccessful_spkids = []
for i, code in enumerate(body_list):
    T_START_body = tm.perf_counter()
    body_data = isu.query_SBDB(str(code))
    print_verbose(f"Retrieving ephemeris uncertainty data for: {body_data['longname']} ({i+1}/{len(body_list)})...")
    
    spkid = body_data['SPKID']
    
    response = isu.retrieve_and_save_ephemeris_uncertainty(
        spkid = spkid,
        file_path = ephem_unc_file_dir,
        file_name = spkid,
        start_time = '1901-01-01',
        stop_time = '2030-01-01',
        time_step = '10d',
        overwrite_existing = True,
    )
    
    T_END_body = tm.perf_counter()
    print_verbose(f"Done ({T_END_body - T_START_body:.3f}s) \n")

total_duration = T_END_body - T_START_overall
if total_duration < 90:
    print_verbose(f"ALL FILES DONE ({total_duration:.3f}s)")
else:
    print_verbose(f"ALL FILES DONE ({total_duration/60:.1f} mins)")
if unsuccessful_spkids:
    print(f"Issues occurred with the following SPKIDs: {unsuccessful_spkids}")
    
#%%

import pickle 

with open(f"{ephem_unc_file_dir}{spkid}.p", 'rb') as fp:
    target_ephem_unc = pickle.load(fp)
    
# Make a column with epoch times using tudat DateTime object
target_ephem_unc["ephemeris_time"] = [DateTime.from_julian_day(jd) for jd in target_ephem_unc["datetime_jd"]]
target_ephem_unc["ephemeris_time_J2000"] = [dt.to_epoch() for dt in target_ephem_unc["ephemeris_time"]]

obs_range_start = [1990, 1, 1]
obs_range_end = [2025, 7, 1]
obs_range_start_J2000 = DateTime(*obs_range_start).epoch() # in seconds since J2000
obs_range_end_J2000 = DateTime(*obs_range_end).epoch() # in seconds since J2000

mask_after_window_start = np.where(target_ephem_unc["ephemeris_time_J2000"] > obs_range_start_J2000, True, False)
mask_before_window_end = np.where(target_ephem_unc["ephemeris_time_J2000"] < obs_range_end_J2000, True, False)
mask_within_obs_range = np.logical_and(mask_after_window_start, mask_before_window_end)

target_ephem_unc_filtered = target_ephem_unc[mask_within_obs_range]
