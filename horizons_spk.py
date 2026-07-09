"""
This script can be used to generate and save SPK files for a set of bodies. It
is based on a script from the Horizons API website
(https://ssd-api.jpl.nasa.gov/doc/horizons.html) 

changes:
☑ make script version that I can run in jupyter interactive terminal
☑ compare using local file and using JPL HorizonsQuery
☑ make function version that can be called in a loop
☑ include timing prints for running this script/function
☑ use SPKID as input. also have an input which determines the filename of the
  SPK file, which should be filing_name_short.
☑ start_time and stop_time could be optional inputs with defaults set to 1800
  and 2199
☑ files should be saved to a specified folder
☑ files should be saved with a specified file name (not SPKID.bsp)
    - [filing_name_short].bsp 
        - e.g. for 101955 Bennu: 101955.bsp
        - e.g. for 152563 (1992 BF): 152563.bsp
        - e.g. for 1995 CR: 1995-CR.bsp
    - could leave out startdate and enddate if using limits 
    - [filing_name_short]_[startdate]_[enddate].bsp 
        - e.g. for 101955 Bennu: 101955_1985-01-01_2035-01-01.bsp
        - e.g. for 152563 (1992 BF): 152563_1985-01-01_2035-01-01.bsp
        - e.g. for 1995 CR: 1995-CR_1985-01-01_2035-01-01.bsp

eventually, run this for all asteroids mentioned in SiMDA
"""

import time as tm
T_START_overall = tm.time()
import IS_utils as isu
import pandas as pd
import numpy as np 

"""
=================== THIS FUNCTION NOW LIVES IN IS_utils ========================

import json
import base64
import requests
def generate_spk_file(
    spkid: str,
    spk_file_path: str,
    spk_file_name: str,
    start_time = '1900-01-01',
    stop_time = '2100-01-01',
    overwrite_existing = False,
):
    ""
    This function is based on a script from the Horizons API website
    (https://ssd-api.jpl.nasa.gov/doc/horizons.html). It can be used to generate
    an SPK file from JPL Horizons and save it locally 
    
    Parameters
    ----------
    spkid: str
        Name of the body of whose SPK file is to be generated. The string must
        be a spice-recognized name or ID. 
    spk_file_path: str
        Path to the folder to which the SPK file will be saved
    spk_file_name: str
        File name of the generated SPK file: [spk_file_name].bsp
    start_time: str
        Date defining the start of the SPK file time interval (ET). Must be
        formatted as 'yyyy-mm-dd', for example '2025-09-25'
    stop_time: str
        Date defining the end of the SPK file time interval (ET). Must be
        formatted as 'yyyy-mm-dd', for example '2025-09-25'
    overwrite_existing: bool
        If a file with name [spk_file_name].bsp already exists in the folder
        given by spk_file_path, it will only be overwritten if
        overwrite_existing is True. Otherwise, the file generation will be
        skipped (default behaviour)
    
    Returns
    ------
    None if file generation was successful. If not, the spkid input is returned.
    ""

    spk_file_name += ".bsp"
    if os.path.exists(spk_file_path+spk_file_name) and overwrite_existing==False:
        print("Skipping SPK file generation to avoid overwrite.")
        return None
                
    # Build the appropriate URL for this API request
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api'
    # IMPORTANT: You must encode the "=" as "%3D" and the ";" as "%3B" in the
    # Horizons COMMAND parameter specification.
    url += "?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
    url += "&COMMAND='DES%3D{}%3B'&START_TIME='{}'&STOP_TIME='{}'".format(spkid, start_time, stop_time)
    
    # Submit the API request and decode the JSON-response:
    response = requests.get(url)
    try:
        data = json.loads(response.text)
    except ValueError:
        print(f"Unable to decode JSON results for SPKID: {spkid}")

    # If the request was valid...
    if (response.status_code == 200):
        
        # If the SPK file was generated, decode it and write it to the output file:
        if "spk" in data:

            try:
                f = open(spk_file_path + spk_file_name, "wb")
            except OSError as err:
                print(f"Unable to open SPK file: '{spk_file_path + spk_file_name}': {err}")
            
            # Decode and write the binary SPK file content:
            f.write(base64.b64decode(data["spk"]))
            f.close()
            print(f"SPK file saved to: {spk_file_path + spk_file_name}")
            return None
    
        # Otherwise, the SPK file was not generated so output an error:
        print(f"ERROR: SPK file not generated for SPKID: {spkid}")
        if "result" in data:
            print(data["result"])
        else:
            print(response.text)
        return spkid

    # If the request was invalid, extract error content and display it:
    elif (response.status_code == 400):
        data = json.loads(response.text)
        if "message" in data:
            print(f"MESSAGE: {data['message']}")
        else:
            print(json.dumps(data, indent=2))
        return spkid

    # Otherwise, some other error occurred:
    else:
        print(f"RESPONSE CODE: {response.status_code}")
        return spkid
"""

# Inputs 
# spkid = '20001566'
# spk_file_name = 'spk_file.bsp'
SPK_file_dir = "data/SPK_files/"
verbose = True 

# Function that prints args if verbose=True and does nothing if verbose=False
print_verbose = print if verbose else lambda *a, **k: None

# Get SiMDA data, filter out comets and TNOS
bodies_SiMDA_filtered = (
    pd.read_csv(isu.SiMDA_file)
    .assign(NUM=lambda x: np.int32(x.NUM)) # ! this produces a warning because the NUM value is empty for some rows 
    .query("DYN != 'COM' & DYN != 'TNO'") # filter out comets and trans-Neptunian objects (COM, TNO)
    # .sort_values(by='MASS', ascending=False) # sort bodies by mass, descending order
    .loc[:, ["NUM", "DESIGNATION", "DIAM", "DYN", "MASS"]]
)
body_list = bodies_SiMDA_filtered['NUM'].to_list()

# body_list = body_list[:5]
# body_list = ["Pluto", 101955]
# Errors for: Pluto, 101955 Bennu

body_list = ['1866']


# Generate files
unsuccessful_spkids = []
for i, code in enumerate(body_list):
    T_START_body = tm.time()
    body_data = isu.query_SBDB(str(code))
    print_verbose(f"Generating SPK file for: {body_data['longname']} ({i+1}/{len(body_list)})...")
    
    spkid = body_data['SPKID']
    # spk_file_name = body_data['filing_name_short']
    
    spkid = isu.generate_spk_file(
        spkid = spkid, 
        spk_file_path = SPK_file_dir,
        spk_file_name = spkid,
        overwrite_existing = True,
    )
    if not (spkid == None):
        unsuccessful_spkids.append(spkid)
    
    T_END_body = tm.time()
    print_verbose(f"Done ({T_END_body - T_START_body:.3f}s) \n")

total_duration = T_END_body - T_START_overall
if total_duration < 60:
    print_verbose(f"ALL FILES DONE ({total_duration:.3f}s)")
else:
    print_verbose(f"ALL FILES DONE ({total_duration/60:.1f} mins)")
if unsuccessful_spkids:
    print(f"Issues occurred with the following SPKIDs: {unsuccessful_spkids}")

#%% SCRIPT VERSION
"""
This script comes from https://ssd-api.jpl.nasa.gov/doc/horizons.html. It can be
invoked from the command line to generate an SPK file from JPL Horizons. For
example, for asteroid 1 Ceres (with SPKID 2000001):
    python horizons_spk.py 2000001. 
"""
_ = 0
# import sys
# import json
# import base64
# import requests

# # Inputs 
# # spkid = 
# spk_filename = 'spk_file.bsp'
# SPK_file_dir = "../data/SPK_files/"

# # Define API URL and SPK filename:
# url = 'https://ssd.jpl.nasa.gov/api/horizons.api'

# # Define the time span: # not necessarily accurate over the full time span
# start_time = '1950-01-01'
# stop_time = '2050-12-31'

# # Get the requested SPK-ID from the command-line:
# if (len(sys.argv)) == 1:
#     print("please specify SPK-ID on the command-line")
#     sys.exit(2)
# spkid = sys.argv[1]

# # Build the appropriate URL for this API request:
# # IMPORTANT: You must encode the "=" as "%3D" and the ";" as "%3B" in the
# #            Horizons COMMAND parameter specification.
# url += "?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
# url += "&COMMAND='DES%3D{}%3B'&START_TIME='{}'&STOP_TIME='{}'".format(spkid, start_time, stop_time)

# # Submit the API request and decode the JSON-response:
# response = requests.get(url)
# try:
#     data = json.loads(response.text)
# except ValueError:
#     print("Unable to decode JSON results")

# # If the request was valid...
# if (response.status_code == 200):
#     #
#     # If the SPK file was generated, decode it and write it to the output file:
#     if "spk" in data:
#         # If a suggested SPK file basename was provided, use it:
#         if "spk_file_id" in data:
#             spk_filename = data["spk_file_id"] + ".bsp"
            
#         try:
#             f = open(SPK_file_dir + spk_filename, "wb")
#         except OSError as err:
#             print("Unable to open SPK file '{0}': {1}".format(SPK_file_dir + spk_filename, err))
        
#         # Decode and write the binary SPK file content:
#         f.write(base64.b64decode(data["spk"]))
#         f.close()
#         print("wrote SPK content to {0}".format(SPK_file_dir + spk_filename))
#         sys.exit()
    
#     # Otherwise, the SPK file was not generated so output an error:
#     print("ERROR: SPK file not generated")
#     if "result" in data:
#         print(data["result"])
#     else:
#         print(response.text)
#     sys.exit(1)

# # If the request was invalid, extract error content and display it:
# if (response.status_code == 400):
#     data = json.loads(response.text)
#     if "message" in data:
#         print("MESSAGE: {}".format(data["message"]))
#     else:
#         print(json.dumps(data, indent=2))

# # Otherwise, some other error occurred:
# print("response code: {0}".format(response.status_code))
# sys.exit(2)