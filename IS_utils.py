"""
This file contains several custom functions with various purposes that are used
throughout the project.
Plotting configurations are also defined in this file to ensure consistent
plotting style.
"""

# Tudat imports
from tudatpy.interface import spice
from tudatpy.dynamics import simulator, environment_setup, propagation_setup
import tudatpy.constants as cnst
from tudatpy.astro import element_conversion
from tudatpy.astro.time_representation import DateTime, julian_day_to_seconds_since_epoch

# JPL SBDB, JPL Horizons, MPC interfaces
from astroquery.jplsbdb import SBDB as astroquerySBDB
from tudatpy.data.horizons import HorizonsQuery
from horizons_extended import HorizonsClass
from tudatpy.data.sbdb import SBDBquery
from tudatpy.data.mpc import BatchMPC

# Other Python imports
import os
import json
import pickle
import base64
import operator
import warnings
import requests
import time as tm
import numpy as np
import pandas as pd
import datetime as dttm
import matplotlib.colors
import matplotlib.pyplot as plt

# Files & directories
ephem_unc_dir = "input_data/ephemeris_uncertainty/"
mass_data_dir = "input_data/mass_data/"
SBDB_data_dir = "input_data/SBDB_data/"
SiMDA_file = "input_data/SiMDA_data/SiMDA_250320.csv"     # from https://astro.kretlow.de/simda/catalog/
SPK_file_dir = "input_data/SPK_files/"
yark_data_dir = "input_data/yarkovsky_data/"
plot_dir = "plots/"
output_data_dir = "output_data/"

# Space telescope codes
spacecraft_codes = {
    # Name          # MPC code      # Horizons code
    "WISE":         ("C51",         "-163"),
    "TESS":         ("C57",         "-95"),
    "NEOSSat":      ("C53",         "-139089"),
    "Non-geocentric Occultation Observation": ("275", None),
    # For spacecraft MPC codes: https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html
    # For spacecraft Horizons codes: https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND=%27*%27
}

### PLOTTING CONFIGURATION
convenience_scale = 1.5
A4_width_INCH = 210 / 25.4 # (210 mm) / (25.4 mm/inch)
overleaf_hscale = 0.75 # proportion of the horizontal span used for text, etc.
textwidth_INCH = A4_width_INCH * overleaf_hscale * convenience_scale
full_width_INCH = 1.0 * textwidth_INCH
twothirds_width_INCH = 0.67 * textwidth_INCH
half_width_INCH = 0.50 * textwidth_INCH

# Fonts
tiny_font_size = 7 * convenience_scale
small_font_size = 8 * convenience_scale
medium_font_size = 10 * convenience_scale
big_font_size = 12 * convenience_scale
huge_font_size = 16 * convenience_scale
# tiny_font_size = 10
# small_font_size = 13
# medium_font_size = 16
# big_font_size = 20
# huge_font_size = 28
plt.rcParams["font.family"] = "Atkinson Hyperlegible"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Atkinson Hyperlegible'
plt.rcParams['mathtext.it'] = 'Atkinson Hyperlegible:italic'
plt.rcParams['mathtext.bf'] = 'Atkinson Hyperlegible:bold'
plt.rcParams['legend.title_fontsize'] = small_font_size
# Style
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
markers = ['v', '^', '<', '>',
           (5, 1, 0), (6, 1, 0), (8, 1, 0), (10, 1, 0),
           'x', 'X', 'P',
           (3, 2, 0), (5, 2, 0), (6, 2, 0),
           's']
# Colour modes
# - Index of 0 (or boolean of False) corresponds to light mode
# - Index of 1 (or boolean of True) corresponds to dark mode
ax_background = ['#f6f6f6', '#080808']
fig_background = ['#ffffff', '#000000']
text_color = ['#000000', '#ffffff']     # also for ticks, spines, etc.
legend_bg_color = ax_background
grid_color = ['#BDBDBD', '#666666']
spine_color = ['#000000', "#b6b6b6"]

cmap_default10 = plt.get_cmap("tab10") # matplotlib preset set of 10 colours
custom_colors_lightmode = [
    '#008FD5', # 0 blue
    '#FF7125', # 1 orange
    '#579621', # 2 green
    '#DC2525', # 3 red
    '#e1aa00', # 4 yellow
    '#DD58B5', # 5 pink
    '#8A3626', # 6 brown
    '#8B8B8B', # 7 grey
    '#27AFBE', # 8 cyan
    '#AC31A5', # 9 purple
]
custom_colors_darkmode = [
    '#008FD5', # 0 blue (same)
    '#FF7125', # 1 orange (same)
    '#579621', # 2 green (same)
    '#DC2525', # 3 red (same)
    'xkcd:sunflower', # 4 yellow (brighter)
    '#FF58CD', # 5 pink (brighter)
    '#965F3D', # 6 brown (lighter)
    '#B8B8B8', # 7 grey (lighter)
    '#2FD5E7', # 8 cyan (brighter)
    '#D609CC', # 9 purple (brighter)
]
cmap_custom10_lightmode = matplotlib.colors.ListedColormap(
    colors = custom_colors_lightmode,
    name = 'tab10_custom_lightmode'
)
cmap_custom10_darkmode = matplotlib.colors.ListedColormap(
    colors = custom_colors_darkmode,
    name = 'tab10_custom_darkmode'
)
cmap_custom10 = [cmap_custom10_lightmode, cmap_custom10_darkmode]





### CUSTOM FUNCTIONS
# Root mean square
def rms(input):
    """
    Calculates the root-mean-square (RMS) of the input array. Input should be
    a numpy array or similar
    """
    return np.sqrt(np.mean(input**2))




# Formatting, fontsizes, colours for standard plots
def set_default_2d_ax_settings(
    ax: matplotlib.axes._axes.Axes,
    colormode
):
    """
    Sets some characteristics of Axes object for a standard 2d plot. This is for
    consistent plotting style, to reduce code clutter and repitition when
    plotting, and to make it easier to produce plots in both colour modes.

    Parameters
    ----------
    ax: matplotlib.axes._axes.Axes
        Axes object to apply settings to
    colormode: bool or float
        Determines while colour mode is used for plotting. 0/False for light
        mode, 1/True for dark mode

    Return
    ------
    ax: matplotlib.axes._axes.Axes
        Modified Axes object
    """

    ax.grid(visible=True, color=grid_color[colormode])
    ax.set_facecolor(ax_background[colormode])
    for spine in ax.spines.values(): # ax borders
        spine.set_edgecolor(spine_color[colormode])
    ax.tick_params(
        axis = 'both',
        which = 'major',
        labelsize = small_font_size,
        labelcolor = text_color[colormode],
        color = spine_color[colormode],
    )
    ax.set_ylabel(
        None,
        fontsize = medium_font_size,
        color = text_color[colormode],
    )
    ax.set_xlabel(
        None,
        fontsize = medium_font_size,
        color = text_color[colormode],
    )
    plt.suptitle( # main title at top, in bold / overall figure title
        None,
        # fontsize = isu.big_font_size, # this must be set together with the title
        # weight = 'bold', # this must be set together with the title
        color = text_color[colormode],
    )
    ax.set_title( # subtitle below main title / subplot title
        None,
        # fontsize = isu.medium_font_size, # this must be set together with the title
        color = text_color[colormode],
    )

    return ax




### Standard input processing
def query_SBDB(body_for_querying, print_body_info = False):
    """
    Uses astroquery to query the JPL Small Body Database (SBDB) for the chosen
    body. Uses a locally saved file if available (i.e. if the data for
    body_for_querying has been queried before)

    More information:
    SBDBClass: https://astroquery.readthedocs.io/en/latest/api/astroquery.jplsbdb.SBDBClass.html#astroquery.jplsbdb.SBDBClass
    Output: https://ssd-api.jpl.nasa.gov/doc/sbdb.html#data-output

    Parameters
    ----------
    body_for_querying: str
        Body whose data will be queried from JPL SBDB. Accepts designation
        number, common name, or provisional designation of asteroid

    Returns
    -------
    body: dict
        Dictionary with various information on the body
    """

    try:
        # See if this body's data has been saved locally after a previous query
        with open(f"{SBDB_data_dir}{body_for_querying}.p", 'rb') as fp:
            body_SBDB = pickle.load(fp)

        # NOTE: It would be better to check how old this file is, and if it is
        # older than 4 weeks (for example), to run a new query and overwrite the
        # existing file. This would ensure the data is reasonably up-to-date.

    except FileNotFoundError:
        # For some strange reason, searching 2020 PP1 via this astropy query
        # instead returns asteroid 412664 (2014 OH196), whose fullname is
        # provided as '412664 (2014 OH196 = 2010 PP1)'. This seems to be
        # incorrect, as 2010 PP1 is not the same asteroid as 412664. It may just
        # be an error in the database. This is the behaviour as of 19 May 2026.
        if body_for_querying == "2020 PP1":
            # Use the SPKID instead
            body_for_querying = "54050999"

        # If body data not available locally, query JPL SBDB
        body_SBDB = astroquerySBDB.query(
            body_for_querying,
            phys = True,              # Include physical parameters if available (D, H, rot., albedo, etc.)
            full_precision = True,    # 16 signigicant digits (instead of 3)
        )
        # Save data locally
        with open(f"{SBDB_data_dir}{body_for_querying}.p", 'wb') as fp:
            pickle.dump(body_SBDB, fp, protocol=pickle.DEFAULT_PROTOCOL)
        # NOTE: data for the same object may be stored more than once if
        # body_for_querying is given in different forms (e.g. designation
        # number, common name, or provisional designation). Also, I'm using
        # Pickle because the Quantity object used by SBDB cannot be serialised
        # using JSON


    # Name, number, provisional designation etc.
    body = dict()
    body['SPKID'] = body_SBDB['object']['spkid']
    if body_SBDB['object']['kind'] == 'an':
        # 'an' means 'numbered asteroid'
        body['designation_number'] = body_SBDB['object']['des']
        body['designation_number_flag'] = True

        # get provisional designation (YYYY AA[NNN])
        body['provisional_designation'] = " ".join(body_SBDB['object']['fullname'].split(" ")[-2:])[1:-1]

        # try to get common name
        try:
            body['common_name'] = " ".join(body_SBDB['object']['shortname'].split(" ")[1:])
            body['common_name_flag'] = True
            body['longname'] = " ".join([body['designation_number'], body['common_name']])
        except KeyError:
            body['common_name'] = None
            body['common_name_flag'] = False
            body['longname'] = f"{body['designation_number']} ({body['provisional_designation']})"
        # labelname is 'designation_number common_name' or just
        # 'designation_number' if common_name is None. e.g. '2062 Aten' or '136617'
        common_name_addition = f"{' '*body['common_name_flag']}{str(body['common_name'])*body['common_name_flag']}"
        body['labelname'] = f"{body['designation_number']}{common_name_addition}"
        body['filing_name_long'] = f"{body['longname'].replace(' ', '-').replace('(', '').replace(')', '')}"
        body['filing_name_short'] = body['designation_number']

    elif body_SBDB['object']['kind'] == 'au':
        # 'au' means 'unnumbered asteroid'
        body['designation_number'] = '[unnumbered]'
        body['designation_number_flag'] = False
        body['common_name'] = None
        body['provisional_designation'] = body_SBDB['object']['fullname'][1:-1]
        body['longname'] = body['provisional_designation']
        body['labelname'] = body['provisional_designation']
        body['filing_name_long'] = f"{body['provisional_designation'].replace(' ', '-')}"
        body['filing_name_short'] = body['filing_name_long']

    # Status
    body['NEO_flag'] = body_SBDB['object']['neo']
    body['PHA_flag'] = body_SBDB['object']['pha']

    # Orbit fit data
    body['epoch_osculation_JD'] = body_SBDB['orbit']['epoch'].value
    body['equinox'] = body_SBDB['orbit']['equinox']
    body['RMS_norm_res'] = float(body_SBDB['orbit']['rms'])
    body['n_obs_used'] = body_SBDB['orbit']['n_obs_used'] # optical plus radar observations
    body['n_dop_obs_used'] = body_SBDB['orbit']['n_dop_obs_used'] # radar Doppler observations
    body['n_del_obs_used'] = body_SBDB['orbit']['n_del_obs_used'] # radar delay observations
    body['obs_arc_DAY'] = float(body_SBDB['orbit']['data_arc'])
    try:
        body['A2_AU_per_DAYsq'] = body_SBDB['orbit']['model_pars']['A2'].value
    except (TypeError, KeyError) as ex:
        body['A2_AU_per_DAYsq'] = None

    try:
        body['A2_unc_AU_per_DAYsq'] = body_SBDB['orbit']['model_pars']['A2_sig'].value
    except (TypeError, KeyError) as ex:
        body['A2_unc_AU_per_DAYsq'] = None

    # Orbital elements
    def assign_except_AE_None(dictkey, value):
        try:
            body[dictkey] = value
        except (AttributeError) as ex:
            body[dictkey] = None

    body['semimajoraxis_AU'] = body_SBDB['orbit']['elements']['a'].value
    body['eccentricity'] = body_SBDB['orbit']['elements']['e']
    body['inclination_DEG'] = body_SBDB['orbit']['elements']['i'].value
    body['RA_ascendingnode_DEG'] = body_SBDB['orbit']['elements']['om'].value
    body['argument_periapsis_DEG'] = body_SBDB['orbit']['elements']['w'].value
    body['mean_anomaly_DEG'] = body_SBDB['orbit']['elements']['ma'].value
    body['time_perihelion_passage_JD'] = body_SBDB['orbit']['elements']['tp'].value # JD = Julian Day
    body['earth_MOID_AU'] = body_SBDB['orbit']['moid'].value
    body['orbitperiod_DAY'] = body_SBDB['orbit']['elements']['per'].value

    # Uncertainty of orbital elements
    assign_except_AE_None('semimajoraxis_unc_AU', body_SBDB['orbit']['elements']['a_sig'].value)
    assign_except_AE_None('eccentricity_unc', body_SBDB['orbit']['elements']['e_sig'])
    assign_except_AE_None('inclination_unc_DEG', body_SBDB['orbit']['elements']['i_sig'].value)
    assign_except_AE_None('RA_ascendingnode_unc_DEG', body_SBDB['orbit']['elements']['om_sig'].value)
    assign_except_AE_None('argument_periapsis_unc_DEG', body_SBDB['orbit']['elements']['w_sig'].value)
    assign_except_AE_None('mean_anomaly_unc_DEG', body_SBDB['orbit']['elements']['ma_sig'].value)
    assign_except_AE_None('time_perihelion_passage_unc_JD', body_SBDB['orbit']['elements']['tp_sig'].value)
    assign_except_AE_None('orbitperiod_unc_DAY', body_SBDB['orbit']['elements']['per_sig'].value)

    # Uncertainty of orbital elements (all NoneType for Pluto, for example, so
    # we need to do try/excepts)

    # Physical parameters
    try:
        body['diameter_KM'] = body_SBDB['phys_par']['diameter'].value
    except (KeyError, AttributeError) as ex:
        body['diameter_KM'] = None

    try:
        body['diameter_unc_KM'] = body_SBDB['phys_par']['diameter_sig'].value
    except (KeyError, AttributeError) as ex:
        body['diameter_unc_KM'] = 'N/A'

    try:
        body['diameter_ref'] = body_SBDB['phys_par']['diameter_ref']
        body['diameter_ref_flag'] = True
    except KeyError:
        body['diameter_ref'] = None
        body['diameter_ref_flag'] = False

    if print_body_info:
        if body['diameter_KM'] == None:
            diameter_information = ""
        else:
            diameter_information = f"""Diameter{'*'*body['diameter_ref_flag']} [km]          : {body['diameter_KM']} +/- {body['diameter_unc_KM']}

{body['diameter_ref_flag']*'*Ref: '}{body['diameter_ref']}
            """

        # Print selected body info
        print(f"""BODY INFORMATION FOR: {body['longname'].upper()} (from JPL SBDB)
    Designation number      : {body['designation_number']}
    Common name             : {body['common_name']}
    Provisional designation : {body['provisional_designation']}
    NEO / PHA status        : {body['NEO_flag']} / {body['PHA_flag']}
    Semi-major axis [AU]    : {round(body['semimajoraxis_AU'],4)} +/- {body['semimajoraxis_unc_AU']}
    Eccentricity [-]        : {round(body['eccentricity'],4)} +/- {body['eccentricity_unc']}
    Orbital period [days]   : {round(body['orbitperiod_DAY'],1)} +/- {body['orbitperiod_unc_DAY']}
    {diameter_information}"""
        )

    return body
#
def get_body_names_from_designations(codes):
    """
    Queries the SBDB database to retrieve name(s) of asteroid(s), given the
    MPC/IAU designation number(s) of the body/bodies. If no common name is
    available, the provisional designation will be retrieved (e.g. '2002 LM60').

    Parameters
    ----------
    codes: list of int, list of str
        The IAU/MPC asteroid number designations of the bodies for which
        names are to be retrieved. For unnumbered objects, this can also be a
        string of the provisional designation (e.g. '1998 SD9')

    Return
    ------
    code_names: list of str
        The retrieved name(s) and/or provisional designations. Order from codes
        is preserved
    code_SBDB: tudatpy.data.sbdb.SBDBquery
        SBDBquery objects
    """

    # Check type of input variable
    if not (isinstance(codes, list) or
            isinstance(codes, tuple) or
            isinstance(codes, np.ndarray)
    ):
        raise TypeError(f"Input must be list, tuple, or numpy.ndarray; {type(codes)} not allowed.")

    code_names = []
    query_objects = []

    for code in codes:
        code_SBDB = SBDBquery(code)
        query_objects.append(code_SBDB)

        try:
            code_SPK_ID = code_SBDB.shortname
            code_name = code_SPK_ID.split(" ")[1:]
            code_name = " ".join(code_name)
            code_names.append(code_name)

        # If no common name, code_SBDB.shortname raises KeyError
        # Instead, get the provisional designation (e.g. 1998 KG3 for object 350462)
        except KeyError:
            try:
                code_SPK_ID = code_SBDB.name
                name_parts = code_SPK_ID.split(" ")
                code_number = int(name_parts[0])
                code_name = " ".join(name_parts[1:])[1:-1] # removes number and brackets from name string
                code_names.append()

            # If object is unnumbered, name string is formatted differently
            # If unnumbered, int(name_parts[0]) raises ValueError and redirects here
            except ValueError:
                code_name = " ".join(name_parts[0:])[1:-1] # removes number and brackets from name string
                code_names.append(code_name)

            # A KeyError occurs if code_SBDB.name doesn't exist (if e.g. object doesn't exist)
            except KeyError as ex:
                ex.add_note(f"The body '{code}' likely couldn't be found in the JPL SBDB")
                raise

    if len(codes)==1:
        return code_names[0], query_objects[0]
    else:
        return code_names, query_objects
#
def format_integrator_string(integrator):
    """
    Processes the integrator input string and returns a string that is nicely
    formatted for plots, labels, etc.
    """

    integrator = integrator.upper()
    integrator = integrator.replace("_", "")
    if integrator == "EULER":
        integrator = "Euler"
    elif integrator == "RKF12":
        integrator = "RKF1(2)"
    elif integrator == "RKF45":
        integrator = "RKF4(5)"
    elif integrator == "RKF56":
        integrator = "RKF5(6)"
    elif integrator == "RKF78":
        integrator = "RKF7(8)"
    elif integrator == "RKF89":
        integrator = "RKF8(9)"
    elif integrator == "RKF108":
        integrator = "RKF10(8)"
    elif integrator == "RKF1210":
        integrator = "RKF12(10)"
    elif integrator == "RKF1412":
        integrator = "RKF14(12)"

    return integrator
#
def process_integrator_input(
    integrator: str,
    print_integrator_info = False
):
    """
    Processes the integrator input string and returns the corresponding
    coefficient set and order.

    Parameters
    ----------
    integrator: str
        Integration method. Valid inputs are:
            - "Euler"
            - "RK[X]"       *   e.g. RK10
            - "RKF[A][B]"   **  e.g. RKF78
        Not case sensitive. Underscores can be inserted for readability, i.e.
        RKF_12_10.

        * Runge-Kutta fixed step method of order X, with X =
            [1,2,3,4,5,6,7,8,9,10,12,14]
        ** Runge-Kutta-Fehlberg variable step method of order A with error
            estimator of order B, with (A,B) = [(1,2), (4,5), (5,6), (7,8),
            (8,9), (10,8), (12,10), (14,12)]

        Coefficient sets available in Tudat but not supported by this function:
        explicit_mid_point, explicit_trapezoid_rule, ralston, ralston_3,
        ralston_4, SSPRK3, three_eight_rule_rk_4, heun_euler, rkdp_87, rkv_89
    verbose: bool
        If True, a summary of the selected integrator is printed

    Returns
    ------
    coefficient_set: tudatpy.kernel.dynamics.propagation_setup.integrator.CoefficientSets
        Coefficient set corresponding to input string
    order_to_use: tudatpy.kernel.dynamics.propagation_setup.integrator.OrderToIntegrate
        Order to integrate corresponding to input string
    """

    integrator = integrator.upper()
    integrator = integrator.replace("_", "")
    order_to_use = propagation_setup.integrator.OrderToIntegrate.lower
    if integrator in ["EULER"]:
        coefficient_set = propagation_setup.integrator.euler_forward
    elif integrator in ["RK1", "RKF12"]:
        coefficient_set = propagation_setup.integrator.rkf_12
    elif integrator in ["RK2"]:
        coefficient_set = propagation_setup.integrator.rkf_12
        order_to_use = propagation_setup.integrator.OrderToIntegrate.higher
    elif integrator in ["RK3"]:
        coefficient_set = propagation_setup.integrator.rk_3
    elif integrator in ["RK4", "RKF45"]:
        coefficient_set = propagation_setup.integrator.rkf_45
    elif integrator in ["RK5", "RKF56"]:
        coefficient_set = propagation_setup.integrator.rkf_56
    elif integrator in ["RK6"]:
        coefficient_set = propagation_setup.integrator.rkf_56
        order_to_use = propagation_setup.integrator.OrderToIntegrate.higher
    elif integrator in ["RK7", "RKF78"]:
        coefficient_set = propagation_setup.integrator.rkf_78
    elif integrator in ["RK8", "RKF89"]:
        coefficient_set = propagation_setup.integrator.rkf_89
    elif integrator in ["RK9"]:
        coefficient_set = propagation_setup.integrator.rkf_89
        order_to_use = propagation_setup.integrator.OrderToIntegrate.higher
    elif integrator in ["RKF108"]:
        coefficient_set = propagation_setup.integrator.rkf_108
    elif integrator in ["RK10", "RKF1210"]:
        coefficient_set = propagation_setup.integrator.rkf_1210
    elif integrator in ["RK12", "RKF1412"]:
        coefficient_set = propagation_setup.integrator.rkf_1412
    elif integrator in ["RK14"]:
        coefficient_set = propagation_setup.integrator.rkf_1412
        order_to_use = propagation_setup.integrator.OrderToIntegrate.higher
    else:
        raise ValueError(f"Selected integrator {integrator} is not recognised.")

    # Print selected integrator info
    if print_integrator_info:
        print(f"""SELECTED INTEGRATOR:
    String                  : {integrator}
    Coefficient set         : {coefficient_set.name}
    Order to use            : {order_to_use.name}
        """)

    return coefficient_set, order_to_use
#
def define_plot_behaviour_lists(
    plot_behaviour: str,
    colormode: int
):
    """
    Defines lists that determine plotting behaviour. Each individual plot only
    recognises "show" or "save" as output options. define_plot_behaviour_lists()
    is useful when the desired outcome is to output more than one version of the
    same plot (with different colormodes or different display types). The
    returns (plot_behaviour_list, colormode_list) can be zipped together and the
    plot can be called in a loop to produce all desired outputs.

    Parameters
    ----------
    plot_behaviour: str
        Valid inputs are "show", "showboth", "save", "both", and "skip".
            - "show": display plot(s) in (interactive) terminal (specified
              colormode only)
            - "showboth": display plot(s) in (interactive) terminal (lightmode
              and darkmode)
            - "save": save plots to files (lightmode and darkmode)
            - "both": display plot(s) in (interactive) terminal (specified
              colormode only) AND save plots to files (lightmode and darkmode)
            - "skip": skip all plotting
    colormode: int
        0 for lightmode, 1 for darkmode. Only relevant if plot_behaviour is
        "show" or "both". If plot_behaviour is "save" or "showboth", both
        colormodes are saved/shown anyway.

    Returns
    ------
    plot_behaviour_list: list
    colormode_list: list
    """

    if plot_behaviour == 'save':
        plot_behaviour_list = ['save', 'save']
        colormode_list = [1,0]
    elif plot_behaviour == 'showboth':
        plot_behaviour_list = ['show', 'show']
        colormode_list = [1,0]
    elif plot_behaviour == 'both':
        plot_behaviour_list = ['show', 'save', 'save']
        colormode_list = [colormode,1,0]
    elif plot_behaviour == 'skip':
        plot_behaviour_list = []
        colormode_list = []
    elif plot_behaviour == 'show':
        plot_behaviour_list = ['show']
        colormode_list = [colormode]
    else:
        raise ValueError("Invalid input for plot_behaviour.")

    return plot_behaviour_list, colormode_list
#
def convert_step_size_DAY_to_str_with_unit(
    step_size_DAY,
    full_unit_name: bool = False,
    also_return_float: bool = False,
    minute_as_m: bool = False,
):
    """
    Converts step_size_DAY to a more readable unit and returns a string with the
    value of the step size (rounded to three decimal places) and the
    corresponding unit. Useful for print statements, plot labels, etc.

    Parameters
    ----------
    step_size_DAY: float
        Step size in days
    full_unit_name: bool
        Default behaviour is to use the abbreviated name of the unit, e.g. 'h'
        for 'hour'. If full_unit_name is True, the full unit name is used
        instead
    also_return_float: bool
        If True, the float value of the step size in the converted units will
        also be returned
    minute_as_m: bool
        If True, the unit for minutes will be formatted just as 'm', as opposed
        to 'min' or 'minutes'

    Returns
    ----------
    step_size_unit_str: str
        String with the converted step size value (rounded to max. 3 decimal
        places) and corresponding unit
    step_size_new_unit: float
        Step size in the converted units
    """

    if step_size_DAY >= 1:
        step_size_new_unit = step_size_DAY
        step_size_unit_str = f"{step_size_new_unit:.4g} d{'ays'*full_unit_name}"
    elif step_size_DAY >= (1/24):
        step_size_new_unit = step_size_DAY*24
        step_size_unit_str = f"{step_size_new_unit:.4g} h{'ours'*full_unit_name}"
    elif (step_size_DAY >= (1/(24*60))) and minute_as_m:
        step_size_new_unit = step_size_DAY*24*60
        step_size_unit_str = f"{step_size_new_unit:.4g} m"
    elif step_size_DAY >= (1/(24*60)):
        step_size_new_unit = step_size_DAY*24*60
        step_size_unit_str = f"{step_size_new_unit:.4g} min{'utes'*full_unit_name}"
    else:
        step_size_new_unit = step_size_DAY*24*60*60
        step_size_unit_str = f"{step_size_new_unit:.5g} s{'econds'*full_unit_name}"

    if also_return_float:
        return step_size_unit_str, step_size_new_unit
    else:
        return step_size_unit_str
#
def define_integrator_and_step_size_DAY_from_eccentricity(e, verbose=False):
    if e < 0.6:
        integrator = "RK8"
        step_size_DAY = 2**-2    # 0.25 day / 6 h
    # if e < 0.4:
    #     integrator = "RK8"
    #     step_size_DAY = 2**0    # 1 day / 24 h
    # elif (e > 0.4) and (e < 0.6):
    #     integrator = "RK8"
    #     step_size_DAY = 2**-1   # 0.5 day / 12 h
    else:
        # e greater than 0.6
        integrator = "RK10"
        step_size_DAY = 2**-3   # 0.125 day / 3 h
    if verbose:
        print(f"    integrator              : {step_size_DAY} day ({step_size_DAY*24} h)\n")
        print(f"    ∆t                      : {step_size_DAY} day ({step_size_DAY*24} h)\n")

    return integrator, step_size_DAY



### Close pass searches
def search_for_close_passes_with_JPL_Horizons(
    target_body_name: str,          # designation number, common name, or provisional designation of asteroid
    n_MBAs_search: int,             # number of MBAs to include when searching for close passes (takes the n_MBAs_search most massive MBAs after excluding the n_MBAs_already_included most massive MBAs). Maximum: 396 - n_MBAs_already_included (as of 11 Jun 2025)
    n_MBAs_already_included: int,   # number of most massive MBAs to exclude from the search process (e.g. if the 8 most massive MBAs are included in the dynamics anyway, we don't need to search for close passes with these 8)
    search_period_initial_time: DateTime, # Tudat DateTime specifying the start of the search period for close passes
    n_orbits: float,                # duration of search period in number of orbits of the target body
    timestep_DAY: float,              # timestep used for querying JPL Horizons (i.e. time resolution of search) [days]
    close_pass_threshold_AU = 1.0,  # distance threshold that defines what is considered a close pass [AU]
    verbose = False,                # choose True to include additional print statements, e.g. with results of close pass search
):
    """
    This functions searches for close approaches between main belt asteroids
    (MBAs) and the given target body during a specified search period. This is
    to help in identifying additional asteroids which may need to be included in
    the dynamic model.
        MBA (mass) data comes from the SiMDA database, stored as a local .csv
    file. Asteroid ephemerides are retrieved from JPL Horizons.

    !! there could be a faster way of doing this, by getting Kepler elements of
    the asteroids and calculating closest approach analytically, rather than
    querying JPL, which is quite slow for many bodies and for long search
    periods.

    Parameters
    ----------
    target_body_name: str
        Subject of the search process (designation number, common name, or
        provisional designation of asteroid). This string will be used in the
        astroquery to JPL SBDB
    n_MBAs_search: int
        Number of MBAs to include in the search process. The SiMDA catalogue is
        first sorted by mass, then the top n_MBAs_already_included most massive
        asteroids are excluded. Of the remaining asteroids, the top n_MBA most
        massive asteroids are included in the search process. Maximum value of
        n_MBAs_search: 396 - n_MBAs_already_included
    n_MBAs_already_included: int
        Number of most massive MBAs to exclude from the search process (e.g. if
        the 8 most massive MBAs are included in the dynamics anyway, we don't
        need to search for close passes with these 8)
    search_period_initial_time: tudatpy.kernel.astro.time_conversion.DateTime
        Tudat DateTime specifying the start of the search period for close passes
    n_orbits: float
        Duration of search period in number of orbits of the target body
    timestep_DAY: float
        Timestep used for querying JPL Horizons (i.e. time resolution of search)
        [days]
    close_pass_threshold_AU: float
        Distance threshold that defines what is considered a close pass [AU].
        Default value is 1.0 AU
    verbose: bool
        Determines whether certain print statements are executed (e.g. with
        results of close pass search)

    Returns
    ----------
    close_passers: dict
        Dictionary where the keys are the designation numbers of the asteroids
        that have close passes with the target body during the search period and
        each value is the minimum distance recorded between the target body and
        the close passing asteroid during the search period
    """

    # Runtime start
    T_START_overall = tm.time()

    # Function that prints args if verbose=True and does nothing if verbose=False
    print_verbose = print if verbose else lambda *a, **k: None
    print_verbose(f"Commencing search for close passes with target body... \n")

    # Initialise
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"

    target_body = query_SBDB(target_body_name)
    target_int = int(target_body['designation_number'])
    search_period_duration_SEC = n_orbits * (target_body['orbitperiod_DAY'] * cnst.JULIAN_DAY)
    search_period_initial_time_J2000 = search_period_initial_time.epoch() # in seconds since J2000
    search_period_final_time_J2000 = search_period_initial_time_J2000 + search_period_duration_SEC

    # Load relevant bodies from SiMDA file
    bodies_SiMDA = (
        pd.read_csv(SiMDA_file)
        .assign(NUM=lambda x: np.int32(x.NUM)) # this produces a warning because the NUM value is empty for some rows
        .query("DYN != 'COM' & DYN != 'TNO'") # filter out comets and trans-Neptunian objects (COM, TNO)
        .query("NUM != @target_int") # remove target body, if present
        .sort_values(by='MASS', ascending=False) # sort bodies by mass, descending order
        .loc[:, ["NUM", "DESIGNATION", "DIAM", "DYN", "MASS"]] # filter out irrelevant data columns
        [n_MBAs_already_included:(n_MBAs_already_included+n_MBAs_search)] # take first n_MBAs_search only, excluding n_MBAs_already_included
    )
    bodies_SiMDA.sort_values(by='NUM') # re-sort by designation number
    largest_MBAs = bodies_SiMDA.NUM.to_list()


    # Retrieve relevant body ephemerides from JPL Horizons
    T_START_JPL_queries = tm.time()
    print_verbose("Retrieving ephemerides from JPL Horizons for close-pass search...")
    interpolator_buffer_DAY = (timestep_DAY) * 2 # Extra buffer needed to ensure interpolator doesn't request epoch outside specified range

    # - Ephemerides for SiMDA asteroids
    MBA_ephemerides = {}
    for code in largest_MBAs:
        # print(code)
        query = HorizonsQuery(
            query_id = f"{code};",
            location = f"@{global_frame_origin}",
            epoch_start = search_period_initial_time_J2000 - interpolator_buffer_DAY*cnst.JULIAN_DAY,
            epoch_end = search_period_final_time_J2000 + interpolator_buffer_DAY*cnst.JULIAN_DAY,
            epoch_step = f"{int(timestep_DAY)}d",
            extended_query = True,
        )
        MBA_ephemerides[code] = query.cartesian(
            # frame_origin = global_frame_origin,
            frame_orientation = global_frame_orientation,
        )

    # - Ephemeris for target body
    target_ephemeris_query = HorizonsQuery(
        query_id = f"{target_body['designation_number']};",
        location = f"@{global_frame_origin}",
        epoch_start = search_period_initial_time_J2000 - interpolator_buffer_DAY*cnst.JULIAN_DAY,
        epoch_end = search_period_final_time_J2000 + interpolator_buffer_DAY*cnst.JULIAN_DAY,
        epoch_step = f"{int(timestep_DAY)}d",
        extended_query = True,
    )
    target_body_ephemeris = target_ephemeris_query.cartesian(
        # frame_origin = global_frame_origin,
        frame_orientation = global_frame_orientation,
    )
    T_END_JPL_queries = tm.time()
    print_verbose(f"Done. ({round(T_END_JPL_queries - T_START_JPL_queries,2)}s) \n")


    # Search for close passes
    T_START_close_pass_search = tm.time()
    print_verbose("Searching for close passes...")

    # - Initialise dictionary of distances between target and MBAs
    MBA_distances_AU = {key: np.zeros(np.shape(target_body_ephemeris)[0]) for key in largest_MBAs}
    close_passers = {} # records which MBAs pass within the threshold distance of the target body and the corresponding closest pass distance (AU)

    # - Perform search
    for idx, target_body_7state in enumerate(target_body_ephemeris):
        epoch = target_body_7state[0]
        target_body_state = target_body_7state[1:]

        for code in largest_MBAs:
            MBA_state = MBA_ephemerides[code][idx][1:]
            dist_MBA_to_target_body_AU = np.linalg.norm(target_body_state[:3] - MBA_state[:3]) / cnst.ASTRONOMICAL_UNIT
            MBA_distances_AU[code][idx] = dist_MBA_to_target_body_AU

            if dist_MBA_to_target_body_AU < close_pass_threshold_AU:
                if code not in close_passers: # add a newly discovered close passer
                    close_passers[code] = dist_MBA_to_target_body_AU
                elif close_passers[code] > dist_MBA_to_target_body_AU: # update the minimum distance
                    close_passers[code] = dist_MBA_to_target_body_AU
    T_END_close_pass_search = tm.time()
    print_verbose(f"Done. ({round(T_END_close_pass_search - T_START_close_pass_search,2)}s) \n")


    # Print results
    if verbose:
        propagation_final_time = search_period_initial_time.add_seconds( search_period_duration_SEC )

        width1 = 12
        width2 = 12
        width3 = 50
        print(f"""{target_body['longname']} CLOSE PASSES:
• Number of MBAs checked  : {n_MBAs_search}
• Close pass threshold    : {close_pass_threshold_AU} AU
• Search period start     : {search_period_initial_time}
• Search period end       : {propagation_final_time}
• Search period duration  : {search_period_duration_SEC / cnst.JULIAN_DAY:.2f} days ({n_orbits} orbits)
• Runtime                 : {round(T_END_close_pass_search - T_START_overall,3)} seconds
RESULTS:
{"#":>{width1}} {"ASTEROID":<{width2}} : DISTANCE OF CLOSEST PASS [AU]"""
        )

        close_passers_sorted = sorted(close_passers.items(), key=operator.itemgetter(1))
        for close_pass in close_passers_sorted:
            common_name = bodies_SiMDA.loc[bodies_SiMDA['NUM'] == close_pass[0], 'DESIGNATION'].iloc[0]
            print(f"{close_pass[0]:>{width1}} {common_name:<{width2}} : {close_pass[1]:.6f}")


    # Return
    return close_passers
    # return close_passers, T_END_close_pass_search - T_START_overall
#
def search_for_close_passes_with_Kepler_elements(
    target_body_name: str,          # designation number, common name, or provisional designation of asteroid
    n_MBAs_search: int,             # number of MBAs to include when searching for close passes (takes the n_MBAs_search most massive MBAs after excluding the n_MBAs_already_included most massive MBAs). Maximum: 396 - n_MBAs_already_included (as of 11 Jun 2025)
    n_MBAs_already_included: int,   # number of most massive MBAs to exclude from the search process (e.g. if the 8 most massive MBAs are included in the dynamics anyway, we don't need to search for close passes with these 8)
    search_period_initial_time: DateTime, # Tudat DateTime specifying the start of the search period for close passes
    n_orbits: float,                # duration of search period in number of orbits of the target body
    timestep_DAY: float,              # timestep used for querying JPL Horizons (i.e. time resolution of search) [days]
    close_pass_threshold_AU = 1.0,  # distance threshold that defines what is considered a close pass [AU]
    verbose = False,                # choose True to include additional print statements, e.g. with results of close pass search
):
    """
    Faster alternative to search_for_close_passes_with_JPL_Horizons(). Instead
    of retrieving ephemerides from JPL Horizons for all asteroids for the entire
    search period (which can be quite slow), this function retrieves the Kepler
    elements of all asteroids from JPL SBDB (or from local files if available),
    and manually calculates the Kepler and Cartesian states at all epochs to
    find any close passes between the target body and any other asteroids. This
    may be less accurate than search_for_close_passes_with_JPL_Horizons() due to
    assuming ideal elliptical Kepler orbits.
        The purpose of the function is to help in identifying close passes which
    may need to be included in the dynamic model. The SiMDA database is still
    used for getting asteroid masses.

    Parameters
    ----------
    target_body_name: str
        Subject of the search process (designation number, common name, or
        provisional designation of asteroid). This string will be used in the
        astroquery to JPL SBDB
    n_MBAs_search: int
        Number of MBAs to include in the search process. The SiMDA catalogue is
        first sorted by mass, then the top n_MBAs_already_included most massive
        asteroids are excluded. Of the remaining asteroids, the top n_MBA most
        massive asteroids are included in the search process. Maximum value of
        n_MBAs_search: 396 - n_MBAs_already_included
    n_MBAs_already_included: int
        Number of most massive MBAs to exclude from the search process (e.g. if
        the 8 most massive MBAs are included in the dynamics anyway, we don't
        need to search for close passes with these 8)
    search_period_initial_time: tudatpy.kernel.astro.time_conversion.DateTime
        Tudat DateTime specifying the start of the search period for close passes
    n_orbits: float
        Duration of search period in number of orbits of the target body
    timestep_DAY: float
        Timestep used for querying JPL Horizons (i.e. time resolution of search)
        [days]
    close_pass_threshold_AU: float
        Distance threshold that defines what is considered a close pass [AU].
        Default value is 1.0 AU
    verbose: bool
        Determines whether certain print statements are executed (e.g. with
        results of close pass search)

    Returns
    ----------
    close_passers: dict
        Dictionary where the keys are the designation numbers of the asteroids
        that have close passes with the target body during the search period and
        each value is the minimum distance recorded between the target body and
        the close passing asteroid during the search period
    """

    # Runtime start
    T_START_overall = tm.time()

    # Function that prints args if verbose=True and does nothing if verbose=False
    print_verbose = print if verbose else lambda *a, **k: None
    print_verbose(f"Commencing search for close passes with target body... \n")

    sun_gravitational_parameter_SI = 1.327124400420322e+20  # [m^3 s^-2], from SPICE !! JPL SBDB might use a slightly different value for this

    target_body = query_SBDB(target_body_name)
    target_int = int(target_body['designation_number'])
    search_period_duration_SEC = n_orbits * (target_body['orbitperiod_DAY'] * cnst.JULIAN_DAY)
    search_period_initial_time_J2000 = search_period_initial_time.epoch()  # 1 Jan 2010 in seconds since J2000
    search_period_final_time_J2000 = search_period_initial_time_J2000 + search_period_duration_SEC

    # Load relevant bodies from SiMDA file
    bodies_SiMDA = (
        pd.read_csv(SiMDA_file)
        .assign(NUM=lambda x: np.int32(x.NUM)) # this produces a warning because the NUM value is empty for some rows
        .query("DYN != 'COM' & DYN != 'TNO'") # filter out comets and trans-Neptunian objects (COM, TNO)
        .query("NUM != @target_int") # remove target body, if present
        .sort_values(by='MASS', ascending=False) # sort bodies by mass, descending order
        .loc[:, ["NUM", "DESIGNATION", "DIAM", "DYN", "MASS"]] # filter out irrelevant data columns
        [n_MBAs_already_included:(n_MBAs_already_included+n_MBAs_search)] # take first n_MBAs_search only, excluding n_MBAs_already_included
    )
    bodies_SiMDA.sort_values(by='NUM') # re-sort by designation number
    MBAs = bodies_SiMDA.NUM.to_list()

    # Initialise variables for storing search outputs
    T_START_SBDB_queries = tm.time()
    print_verbose("Retrieving body data from JPL SBDB for close-pass search...")
    all_bodies = [target_int] + MBAs
    all_bodies_data_dicts = {}
    search_epochs = np.arange(search_period_initial_time_J2000, search_period_final_time_J2000, (timestep_DAY * cnst.JULIAN_DAY))
    kepler_state_histories_dict = {key: np.zeros( (len(search_epochs), 7) ) for key in all_bodies}
    cartesian_state_histories_dict = {key: np.zeros( (len(search_epochs), 6) ) for key in all_bodies}
    distance_to_target_body_AU_histories_dict = {key: np.zeros( len(search_epochs) ) for key in all_bodies}

    # Define Kepler state at search_period_initial_time for all bodies
    for designation_number in all_bodies:
        # SBDB query
        body_data = query_SBDB(str(designation_number))

        # Give warning if a body's orbit was computed with a different reference system
        if body_data['equinox'] != "J2000":
            warnings.warn(f"Warning: Non-standard reference frame used in SBDB for orbit definition of body {designation_number}")

        # Calculate true anomaly for current MBA at search_period_initial_time
        time_perihelion_passage_J2000 = julian_day_to_seconds_since_epoch(body_data['time_perihelion_passage_JD'])
        delta_mean_anomaly_RAD = element_conversion.elapsed_time_to_delta_mean_anomaly(
            (search_period_initial_time_J2000 - time_perihelion_passage_J2000),
            sun_gravitational_parameter_SI,
            body_data['semimajoraxis_AU'] * cnst.ASTRONOMICAL_UNIT,
        )
        mean_anomaly_at_initial_time_RAD = delta_mean_anomaly_RAD % (2*np.pi)
        true_anomaly_at_initial_time_RAD = element_conversion.mean_to_true_anomaly(
            eccentricity = body_data['eccentricity'],
            mean_anomaly = mean_anomaly_at_initial_time_RAD,
        )

        # Save Kepler state (+ mean anomaly) of current MBA at search_period_initial_time into kepler_state_histories_dict
        kepler_state_at_initial_time = [
            body_data['semimajoraxis_AU'] * cnst.ASTRONOMICAL_UNIT,
            body_data['eccentricity'],
            np.radians(body_data['inclination_DEG']),
            np.radians(body_data['argument_periapsis_DEG']),
            np.radians(body_data['RA_ascendingnode_DEG']),
            true_anomaly_at_initial_time_RAD,
            mean_anomaly_at_initial_time_RAD,
        ]
        kepler_state_histories_dict[designation_number][0] = kepler_state_at_initial_time
        # Save Cartesian state of current MBA at search_period_initial_time into cartesian_state_histories_dict
        cartesian_state_at_initial_time = element_conversion.keplerian_to_cartesian(
            kepler_state_at_initial_time[:6],
            sun_gravitational_parameter_SI,
        )
        cartesian_state_histories_dict[designation_number][0] = cartesian_state_at_initial_time
        # Calculate and save distance between target body and current MBA at search_period_initial_time
        distance_to_target_body_at_initial_time_AU = np.linalg.norm(cartesian_state_at_initial_time[:3] - cartesian_state_histories_dict[target_int][0][:3]) / cnst.ASTRONOMICAL_UNIT
        distance_to_target_body_AU_histories_dict[designation_number][0] = distance_to_target_body_at_initial_time_AU
        # Save body_data dict
        all_bodies_data_dicts[designation_number] = body_data

    T_END_SBDB_queries = tm.time()
    print_verbose(f"Done. ({round(T_END_SBDB_queries - T_START_SBDB_queries,2)}s) \n")

    T_START_search = tm.time()
    print_verbose("Searching for close passes...")

    # Loop through search period with timestep_DAY
    for i, epoch in enumerate(search_epochs[1:]):
        i = i + 1 # skipping the initial epoch (basically means i is 1-indexed)
        # loop through all bodies
        for designation_number in all_bodies:
            # calculate true anomaly for current MBA at current epoch
            delta_mean_anomaly_RAD = element_conversion.elapsed_time_to_delta_mean_anomaly(
                (timestep_DAY * cnst.JULIAN_DAY),
                sun_gravitational_parameter_SI,
                all_bodies_data_dicts[designation_number]['semimajoraxis_AU'] * cnst.ASTRONOMICAL_UNIT,
            )
            mean_anomaly_at_current_epoch_RAD = kepler_state_histories_dict[designation_number][i-1][6] + delta_mean_anomaly_RAD
            true_anomaly_at_current_epoch_RAD = element_conversion.mean_to_true_anomaly(
                eccentricity = all_bodies_data_dicts[designation_number]['eccentricity'],
                mean_anomaly = mean_anomaly_at_current_epoch_RAD,
            )
            # save Kepler state (+ mean anomaly) at current epoch
            kepler_state_at_current_epoch = [
                kepler_state_histories_dict[designation_number][0][0],  # constant (semi-major axis)
                kepler_state_histories_dict[designation_number][0][1],  # constant (eccentricity)
                kepler_state_histories_dict[designation_number][0][2],  # constant (inclination)
                kepler_state_histories_dict[designation_number][0][3],  # constant (argument of periapsis)
                kepler_state_histories_dict[designation_number][0][4],  # constant (RAAN)
                true_anomaly_at_current_epoch_RAD,
                mean_anomaly_at_current_epoch_RAD,
            ]
            kepler_state_histories_dict[designation_number][i] = kepler_state_at_current_epoch
            # save Cartesian state at current epoch
            cartesian_state_at_current_epoch = element_conversion.keplerian_to_cartesian(
                kepler_state_at_current_epoch[:6],
                sun_gravitational_parameter_SI,
            )
            cartesian_state_histories_dict[designation_number][i] = cartesian_state_at_current_epoch
            # calculate distance between target body and current MBA at current epoch
            distance_to_target_body_AU = np.linalg.norm(cartesian_state_at_current_epoch[:3] - cartesian_state_histories_dict[target_int][i][:3]) / cnst.ASTRONOMICAL_UNIT
            distance_to_target_body_AU_histories_dict[designation_number][i] = distance_to_target_body_AU

    min_distances_to_target_body_AU = {
        designation_number: np.min(distance_to_target_body_AU_histories_dict[designation_number]) for designation_number in all_bodies[1:]
    }
    close_passers = {
        key: value for key, value in min_distances_to_target_body_AU.items() if value < close_pass_threshold_AU
    }

    T_END_search = tm.time()
    print_verbose(f"Done. ({round(T_END_search - T_START_search,2)}s) \n")

    # Print results
    if verbose:
        propagation_final_time = search_period_initial_time.add_seconds( search_period_duration_SEC )

        width1 = 12
        width2 = 12
        width3 = 50
        print(f"""{target_body['longname']} CLOSE PASSES:
• Number of MBAs checked  : {n_MBAs_search}
• Close pass threshold    : {close_pass_threshold_AU} AU
• Search period start     : {search_period_initial_time}
• Search period end       : {propagation_final_time}
• Search period duration  : {search_period_duration_SEC / cnst.JULIAN_DAY:.2f} days ({n_orbits} orbits)
• Runtime                 : {round(T_END_search - T_START_overall,3)} seconds
RESULTS: ({len(close_passers)} close passes found)
{"#":>{width1}} {"ASTEROID":<{width2}} : DISTANCE OF CLOSEST PASS [AU]"""
        )

        close_passers_sorted = sorted(close_passers.items(), key=operator.itemgetter(1))
        for close_pass in close_passers_sorted:
            common_name = bodies_SiMDA.loc[bodies_SiMDA['NUM'] == close_pass[0], 'DESIGNATION'].iloc[0]
            print(f"{close_pass[0]:>{width1}} {common_name:<{width2}} : {close_pass[1]:.6f}")


    return close_passers
    # return close_passers, T_END_search - T_START_overall




### Retrieve and save JPL Horizons data
def generate_spk_file(
    spkid: str,
    spk_file_path: str,
    spk_file_name: str,
    start_time = '1900-01-01',
    stop_time = '2100-01-01',
    overwrite_existing = False,
):
    """
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
    """

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
#
def retrieve_and_save_ephemeris_uncertainty(
    spkid: str,
    file_path: str,
    file_name: str,
    start_time = '1901-01-01',
    stop_time = '2030-01-01',
    time_step = '10d',
    overwrite_existing = False,
):
    """
    This function can be used to generate retrieve JPL Horizons ephemeris
    uncertainty information (in XYZ and RTN frames) and save it to a file using
    pickle

    Parameters
    ----------
    spkid: str
        Name of the body of whose ephemeris uncertainty data is to be generated.
        The string must be a spice-recognized name or ID.
    file_path: str
        Path to the folder to which the data file will be saved
    file_name: str
        File name of the generated ephemeris uncertainty file: [file_name].p
    start_time: str
        Date defining the start of the ephemeris uncertainty time interval (ET).
        Must be formatted as 'yyyy-mm-dd', for example '2025-09-25'
    stop_time: str
        Date defining the end of the ephemeris uncertainty time interval (ET).
        formatted as 'yyyy-mm-dd', for example '2025-09-25'
    time_step: str
        Defines the time interval between rows of the ephemeris uncertainty data
        table. Must be formatted as an integer followed by y, d, m, or s (for
        year, day, minute, or second), for example '10d' or '60m'
    overwrite_existing: bool
        If a file with name [file_name].p already exists in the folder given by
        file_path, it will only be overwritten if overwrite_existing is True.
        Otherwise, the file generation will be skipped (default behaviour)

    Returns
    ------
    None if file generation was successful or skipped. If a problem occurrs, the
    spkid input is returned.
    """

    file_name += ".p"
    if os.path.exists(file_path+file_name) and overwrite_existing==False:
        print("Skipping ephemeris uncertainty file generation to avoid overwrite.")
        return None

    # try:
    # Create HorizonsClass object
    HC_obj = HorizonsClass(
        spkid,
        location = f"@Sun",
        id_type = "designation",
        epochs = {
            "start": start_time,
            "stop": stop_time,
            "step": time_step,
        },
    )
    # Request Cartesian XYZ position and velocity uncertainties in ECLIPJ2000 frame, Sun-centered
    # These are implicitly 3-sigma uncertainties, see https://ssd.jpl.nasa.gov/horizons/manual.html#longterm:~:text=Output%20for%20asteroids%20and%20comets%20can%20include%20formal%20%2B%2F%2D%203%2Dstandard%2Ddeviation%20statistical%20orbit%20uncertainty%20quantities
    body_ephem_unc_table = HC_obj.vectors_uncertainty(
        # 2: State vector {x,y,z,Vx,Vy,Vz}
        # x: XYZ uncertainties
        # a: ACN uncertainties (along-track, cross-track, normal)
        # r: RTN uncertainties (radial, transverse, normal)
        # See: https://ssd-api.jpl.nasa.gov/doc/horizons.html#vec_table
        vector_table = "2xr",
        refplane = "earth",
    )

    # Convert table units to SI (from AU, AU/day, etc.)
    for col in body_ephem_unc_table.colnames:
        if col in ["targetname", "datetime_jd", "datetime_str", "H", "G", "ephemeris_time"]:
            continue
        body_ephem_unc_table[col] = body_ephem_unc_table[col].quantity.si

    # Save to file
    save_with_pickle(body_ephem_unc_table, file_path, file_name)
    print(f"Ephemeris uncertainty file saved to: {file_path + file_name}")
    return None

    # except as ex:




### File saving and loading
def save_with_pickle(obj_to_save, save_path, file_name):
    with open(f"{save_path}{file_name}", 'wb') as fp:
        pickle.dump(obj_to_save, fp, protocol=pickle.DEFAULT_PROTOCOL)
#
def load_with_pickle(save_path, file_name):
    with open(f"{save_path}{file_name}", 'rb') as fp:
        loaded_obj = pickle.load(fp)
    return loaded_obj
#
def save_estimation_output_attributes(estimation_output, data_save_path):
    # Some attributes are not saved, see explanations in
    # Activity_logs/data_to_save_info.txt
    # Lines that are commented out are because the file sizes of those
    # variables were large and their priority was low

    # Create directory
    EO_save_path = data_save_path + 'estimation_output/'
    if not os.path.exists(EO_save_path):
        os.makedirs(EO_save_path)

    # Attributes directly in EstimationOutput object
    eo = estimation_output
    save_with_pickle(eo.best_iteration, EO_save_path, "best_iteration.p")
    save_with_pickle(eo.consider_covariance, EO_save_path, "consider_covariance.p")
    save_with_pickle(eo.consider_covariance_contribution, EO_save_path, "consider_covariance_contribution.p")
    save_with_pickle(eo.consider_normalization_terms, EO_save_path, "consider_normalization_terms.p")
    save_with_pickle(eo.correlations, EO_save_path, "correlations.p")
    save_with_pickle(eo.covariance, EO_save_path, "covariance.p")
    save_with_pickle(eo.covariance_with_consider_parameters, EO_save_path, "covariance_with_consider_parameters.p")
    # save_with_pickle(eo.design_matrix, EO_save_path, "design_matrix.p")
    save_with_pickle(eo.design_matrix_consider_parameters, EO_save_path, "design_matrix_consider_parameters.p")
    save_with_pickle(eo.exception_during_inversion, EO_save_path, "exception_during_inversion.p")
    save_with_pickle(eo.exception_during_propagation, EO_save_path, "exception_during_propagation.p")
    save_with_pickle(eo.final_parameters, EO_save_path, "final_parameters.p")
    save_with_pickle(eo.final_residuals, EO_save_path, "final_residuals.p")
    save_with_pickle(eo.formal_errors, EO_save_path, "formal_errors.p")
    save_with_pickle(eo.inverse_covariance, EO_save_path, "inverse_covariance.p")
    save_with_pickle(eo.inverse_normalized_covariance, EO_save_path, "inverse_normalized_covariance.p")
    save_with_pickle(eo.normalization_terms, EO_save_path, "normalization_terms.p")
    save_with_pickle(eo.normalized_covariance, EO_save_path, "normalized_covariance.p")
    save_with_pickle(eo.normalized_covariance_with_consider_parameters, EO_save_path, "normalized_covariance_with_consider_parameters.p")
    # save_with_pickle(eo.normalized_design_matrix, EO_save_path, "normalized_design_matrix.p")
    save_with_pickle(eo.normalized_design_matrix_consider_parameters, EO_save_path, "normalized_design_matrix_consider_parameters.p")
    save_with_pickle(eo.parameter_history, EO_save_path, "parameter_history.p")
    save_with_pickle(eo.residual_history, EO_save_path, "residual_history.p")
    # save_with_pickle(eo.weighted_design_matrix, EO_save_path, "weighted_design_matrix.p")
    # save_with_pickle(eo.weighted_normalized_design_matrix, EO_save_path, "weighted_normalized_design_matrix.p")

    # Attributes of each estimation iteration
    for iter_idx, var_sim_result in enumerate(eo.simulation_results_per_iteration):
        # var_sim_result is a SingleArcVariationalSimulationResults object
        iter_save_path = EO_save_path + f'simulation_results_per_iteration_{iter_idx}/'
        if not os.path.exists(iter_save_path):
            os.makedirs(iter_save_path)
        # save_with_pickle(var_sim_result.sensitivity_matrix_history, iter_save_path, "sensitivity_matrix_history.p")
        # save_with_pickle(var_sim_result.state_transition_matrix_history, iter_save_path, "state_transition_matrix_history.p")

        # dyn_result is a SingleArcSimulationResults object
        dyn_result = var_sim_result.dynamics_results
        dyn_result_save_path = iter_save_path + f'dynamics_results/'
        if not os.path.exists(dyn_result_save_path):
            os.makedirs(dyn_result_save_path)
        # save_with_pickle(dyn_result.cumulative_computation_time_history, dyn_result_save_path, "cumulative_computation_time_history.p")
        # save_with_pickle(dyn_result.cumulative_number_of_function_evaluations_history, dyn_result_save_path, "cumulative_number_of_function_evaluations_history.p")
        save_with_pickle(dyn_result.dependent_variable_history, dyn_result_save_path, "dependent_variable_history.p")
        save_with_pickle(dyn_result.dependent_variable_history_time_object, dyn_result_save_path, "dependent_variable_history_time_object.p")
        save_with_pickle(dyn_result.dependent_variable_ids, dyn_result_save_path, "dependent_variable_ids.p")
        save_with_pickle(dyn_result.initial_and_final_times, dyn_result_save_path, "initial_and_final_times.p")
        save_with_pickle(dyn_result.integration_completed_successfully, dyn_result_save_path, "integration_completed_successfully.p")
        # save_with_pickle(dyn_result.ordered_dependent_variable_settings, dyn_result_save_path, "ordered_dependent_variable_settings.p")
        save_with_pickle(dyn_result.processed_state_ids, dyn_result_save_path, "processed_state_ids.p")
        save_with_pickle(dyn_result.propagated_state_ids, dyn_result_save_path, "propagated_state_ids.p")
        save_with_pickle(dyn_result.propagated_state_vector_length, dyn_result_save_path, "propagated_state_vector_length.p")
        save_with_pickle(dyn_result.propagation_is_performed, dyn_result_save_path, "propagation_is_performed.p")
        save_with_pickle(dyn_result.state_history, dyn_result_save_path, "state_history.p")
        # save_with_pickle(dyn_result.state_history_time_object, dyn_result_save_path, "state_history_time_object.p")
        save_with_pickle(dyn_result.total_computation_time, dyn_result_save_path, "total_computation_time.p")
        save_with_pickle(dyn_result.total_number_of_function_evaluations, dyn_result_save_path, "total_number_of_function_evaluations.p")
        # save_with_pickle(dyn_result.unordered_dependent_variable_settings, dyn_result_save_path, "unordered_dependent_variable_settings.p")
        # save_with_pickle(dyn_result.unprocessed_state_history, dyn_result_save_path, "unprocessed_state_history.p")
        # save_with_pickle(dyn_result.unprocessed_state_history_time_object, dyn_result_save_path, "unprocessed_state_history_time_object.p")
#
def save_observation_collection_attributes(observation_collection, data_save_path):
    # Some attributes are not saved, see explanations in
    # Activity_logs/data_to_save_info.txt
    # Lines that are commented out are because they contain objects that are
    # not pickle-able (and due to large file size for oc.concatenated_times_objects)

    # Create directory
    OC_save_path = data_save_path + 'observation_collection/'
    if not os.path.exists(OC_save_path):
        os.makedirs(OC_save_path)

    # Attributes in ObservationCollection object
    oc = observation_collection
    save_with_pickle(oc.concatenated_link_definition_ids, OC_save_path, "concatenated_link_definition_ids.p")
    save_with_pickle(oc.concatenated_observations, OC_save_path, "concatenated_observations.p")
    save_with_pickle(oc.concatenated_times, OC_save_path, "concatenated_times.p")
    # save_with_pickle(oc.concatenated_times_objects, OC_save_path, "concatenated_times_objects.p")
    save_with_pickle(oc.concatenated_weights, OC_save_path, "concatenated_weights.p")
    # save_with_pickle(oc.link_definition_ids, OC_save_path, "link_definition_ids.p")
    # save_with_pickle(oc.link_definitions_per_observable, OC_save_path, "link_definitions_per_observable.p")
    # save_with_pickle(oc.link_ends_per_observable_type, OC_save_path, "link_ends_per_observable_type.p")
    save_with_pickle(oc.observable_type_start_index_and_size, OC_save_path, "observable_type_start_index_and_size.p")
    save_with_pickle(oc.observation_set_start_index_and_size, OC_save_path, "observation_set_start_index_and_size.p")
    save_with_pickle(oc.observation_vector_size, OC_save_path, "observation_vector_size.p")
    # save_with_pickle(oc.sorted_observation_sets, OC_save_path, "sorted_observation_sets.p")
    save_with_pickle(oc.sorted_per_set_time_bounds, OC_save_path, "sorted_per_set_time_bounds.p")
    save_with_pickle(oc.time_bounds, OC_save_path, "time_bounds.p")
    save_with_pickle(oc.time_bounds_time_object, OC_save_path, "time_bounds_time_object.p")
#
def save_batch_attributes(batch, data_save_path):
    # Create directory
    batch_save_path = data_save_path + 'batch/'
    if not os.path.exists(batch_save_path):
        os.makedirs(batch_save_path)

    # Attributes in BatchMPC object
    b = batch
    save_with_pickle(b._MPC_space_telescopes, batch_save_path, "_MPC_space_telescopes.p")
    save_with_pickle(b.size, batch_save_path, "size.p")
    save_with_pickle(b.epoch_end, batch_save_path, "epoch_end.p")
    save_with_pickle(b.epoch_start, batch_save_path, "epoch_start.p")
    save_with_pickle(b.MPC_objects, batch_save_path, "MPC_objects.p")
    save_with_pickle(b.bands, batch_save_path, "bands.p")
    save_with_pickle(b.observatories, batch_save_path, "observatories.p")
    save_with_pickle(b.space_telescopes, batch_save_path, "space_telescopes.p")
    save_with_pickle(b.bodies_created, batch_save_path, "bodies_created.p")
    save_with_pickle(b.table, batch_save_path, "table.p")
    # Outputs of useful BatchMPC methods
    save_with_pickle(b.observatories_table(), batch_save_path, "observatories_table.p")




### Read Yarkovsky data
def load_Fenucci_etal2024_tabB1():
    Fenucci_etal2024_tabB1 = (
        pd.read_fwf(
            yark_data_dir + "Fenucci-et-al-2024_tableB1.dat",
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
    return Fenucci_etal2024_tabB1
#
def load_Greenberg_etal2020_tab():
    Greenberg_etal2020_tab = pd.read_fwf(
        yark_data_dir + "Greenberg-at-al-2020_Yarkovsky.dat",
        widths = [
            7, 17, 4, 5, 4, 2, 5, 6, 3, 8, 6, 6, 8, 6, 6, 5, 5, 9
        ],
        names = [           # From Greenberg-et-al-2020_Yarkovsky.txt
            'desig',        # ---       Designation number
            'name',         # ---       Common name / provisional designation
            'f_Name',       # ---       Flag(s) on name (B = Binary or triple asteroid; S = Anomalously high {xi} value, {xi} > 0.5; D = Yarkovsky detection predicted to be weaker due to the time span or quantity of astrometry (Section 5.1))
            'a',            # AU        Semi-major axis
            'e',            # ---       Orbital eccentricity
            'f_D',          # ---       Flag on D (* = Diameter inferred from H-magnitude via Equation 13 when the taxonomic type is not available from the Small Body Database. T = Diameter inferred from H-magnitude via Equation 13 when the taxonomic type is available from the Small Body Database.)
            'D',            # km        Diameter
            'No',           # ---       Number of optical measurements used in the solution
            'Nr',           # ---       Number of radar measurements used in the solution
            'da/dt',        # 10-4AU/My Orbit-averaged drift in semi-major axis (based on optical observations only)
            'e_da/dt',      # 10-4AU/My One-standard-deviation in da/dt (based on optical observations only)
            'p',            # 10-4AU/My p-value (based on optical observations only). Used in distinguishing between models using optical data only (p) and optical plus radar data (p_r)
            'da/dt_r',      # 10-4AU/My Orbit-averaged drift in semi-major axis (based on optical and radar observations (Nr > 0), otherwise blank)
            'e_da/dt_r',    # 10-4AU/My One-standard-deviation in da/dt_r (based on optical and radar observations (Nr > 0), otherwise blank)
            'p_r',          # 10-4AU/My p-value (based on optical and radar observations (Nr > 0), otherwise blank). Used in distinguishing between models using optical data only (p) and optical plus radar data (p_r)
            'sY',           # ---       Yarkovsky sensitivity parameter (Of Nuget et al. (2012) [2012AJ....144...60N])
            'xi',           # ---       Yarkovsky efficiency (Which was computed with a bulk density that was extracted from the Small Body Database, if available, or inferred from the spectral type, if available (Section 10))
            'Arc'           # yr        Length of observational arc
        ]
    )
    return Greenberg_etal2020_tab
#
def get_Yarkovsky_parameter(body, verbose):
    Fenucci_etal2024_tabB1 = load_Fenucci_etal2024_tabB1()
    # Greenberg_etal2020_tab = load_Greenberg_etal2020_tab()

    # Try to get Yarkovsky parameter (A2) from Fenucci et al 2024 table B1
    try:
        # Try first with body's permanent designation number
        Fenucci_body_data = Fenucci_etal2024_tabB1.loc[
            Fenucci_etal2024_tabB1['Asteroid'] == body['designation_number']
        ]
        A2_target_body_AU_per_DAYsq = Fenucci_body_data['A2'].item()
        yark_source = "Fenucci et al. (2024) table B1"

    except ValueError:
        # If no match with permanent designation number, try provisional
        # designation number
        Fenucci_body_data = Fenucci_etal2024_tabB1.loc[
            Fenucci_etal2024_tabB1['Asteroid'] == body['provisional_designation']
        ]
        A2_target_body_AU_per_DAYsq = Fenucci_body_data['A2'].item()
        yark_source = "Fenucci et al. (2024) table B1"

    except KeyError as ex:
        ex.add_note(f"No A2 value found in Fenucci_et_al_2024_tabB1 for '{body['longname']}'")
        raise

        # !!! If not match in Fenucci table, search for body in Greenberg table?
        # Then convert da/dt (or da/dt_r) from 1e-4 AU/My

        # Farnocchia et al. (2013) has A2 values for 54509 YORP and 1620 Geographos,
        # which Fenucci et al (2024) is missing

        # if estimate_A2:
        #     # If target body not found in table(s), set a default value for A2 estimation
        #     print(f"Using default A2 value of -1e-13 [m/s^2] for initial guess")
        #     A2_target_body = -1e-13
        #     yark_source = "default"
        # else:
        #     print_default(f"Turning on A2 estimation and using default A2 value of -1e-13 [m/s^2] for initial guess")
        #     estimate_A2 = True

    # Convert from [AU/day^2] to SI [m/s^2]
    A2_target_body = A2_target_body_AU_per_DAYsq * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2)
    A2_target_body_unc = Fenucci_body_data['e_A2'].item() * (cnst.ASTRONOMICAL_UNIT / cnst.JULIAN_DAY**2)
    if verbose:
        print(f"""YARKOVSKY INFORMATION FOR: {body['longname'].upper()}
    Source:                 : {yark_source}
    A2 parameter [m/s^2]    : {A2_target_body:.4g} +/- {A2_target_body_unc:.4g}
    Signal-to-noise ratio   : {Fenucci_body_data['SNR'].item():.4g}
    Observation arc [years] : {Fenucci_body_data['deltat'].item():.4g}""")
    return A2_target_body, A2_target_body_unc, Fenucci_body_data, yark_source




### Environment setup
def define_large_body_accelerations(EIH_needed):
    if EIH_needed:
        # Large body accelerations with Einstein-Infeld-Hoffman (LB+EIH) (Sun + 8 planets + Moon + EIH)
        accelerations_large_bodies = {
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
    else:
        # Large body accelerations (LB) (Sun + 8 planets + Moon)
        accelerations_large_bodies = {
            "Sun": [
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True),
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
    return accelerations_large_bodies
#
def set_propagation_start_from_first_observation_epoch(first_obs_epoch_J2000):
    # Set start time of propagation to beginning of day before first observation
    # (min. 24 hours back, max. 48 hours back)
    propagation_start_DateTime = DateTime.from_epoch(first_obs_epoch_J2000).add_days(-1)
    propagation_start_DateTime = DateTime(
        propagation_start_DateTime.year,
        propagation_start_DateTime.month,
        propagation_start_DateTime.day,
        0, # hour
        0, # minute
    )
    propagation_start_J2000 = propagation_start_DateTime.epoch() # in seconds since J2000
    return propagation_start_DateTime, propagation_start_J2000
#
def set_propagation_end_from_last_observation_epoch(last_obs_epoch_J2000):
    # Set start time of propagation to end of day after last observation
    # (min. 24 hours forward, max. 48 hours forward)
    propagation_end_DateTime = DateTime.from_epoch(last_obs_epoch_J2000).add_days(2)
    propagation_end_DateTime = DateTime(
        propagation_end_DateTime.year,
        propagation_end_DateTime.month,
        propagation_end_DateTime.day,
        0, # hour
        0, # minute
    )
    propagation_end_J2000 = propagation_end_DateTime.epoch() # in seconds since J2000
    return propagation_end_DateTime, propagation_end_J2000



### Propagate body chosen integrator and range of
def propagate_body_with_integrator(
    # Target body
    body_to_propagate: str,     # designation number, common name, or provisional designation of asteroid
    # Integrator
    integrator: str,            # see "Options for 'integrator'" below for formatting
    step_size_hi_power: int,    # max. Δt = 2^step_size_hi_power [days]
    step_size_lo_power: int,    # min. Δt = 2^step_size_lo_power [days]
    # Observations
    obs_range_start: list[int] = [1900, 1, 1],  # start of time range to consider for astrometric observations of body_to_propagate. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.
    obs_range_end: list[int] = [2026, 7, 1],    # end of time range to consider for astrometric observations of body_to_propagate. Format is [year, month, day, hour, minute, second, microsecond]. First 3 elements are mandatory, rest are optional.
    # Dynamic model
    dynamic_model_def: str = "LB+LMBA+AB+Yark", # define the acceleration model (start with "LB", add "+EIH", "+LMBA", "+CP", "+AB", "+Yark" as desired). "+CP" NOT RECOMMENDED!
    n_LMBAs: int = 8,                          # number of MBAs whose gravitation is to be included in dynamics (takes the n_LMBAs most massive MBAs). Only relevant if 'LMBA' is included in dynamic_model_def
    # Frames
    central_body: str = "Sun",
    global_frame_origin: str = "SSB",
    global_frame_orientation: str = "J2000",
    # Plotting & Printing
    verbose: bool = False,                          # choose True to include additional supplementary print statements
    quiet: bool = False,                            # choose True to suppress all print statements
    colormode: int = 0,                             # 0 for lightmode, 1 for darkmode when showing plots (both colormodes are saved when plot_behaviour is 'save' anyway)
    plot_behaviour: str = 'save',                   # 'save' to save all plots as PDFs. 'show' to display them. 'both' to save and display. 'showboth' to display plots in both light and darkmode. 'skip' to skip all plotting
    integrator_selection_dir: str = "integrator-selection/", # base directory for saving run outputs (plots and data)
    timestamp_dir: str = None,                      # Timestamped directory in which to save outputs. If None, results are saved into a newly created timestamped directory
    save_state_histories_to_files: bool = False,    # choose True to save state histories of each time step's propagation to a file using pickle
):
    """
    This script propagates the motion of a chosen body for the period within
    [obs_range_start, obs_range_end] for which MPC observations of this object
    are available, using the selected numerical integration method and time
    steps (Δt), as determined by the step_size_hi_power and step_size_lo_power
    inputs.

    PLOTS:
    - Plot 1: Position error over time. An error curve is plotted for each
      propagation except for the one that used the smallest Δt. Error for a
      propagation with time step Δt is approximated as the position difference
      compared to the propagation that used Δt/2
    - Plot 2: Max. and RMS of position error over time (as calculated for Plot
      1) vs Δt

    Parameters
    ----------
    body_to_propagate: str
        Asteroid to propagate (designation number, common name, or provisional
        designation of asteroid). This string will be used in the astroquery to
        JPL SBDB
    integrator: str
        Numerical integration method to use (not case sensitive). Only fixed
        step integration methods allowed. See process_integrator_input() for
        formatting and available methods
    step_size_hi_power: int
        Highest Δt used in propagations is Δt = 2^step_size_hi_power [days]
    step_size_lo_power: int
        Lowest Δt used in propagations is Δt = 2^step_size_lo_power [days]
    obs_range_start: list[int] = [1900, 1, 1]
        Start of time range to consider for astrometric observations of
        body_to_propagate. Format is [year, month, day, hour, minute, second,
        microsecond]. First 3 elements are mandatory, rest are optional.
    obs_range_end: list[int] = [2026, 7, 1]
        End of time range to consider for astrometric observations of
        body_to_propagate. Format is [year, month, day, hour, minute, second,
        microsecond]. First 3 elements are mandatory, rest are optional.
    dynamic_model_def: str = "LB+LMBA+AB+Yark"
        Definition of the acceleration model to use (start with "LB", add
        "+EIH", "+LMBA", "+AB", "+Yark" as desired). Note: "+CP" not available
    n_LMBAs: int = 8
        Number of MBAs whose gravitation is to be included in dynamics (takes
        the n_LMBAs most massive MBAs). Only relevant if 'LMBA' is included in
        dynamic_model_def
    central_body: str = "Sun"
        Central body for propagation
    global_frame_origin: str = "SSB"
        Frame origin for propagation
    global_frame_orientation: str = "J2000"
        Frame orientation for propagation
    verbose: bool = False
        If True, include additional print statements
    quiet: bool = False
        If True, suppress all print statements
    colormode: int = 0
        0 for lightmode, 1 for darkmode when showing plots (both colormodes are
        saved when plot_behaviour is 'save' anyway)
    plot_behaviour: str = 'save'
        'save' to save all plots as PDFs. 'show' to display them. 'both' to save
        and display. 'showboth' to display plots in both light and darkmode
    integrator_selection_dir: str = "integrator-selection/"
        Base directory within "output_data/" and/or "plots/" for saving run
        outputs (plots and data)
    save_state_histories_to_files: bool = False
        If True, state histories of each time step's propagation are saved to a
        file using pickle

    Returns
    ------
    body: dict
        Dictionary containing information on the selected body, as taken from
        JPL SBDB
    function_evaluations: np.ndarray
        Number of function evaluations performed for each numerical propagation
    epoch_sets: list of np.ndarray
        Epochs at which the state was evaluated; one set for each numerical
        propagation (except the one with the smallest Δt). None if plotting
        turned off
    pos_error_mag_sets:
        Position error at each epoch. Calculated w.r.t. the numerical
        propagation that use a timestep half as large. One set for each
        numerical propagation (except the one with the smallest Δt). None if
        plotting turned off
    pos_error_RMSs:
        RMS of position error over time. One value for each numerical
        propagation (except the one with the smallest Δt). None if plotting
        turned off
    """


    ######################## 2. Input processing ###############################
    #region 2. Input processing

    ## Printing setup
    if verbose and quiet:
        raise ValueError("Verbose mode and quiet mode have both been selected. Choose one or neither.")
    # Function that prints args if quiet=False and does nothing if quiet=True
    print_default = print if not quiet else lambda *a, **k: None
    # Function that prints args if verbose=True and does nothing if verbose=False
    print_verbose = print if verbose else lambda *a, **k: None
    T_START_setup = tm.perf_counter()

    ## Get body data from JPL SBDB and print selected info
    body = query_SBDB(body_to_propagate, print_body_info=verbose)
    body_dir = f"{body['filing_name_long']}/"

    ## Integrator
    integrator = format_integrator_string(integrator)
    coefficient_set, order_to_use = process_integrator_input(integrator, print_integrator_info=verbose)
    # List of Δt
    step_sizes_DAY = np.logspace(
        step_size_lo_power, # must be increasing, must be doubling
        step_size_hi_power,
        step_size_hi_power - step_size_lo_power + 1,
        base = 2,
    )
    print_default(f"SETTING UP PROPAGATIONS... | {integrator} | {body['longname']}")

    ## Observation range time variables
    obs_range_start_J2000 = DateTime(*obs_range_start).epoch() # in seconds since J2000
    obs_range_end_J2000 = DateTime(*obs_range_end).epoch() # in seconds since J2000

    ## Process dynamic model definition
    EIH_needed = True if "EIH" in dynamic_model_def else False
    LMBAs_needed = True if "LMBA" in dynamic_model_def else False
    n_LMBAs = 0 if LMBAs_needed == False else n_LMBAs # set n_LMBAs to 0 if not needed
    ABs_needed = True if "AB" in dynamic_model_def else False
    Yark_needed = True if "Yark" in dynamic_model_def else False

    ## Yarkovsky parameter
    if Yark_needed:
        A2_target_body, A2_target_body_unc, _, _ = get_Yarkovsky_parameter(body, verbose)

    ## Plotting behaviour
    if plot_behaviour == 'save':
        colormode_list = [1,0]
        plot_behaviour_list = ['save', 'save']
    elif plot_behaviour == 'showboth':
        colormode_list = [1,0]
        plot_behaviour_list = ['show', 'show']
    elif plot_behaviour == 'both':
        colormode_list = [colormode,1,0]
        plot_behaviour_list = ['show', 'save', 'save']
    elif plot_behaviour == 'skip':
        colormode_list = []
        plot_behaviour_list = []
    elif plot_behaviour == 'show':
        colormode_list = [colormode]
        plot_behaviour_list = ['show']
    else:
        raise ValueError("Invalid input for plot_behaviour.")

    ## Create dictionary of all relevant inputs, to be saved to a file later
    input_dict = {
        'body_to_propagate': body_to_propagate,
        # Observations
        'obs_range_start': obs_range_start,
        'obs_range_end': obs_range_end,
        # Dynamic model
        'dynamic_model_def': dynamic_model_def,
        'n_LMBAs': n_LMBAs,
        'A2_target_body': A2_target_body,
        'A2_target_body_unc': A2_target_body_unc,
        # Frames
        'central_body': central_body,
        'global_frame_origin': global_frame_origin,
        'global_frame_orientation': global_frame_orientation,
        # Integrator
        'integrator': integrator,
        'step_size_lo_power': step_size_lo_power,
        'step_size_hi_power': step_size_hi_power,
        'step_sizes_DAY': list(step_sizes_DAY),
        # Plotting & printing
        'verbose': verbose,
        'quiet': quiet,
        'colormode': colormode,
        'plot_behaviour': plot_behaviour,
        'integrator_selection_dir': integrator_selection_dir,
        'save_state_histories_to_files': save_state_histories_to_files,
    }


    ##################### 3. Retrieve MPC observations #########################
    #region 3.Retrieve MPC

    ## Retrieve observations to get propagation start and end times
    T_START_MPC = tm.perf_counter()
    print_verbose(f"Retrieving MPC observations for {body['longname']} (SPKID: {body['SPKID']}) in range {obs_range_start[0]}-{obs_range_start[1]}-{obs_range_start[2]} to {obs_range_end[0]}-{obs_range_end[1]}-{obs_range_end[2]}...")
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
    # Filter out spacecraft observations
    spacecraft_in_batch = batch.observatories_table(only_space_telescopes=True).Name.to_list()
    spacecraft_to_exclude = list(set(spacecraft_in_batch))
    spacecraft_to_exclude_MPC_codes = [spacecraft_codes[name][0] for name in spacecraft_to_exclude]
    batch.filter(
        observatories_exclude = spacecraft_to_exclude_MPC_codes,
    )

    ## Time variables
    # Set start time of propagation to beginning of day before first observation (min. 24 hours back)
    propagation_start_DateTime = DateTime.from_epoch(batch.epoch_start).add_days(-1)
    propagation_start_DateTime = DateTime(propagation_start_DateTime.year, propagation_start_DateTime.month, propagation_start_DateTime.day, 0, 0)
    propagation_start_J2000 = propagation_start_DateTime.epoch() # in seconds since J2000
    input_dict['propagation_start_J2000'] = propagation_start_J2000
    # Set end time of propagation to end of day after last observation (min. 24 hours forward)
    propagation_end_DateTime = DateTime.from_epoch(batch.epoch_end).add_days(2)
    propagation_end_DateTime = DateTime(propagation_end_DateTime.year, propagation_end_DateTime.month, propagation_end_DateTime.day, 0, 0)
    propagation_end_J2000 = propagation_end_DateTime.epoch() # in seconds since J2000
    input_dict['propagation_end_J2000'] = propagation_end_J2000
    # Other time variables
    obs_arc_YEAR = (batch.epoch_end - batch.epoch_start) / cnst.JULIAN_YEAR
    propagation_duration_SEC = propagation_end_J2000 - propagation_start_J2000
    propagation_duration_YEAR = propagation_duration_SEC / cnst.JULIAN_YEAR
    ephem_step_size_DAY = 1.0
    time_buffer = 2 * 31 * cnst.JULIAN_DAY  # 2 month time buffer to avoid interpolation errors

    T_END_MPC = tm.perf_counter()
    print_verbose(f"Done ({T_END_MPC - T_START_MPC:.3f}s)\n")

    ## Print some info
    print_default(f"Propagation start and end dates: {propagation_start_DateTime.year}-{propagation_start_DateTime.month}-{propagation_start_DateTime.day} to {propagation_end_DateTime.year}-{propagation_end_DateTime.month}-{propagation_end_DateTime.day} (duration: {propagation_duration_YEAR:.4g} years)")
    # print_default(f"Observation arc: {obs_arc_YEAR:.4g} years")



    ################# 4.1 Environment setup (define bodies) ####################
    #region 4.1 Env. bodies

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
    # Load SPICE kernels
    spice.clear_kernels()
    spice.load_standard_kernels()

    ## SiMDA bodies: largest main-belt asteroids (LMBAs) and close passers (CPs)
    target_int = int(body['designation_number']) if body['designation_number'] != '[unnumbered]' else 0
    bodies_SiMDA_filtered = (
        pd.read_csv(SiMDA_file)
        .query("DYN != 'COM' & DYN != 'TNO'") # filter out comets and trans-Neptunian objects (COM, TNO). Remaining types are AMO, APO, MBA, NEA, OMB (see https://pdssbn.astro.umd.edu/data_other/objclass.shtml)
        .assign(NUM=lambda x: np.int32(x.NUM))
        .query("NUM != @target_int") # remove propagated body, if present
        .sort_values(by='MASS', ascending=False) # sort bodies by mass, descending order
        .loc[:, ["NUM", "DESIGNATION", "DIAM", "DYN", "MASS"]]
    )
    bodies_SiMDA_largest = bodies_SiMDA_filtered.head(n_LMBAs) # take n_LMBAs most massive
    largest_MBAs = bodies_SiMDA_largest.NUM.to_list()
    largest_MBAs_masses = bodies_SiMDA_largest.MASS.to_list()

    # Get SBDB body data for all SiMDA bodies (nice to have common name, etc. for plotting)
    included_MBAs_data_dict = {}
    for designation_number in largest_MBAs:
        LMBA_data = query_SBDB(str(designation_number))
        included_MBAs_data_dict[designation_number] = LMBA_data

    ## Additional bodies
    if ABs_needed:
        bodies_additional_JPL = [("999", "Pluto")] # format for each body is (JPL Horizons id, common name)
        bodies_additional_masses_KG = [1.3025e22] # !!! source?
    else:
        bodies_additional_JPL = []
        bodies_additional_masses_KG = []


    ############### 4.1 Environment setup (get ephemerides) ####################
    #region 4.2 Env. ephem.

    T_START_get_ephemerides = tm.perf_counter()
    print_verbose("Loading ephemerides...")
    ephem_step_size_MIN = (ephem_step_size_DAY * cnst.JULIAN_DAY) / 60
    timestep_horizons_MIN = ephem_step_size_MIN * 5.0 # Longer time step for non-target body HorizonsQueries
    JPLHQ_location = central_body

    # Ephemerides for SiMDA asteroids
    print_verbose("...for SiMDA bodies (LMBAs)...")
    MBA_ephemerides = {}
    for code in largest_MBAs:
        # Load ephemeris from SPK file
        spice.load_kernel(SPK_file_dir + included_MBAs_data_dict[code]['SPKID'] + ".bsp" )
        print_verbose(f"    SPK file of {included_MBAs_data_dict[code]['longname']} loaded. (SPKID: {included_MBAs_data_dict[code]['SPKID']})")
        MBA_ephemerides[code] = environment_setup.ephemeris.direct_spice(
            global_frame_origin,
            global_frame_orientation,
            included_MBAs_data_dict[code]['SPKID']
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
    if not os.path.exists(SPK_file_dir + body['SPKID'] + ".bsp"):
        # If no SPK file for this body exists already, then generate it
        spkid = generate_spk_file(
            spkid = body['SPKID'],
            spk_file_path = SPK_file_dir,
            spk_file_name = body['SPKID'],
        )
    spice.load_kernel(SPK_file_dir + body['SPKID'] + ".bsp")
    print_verbose(f"SPK file of {body['longname']} loaded. (SPKID: {body['SPKID']})")
    target_body_ephemeris = environment_setup.ephemeris.direct_spice(
        global_frame_origin, global_frame_orientation, body['SPKID']
    )
    target_body_ephemeris = environment_setup.ephemeris.tabulated_from_existing(
        target_body_ephemeris,
        start_time = propagation_start_J2000 - time_buffer,
        end_time = propagation_end_J2000 + time_buffer,
        time_step = ephem_step_size_DAY * cnst.JULIAN_DAY,
    )

    T_END_get_ephemerides = tm.perf_counter()
    print_verbose(f"Ephemerides loaded ({T_END_get_ephemerides - T_START_get_ephemerides:.3f}s) \n")


    ########### 4.3 Environment setup (create system of bodies) ################
    #region 4.3 Env. system

    # Add SPICE bodies
    body_settings = environment_setup.get_default_body_settings(
        bodies_SPICE, global_frame_origin, global_frame_orientation
    )
    # Add SiMDA asteroids to body settings (ephemerides and gravity fields)
    for code, mass in zip(largest_MBAs, largest_MBAs_masses):
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
    body_settings.get(body['SPKID']).ephemeris_settings = target_body_ephemeris
    # define gravity field for target body but set mass to zero (needed for EIH)
    body_settings.get(body['SPKID']).gravity_field_settings = (
            environment_setup.gravity_field.central(0 * cnst.GRAVITATIONAL_CONSTANT)
    )

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
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name = body['SPKID'],
        observer_body_name = central_body,
        reference_frame_name = global_frame_orientation,
        aberration_corrections = "NONE",
        ephemeris_time = propagation_start_J2000,
    )


    ##################### 5. Acceleration settings #############################
    #region 5.Acc. settings

    # Large body accelerations (LB) (Sun + 8 planets + Moon)
    accelerations_large_bodies = define_large_body_accelerations(EIH_needed)

    # Largest MBAs body accelerations
    accelerations_LMBAs = {
        included_MBAs_data_dict[code]['SPKID']: \
            [propagation_setup.acceleration.point_mass_gravity()]
            for code in largest_MBAs
    }
    # Additional body accelerations
    accelerations_ABs = {
        str(name): \
            [propagation_setup.acceleration.point_mass_gravity()]
            for name in bodies_additional
    }
    # Yarkovsky acceleration
    if Yark_needed:
        sun_accelerations_list = \
            [propagation_setup.acceleration.yarkovsky(A2_target_body)] + \
                accelerations_large_bodies["Sun"]
        accelerations_yarkovsky = {
            "Sun": sun_accelerations_list,
        }
        accelerations_large_bodies = accelerations_large_bodies | accelerations_yarkovsky
    # Combine
    accelerations_all = accelerations_large_bodies | accelerations_LMBAs | accelerations_ABs

    # Accelerations settings
    acceleration_settings = {}
    for body_to_propagate in bodies_to_propagate:
        acceleration_settings[str(body_to_propagate)] = accelerations_all

    # Create the acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )


    ###################### 6. Propagation settings #############################
    #region 6.Prop. settings

    # Termination settings
    termination_settings = propagation_setup.propagator.time_termination(
        termination_time = propagation_end_J2000,
        terminate_exactly_on_final_condition = True
    )
    # Dependent variables to save
    dependent_variables_to_save = [
        # propagation_setup.dependent_variable.relative_position("Jupiter", "Sun"),
        # propagation_setup.dependent_variable.relative_velocity("Jupiter", "Sun"),
        # propagation_setup.dependent_variable.keplerian_state(packed_format_MPC_code, "Sun"),
        # propagation_setup.dependent_variable.single_acceleration(
        #     propagation_setup.acceleration.yarkovsky_acceleration_type, str(packed_format_MPC_code), central_body,
        # ),
    ]
    setup_done_timestamp = dttm.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    T_END_setup = tm.perf_counter()
    print_default(f"SETUP DONE ({T_END_setup - T_START_setup:.3f}s) | {setup_done_timestamp} \n")



    ####################### 7. Loop & propagate ################################
    #region 7. Loop

    T_START_propagation_loop = tm.perf_counter()
    state_histories = []
    function_evaluations = np.zeros(len(step_sizes_DAY))
    epoch_sets = []
    pos_error_mag_sets = []
    pos_error_RMSs = []
    pos_error_maxes = []

    for idx, current_step_size_DAY in enumerate(step_sizes_DAY):
        T_START_int_setting = tm.perf_counter()

        # Print to identify current run
        current_idx_timestamp = dttm.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        step_size_unit_str = convert_step_size_DAY_to_str_with_unit(
            current_step_size_DAY, full_unit_name=True)
        print_default(f"Propagating dynamics for Δt = {step_size_unit_str} | {current_idx_timestamp}")

        # Integrator settings (fixed step)
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step = current_step_size_DAY * cnst.JULIAN_DAY,
            coefficient_set = coefficient_set,
            order_to_use = order_to_use,
        )

        # Propagation settings
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
        if verbose:
            console_print_settings = propagator_settings.print_settings
            console_print_settings.print_propagation_clock_time = True
            console_print_settings.print_number_of_function_evaluations = True
            console_print_settings.print_state_indices = True if idx == 0 else False
            console_print_settings.print_termination_reason = True if idx == 0 else False
            # console_print_settings.results_print_frequency_in_steps = 20

        # Propagate dynamics
        dynamics_simulator = simulator.create_dynamics_simulator(
            bodies,
            propagator_settings,
        )
        state_history = dynamics_simulator.state_history
        if save_state_histories_to_files:
            state_histories.append(state_history)
        n_function_evaluations = max(
            dynamics_simulator.cumulative_number_of_function_evaluations.values()
        )
        function_evaluations[idx] = n_function_evaluations

        # Calculate errors for plots
        if idx != 0:
            # Get epochs of state histories for current and previous time step
            previous_ts_epochs = np.array(list(previous_ts_state_history.keys()))[::2] # Every 2nd, since there are twice as many
            current_ts_epochs = np.array(list(state_history.keys()))[:len(previous_ts_epochs)] # Ensure lengths are equal

            # Calculate magnitude of position error for each epoch
            pos_error_mags = [
                np.linalg.norm(
                    (state_history[c_epoch] - previous_ts_state_history[p_epoch])[0:3]
                ) for (p_epoch, c_epoch) in zip(previous_ts_epochs, current_ts_epochs)
            ]
            pos_error_RMS = rms(np.array(pos_error_mags))
            pos_error_max = max(pos_error_mags)

            pos_error_mag_sets.append(pos_error_mags)
            pos_error_RMSs.append(pos_error_RMS)
            pos_error_maxes.append(pos_error_max)
            epoch_sets.append(current_ts_epochs)

        previous_ts_state_history = state_history

        T_END_int_setting = tm.perf_counter()
        print_default(f"Done ({T_END_int_setting - T_START_int_setting:.3f}s) \n")

    print_default(f"PROPAGATIONS DONE | {integrator} | {body['longname']} | {(T_END_int_setting - T_START_propagation_loop)/60:.3f} mins")



    ################### 8.1 Save inputs & outputs ##############################
    #region 8.1 Save in/outputs

    if timestamp_dir:
        # Output data: If timestamp_dir is given as input, then output data from
        # multiple runs will likely be saved to this directory, so each run gets
        # its own subdirectory, named according to the integrator
        integrator_dir = f"{integrator}/"

        # Plots: If timestamp_dir is given as input, then plots from multiple
        # runs will likely be saved to this directory, so all inputs JSONs are
        # named according to the integrator and saved to a single subdirectory
        inputs_dir = "inputs/"
        inputs_JSON_filename = f"_inputs_{integrator}.json"
    else:
        # If timestamp_dir is NOT given as input, then create one. No
        # subdirectories needed for integrator or inputs
        # Timestamp string to uniquely identify this run
        # e.g. formats 28 November 2025, 1:57:01pm as '251128-135701'
        timestamp_str = dttm.datetime.now().strftime("%y%m%d-%H%M%S")
        timestamp_dir = f"{timestamp_str}/"
        integrator_dir = ""
        inputs_dir = ""
        inputs_JSON_filename = f"_inputs.json"

    # Save results of propagations to files
    if save_state_histories_to_files:
        output_data_save_path = output_data_dir + integrator_selection_dir + \
            body_dir + timestamp_dir + integrator_dir
        if not os.path.exists(output_data_save_path):
            os.makedirs(output_data_save_path)

        # Save inputs that went into this run to JSON file to output data directory
        with open(output_data_save_path + "_inputs.json", "w") as f:
            json.dump([input_dict], f, indent=4)

        for idx, current_power in enumerate(
            np.arange(step_size_lo_power, step_size_hi_power + 1)
        ):
            # Save state histories using pickle
            save_with_pickle(
                state_histories[idx],
                output_data_save_path,
                f"state_history_{current_power}.p"
            )
        print_default(f"State history data saved: \n{output_data_save_path}")

    # Save inputs to plots folder
    if 'save' in plot_behaviour_list:
        # Define and create folder for saving plots
        plots_save_path = plot_dir + integrator_selection_dir + body_dir + timestamp_dir
        if not os.path.exists(plots_save_path):
            os.makedirs(plots_save_path)

        # Save inputs that went into this run to JSON to plot directory
        inputs_save_path = plots_save_path + inputs_dir
        if not os.path.exists(inputs_save_path):
            os.makedirs(inputs_save_path)
        with open(inputs_save_path + inputs_JSON_filename, "w") as f:
            json.dump([input_dict], f, indent=4)
    else:
        plots_save_path = None

    return body, \
        function_evaluations, \
        epoch_sets, \
        pos_error_mag_sets, \
        pos_error_RMSs, \
        pos_error_maxes, \
        plots_save_path
#
def plot_position_error_per_timestep(
    epoch_sets,
    pos_error_mag_sets,
    step_size_hi_power: int,
    step_size_lo_power: int,
    integrator: str,
    body,
    arc_YEAR,
    integrator_error_threshold_KM: float = None,
    colormode: int = 0,
    plot_behaviour: str = "show",
    plots_save_path: str = None,
):

    # List of Δt
    step_sizes_DAY = np.logspace(
        step_size_lo_power, # must be increasing, must be doubling
        step_size_hi_power,
        step_size_hi_power - step_size_lo_power + 1,
        base = 2,
    )
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.67])*twothirds_width_INCH,
        facecolor = fig_background[colormode],
        layout = 'constrained',
    )
    for i, current_step_size_DAY in reversed(list(enumerate(step_sizes_DAY[1:]))):
        # Make Δt more readable
        step_size_unit_str = convert_step_size_DAY_to_str_with_unit(current_step_size_DAY)
        # Approximate years for plot ticks
        epochs_plot = (epoch_sets[i] / cnst.JULIAN_YEAR) + 2000

        # Plot line
        ax.plot(
            epochs_plot,
            np.array(pos_error_mag_sets[i]) / 1000, # in km
            label = f"Δt = {step_size_unit_str}",
            color = cmap_custom10[colormode](i%10),
            linestyle = linestyles[i//10],
            linewidth = 2.0,
            zorder = 2.5,
        )

    # Set default ax settings
    ax = set_default_2d_ax_settings(ax, colormode)
    # Other axis settings
    ax.set_yscale('log')
    if integrator_error_threshold_KM:
        ax.axhline(
            integrator_error_threshold_KM,
            color = text_color[colormode],
            linewidth = 2.0,
            linestyle = 'dotted',
            label = "Threshold",
            zorder = 2.1,
        )
    # Legend, labels, titles
    ax.legend(
        loc = 'center left',
        bbox_to_anchor = (1.01, 0.5),
        markerscale = 1.2,
        fontsize = small_font_size,
        labelcolor = text_color[colormode],
        facecolor = legend_bg_color[colormode],
    )
    ax.set_ylabel(f"Error [km]")
    ax.set_xlabel("Year")
    # plt.suptitle(
    #     f"Position Error over Time: {body['longname']}",
    #     fontsize = big_font_size,
    #     weight = 'bold',
    # )
    title_tex_Body = f"$\\bf{{Body:}}$"
    title_tex_Integrator = f"$\\bf{{Integrator:}}$"
    fig.suptitle(
        f"{title_tex_Body} {body['longname']},  {title_tex_Integrator} {integrator}",
        fontsize = medium_font_size,
    )
    ax.set_title(
        fr"$e$ = {body['eccentricity']:.3f},  $a$ = {body['semimajoraxis_AU']:.3f} AU,  arc = {arc_YEAR:.3g} y",
        fontsize = small_font_size,
    )

    if plot_behaviour == 'show':
        plt.show(block=False)
        plt.pause(0.001)
    elif plot_behaviour == 'save':
        save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_name = f"error_vs_time_{integrator}.pdf"
        plt.savefig(save_path + fig_name, format="pdf")
        plt.close()
        print(f"Error vs. time {'darkmode '*colormode}figure saved: \n{save_path+fig_name}")
#
def plot_max_and_RMS_error_vs_delta_t(
    step_size_hi_power: int,
    step_size_lo_power: int,
    pos_error_RMSs,
    pos_error_maxes,
    integrator: str,
    body,
    arc_YEAR,
    integrator_error_threshold_KM: float = None,
    colormode: int = 0,
    plot_behaviour: str = "show",
    plots_save_path: str = None,
):
    # List of Δt
    step_sizes_DAY = np.logspace(
        step_size_lo_power, # must be increasing, must be doubling
        step_size_hi_power,
        step_size_hi_power - step_size_lo_power + 1,
        base = 2,
    )
    fig, ax = plt.subplots(
        1, 1,
        figsize = np.array([1.0,0.85])*half_width_INCH,
        facecolor = fig_background[colormode],
        layout = 'constrained',
    )
    pos_error_maxes = np.array(pos_error_maxes) / 1000 # in km
    ax.plot(
        step_sizes_DAY[1:] * 24, # in h
        pos_error_maxes,  # in km
        label = "Max. pos. error",
        marker = 'o',
        markersize = 9,
        linewidth = 2.5,
        color = cmap_custom10[colormode](0),
        zorder = 2.5,
    )
    pos_error_RMSs = np.array(pos_error_RMSs) / 1000 # in km
    ax.plot(
        step_sizes_DAY[1:] * 24, # in h
        pos_error_RMSs,   # in km
        label = "RMS of pos. error",
        marker = 's',
        markersize = 9,
        linewidth = 2.5,
        linestyle = '--',
        color = cmap_custom10[colormode](1),
        zorder = 2.5,
    )

    # Set default ax settings
    ax = set_default_2d_ax_settings(ax, colormode)
    # Other axis settings
    ax.set_yscale('log')
    ax.set_xscale('log')
    if integrator_error_threshold_KM:
        ax.axhline(
            integrator_error_threshold_KM,
            color = text_color[colormode],
            linewidth = 2.0,
            linestyle = 'dotted',
            label = "Threshold",
            zorder = 2.1,
        )
    ax.margins(x=0.10)
    lowest_point = min(pos_error_maxes.min(), pos_error_RMSs.min()) # in km
    # highest_point = max(pos_error_maxes.max(), pos_error_RMSs.max()) # in km
    y_min, y_max = ax.get_ylim()
    default_y_margin_log = np.log10(lowest_point) - np.log10(y_min)
    new_y_min_log = np.log10(lowest_point) - (5.0 * default_y_margin_log)
    ax.set_ylim((10**new_y_min_log, y_max))

    # Annotate all points with time steps with readable units
    high_text_point = 10**(np.log10(lowest_point) - (2.5 * default_y_margin_log))
    low_text_point = 10**(np.log10(lowest_point) - (4.5 * default_y_margin_log))
    # List that alternates between high_text_point and low_text_point
    text_points = [None]*len(pos_error_maxes)
    text_points[::2] = [high_text_point] * len(text_points[::2])
    text_points[1::2] = [low_text_point] * len(text_points[1::2])
    for idx, (x,y) in enumerate(zip(
        np.array(step_sizes_DAY[1:]) * 24, # in h
        text_points,
    )):
        step_size_unit_str = convert_step_size_DAY_to_str_with_unit(
            step_sizes_DAY[idx+1],
            minute_as_m = True,
        )
        ax.annotate(
            step_size_unit_str, # text
            (x,y),              # coordinates
            textcoords = "offset points", # how to position the text
            xytext = (0,0), # distance from text to point (x,y)
            # xytext = (x,y*0.3), # distance from text to point (x,y)
            ha = 'center',      # horizontal alignment can be left, right or center
            fontsize = tiny_font_size,
            color = text_color[colormode],
        )
        ax.plot(
            [x,x],
            [y*10, pos_error_maxes[idx]],
            color = text_color[colormode],
            linestyle = '--',
            linewidth = 1.0,
            alpha = 0.5,
            zorder = 2.1,
        )


    # Legend, labels, titles
    ax.legend(
        loc = 'upper left',
        markerscale = 1.2,
        fontsize = small_font_size,
        labelcolor = text_color[colormode],
        facecolor = legend_bg_color[colormode],
    )
    ax.set_ylabel(f"Error [km]")
    ax.set_xlabel("Δt [h]")
    title_tex_Body = f"$\\bf{{Body:}}$"
    title_tex_Integrator = f"$\\bf{{Integrator:}}$"
    fig.suptitle(
        f"{title_tex_Body} {body['longname']},  {title_tex_Integrator} {integrator}",
        fontsize = medium_font_size,
    )
    ax.set_title(
        fr"$e$ = {body['eccentricity']:.3f},  $a$ = {body['semimajoraxis_AU']:.3f} AU,  arc = {arc_YEAR:.3g} y",
        fontsize = small_font_size,
    )
    # fig.set_tight_layout(True)
    if plot_behaviour == 'show':
        plt.show()
    elif plot_behaviour == 'save':
        save_path = plots_save_path + "darkmode/" if colormode else plots_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_name = f"error_vs_timestep_{integrator}.pdf"
        plt.savefig(save_path + fig_name, format="pdf")
        plt.close()
        print(f"Error vs. Δt {'darkmode '*colormode}figure saved: \n{save_path+fig_name}")


#%% Compare close-pass search methods
### Compare close-pass search methods
# To run this cell, change the returns of the two functions to include the runtimes

# file_directory_path = os.path.dirname(__file__)
# os.chdir(file_directory_path) # ensure that the cwd is the subfolder where this file is

# target_body_name = "Aten"
# _ = query_SBDB(target_body_name, True) # show body info before running
# n_MBAs_search = 50
# n_MBAs_already_included = 0
# search_period_initial_time = DateTime(2010, 1, 1)
# n_orbits = 1.1
# timestep_DAY = 20.0
# close_pass_threshold_AU = 1.0
# verbose = True


# close_passers_Horizons, runtime_Horizons_SEC = search_for_close_passes_with_JPL_Horizons(
#     target_body_name = target_body_name,
#     n_MBAs_search = n_MBAs_search,
#     n_MBAs_already_included = n_MBAs_already_included,
#     search_period_initial_time = search_period_initial_time,
#     n_orbits = n_orbits,
#     timestep_DAY = timestep_DAY,
#     close_pass_threshold_AU = close_pass_threshold_AU,
#     verbose = verbose,
# )

# close_passers_Kepler, runtime_Kepler_SEC = search_for_close_passes_with_Kepler_elements(
#     target_body_name = target_body_name,
#     n_MBAs_search = n_MBAs_search,
#     n_MBAs_already_included = n_MBAs_already_included,
#     search_period_initial_time = search_period_initial_time,
#     n_orbits = n_orbits,
#     timestep_DAY = timestep_DAY,
#     close_pass_threshold_AU = close_pass_threshold_AU,
#     verbose = verbose,
# )

# print(f"")
# print(f"runtime Horizons: {runtime_Horizons_SEC} s")
# print(f"runtime Kepler: {runtime_Kepler_SEC} s")
# print(f"runtime ratio Horizons/Kepler: {runtime_Horizons_SEC/runtime_Kepler_SEC}")
# print(f"")
# found_by_Horizons_set = set(list(close_passers_Horizons.keys()))
# found_by_Kepler_set = set(list(close_passers_Kepler.keys()))
# found_by_both = list( found_by_Horizons_set & found_by_Kepler_set )
# found_by_Horizons_only = list( found_by_Horizons_set - found_by_Kepler_set )
# found_by_Kepler_only = list( found_by_Kepler_set - found_by_Horizons_set )
# print(f"close passes found by Horizons only: {len(found_by_Horizons_only)} ({found_by_Horizons_only})")
# print(f"close passes found by Kepler only: {len(found_by_Kepler_only)} ({found_by_Kepler_only})")

# diff_min_dist = []
# diff_lim = 0.02
# for body in found_by_both:
#     Horizons_min_dist = close_passers_Horizons[body]
#     Kepler_min_dist = close_passers_Kepler[body]

#     if (Kepler_min_dist > (1+diff_lim)*Horizons_min_dist) or (Kepler_min_dist < (1-diff_lim)*Horizons_min_dist):
#         diff_min_dist.append(body)

# print(f"")
# print(f"close passes found by both: {len(found_by_both)} ({found_by_both})")
# print(f"found by both but min. dist. differs by >{diff_lim*100}%: {len(diff_min_dist)} ({diff_min_dist})")
