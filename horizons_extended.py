"""
The code in this file adds extra functionality to HorizonsClass to allow for
retrieving ephemeris uncertainty information for bodies

Based on code by @megargayu on GitHub, see
https://github.com/astropy/astroquery/pull/3273 

This file was put together and sent to me by Lars Hinüber (@larshinueber)
"""


import warnings
from collections import OrderedDict

import astroquery
from astropy.io import ascii
from astropy.table import Column, Table
from astropy.time import Time
from astroquery.jplhorizons import HorizonsClass, conf
from numpy import isnan, nan, ndarray
from requests import Response

# 2. third party imports
from requests.exceptions import HTTPError

astroquery.jplhorizons.Conf.vec_columns = astroquery.jplhorizons.Conf.vec_columns | {
    "X_s": ("x_s", "AU"),
    "Y_s": ("y_s", "AU"),
    "Z_s": ("z_s", "AU"),
    "VX_s": ("vx_s", "AU/d"),
    "VY_s": ("vy_s", "AU/d"),
    "VZ_s": ("vz_s", "AU/d"),
    "A_s": ("a_s", "AU"),
    "C_s": ("c_s", "AU"),
    "N_s": ("n_s", "AU"),
    "VA_s": ("va_s", "AU/d"),
    "VC_s": ("vc_s", "AU/d"),
    "VN_s": ("vn_s", "AU/d"),
    "R_s": ("r_s", "AU"),
    "T_s": ("t_s", "AU"),
    # N_s is repeated here.
    "VR_s": ("vr_s", "AU/d"),
    "VT_s": ("vt_s", "AU/d"),
    # VN_s is repeated here.
    # Note: A_s is duplicated for POS (p) and ACN (a) uncertainties. It
    # is up to the user to differentiate them! (They have the same units.)
    "D_s": ("d_s", "AU"),
    # R_s is repeated here.
    "VA_RA_s": ("va_ra_s", "AU/d"),
    "VD_DEC_s": ("va_dec_s", "AU/d"),
    # VR_s is repeated here.
}


def _parse_result_unc(self, response, verbose=None):
    """
    Parse query result to a `~astropy.table.Table` object.


    Parameters
    ----------

    response : `~requests.Response`
        Response from server.


    Returns
    -------

    data : `~astropy.table.Table`

    """

    self.last_response = response
    try:
        response.raise_for_status()
    except HTTPError:
        # don't cache any HTTP errored queries (especially when the API is down!)
        try:
            self._last_query.remove_cache_file(self.cache_location)
        except OSError:
            # this is allowed: if `cache` was set to False, this
            # won't be needed
            pass
        raise

    self._raw_response = response.text

    # return raw response, if desired
    if self.return_raw:
        # reset return_raw flag
        self.return_raw = False
        return self._raw_response

    # split response by line break
    src = response.text.split("\n")

    data_start_idx = 0
    data_end_idx = 0
    H, G = nan, nan
    M1, M2, k1, k2, phcof = nan, nan, nan, nan, nan
    headerline = []
    centername = ""
    for idx, line in enumerate(src):
        # read in ephemerides header line; replace some field names
        if self.query_type == "ephemerides" and "Date__(UT)__HR:MN" in line:
            headerline = str(line).split(",")
            headerline[2] = "solar_presence"
            headerline[3] = (
                "lunar_presence" if "Earth" in centername else "interfering_body"
            )
            headerline[-1] = "_dump"
            if isinstance(self.id, dict) or str(self.id).startswith("g:"):
                headerline[4] = "nearside_flag"
                headerline[5] = "illumination_flag"
        # read in elements header line
        elif self.query_type == "elements" and "JDTDB," in line:
            headerline = str(line).split(",")
            headerline[-1] = "_dump"
        # identify end of data block
        if "$$EOE" in line:
            data_end_idx = idx
        # identify start of data block
        if "$$SOE" in line:
            data_start_idx = idx + 1

            # read in vectors header line
            # reading like this helps fix issues with commas after JDTDB
            if self.query_type == "vectors":
                headerline_raw = str(src[idx - 2]).replace("JDTDB,", "JDTDB")
                headerline = [
                    "            JDTDB",
                    *str(headerline_raw).split("JDTDB")[1].split(","),
                ]
                headerline[-1] = "_dump"
        # read in targetname
        if "Target body name" in line:
            targetname = line[18:50].strip()
        # read in center body name
        if "Center body name" in line:
            centername = line[18:50].strip()
        # read in H and G (if available)
        if "rotational period in hours)" in line:
            HGline = src[idx + 2].split("=")
            if "B-V" in HGline[2] and "G" in HGline[1]:
                try:
                    H = float(HGline[1].rstrip("G"))
                    G = float(HGline[2].rstrip("B-V"))
                except ValueError:
                    H = nan
                    G = nan
        # read in M1, M2, k1, k2, and phcof (if available)
        if "Comet physical" in line:
            HGline = src[idx + 2].split("=")
            try:
                M1 = float(HGline[1].rstrip("M2"))
                k1 = float(HGline[3].rstrip("k2"))
            except ValueError:
                M1 = nan
                k1 = nan
            try:
                M2 = float(HGline[2].rstrip("k1"))
                k2 = float(HGline[4].rstrip("PHCOF"))
            except ValueError:
                M2 = nan
                k2 = nan
            try:
                phcof = float(HGline[5])
            except ValueError:
                phcof = nan
        # catch unambiguous names
        if (
            "Multiple major-bodies match string" in line
            or "Matching small-bodies:" in line
        ) and ("No matches found" not in src[idx + 1]):
            for i in range(idx + 2, len(src), 1):
                if ("To SELECT, enter record" in src[i]) or (
                    "make unique selection." in src[i]
                ):
                    end_idx = i
                    break
            raise ValueError(
                (
                    "Ambiguous target name; provide "
                    "unique id:\n%s" % "\n".join(src[idx + 2 : end_idx])
                )
            )
        # catch unknown target
        if "Matching small-bodies" in line and "No matches found" in src[idx + 1]:
            raise ValueError(
                ("Unknown target ({:s}). Maybe try different id_type?").format(self.id)
            )
        # catch any unavailability of ephemeris data
        if "No ephemeris for target" in line:
            errormsg = line[line.find("No ephemeris for target") :]
            errormsg = errormsg[: errormsg.find("\n")]
            raise ValueError("Horizons Error: {:s}".format(errormsg))
        # catch elements errors
        if "Cannot output elements" in line:
            errormsg = line[line.find("Cannot output elements") :]
            errormsg = errormsg[: errormsg.find("\n")]
            raise ValueError("Horizons Error: {:s}".format(errormsg))
        # catch date error
        if "Cannot interpret date" in line:
            errormsg = line[line.find("Cannot interpret date") :]
            errormsg = errormsg[: errormsg.find("\n")]
            raise ValueError("Horizons Error: {:s}".format(errormsg))
        if "INPUT ERROR" in line:
            headerline = []
            break

    if headerline == []:
        err_msg = "".join(src[data_start_idx:data_end_idx])
        if len(err_msg) > 0:
            raise ValueError("Query failed with error message:\n" + err_msg)
        else:
            raise ValueError(
                (
                    "Query failed without known error message; "
                    "received the following response:\n"
                    "{}"
                ).format(response.text)
            )
    # strip whitespaces from column labels
    headerline = [h.strip() for h in headerline]

    # add numbers to duplicates
    headerline_seen = {}  # format - column_name: [headerline_idx, count]
    dup_col_to_orig = {}  # format - remapped_column_name: [original_column_name, index], used for later processing
    for i, col in enumerate(headerline):
        if col in headerline_seen:
            headerline_seen[col][1] += 1
            headerline[headerline_seen[col][0]] = f"{col}_1"
            dup_col_to_orig[f"{col}_1"] = [col, 1]

            headerline[i] = f"{col}_{headerline_seen[col][1]}"
            dup_col_to_orig[headerline[i]] = [col, headerline_seen[col][1]]
        else:
            headerline_seen[col] = [i, 1]

    # remove all 'Cut-off' messages
    raw_data = [
        line for line in src[data_start_idx:data_end_idx] if "Cut-off" not in line
    ]

    # read in data
    data = ascii.read(
        raw_data,
        names=headerline,
        fill_values=[(".n.a.", "0"), ("n.a.", "0")],
        fast_reader=False,
    )
    # force to a masked table
    data = Table(data, masked=True)

    # convert data to QTable
    # from astropy.table import QTable
    # data = QTable(data)
    # does currently not work, unit assignment in columns creates error
    # results in:
    # TypeError: The value must be a valid Python or Numpy numeric type.

    # remove last column as it is empty
    data.remove_column("_dump")

    # add targetname and physical properties as columns
    data.add_column(Column([targetname] * len(data), name="targetname"), index=0)
    if not isnan(H):
        data.add_column(Column([H] * len(data), name="H"), index=3)
    if not isnan(G):
        data.add_column(Column([G] * len(data), name="G"), index=4)
    if not isnan(M1):
        data.add_column(Column([M1] * len(data), name="M1"), index=3)
    if not isnan(M2):
        data.add_column(Column([M2] * len(data), name="M2"), index=4)
    if not isnan(k1):
        data.add_column(Column([k1] * len(data), name="k1"), index=5)
    if not isnan(k2):
        data.add_column(Column([k2] * len(data), name="k2"), index=6)
    if not isnan(phcof):
        data.add_column(Column([phcof] * len(data), name="phasecoeff"), index=7)

    # replace missing airmass values with 999 (not observable)
    if self.query_type == "ephemerides" and "a-mass" in data.colnames:
        data["a-mass"] = data["a-mass"].filled(999)

    # set column definition dictionary
    if self.query_type == "ephemerides":
        column_defs = conf.eph_columns
    elif self.query_type == "elements":
        column_defs = conf.elem_columns
    elif self.query_type == "vectors":
        column_defs = conf.vec_columns
    else:
        raise TypeError("Query type unknown.")

    # set column units
    rename = []
    for col in data.columns:
        # fetch from original definition, not remapped
        col_unit = column_defs[
            dup_col_to_orig[col][0] if col in dup_col_to_orig.keys() else col
        ]

        data[col].unit = col_unit[1]
        if data[col].name != col_unit[0]:
            rename.append(data[col].name)

    # rename columns
    for col in rename:
        try:
            if col in dup_col_to_orig.keys():  # preserve index on duplicate columns
                to_rename = f"{column_defs[dup_col_to_orig[col][0]][0]}_{dup_col_to_orig[col][1]}"
            else:
                to_rename = column_defs[col][0]

            data.rename_column(data[col].name, to_rename)
        except KeyError:
            pass

    return data


HorizonsClass._parse_result = _parse_result_unc


# @async_to_sync
def vectors_uncertainty(
    self,
    *,
    get_query_payload=False,
    closest_apparition=False,
    no_fragments=False,
    get_raw_response=False,
    cache=True,
    refplane="ecliptic",
    vector_table="3",
    aberrations="geometric",
    delta_T=False,
):
    """
    Query JPL Horizons for state vectors.

    .. deprecated:: 0.4.7
        The ``get_raw_response`` keyword argument is deprecated.  The
        `~HorizonsClass.vectors_async` method will return a raw response.

    The ``location`` parameter in ``HorizonsClass`` refers in this case to
    the center body relative to which the vectors are provided.

    The following table lists the values queried, their definitions, data
    types, units, and original Horizons designations (where available). For
    more information on the definitions of these quantities, please refer to
    the `Horizons User Manual <https://ssd.jpl.nasa.gov/horizons/manual.html>`_.

    +------------------+-----------------------------------------------+
    | Column Name      | Definition                                    |
    +==================+===============================================+
    | targetname       | official number, name, designation (string)   |
    +------------------+-----------------------------------------------+
    | H                | absolute magnitude in V band (float, mag)     |
    +------------------+-----------------------------------------------+
    | G                | photometric slope parameter (float)           |
    +------------------+-----------------------------------------------+
    | M1               | comet total abs mag (float, mag, ``M1``)      |
    +------------------+-----------------------------------------------+
    | M2               | comet nuclear abs mag (float, mag, ``M2``)    |
    +------------------+-----------------------------------------------+
    | k1               | total mag scaling factor (float, ``k1``)      |
    +------------------+-----------------------------------------------+
    | k2               | nuclear mag scaling factor (float, ``k2``)    |
    +------------------+-----------------------------------------------+
    | phasecoeff       | comet phase coeff (float, mag/deg, ``PHCOFF``)|
    +------------------+-----------------------------------------------+
    | datetime_str     | epoch Date (str, ``Calendar Date (TDB)``)     |
    +------------------+-----------------------------------------------+
    | datetime_jd      | epoch Julian Date (float, ``JDTDB``)          |
    +------------------+-----------------------------------------------+
    | delta_T          | time-varying difference between TDB and UT    |
    |                  | (float, ``delta-T``, optional)                |
    +------------------+-----------------------------------------------+
    | x                | x-component of position vector                |
    |                  | (float, au, ``X``)                            |
    +------------------+-----------------------------------------------+
    | y                | y-component of position vector                |
    |                  | (float, au, ``Y``)                            |
    +------------------+-----------------------------------------------+
    | z                | z-component of position vector                |
    |                  | (float, au, ``Z``)                            |
    +------------------+-----------------------------------------------+
    | vx               | x-component of velocity vector (float, au/d,  |
    |                  | ``VX``)                                       |
    +------------------+-----------------------------------------------+
    | vy               | y-component of velocity vector (float, au/d,  |
    |                  | ``VY``)                                       |
    +------------------+-----------------------------------------------+
    | vz               | z-component of velocity vector (float, au/d,  |
    |                  | ``VZ``)                                       |
    +------------------+-----------------------------------------------+
    | lighttime        | one-way lighttime (float, d, ``LT``)          |
    +------------------+-----------------------------------------------+
    | range            | range from coordinate center (float, au,      |
    |                  | ``RG``)                                       |
    +------------------+-----------------------------------------------+
    | range_rate       | range rate (float, au/d, ``RR``)              |
    +------------------+-----------------------------------------------+


    Parameters
    ----------

    closest_apparition : boolean, optional
        Only applies to comets. This option will choose the closest
        apparition available in time to the selected epoch; default: False.
        Do not use this option for non-cometary objects.

    no_fragments : boolean, optional
        Only applies to comets. Reject all comet fragments from selection;
        default: False. Do not use this option for non-cometary objects.

    get_query_payload : boolean, optional
        When set to `True` the method returns the HTTP request parameters as
        a dict, default: False

    get_raw_response: boolean, optional
        Return raw data as obtained by JPL Horizons without parsing the data
        into a table, default: False

    refplane : string
        Reference plane for all output quantities: ``'ecliptic'`` (ecliptic
        and mean equinox of reference epoch), ``'earth'`` (Earth mean
        equator and equinox of reference epoch), or ``'body'`` (body mean
        equator and node of date); default: ``'ecliptic'``.

        See :ref:`Horizons Reference Frames <jpl-horizons-reference-frames>`
        in the astroquery documentation for details.

    aberrations : string, optional
        Aberrations to be accounted for: [``'geometric'``,
        ``'astrometric'``, ``'apparent'``]. Default: ``'geometric'``

    delta_T : boolean, optional
        Triggers output of time-varying difference between TDB and UT
        time-scales. Default: False

    cache : bool
        Defaults to True. If set overrides global caching behavior.
        See :ref:`caching documentation <astroquery_cache>`.


    Returns
    -------

    response : `requests.Response`
        The response of the HTTP request.


    Examples
    --------

    >>> from astroquery.jplhorizons import Horizons
    >>> obj = Horizons(id='2012 TC4', location='257',
    ...                epochs={'start': '2017-10-01',
    ...                        'stop': '2017-10-02',
    ...                        'step': '10m'})
    >>> vec = obj.vectors()  # doctest: +REMOTE_DATA
    >>> print(vec)  # doctest: +SKIP
    targetname  datetime_jd  ...      range          range_rate
        ---           d       ...        AU             AU / d
    ---------- ------------- ... --------------- -----------------
    (2012 TC4)     2458027.5 ... 0.0429332099306 -0.00408018711862
    (2012 TC4) 2458027.50694 ... 0.0429048742906 -0.00408040726527
    (2012 TC4) 2458027.51389 ... 0.0428765385796 -0.00408020747595
    (2012 TC4) 2458027.52083 ... 0.0428482057142  -0.0040795878561
    (2012 TC4) 2458027.52778 ...  0.042819878607 -0.00407854931543
    (2012 TC4) 2458027.53472 ... 0.0427915601617  -0.0040770935665
            ...           ... ...             ...               ...
    (2012 TC4) 2458028.45833 ... 0.0392489462501 -0.00405496595173
    (2012 TC4) 2458028.46528 ...   0.03922077771 -0.00405750632914
    (2012 TC4) 2458028.47222 ...  0.039192592935 -0.00405964084539
    (2012 TC4) 2458028.47917 ...  0.039164394759 -0.00406136516755
    (2012 TC4) 2458028.48611 ... 0.0391361860433 -0.00406267574646
    (2012 TC4) 2458028.49306 ... 0.0391079696711  -0.0040635698239
    (2012 TC4)     2458028.5 ... 0.0390797485422 -0.00406404543822
    Length = 145 rows

    """

    URL = conf.horizons_server

    # check for required information and assemble commandline stub
    if self.id is None:
        raise ValueError("'id' parameter not set. Query aborted.")
    elif isinstance(self.id, dict):
        commandline = self._format_id_coords(self.id)
    else:
        commandline = str(self.id)
    if self.location is None:
        self.location = "500@10"
    if self.epochs is None:
        self.epochs = Time.now().jd
    # expand commandline based on self.id_type
    if self.id_type in ["designation", "name", "asteroid_name", "comet_name"]:
        commandline = {
            "designation": "DES=",
            "name": "NAME=",
            "asteroid_name": "ASTNAM=",
            "comet_name": "COMNAM=",
        }[self.id_type] + commandline
    if self.id_type in ["smallbody", "asteroid_name", "comet_name", "designation"]:
        commandline += ";"
        if isinstance(closest_apparition, bool):
            if closest_apparition:
                commandline += " CAP;"
        else:
            commandline += " CAP{:s};".format(closest_apparition)
        if no_fragments:
            commandline += " NOFRAG;"
    # configure request_payload for vectors query
    request_payload = OrderedDict(
        [
            ("format", "text"),
            ("EPHEM_TYPE", "VECTORS"),
            ("OUT_UNITS", "AU-D"),
            ("COMMAND", '"' + commandline + '"'),
            ("CSV_FORMAT", ('"YES"')),
            (
                "REF_PLANE",
                {
                    "ecliptic": "ECLIPTIC",
                    "earth": "FRAME",
                    "frame": "FRAME",
                    "body": "'BODY EQUATOR'",
                }[refplane],
            ),
            ("REF_SYSTEM", "ICRF"),
            ("TP_TYPE", "ABSOLUTE"),
            ("VEC_LABELS", "YES"),
            (
                "VEC_CORR",
                {"geometric": '"NONE"', "astrometric": '"LT"', "apparent": '"LT+S"'}[
                    aberrations
                ],
            ),
            ("VEC_TABLE", vector_table),
            ("VEC_DELTA_T", {True: "YES", False: "NO"}[delta_T]),
            ("OBJ_DATA", "YES"),
        ]
    )
    if isinstance(self.location, dict):
        request_payload = dict(
            **request_payload, **self._location_to_params(self.location)
        )
    else:
        request_payload["CENTER"] = "'" + str(self.location) + "'"
    # parse self.epochs
    if isinstance(self.epochs, (list, tuple, ndarray)):
        request_payload["TLIST"] = "\n".join([str(epoch) for epoch in self.epochs])
    elif isinstance(self.epochs, dict):
        if (
            "start" not in self.epochs
            or "stop" not in self.epochs
            or "step" not in self.epochs
        ):
            raise ValueError("'epochs' must contain start, stop, step")
        request_payload["START_TIME"] = (
            '"' + self.epochs["start"].replace("'", "") + '"'
        )
        request_payload["STOP_TIME"] = '"' + self.epochs["stop"].replace("'", "") + '"'
        request_payload["STEP_SIZE"] = '"' + self.epochs["step"].replace("'", "") + '"'

    else:
        # treat epochs as a list
        request_payload["TLIST"] = str(self.epochs)

    self.query_type = "vectors"

    # return request_payload if desired
    if get_query_payload:
        return request_payload

    # set return_raw flag, if raw response desired
    if get_raw_response:
        self.return_raw = True

    # query and parse
    response = self._request(
        "GET", URL, params=request_payload, timeout=self.TIMEOUT, cache=cache
    )
    self.uri = response.url

    # check length of uri
    if len(self.uri) >= 2000:
        warnings.warn(
            (
                "The uri used in this query is very long "
                "and might have been truncated. The results of "
                "the query might be compromised. If you queried "
                "a list of epochs, consider querying a range."
            )
        )

    if isinstance(response, Response):
        response.raise_for_status()

    result = self._parse_result(response)
    self.table = result
    return result


HorizonsClass.vectors_uncertainty = vectors_uncertainty

Horizons = HorizonsClass()
