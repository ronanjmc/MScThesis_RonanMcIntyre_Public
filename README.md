# MScThesis_RonanMcIntyre_Public
Project code used to generate the results for my thesis, titled: "Near-Earth Asteroid Orbit Determination &amp; Yarkovsky Effect Detection"

Overview of project files:
- `Thesis_Final_2026-07-19.pdf` - The final thesis document, in PDF format. This document is also available in the [TU Delft repository](https://resolver.tudelft.nl/uuid:56640d62-64f6-4041-bf46-9ffe02394a1b)

- `acceleration_scenario_comparison.py` - Compares different dynamical acceleration scenarios for a target asteroid, such as different combinations of solar-system perturbations and Yarkovsky effects. It propagates orbits, compares them against JPL Horizons, and generates plots showing position error and residual behavior.

- `ephemeris_uncertainty_retrieval.py` - Downloads and saves ephemeris uncertainty data for a list of bodies using Horizons-based utilities.

- `est_w_mpc_single.py` - Runs an estimation for a single asteroid. Useful for testing or developing changes before applying them to `est_w_mpc_large.py`.

- `est_w_mpc_large.py` - Runs estimations for a set of asteroids.

- `estimation_plots.py` - Contains functions for generating plots/figures of the outputs of the estimations. It provides functions for residual plots, position error plots, formal error plots, correlation matrices, etc.

- `horizons_spk.py` - A helper script for generating SPK files from JPL Horizons. Can be used to generate SPK files for many bodies at once.

- `horizons_extended.py` - An extension to the Horizons query interface that adds support for parsing and retrieving ephemeris uncertainty information from JPL Horizons responses.

- `integrator_selection.py` - Evaluates the performance of different numerical integrators and time steps. Propagates bodies under the same conditions with several integrators and compares error versus computational cost.

- `IS_utils.py` - The shared utility module for the project. It contains many helper functions for querying SBDB and Horizons, loading data, generating SPK files, formatting inputs, and supporting plotting and file I/O.

- `mpc_custom_version.py` - A custom version of Tudat’s mpc.py, with different handling of the custom_name parameter.

- `residual_cutoff_selection.py` - Tests different residual filtering thresholds for estimation runs. It helps assess how aggressive residual filtering should be when filtering observation data before estimation.

- `yark_results_processing.py` - Processes and analyzes the outputs from the large estimation runs, especially for investigating Yarkovsky-related results. It is focused on summarizing and plotting the results from those batch runs.


Overview of project subfolders:

- `input_data`:
    - `SPK_files` - Ephemeris files (SPK) that are generated from JPL Horizons are stored here.*
    - `ephemeris_uncertainty` - The uncertainty files of the JPL Horizons ephemerides are stored here.*
    - `SBDB_data` - Pickle files of the JPL SBDB queries are stored here.*
    - `SiMDA_data` - Mass data for large asteroids is stored here.
    - `yarkovsky_data` - Contains table B.1 of Fenucci et al. (2024), whose data is used for comparison against the estimations of this thesis.

- `output_data` - Where the results of various scripts are stored.

- `plots` - Where plots generated from outputs are stored.

*This folder is empty on GitHub because of the large number of files and/or large file sizes.

For any questions about the files in this repo, contact ronan.j.mcintyre@gmail.com
