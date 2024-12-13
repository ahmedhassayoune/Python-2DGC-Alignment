import glob
import math
import os
import subprocess
import sys
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pandas as pd

path_to_scr_folder = os.path.join(os.path.dirname(os.path.abspath("")), "src")
sys.path.append(path_to_scr_folder)

import projection
from peak_detection import peak_detection
from read_chroma import read_chromato_and_chromato_cube


def swpa_peak_alignment(chromato_dir, ref_filename, swpa_input_path, mod_time=1.25):
    r"""Aligns chromatogram peaks using SWPA.
    Given a reference chromatogram and a set of target chromatograms,
    aligns the peaks of the target chromatograms with the peaks of the reference chromatogram.

    Parameters
    ----------
    chromato_dir : str
        Path to the directory containing the chromatogram files
    ref_filename : str
        Name of the reference chromatogram file.
    swpa_input_path : str
        Path to the directory where the SWPA input files will be saved.
    mod_time : float
        The modulation time of the chromatograms.

    Returns
    -------
    DataFrame
        DataFrame containing the matched peaks.
    """
    dict_chromatos = {}

    # Create SWPA input files
    swpa_input(chromato_dir, swpa_input_path, dict_chromatos, mod_time=mod_time)

    matches = run_swpa_script(
        swpa_input_path, ref_filename, dict_chromatos, mod_time=mod_time
    )
    return matches


def swpa_input(
    chromato_dir, swpa_input_path, dict_chromatos, peak_detection_thr=5, mod_time=1.25
):
    """
    Preprocess chromatograms and save them as CSV input files for SWPA script.

    Parameters:
    ----------
    chromato_dir : str
        The directory path where the chromatogram files are located.
    swpa_input_path : str
        The directory path where the SWPA input files will be saved.
    dict_chromatos : dict
        A dictionary to store the chromatogram data for each chromatogram file.
    peak_detection_thr : float
        The threshold for peak detection.
    mod_time : float
        The modulation time of the chromatograms.
    """

    files = os.listdir(chromato_dir)
    range_min, range_max = get_mass_range_from_chromatos(chromato_dir)

    for i, file in enumerate(files):
        # read chromatogram and correct baseline
        chromato, time_rn, chromato_cube, sigma, mass_range = (
            read_chromato_and_chromato_cube(
                os.path.join(chromato_dir, file), mod_time=mod_time, pre_process=True
            )
        )

        # get basename of file without extension
        basename = file.split(".")[0]
        dict_chromatos[basename] = (chromato, time_rn)

        # detect peaks
        min_seuil = peak_detection_thr * sigma * 100 / np.max(chromato)
        coordinates = peak_detection(
            (chromato, time_rn, None), None, chromato_cube, min_seuil, None
        )
        coordinates = np.array(sorted(coordinates, key=lambda x: x[0] + x[1]))
        projected_coordinates = projection.matrix_to_chromato(
            points=coordinates,
            time_rn=time_rn,
            mod_time=mod_time,
            chromato_dim=chromato.shape,
        )

        frange_min, frange_max = mass_range
        rows = []
        # Loop through projected coordinates
        for (rt1_idx, rt2_idx), (rt1, rt2) in zip(coordinates, projected_coordinates):
            # Extract spectral data (X columns)
            spectral_data = chromato_cube[:, rt1_idx, rt2_idx]

            # Add zero paddings to the spectral data
            spectral_data = np.pad(
                spectral_data,
                (frange_min - range_min, range_max - frange_max),
                mode="constant",
                constant_values=0,
            )

            # Create a row with RT1, RT2, and the spectral data
            row = [rt1, rt2] + spectral_data.tolist()

            rows.append(row)

        # Create DataFrame from rows
        column_names = ["t1", "t2"] + [f"X{i}" for i in range(range_min, range_max + 1)]
        df = pd.DataFrame(rows, columns=column_names)

        # Save DataFrame to CSV
        output_file = os.path.join(swpa_input_path, f"{file.split('.')[0]}.csv")
        df.to_csv(output_file, index=False)

        print(f"Saved files {i+1}/{len(files)}: {output_file}")


def run_swpa_script(swpa_input_path, ref_filename, dict_chromatos, mod_time=1.25):
    """
    Run the SWPA script to align chromatograms.

    Parameters:
    ----------
    swpa_input_path : str
        The directory path where the SWPA input files are located.
    ref_filename : str
        Name of the reference chromatogram file.
    dict_chromatos : dict
        A dictionary containing the chromatogram data for each chromatogram file.
    mod_time : float
        The modulation time of the chromatograms.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the concatenated data from all matching CSV files
    """
    # compute script path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    swpa_script_path = os.path.join(base_dir, "swpa_peak_alignment.r")

    # prepare arguments
    ref = os.path.join(swpa_input_path, f"{ref_filename}.csv")
    targets = glob.glob(os.path.join(swpa_input_path, "*.csv"))
    targets.pop(targets.index(ref))

    # create directory for output files using date and time as name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(swpa_input_path, f"{current_time}_out")
    os.makedirs(output_dir)

    print("\nScheduling SWPA script with the following arguments:")
    print(f"\t- Output directory: {output_dir}")
    print(f"\t- Reference: {ref}")
    print(f"\t- Targets: {targets}\n")

    # run SWPA script
    p = subprocess.Popen(
        ["Rscript", swpa_script_path, output_dir, ref] + targets, cwd=base_dir
    )
    p.wait()

    matches = load_and_merge_csvs(
        output_dir, ref_filename, dict_chromatos, mod_time=mod_time
    )
    return matches


def load_and_merge_csvs(output_dir, ref_filename, dict_chromatos, mod_time=1.25):
    """
    Load and merge all CSV files in the specified directory.

    This function reads all CSV files produced by the SWPA script in the specified directory,
    concatenates them into a single DataFrame, and adds an additional column that indicates
    the chromatogram name from which each row originated.

    Parameters:
    ----------
    output_dir : str
        The directory path where the swpa output CSV files are located.
    ref_filename : str
        Name of the reference chromatogram file.
    dict_chromatos : dict
        A dictionary containing the chromatogram data for each chromatogram file.
    mod_time : float
        The modulation time of the chromatograms.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the concatenated data from all matching CSV files
    """
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    df_list = []

    ref_chromato, ref_time_rn = dict_chromatos[ref_filename]

    for file in csv_files:
        df = pd.read_csv(file, index_col=0)

        # Add a column with the filename
        cdf_filename = os.path.basename(file).split("_aligned.csv")[0]
        df.insert(0, "filename", cdf_filename)
        df_list.append(df)

        chromato, time_rn = dict_chromatos[cdf_filename]

        # Add target chromatogram matrix coordinates to the DataFrame
        tpoints = np.array(df[["tt1", "tt2"]])
        matrix_coord = projection.chromato_to_matrix(
            points=tpoints,
            time_rn=time_rn,
            mod_time=mod_time,
            chromato_dim=chromato.shape,
        )
        df.insert(3, "tp1", matrix_coord[:, 0])
        df.insert(4, "tp2", matrix_coord[:, 1])

        # Add reference chromatogram matrix coordinates to the DataFrame
        rpoints = np.array(df[["rt1", "rt2"]])
        matrix_coord = projection.chromato_to_matrix(
            points=rpoints,
            time_rn=ref_time_rn,
            mod_time=mod_time,
            chromato_dim=ref_chromato.shape,
        )
        df.insert(7, "rp1", matrix_coord[:, 0])
        df.insert(8, "rp2", matrix_coord[:, 1])

    # Concatenate all the DataFrames into one big DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)

    # Sort the DataFrame by descending combined score
    sim_norm = (merged_df["sim"] - merged_df["sim"].min()) / (
        merged_df["sim"].max() - merged_df["sim"].min()
    )
    dist_norm = 1 - (merged_df["dist"] - merged_df["dist"].min()) / (
        merged_df["dist"].max() - merged_df["dist"].min()
    )

    # Combine the normalized columns into a single score
    merged_df["combined_score"] = sim_norm + dist_norm

    # Sort by filename and combined score
    merged_df.sort_values(
        by=["filename", "combined_score"], ascending=[True, False], inplace=True
    )
    return merged_df


def get_mass_range_from_chromatos(chromato_dir):
    """
    Get the mass range of all chromatograms in the specified directory.

    Parameters:
    ----------
    chromato_dir : str
        The directory path where the chromatogram files are located.

    Returns:
    -------
    int, int
        The minimum and maximum mass values of all chromatograms in the directory.
    """
    files = os.listdir(chromato_dir)

    # Get the mass range of all chromatograms
    range_min, range_max = sys.maxsize, -sys.maxsize
    for file in files:
        ds = nc.Dataset(os.path.join(chromato_dir, file))
        range_min = min(math.ceil(ds["mass_range_min"][0]), range_min)
        range_max = max(math.floor(ds["mass_range_max"][1]), range_max)

    return range_min, range_max
