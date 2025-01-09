import glob
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pandas as pd

path_to_scr_folder = os.path.join(os.path.dirname(os.path.abspath("")), "src")
sys.path.append(path_to_scr_folder)

import projection
from peak_detection import peak_detection
from read_chroma import read_chromato_and_chromato_cube


def swpa_peak_alignment(input_dir, ref_filename, mod_time=1.25, peak_detection_thr=5):
    """
    Preprocess chromatograms and save them as CSV input files for SWPA script.

    Parameters:
    ----------
    input_dir : str
        The directory path where the chromatogram files are located.
    ref_filename : str
        Name of the reference chromatogram file.
    mod_time : float
        The modulation time of the chromatograms.
    peak_detection_thr : float
        The threshold for peak detection.
    """

    all_files = glob.glob(os.path.join(input_dir, "*.cdf"))
    ranges = get_mass_range_from_chromatos(all_files)

    ref_file = os.path.join(input_dir, ref_filename)
    target_files = [
        file for file in all_files if file != ref_file
    ]  # remove ref chromatogram from targets

    with tempfile.TemporaryDirectory() as temp_dir:
        dict_chromatos = {}

        # First, convert the reference chromatogram to CSV
        convert_chromato_to_csv(
            ref_file,
            temp_dir,
            ranges,
            dict_chromatos=dict_chromatos,
            mod_time=mod_time,
            peak_detection_thr=peak_detection_thr,
        )
        print("Converted reference chromatogram to CSV")

        # Next, convert the target chromatograms to CSV
        for i, file in enumerate(target_files):
            convert_chromato_to_csv(
                file,
                temp_dir,
                ranges,
                dict_chromatos=dict_chromatos,
                mod_time=mod_time,
                peak_detection_thr=peak_detection_thr,
            )
            print(f"Converted {i + 1}/{len(target_files)} chromatograms to CSV")

        matches = run_swpa_script(
            temp_dir, ref_filename, dict_chromatos, mod_time=mod_time
        )
        return matches


def convert_chromato_to_csv(
    chromato_path,
    output_dir,
    ranges,
    dict_chromatos=None,
    mod_time=1.25,
    peak_detection_thr=5,
):
    """
    Convert a chromatogram file to a CSV file.

    Parameters:
    ----------
    chromato_path : str
        The path to the chromatogram file.
    output_dir : str
        The directory path where the CSV file will be saved.
    ranges : tuple
        The min/max mass range of the chromatograms.
    dict_chromatos : dict
        A dictionary to store the chromatogram shapes and time ranges.
    mod_time : float
        The modulation time of the chromatograms.
    peak_detection_thr : float
        The threshold for peak detection.
    """
    range_min, range_max = ranges
    chromato, time_rn, chromato_cube, sigma, mass_range = (
        read_chromato_and_chromato_cube(
            chromato_path, mod_time=mod_time, pre_process=True
        )
    )
    chromato_basename = os.path.basename(chromato_path).split(".")[0]
    if dict_chromatos is not None:
        dict_chromatos[chromato_basename] = (chromato.shape, time_rn)

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

    rows = []
    pad_left, pad_right = mass_range[0] - range_min, range_max - mass_range[1]
    # Loop through projected coordinates
    for (rt1_idx, rt2_idx), (rt1, rt2) in zip(coordinates, projected_coordinates):
        # Extract spectral data (X columns)
        spectral_data = chromato_cube[:, rt1_idx, rt2_idx]

        # Add zero paddings to the spectral data
        spectral_data = np.pad(
            spectral_data,
            (pad_left, pad_right),
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
    output_file = os.path.join(output_dir, f"{chromato_basename}.csv")
    df.to_csv(output_file, index=False)


def run_swpa_script(csv_path, ref_filename, dict_chromatos, mod_time=1.25):
    """
    Run the SWPA script to align chromatograms.

    Parameters:
    ----------
    csv_path : str
        The directory path where the SWPA input CSV files are located.
    ref_filename : str
        Name of the reference chromatogram file.
    dict_chromatos : dict
        A dictionary containing the chromatogram shapes and time ranges.
    mod_time : float
        The modulation time of the chromatograms.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the concatenated data from all matching CSV files
    """
    # prepare arguments
    ref = os.path.join(csv_path, f"{ref_filename.split('.')[0]}.csv")
    targets = glob.glob(os.path.join(csv_path, "*.csv"))
    targets.pop(targets.index(ref))  # remove ref chromatogram from targets

    # create directory for output files using date and time as name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(csv_path, f"{current_time}_out")
    os.makedirs(output_dir)

    print("\nScheduling SWPA script with the following arguments:")
    print(f"\t- Output directory: {output_dir}")
    print(f"\t- Reference: {ref}")
    print(f"\t- Targets: {targets}\n")

    # compute script path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    swpa_script_path = os.path.join(base_dir, "swpa_peak_alignment.r")

    # run SWPA script
    p = subprocess.Popen(
        ["Rscript", swpa_script_path, output_dir, ref] + targets, cwd=base_dir
    )
    p.wait()

    matches = load_and_merge_csvs(
        output_dir, ref_filename, dict_chromatos, mod_time=mod_time
    )
    return matches


def load_and_merge_csvs(output_csv_dir, ref_filename, dict_chromatos, mod_time=1.25):
    """
    Load and merge all CSV files in the specified directory.

    This function reads all CSV files produced by the SWPA script in the specified directory,
    concatenates them into a single DataFrame, and adds an additional column that indicates
    the chromatogram name from which each row originated.

    Parameters:
    ----------
    output_csv_dir : str
        The directory path where the swpa output CSV files are located.
    ref_filename : str
        Name of the reference chromatogram file.
    dict_chromatos : dict
        A dictionary containing the chromatogram shapes and time ranges.
    mod_time : float
        The modulation time of the chromatograms.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the concatenated data from all matching CSV files
    """
    csv_files = glob.glob(os.path.join(output_csv_dir, "*.csv"))
    df_list = []

    ref_chromato_shape, ref_time_rn = dict_chromatos[ref_filename.split(".")[0]]

    for file in csv_files:
        df = pd.read_csv(file, index_col=0)

        # Add a column with the filename
        cdf_filename = os.path.basename(file).split(".csv")[0]
        df.insert(0, "filename", cdf_filename)

        chromato_shape, time_rn = dict_chromatos[cdf_filename]

        # Add target chromatogram matrix coordinates to the DataFrame
        tpoints = np.array(df[["tt1", "tt2"]])
        matrix_coord = projection.chromato_to_matrix(
            points=tpoints,
            time_rn=time_rn,
            mod_time=mod_time,
            chromato_dim=chromato_shape,
        )
        df.insert(3, "tp1", matrix_coord[:, 0])
        df.insert(4, "tp2", matrix_coord[:, 1])

        # Add reference chromatogram matrix coordinates to the DataFrame
        rpoints = np.array(df[["rt1", "rt2"]])
        matrix_coord = projection.chromato_to_matrix(
            points=rpoints,
            time_rn=ref_time_rn,
            mod_time=mod_time,
            chromato_dim=ref_chromato_shape,
        )
        df.insert(7, "rp1", matrix_coord[:, 0])
        df.insert(8, "rp2", matrix_coord[:, 1])

        # Add the DataFrame to the list
        df_list.append(df)

    # Concatenate all the DataFrames into one big DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)

    # Sort the DataFrame by descending combined score of similarity and distance
    sim_norm = (merged_df["sim"] - merged_df["sim"].min()) / (
        merged_df["sim"].max() - merged_df["sim"].min()
    )
    dist_norm = (merged_df["dist"] - merged_df["dist"].min()) / (
        merged_df["dist"].max() - merged_df["dist"].min()
    )

    # Combine the normalized columns into a single score
    merged_df["combined_sim_dist"] = sim_norm + dist_norm

    # Sort by filename and combined score
    merged_df.sort_values(
        by=["filename", "combined_sim_dist"], ascending=[True, False], inplace=True
    )
    return merged_df


def get_mass_range_from_chromatos(chromatos_path):
    """
    Get the mass range of all chromatograms in the specified directory.

    Parameters:
    ----------
    chromatos_path : str
        The paths to the chromatogram files.

    Returns:
    -------
    int, int
        The minimum and maximum mass values of all chromatograms in the directory.
    """
    # Get the mass range of all chromatograms
    range_min, range_max = sys.maxsize, -sys.maxsize
    for filepath in chromatos_path:
        ds = nc.Dataset(filepath, "r")
        range_min = min(math.ceil(ds["mass_range_min"][:].min()), range_min)
        range_max = max(math.floor(ds["mass_range_max"][:].max()), range_max)

    return range_min, range_max
