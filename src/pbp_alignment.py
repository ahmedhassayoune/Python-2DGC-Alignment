import os
import time
import warnings

import numpy as np
from netCDF4 import Dataset  # For reading/writing NetCDF files
from scipy.interpolate import griddata, interp1d

#TODO: add comment to tell automatic transformation if using peaks instead of pixels
#TODO: check if we should close something with netCDF4
#TODO: handle global parameters in a better way (use dictionary)
#TODO: implement natural-neighbor interpolation
#TODO: implement DualSibson model

# ---------------------------------------------------------------------
# INSTRUMENT PARAMETERS
PRECISION = 0.01  # Precision for rounding of M/S values
INTTHRESHOLD = 0  # Threshold value for signal intensity
NBPIX2NDD_TARGET = 160  # Number of pixels for 2nd dimension (Target chromatogram)
NBPIX2NDD_REF = 160  # Number of pixels for 2nd dimension (Reference chromatogram)
DRIFTMS = 0  # Drift mass spectrum adjustment

# MODEL CHOICE PARAMETERS
TYPICAL_PEAK_WIDTH = [1, 5]  # Typical width of a peak in first and second dimension
MODEL_CHOICE = "normal"  # Model choice ('normal' or 'DualSibson')
UNITS = "pixel"  # Units for typical peak width and alignment points

# I/O PARAMETERS
OUTPUT_PATH = "/home/ahassayoune/GCxGC-MS-alignment/users/output/"
INPUT_PATH = "/home/ahassayoune/GCxGC-MS-alignment/users/input/"
REFERENCE_CHROMATOGRAM_FILE = "Test_reference_chromatogram.cdf"
TARGET_CHROMATOGRAM_FILE = "Test_target_chromatogram.cdf"
REFERENCE_ALIGNMENT_PTS_FILE = "Alignment_pts_Reference.csv"
TARGET_ALIGNMENT_PTS_FILE = "Alignment_pts_Target.csv"
# ---------------------------------------------------------------------


def open_chromatogram(FileName, Intthreshold=INTTHRESHOLD, driftMS=DRIFTMS):
    # Open the CDF file
    with Dataset(FileName, "r") as cdf:
        # Retrieve data from the file
        flag = cdf.variables["flag_count"][:]
        scantime = cdf.variables["scan_acquisition_time"][:]
        scannum = cdf.variables["actual_scan_number"][:]
        medmsmax = cdf.variables["mass_range_max"][:]
        medmsmin = cdf.variables["mass_range_min"][:]
        ionid = cdf.variables["scan_index"][:]
        eachscannum = cdf.variables["point_count"][:]
        MStotint = cdf.variables["total_intensity"][:]
        MSvalue = cdf.variables["mass_values"][:] + driftMS
        MSint = cdf.variables["intensity_values"][:]

    # Compute Time parameters
    Timepara = scantime[np.abs(eachscannum) < np.iinfo(np.int32).max]
    RTini = np.min(Timepara) / 60
    RTruntime = np.max(Timepara) / 60 - RTini
    SamRate = 1 / np.mean(Timepara[1:] - Timepara[:-1])

    # Initialize data arrays
    pixelnum = len(scannum)
    maxscannum = np.max(eachscannum)
    MSdatabox = np.zeros((pixelnum, maxscannum + 1))
    MSvaluebox = np.zeros((pixelnum, maxscannum + 1))
    MSintbox = np.zeros((pixelnum, maxscannum + 1))

    # Populate MS data boxes
    for inum in range(pixelnum):
        if abs(eachscannum[inum]) < np.iinfo(np.int32).max:
            initial = np.sum(eachscannum[:inum]) - eachscannum[inum] + 1
            acqrange = (
                np.arange(
                    min(initial, initial + eachscannum[inum] - 1),
                    max(initial, initial + eachscannum[inum] - 1) + 1,
                )
                if eachscannum[inum] != 0
                else []
            )

            remainrep = np.zeros(maxscannum - eachscannum[inum] + 1)
            MSvaluebox[inum, :] = np.concatenate([MSvalue[acqrange], remainrep])
            MSintbox[inum, :] = np.concatenate([MSint[acqrange], remainrep])

    # Apply intensity threshold
    MSvaluebox[MSintbox < Intthreshold] = 0
    MSintbox[MSintbox < Intthreshold] = 0

    # Organize output into a dictionary
    Chromato = {
        "flag": flag,
        "scantime": scantime,
        "scannum": scannum,
        "medmsmax": medmsmax,
        "medmsmin": medmsmin,
        "eachscannum": eachscannum,
        "MStotint": MStotint,
        "MSvalue": MSvalue,
        "MSint": MSint,
        "Timepara": Timepara,
        "RTini": RTini,
        "RTruntime": RTruntime,
        "SamRate": SamRate,
        "MSthreshold": Intthreshold,
        "pixelnum": pixelnum,
        "maxscannum": maxscannum,
        "minscannum": np.min(eachscannum),
        "MSdatabox": MSdatabox,
        "MSvaluebox": MSvaluebox,
        "MSintbox": MSintbox,
        "ionid": ionid,
    }

    return Chromato


def load_chromatograms(input_path, target_file, reference_file):
    target_chromato_path = os.path.join(input_path, target_file)
    reference_chromato_path = os.path.join(input_path, reference_file)

    if not os.path.exists(target_chromato_path):
        raise FileNotFoundError(
            f"Target chromatogram file {target_chromato_path} does not exist."
        )
    if not os.path.exists(reference_chromato_path):
        raise FileNotFoundError(
            f"Reference chromatogram file {reference_chromato_path} does not exist."
        )

    ChromatoTarget = open_chromatogram(target_chromato_path)
    ChromatoRef = open_chromatogram(reference_chromato_path)

    return ChromatoTarget, ChromatoRef


def time_to_pix(rttime, mod, freq, isot=0):
    """
    Converts retention times from units of time to units of pixels.

    Parameters:
    - rttime: numpy array of shape (n, 2), where each row is [retention time 1, retention time 2]
    - mod: Modulation rate, in seconds
    - freq: Sampling frequency, in Hz
    - isot: Optional suppression time at the beginning of the chromatogram, in minutes (default is 0)

    Returns:
    - rtpix: numpy array of shape (n, 2), where each row corresponds to [pixel 1, pixel 2]
    """
    # Ensure `isot` is in minutes if not provided
    rtpix = np.zeros_like(rttime, dtype=int)
    # Calculate pixels based on time values
    rtpix[:, 1] = np.round(rttime[:, 1] * freq).astype(int)
    rtpix[:, 0] = np.round((rttime[:, 0] - isot) * 60 / mod).astype(int)

    return rtpix


def load_alignment_points(
    reference_file,
    target_file,
    input_path,
    output_path,
    ChromatoRef,
    ChromatoTarget,
    units=UNITS,
    NbPix2ndD_ref=NBPIX2NDD_REF,
    NbPix2ndD_target=NBPIX2NDD_TARGET,
    typical_peak_width=TYPICAL_PEAK_WIDTH,
):
    """
    Imports and processes alignment points for reference and target chromatograms.

    Parameters:
    - reference_file: str, name of the reference alignment points file
    - target_file: str, name of the target alignment points file
    - input_path: str, path to the input directory
    - output_path: str, path to the output directory
    - units: str, whether alignment points are in 'time' or 'pixels'
    - time_to_pix: function to convert time to pixels
    - NbPix2ndD_ref: number of pixels in the second dimension for the reference
    - NbPix2ndD_target: number of pixels in the second dimension for the target
    - ChromatoRef: chromatogram object with `SamRate` and `RTini` attributes for the reference
    - ChromatoTarget: chromatogram object with `SamRate` and `RTini` attributes for the target
    - typical_peak_width: typical peak width in time to be converted

    Returns:
    - Reference_peaks, Target_peaks, typical_peak_width: numpy arrays
    """

    # Helper function to find and load a file
    def load_file(filename, input_path, output_path):
        input_full_path = os.path.join(input_path, filename)
        output_full_path = os.path.join(output_path, filename)

        # Check if the file exists in the input or output path
        if os.path.exists(input_full_path):
            data = np.genfromtxt(input_full_path, delimiter=",")
        elif os.path.exists(output_full_path):
            warnings.warn(
                f"The file {filename} was found in the output_path, not in input_path. Loading from output_path."
            )
            data = np.genfromtxt(output_full_path, delimiter=",")
        else:
            raise FileNotFoundError(
                f"The file {filename} does not exist in either input or output paths."
            )

        # Ensure data has two columns and return
        if data.shape[1] == 2:
            return data
        else:
            raise ValueError(
                f"The file {filename} does not have 2 columns as expected."
            )

    # Load reference and target peaks
    reference_peaks = load_file(reference_file, input_path, output_path)
    target_peaks = load_file(target_file, input_path, output_path)

    if units.lower() == "time":
        # Convert time units to pixel units
        target_peaks = time_to_pix(
            target_peaks,
            NbPix2ndD_target / ChromatoTarget["SamRate"],
            ChromatoTarget["SamRate"],
            ChromatoTarget["RTini"],
        )
        reference_peaks = time_to_pix(
            reference_peaks,
            NbPix2ndD_ref / ChromatoRef["SamRate"],
            ChromatoRef["SamRate"],
            ChromatoRef["RTini"],
        )
        typical_peak_width = time_to_pix(
            typical_peak_width,
            NbPix2ndD_ref / ChromatoRef["SamRate"],
            ChromatoRef["SamRate"],
            ChromatoRef["RTini"],
        )
        global TYPICAL_PEAK_WIDTH
        TYPICAL_PEAK_WIDTH = typical_peak_width

    return reference_peaks, target_peaks


def reshape_tic(MStotint, NbPix2ndD):
    """
    Reshapes the MStotint array into a 2D array with rows of size NbPix2ndD,
    padding with zeros if necessary.
    
    Parameters:
        MStotint (numpy.ndarray): Input 1D array of intensities.
        NbPix2ndD (int): Number of rows for the reshaped array.
    
    Returns:
        numpy.ndarray: Reshaped 2D array.
    """
    # Calculate the padding length to make the array divisible by NbPix2ndD
    pad_length = NbPix2ndD - (len(MStotint) % NbPix2ndD)
    if pad_length == NbPix2ndD:  # No padding needed
        pad_length = 0
    
    # Pad the array with zeros
    padded_MStotint = np.pad(MStotint, (0, pad_length), mode='constant')
    
    # Reshape the padded array
    return np.reshape(padded_MStotint, (NbPix2ndD, -1), order='F')


def MSdataRound_v2(MSvaluebox, MSintbox, Precision=PRECISION):
    """
    Rounds m/z values and combines intensity values for the same m/z in the MS data.

    Parameters:
    - MSvaluebox: numpy array of shape (n, m) containing m/z values
    - MSintbox: numpy array of shape (n, m) containing intensity values corresponding to m/z values
    - Precision: float, the precision to which the m/z values should be rounded (e.g. 0.001)

    Returns:
    - MSvalueboxRounded: numpy array of shape (n, k), rounded m/z values
    - MSintboxRounded: numpy array of shape (n, k), summed intensity values corresponding to the rounded m/z values
    """
    # Round m/z values according to the specified precision
    MSvalueboxII = np.round(MSvaluebox * (1 / Precision)) * Precision
    MSintboxII = MSintbox.copy()  # Keep intensity values unchanged

    # Prepare the output matrices (A for intensities, B for m/z values)
    A = np.zeros_like(MSvalueboxII)
    B = np.zeros_like(MSvalueboxII)

    # Loop through each row (each chromatogram)
    for kt in range(MSvalueboxII.shape[0]):
        # Initialize the first element in the row
        B[kt, 0] = MSvalueboxII[kt, 0]
        A[kt, 0] = MSintboxII[kt, 0]

        # Counter to track the number of unique m/z values in the row
        Cnt = 0

        # Loop through the rest of the columns (m/z values in the current chromatogram row)
        for rr in range(MSvalueboxII.shape[1]):
            # If the m/z value already exists in B, add the intensity to A
            if MSvalueboxII[kt, rr] == B[kt, Cnt]:
                A[kt, Cnt] += MSintboxII[kt, rr]
            else:
                # If not, move to the next position in B and A
                Cnt += 1
                B[kt, Cnt] = MSvalueboxII[kt, rr]
                A[kt, Cnt] = MSintboxII[kt, rr]

    # Remove trailing zeros in B and A
    MaxNotZero2 = np.max(np.sum(B != 0, axis=1))

    # Trim the arrays to remove excess zero elements
    MSvalueboxRounded = B[:, :MaxNotZero2]
    MSintboxRounded = A[:, :MaxNotZero2]

    return MSvalueboxRounded, MSintboxRounded


def align_2d_chrom_ms_v5(
    Ref, Other, Peaks_Ref, Peaks_Other, MSvaluebox, MSintbox, NbPix2ndD, **kwargs
):
    """
    Aligns 2D chromatograms with MS data.

    Parameters:
        Ref (np.ndarray): Reference chromatogram of size [m, n].
        Other (np.ndarray): Target chromatogram of size [m, n].
        Peaks_Ref (np.ndarray): Alignment points in the reference chromatogram.
        Peaks_Other (np.ndarray): Alignment points in the target chromatogram.
        MSvaluebox (np.ndarray): m/z values of ions.
        MSintbox (np.ndarray): Corresponding intensity values of ions.
        NbPix2ndD (int): Number of pixels per 2nd dimension column.
        kwargs (dict): Optional arguments:
            - Interp_meth (str): Interpolation method to use.
            - PowerFactor (float): Weighting factor for deformation correction.
            - Peak_widths (list): Expected widths of peaks.
            - InterPixelInterpMeth (str): Method for inter-pixel interpolation.
            - model_choice (str): Choice of model for alignment: 'normal' or 'DualSibson'.

    Returns:
        dict: A dictionary containing alignment results:
            - AlignedMSvaluebox (np.ndarray)
            - AlignedMSintbox (np.ndarray)
            - Alignedeachscannum (np.ndarray)
            - Alignedmedionid (np.ndarray)
            - Aligned (np.ndarray)
            - Displacement (np.ndarray)
            - Deform_output (np.ndarray)
    """

    # Default values for optional arguments
    Interp_meth = kwargs.get("Interp_meth", "natural-neighbor")
    PowerFactor = kwargs.get("PowerFactor", 2.0)
    Peak_widths = kwargs.get("Peak_widths", [1, 1])
    InterPixelInterpMeth = kwargs.get("InterPixelInterpMeth", "linear")
    model_choice = kwargs.get("model_choice", "normal")

    if model_choice == "normal":
        Aligned, Displacement, Deform_output = align_chromato(
            Ref=Ref,
            Target=Other,
            Peaks_Ref=Peaks_Ref,
            Peaks_Target=Peaks_Other,
            DisplacementInterpMeth=Interp_meth,
            PowerFactor=PowerFactor,
            Peak_widths=Peak_widths,
            InterPixelInterpMeth=InterPixelInterpMeth,
        )
    elif model_choice == "DualSibson":
        # TODO: implement DualSibson model
        # Aligned, Displacement, Deform_output = (
        #     alignChromato_with_SibsonInterp_also_1stD()
        # )
        pass
    else:
        raise ValueError(f"Invalid model choice: {model_choice}.")

    MSpixelsInds = np.arange(MSvaluebox.shape[0])
    AlignedInds = [np.zeros_like(MSpixelsInds) for _ in range(4)]
    AlignedMSvaluebox = [np.zeros_like(MSvaluebox) for _ in range(4)]
    AlignedMSintbox = [np.zeros_like(MSintbox) for _ in range(4)]
    Interp_distr = np.zeros_like(MSpixelsInds, dtype=float)
    Interp_dists = np.zeros_like(MSpixelsInds, dtype=float)
    Interp_distt = np.zeros_like(MSpixelsInds, dtype=float)
    Interp_distu = np.zeros_like(MSpixelsInds, dtype=float)

    Defm = [np.zeros_like(MSpixelsInds, dtype=float) for _ in range(4)]

    # -------------------------
    FrstDFlag, ScndDFlag = 0, 0
    for ht in range(len(AlignedInds[0])):
        # Compute r, s, t, u for interpolation
        Interp_distr[ht] = Displacement[ScndDFlag, FrstDFlag, 1] - np.floor(
            Displacement[ScndDFlag, FrstDFlag, 1]
        )
        Interp_dists[ht] = -Displacement[ScndDFlag, FrstDFlag, 1] + np.ceil(
            Displacement[ScndDFlag, FrstDFlag, 1]
        )
        Interp_distt[ht] = Displacement[ScndDFlag, FrstDFlag, 0] - np.floor(
            Displacement[ScndDFlag, FrstDFlag, 0]
        )
        Interp_distu[ht] = -Displacement[ScndDFlag, FrstDFlag, 0] + np.ceil(
            Displacement[ScndDFlag, FrstDFlag, 0]
        )

        # Update pixel counts
        if (ht+1) % NbPix2ndD != 0:
            ScndDFlag += 1
        else:
            FrstDFlag += 1
            ScndDFlag = 0

    # Correct indices that are outside the chromatogram
    Interp_distr[Interp_distr == 0] = 0.5
    Interp_dists[Interp_dists == 0] = 0.5
    Interp_distt[Interp_distt == 0] = 0.5
    Interp_distu[Interp_distu == 0] = 0.5
    # -------------------------

    # Process for each corner (a, b, c, d)
    for corner in range(4):
        FrstDFlag, ScndDFlag = 0, 0
        aligned_indices = AlignedInds[corner]
        MSvaluebox_aligned = AlignedMSvaluebox[corner]
        MSintbox_aligned = AlignedMSintbox[corner]
        Def = Defm[corner]

        for ht in range(len(aligned_indices)):
            aligned_indices[ht], Def[ht] = compute_alignment(
                ht, Displacement, NbPix2ndD, ScndDFlag, FrstDFlag, Deform_output, corner
            )

            # Update pixel counts
            if (ht+1) % NbPix2ndD != 0:
                ScndDFlag += 1
            else:
                FrstDFlag += 1
                ScndDFlag = 0

        # Correct indices and handle out-of-bounds
        aligned_indices[aligned_indices > np.max(MSpixelsInds)] = 0
        aligned_indices[aligned_indices < 0] = 0

        # Populate MS values and intensities
        LpInds = np.arange(len(aligned_indices))
        LpInds2 = LpInds[aligned_indices != 0]

        del LpInds

        for ht in LpInds2:
            MSvaluebox_aligned[ht, :] = MSvaluebox[aligned_indices[ht], :]
            MSintbox_aligned[ht, :] = (
                MSintbox[aligned_indices[ht], :]
                * Interp_dists[ht]
                * Interp_distu[ht]
                * Def[ht]
            )

        del LpInds2

    del aligned_indices

    # Put the 4 corners interpolated values in the matrices
    AlignedMSvalueboxI = np.concatenate(AlignedMSvaluebox, axis=1)
    AlignedMSintboxI = np.concatenate(AlignedMSintbox, axis=1)

    # Remove zeros temporarily by replacing them with the maximum integer value
    AlignedMSvalueboxI[AlignedMSvalueboxI == 0] = np.iinfo(np.int32).max

    # Sort the values in ascending order (big values corresponding to zeros go at the end)
    # Sorting by rows
    IX = np.argsort(AlignedMSvalueboxI, axis=1)
    AlignedMSvalueboxI.sort(axis=1)

    # Re-put zeros instead of the big values
    AlignedMSvalueboxI[AlignedMSvalueboxI == np.iinfo(np.int32).max] = 0

    # Put the values in AlignedMSintboxI in the corresponding order
    AlignedMSintboxI = np.take_along_axis(AlignedMSintboxI, IX, axis=1)

    # Find the max of non-zero values (to remove useless zeros)
    MaxNotZero = np.max(np.sum(AlignedMSintboxI != 0, axis=1))

    # Remove useless zeros
    AlignedMSvalueboxII = AlignedMSvalueboxI[:, :MaxNotZero]
    AlignedMSintboxII = AlignedMSintboxI[:, :MaxNotZero]

    # Clear variables that are no longer needed
    del AlignedMSintboxI, AlignedMSvalueboxI

    # -- Step 3: For each pixel, only keep each m/z value once, summing corresponding intensity values
    # Initialize matrices for aggregated m/z values and intensities
    unique_mz_values = np.zeros_like(AlignedMSvalueboxII)
    aggregated_intensities = np.zeros_like(AlignedMSvalueboxII)

    for kt in range(AlignedMSvalueboxII.shape[0]):
        # Get unique m/z values and their indices
        unique_mz, indices = np.unique(AlignedMSvalueboxII[kt, :], return_inverse=True)
        # Aggregate corresponding intensities
        aggregated_values = np.zeros_like(unique_mz, dtype=AlignedMSintboxII.dtype)
        np.add.at(aggregated_values, indices, AlignedMSintboxII[kt, :])

        # Update results in the final arrays
        unique_mz_values[kt, : len(unique_mz)] = unique_mz
        aggregated_intensities[kt, : len(unique_mz)] = aggregated_values

    # Remove useless zeros
    MaxNotZero2 = np.max(np.sum(unique_mz_values != 0, axis=1))

    final_mz_values = unique_mz_values[:, :MaxNotZero2]
    final_intensities = aggregated_intensities[:, :MaxNotZero2]

    Alignedeachscannum = np.full(
        (final_intensities.shape[0], 1), final_intensities.shape[1]
    )
    Alignedmedionid = np.cumsum(Alignedeachscannum)

    return {
        "MSvaluebox": final_mz_values,
        "MSintbox": final_intensities,
        "eachscannum": Alignedeachscannum,
        "ionid": Alignedmedionid,
        "Aligned": Aligned,
        "Displacement": Displacement,
        "Deform_output": Deform_output,
    }


def compute_alignment(
    ht, Displacement, NbPix2ndD, ScndDFlag, FrstDFlag, Deform_output, corner
):
    """
    Computes the alignment for a given corner (a, b, c, or d).
    """
    if corner == 0:  # Corner a (floor-floor)
        aligned_index = np.floor(
            Displacement[ScndDFlag, FrstDFlag, 0]
        ) + NbPix2ndD * np.floor(Displacement[ScndDFlag, FrstDFlag, 1])
    elif corner == 1:  # Corner b (ceil-floor)
        aligned_index = np.floor(
            Displacement[ScndDFlag, FrstDFlag, 0]
        ) + NbPix2ndD * np.ceil(Displacement[ScndDFlag, FrstDFlag, 1])
    elif corner == 2:  # Corner c (floor-ceil)
        aligned_index = np.ceil(
            Displacement[ScndDFlag, FrstDFlag, 0]
        ) + NbPix2ndD * np.floor(Displacement[ScndDFlag, FrstDFlag, 1])
    else:  # Corner d (ceil-ceil)
        aligned_index = np.ceil(
            Displacement[ScndDFlag, FrstDFlag, 0]
        ) + NbPix2ndD * np.ceil(Displacement[ScndDFlag, FrstDFlag, 1])

    Def = (
        Deform_output[ScndDFlag, FrstDFlag, 0]
        * Deform_output[ScndDFlag, FrstDFlag, 1]
        / 4
    )
    return aligned_index + ht, Def


def align_chromato(Ref, Target, Peaks_Ref, Peaks_Target, **kwargs):
    """
    Aligns the target chromatogram to the reference chromatogram.

    Parameters:
        ref (numpy.ndarray): Reference chromatogram (2D matrix).
        target (numpy.ndarray): Target chromatogram (2D matrix).
        Peaks_Ref (numpy.ndarray): Positions of alignment points in ref (Nx2 matrix).
        Peaks_Target (numpy.ndarray): Positions of alignment points in target (Nx2 matrix).
        **kwargs: Optional parameters:
            - 'DisplacementInterpMeth' (str): Interpolation method, default 'natural-neighbor'.
            - 'PowerFactor' (float): Weighting factor for inverse distance, default 2.
            - 'Peak_widths' (list): Typical peak widths [width_1st_dim, width_2nd_dim], default [1, 1].
            - 'InterPixelInterpMeth' (str): Interpolation method for intensity, default 'cubic'.

    Returns:
        aligned (numpy.ndarray): Aligned chromatogram.
        displacement (numpy.ndarray): Displacement matrix [m, n, 2].
        deform_output (numpy.ndarray): Deformation correction matrix [m, n, 2].
    """
    # Default optional arguments
    displacement_interp_meth = kwargs.get("DisplacementInterpMeth", "natural-neighbor")
    PowerFactor = kwargs.get("PowerFactor", 2)
    peak_widths = kwargs.get("Peak_widths", [1, 1])
    InterPixelInterpMeth = kwargs.get("InterPixelInterpMeth", "cubic")

    # Ensure ref and target have the same size
    Ref, Target = equalize_size(Ref, Target)

    # Normalize distances
    PeakWidth2ndD = peak_widths[1]
    PeakWidth1stD = peak_widths[0]

    # Compute displacement
    Peaks_displacement = Peaks_Target - Peaks_Ref

    # Initialize output arrays
    Aligned = np.zeros_like(Target)
    Displacement = np.zeros((Target.shape[0], Target.shape[1], 2))

    # -- Compute displacement of peaks and interpolate (1st dim: linear, 2nd dim: natural-neighbor)
    # Initialize Displacement2 array
    Displacement2 = np.zeros((2, 2, 2))

    padding_w_lower = np.floor(0.05 * Aligned.shape[0])
    padding_w_upper = np.ceil(0.05 * Aligned.shape[0])
    padding_x_lower = np.floor(0.05 * Aligned.shape[1])
    padding_x_upper = np.ceil(0.05 * Aligned.shape[1])

    # Compute Displacement2 based on the alignment and reference peaks
    # w: 2nd dimension, x: 1st dimension
    for w in (-padding_w_lower, Aligned.shape[0] + padding_w_upper):
        for x in (-padding_x_lower, Aligned.shape[1] + padding_x_upper):
            # Compute the distance vector for the pixel
            Distance_vec = np.array([w, x]) - np.flip(Peaks_Ref, axis=1)
            Distance = np.sqrt(
                Distance_vec[:, 0] ** 2
                + (Distance_vec[:, 1] * PeakWidth2ndD / PeakWidth1stD) ** 2
            )

            # Compute the displacement using a weighted mean
            weight = 1 / (Distance**PowerFactor)
            d2w = (w + int(padding_w_lower)) // (
                Aligned.shape[0] + int(padding_w_lower) + int(padding_w_upper)
            )
            d2x = (x + int(padding_x_lower)) // (
                Aligned.shape[1] + int(padding_x_lower) + int(padding_x_upper)
            )
            Displacement2[int(d2w), int(d2x), :] = np.sum(
                np.flip(Peaks_displacement, axis=1) * (weight[:, np.newaxis]), axis=0
            ) / np.sum(weight)

    # Add a peak at each corner (around the chromatogram with an offset)
    Peaks_Ref = np.vstack(
        [
            Peaks_Ref,
            # base corners with offsets
            np.array(
                [
                    [-padding_x_lower, -padding_w_lower],
                    [-padding_x_lower, Aligned.shape[0] + padding_w_upper],
                    [Aligned.shape[1] + padding_x_upper, -padding_w_lower],
                    [
                        Aligned.shape[1] + padding_x_upper,
                        Aligned.shape[0] + padding_w_upper,
                    ],
                ]
            ),
        ]
    )

    # Add corresponding displacement values for the added peaks
    Peaks_displacement = np.vstack(
        [
            Peaks_displacement,
            np.array(
                [
                    [Displacement2[0, 0, 1], Displacement2[0, 0, 0]],
                    [Displacement2[1, 0, 1], Displacement2[1, 0, 0]],
                    [Displacement2[0, 1, 1], Displacement2[0, 1, 0]],
                    [Displacement2[1, 1, 1], Displacement2[1, 1, 0]],
                ]
            ),
        ]
    )

    points = np.column_stack(
        (Peaks_Ref[:, 1], Peaks_Ref[:, 0] * PeakWidth2ndD / PeakWidth1stD)
    )

    gridw, gridx = np.mgrid[
        -int(padding_w_lower) : Aligned.shape[0] + int(padding_w_upper),
        -int(padding_x_lower * PeakWidth2ndD / PeakWidth1stD) : int((Aligned.shape[1] + padding_x_upper)* PeakWidth2ndD / PeakWidth1stD),
    ]

    Fdist1 = griddata(points, values=Peaks_displacement[:, 1], xi=(gridw, gridx), method="cubic")

    # TODO: natural-neighbor
    # Perform the natural-neighbor interpolation on 2nd dimension of the displacement
    # Fdist1 = NearestNDInterpolator(Peaks_Ref[:, [1, 0]] * PeakWidth2ndD / PeakWidth1stD, Peaks_displacement[:, 1], fill_value='extrapolate')

    Hep = np.vstack([Peaks_Ref[:-4, 0], Peaks_displacement[:-4, 0]]).T
    Hep1 = Hep[:, 0]
    Hep2 = Hep[:, 1]
    Herp = []

    for hHh in range(len(Hep)):
        Herp.append([Hep1[hHh], np.mean(Hep2[Hep1 == Hep1[hHh]])])

    Herp = np.array(Herp)

    Hap = []
    k = 0
    puet = 1

    for kui in range(len(Herp)):
        for zui in range(len(Hap)):
            if Herp[kui, 0] == Hap[zui][0]:
                puet = 0
                Peaks_displacement[:-4][Herp[:, 0] == Hap[zui][0], 0] = Herp[zui, 1]
        if puet:
            Hap.append(Herp[kui])
            k += 1
        else:
            puet = 1

    Hap = np.array(Hap)

    # Linear interpolation inside the convex hull
    Hum = interp1d(Hap[:, 0], Hap[:, 1], kind="linear", fill_value="extrapolate")(
        np.arange(Target.shape[1])
    )

    # Linear extrapolation for outside the convex hull
    Pks, Displ1D = Peaks_Ref[:-4, 0], Peaks_displacement[:-4, 0]
    minPks, maxPks = np.min(Pks), np.max(Pks)
    Hum2 = interp1d(
        [
            minPks,
            maxPks
        ],
        [
            np.mean(Displ1D[Pks == minPks]),
            np.mean(Displ1D[Pks == maxPks]),
        ],
        kind="linear",
        fill_value="extrapolate",
    )(np.arange(Target.shape[1]+2))

    # Populate the displacement matrix
    for w in range(Aligned.shape[0]):
        for x in range(Aligned.shape[1]):
            if Aligned[w, x] == 0:
                if min(Peaks_Ref[:-4, 0]) <= x <= max(Peaks_Ref[:-4, 0]):
                    Displacement[w, x, :] = [
                        Fdist1[w, int(x * PeakWidth2ndD / PeakWidth1stD)],
                        Hum[x],
                    ]
                else:
                    Displacement[w, x, :] = [
                        Fdist1[w, int(x * PeakWidth2ndD / PeakWidth1stD)],
                        Hum2[x],
                    ]

    # Prepare the output grid (X,Y,Z)
    X = np.ones((Ref.shape[0] * 2, 1)) * np.arange(Ref.shape[1])
    Y = np.arange(
        -round(Ref.shape[0] / 2),
        Ref.shape[0] + (Ref.shape[0] - round(Ref.shape[0] / 2)),
    ).reshape(-1, 1) * np.ones((1, Ref.shape[1])) #TODO: check reshape order

    Z = np.zeros((Target.shape[0] * 2, Target.shape[1]))
    Z[: int(np.floor(Target.shape[0] / 2)), 1:] = Target[
        int(np.floor(Target.shape[0] / 2)) :, :-1
    ]
    Z[
        int(np.round(Target.shape[0] / 2)) : int(np.round(Target.shape[0] / 2))
        + Target.shape[0],
        :,
    ] = Target
    Z[int(np.floor(Target.shape[0] / 2)) + Target.shape[0] :, :-1] = Target[
        : int(np.ceil(Target.shape[0] / 2)), 1:
    ]

    def interpolate_2d(X, Y, Z, Xq, Yq, method):
        points = np.column_stack((X.ravel(), Y.ravel()))
        values = Z.ravel()
        queries = np.column_stack((Xq.ravel(), Yq.ravel()))
        Aligned = griddata(points, values, queries, method=method, fill_value=0)
        return Aligned.reshape(Xq.shape) #TODO: check reshape order

    mid_idx = round(Target.shape[0] / 2)
    Xq = X[mid_idx : (mid_idx + Target.shape[0]), :] + Displacement[:, :, 1]
    Yq = Y[mid_idx : (mid_idx + Target.shape[0]), :] + Displacement[:, :, 0]

    if InterPixelInterpMeth in ["spline", "cubic", "linear"]:
        method = "cubic" if InterPixelInterpMeth == "spline" else InterPixelInterpMeth
        Aligned = interpolate_2d(X, Y, Z, Xq, Yq, method)
    else:
        raise ValueError(f"Unsupported interpolation method: {InterPixelInterpMeth}")

    Aligned[np.isnan(Aligned)] = 0
    for k in range(len(Peaks_displacement) - 4):
        Displacement[int(Peaks_Ref[k, 1]), int(Peaks_Ref[k, 0]), :] = Peaks_displacement[k, ::-1]

    # Initialize deformation arrays
    Deform1 = np.zeros_like(Aligned)
    Deform2 = np.zeros_like(Aligned)

    # Extend Displacement with borders
    Displacement_Extended = np.zeros((Aligned.shape[0] + 2, Aligned.shape[1] + 2, 2))
    Displacement_Extended[1:-1, 1:-1, :] = Displacement

    points = np.column_stack(
        (Peaks_Ref[:, 1] + 1, (Peaks_Ref[:, 0] + 1) * PeakWidth2ndD / PeakWidth1stD)
    )
    Fdist1bis = griddata(
        points, values=Peaks_displacement[:, 1], xi=(gridw, gridx), method="cubic"
    )
    # TODO: Fdist1bis = naturalneighbor

    for w in [0, Aligned.shape[0] + 1]:
        for x in range(Aligned.shape[1] + 1):
            if min(Peaks_Ref[:-4, 0]) <= x <= max(Peaks_Ref[:-4, 0]):
                Displacement_Extended[w, x, :] = [
                    Fdist1bis[w, int(x * PeakWidth2ndD / PeakWidth1stD)],
                    Hum[x],
                ]
            else:
                Displacement_Extended[w, x, :] = [
                    Fdist1bis[w, int(x * PeakWidth2ndD / PeakWidth1stD)],
                    Hum2[x],
                ]

    for w in range(1, Aligned.shape[0]):
        for x in [0, Aligned.shape[1] + 1]:
            if min(Peaks_Ref[:-4, 0]) <= x <= max(Peaks_Ref[:-4, 0]):
                Displacement_Extended[w, x, :] = [
                    Fdist1bis[w, int(x * PeakWidth2ndD / PeakWidth1stD)],
                    Hum[x],
                ]
            else:
                Displacement_Extended[w, x, :] = [
                    Fdist1bis[w, int(x * PeakWidth2ndD / PeakWidth1stD)],
                    Hum2[x],
                ]

    # Compute Deformation
    for w in range(Aligned.shape[0]):
        for x in range(Aligned.shape[1]):
            Deform1[w, x] = 2 + (
                -Displacement_Extended[w, x + 1, 0]
                + Displacement_Extended[w + 2, x + 1, 0]
            )
            Deform2[w, x] = 2 + (
                -Displacement_Extended[w + 1, x, 1]
                + Displacement_Extended[w + 1, x + 2, 1]
            )

    # Correct for negative deformations
    if np.any(Deform1 < 0) or np.any(Deform2 < 0):
        print("Warning: Negative deformation detected. Adjusting to zero.")
        Deform1[Deform1 < 0] = 0
        Deform2[Deform2 < 0] = 0

    # Apply deformation correction
    Aligned *= (Deform1 * Deform2) / 4

    SzE = (*Deform1.shape, 2)
    Deform_output = np.zeros(SzE)
    Deform_output[:, :, 0] = Deform1
    Deform_output[:, :, 1] = Deform2

    return Aligned, Displacement, Deform_output


def equalize_size(chromato1, chromato2):
    """
    Equalizes the size of two GCxGC chromatograms by adding zeros where needed.

    Parameters:
        chromato1 (numpy.ndarray): First chromatogram.
        chromato2 (numpy.ndarray): Second chromatogram.

    Returns:
        tuple: Two chromatograms of equal size (numpy.ndarrays).
    """
    if chromato1.shape != chromato2.shape:
        # Determine the maximum size for the output chromatograms
        max_shape = np.maximum(chromato1.shape, chromato2.shape)

        # Initialize new chromatograms with zeros
        chromato1_eq = np.zeros(max_shape)
        chromato2_eq = np.zeros(max_shape)

        # Fill in the original data
        chromato1_eq[: chromato1.shape[0], : chromato1.shape[1]] = chromato1
        chromato2_eq[: chromato2.shape[0], : chromato2.shape[1]] = chromato2
    else:
        chromato1_eq = chromato1
        chromato2_eq = chromato2

    return chromato1_eq, chromato2_eq


def save_chromatogram(FileName, chromato_obj):
    # Flatten the MSvaluebox and MSintbox arrays and remove zeros
    MSvalueboxLine = chromato_obj["MSvaluebox"].T
    MSintboxLine = chromato_obj["MSintbox"].T
    MSvalueboxLine = MSvalueboxLine.flatten()
    MSintboxLine = MSintboxLine.flatten()
    chromato_obj["MSvalueboxLine"] = MSvalueboxLine[MSvalueboxLine != 0]
    chromato_obj["MSintboxLine"] = MSintboxLine[MSvalueboxLine != 0]

    # Create a new NetCDF file with 64-bit offset
    with Dataset(FileName, "w", format="NETCDF4") as ncnew:
        # Define dimensions
        scan_number = ncnew.createDimension(
            "scan_number", len(chromato_obj["scantime"])
        )
        point_number = ncnew.createDimension(
            "point_number", len(chromato_obj["MSintboxLine"])
        )

        # Define variables
        vardimfirst = ncnew.createVariable("flag_count", "i4", (scan_number,))
        vardimA = ncnew.createVariable("scan_acquisition_time", "f8", (scan_number,))
        vardimB = ncnew.createVariable("actual_scan_number", "i4", (scan_number,))
        vardimC = ncnew.createVariable("mass_range_max", "f8", (scan_number,))
        vardimD = ncnew.createVariable("mass_range_min", "f8", (scan_number,))
        vardimE = ncnew.createVariable("scan_index", "i4", (scan_number,))
        vardimF = ncnew.createVariable("point_count", "i4", (scan_number,))
        vardimG = ncnew.createVariable("total_intensity", "f8", (scan_number,))
        vardimH = ncnew.createVariable("mass_values", "f8", (point_number,))
        vardimI = ncnew.createVariable("intensity_values", "f8", (point_number,))

        # Write data to variables
        vardimfirst[:] = chromato_obj["flag"]
        vardimA[:] = chromato_obj["scantime"]
        vardimB[:] = chromato_obj["scannum"]
        vardimC[:] = chromato_obj["medmsmax"]
        vardimD[:] = chromato_obj["medmsmin"]
        vardimE[:] = chromato_obj["ionid"]
        vardimF[:] = chromato_obj["eachscannum"]
        vardimG[:] = chromato_obj["MStotint"]
        vardimH[:] = chromato_obj["MSvalueboxLine"]
        vardimI[:] = chromato_obj["MSintboxLine"]


if __name__ == "__main__":
    print("Running the main script.")
    print("Loading chromatograms and alignment points.")
    ChromatoTarget, ChromatoRef = load_chromatograms(
        input_path=INPUT_PATH,
        target_file=TARGET_CHROMATOGRAM_FILE,
        reference_file=REFERENCE_CHROMATOGRAM_FILE,
    )
    print("Chromatograms loaded successfully.")

    print("Loading alignment points.")
    reference_peaks, target_peaks = load_alignment_points(
        reference_file=REFERENCE_ALIGNMENT_PTS_FILE,
        target_file=TARGET_ALIGNMENT_PTS_FILE,
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        ChromatoRef=ChromatoRef,
        ChromatoTarget=ChromatoTarget,
    )
    #TODO: to remove
    reference_peaks -= 1
    target_peaks -= 1
    print("Alignment points loaded successfully.")

    print("Reshaping chromatogram data.")
    target_TIC = reshape_tic(ChromatoTarget["MStotint"], NBPIX2NDD_TARGET)
    ref_TIC = reshape_tic(ChromatoRef["MStotint"], NBPIX2NDD_REF)
    print("Chromatogram data reshaped successfully.")

    print("Rounding MS data.")
    time_start = time.time()
    ChromatoTarget["MSvaluebox"], ChromatoTarget["MSintbox"] = MSdataRound_v2(
        ChromatoTarget["MSvaluebox"], ChromatoTarget["MSintbox"]
    )
    time_end = time.time()
    print("MS data rounded successfully. Time taken:", time_end - time_start)

    aligned_result = align_2d_chrom_ms_v5(
        Ref=ref_TIC,
        Other=target_TIC,
        Peaks_Ref=reference_peaks,
        Peaks_Other=target_peaks,
        MSvaluebox=ChromatoTarget["MSvaluebox"],
        MSintbox=ChromatoTarget["MSintbox"],
        NbPix2ndD=NBPIX2NDD_REF,
        Peak_widths=TYPICAL_PEAK_WIDTH,
        model_choice=MODEL_CHOICE,
    )

    Alignedeachscannum = np.sum(aligned_result["MSvaluebox"] != 0, axis=1)
    Alignedionid = np.cumsum(Alignedeachscannum)
    AlignedMStotint = np.sum(aligned_result["MSintbox"], axis=1)

    aligned_result["scannum"] = ChromatoTarget["scannum"]
    aligned_result["flag"] = ChromatoTarget["flag"]
    aligned_result["medmsmax"] = ChromatoTarget["medmsmax"]
    aligned_result["medmsmin"] = ChromatoTarget["medmsmin"]
    aligned_result["scantime"] = ChromatoTarget["scantime"]
    aligned_result["eachscannum"] = Alignedeachscannum
    aligned_result["ionid"] = Alignedionid
    aligned_result["MStotint"] = AlignedMStotint

    print("Saving aligned chromatogram.")
    output_file_name = os.path.join(
        OUTPUT_PATH, os.path.splitext(TARGET_CHROMATOGRAM_FILE)[0] + "_ALIGNED.cdf"
    )

    save_chromatogram(output_file_name, aligned_result)
    print("Aligned chromatogram saved successfully.")
