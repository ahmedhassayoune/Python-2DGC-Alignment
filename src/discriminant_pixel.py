import math
import numpy as np
import os
import read_chroma
import baseline_correction
import plot
import projection
from skimage.feature import peak_local_max
import matching
import utils

def compute_map(chromatos, vij):
    shape = chromatos[0].shape
    vij[vij == 0] = -1
    chromato_map = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pij = chromatos[:,i,j]
            pij_minus_pij_mean = pij - np.mean(pij)
            vij_minus_vij_mean = vij - np.mean(vij)
            den = (math.sqrt(np.sum(pij_minus_pij_mean * pij_minus_pij_mean)) * np.sum(vij_minus_vij_mean * vij_minus_vij_mean))
            if (den == 0):
                rij = 0
            else:
                rij = np.sum(pij_minus_pij_mean * vij_minus_vij_mean) / den
            chromato_map[i][j] = rij
    return chromato_map
        
def read_aligned_chromato(filename):
    res = []
    with open(filename, 'r') as f:
        for lines in f.readlines():
            lines = lines.replace("\n","").split(" ")
            res.append(lines)
    return np.array(res).astype(float)

def read_aligned_chromatos(CORRECTED_CHROMA_PATH):
    corrected_chroma_files = os.listdir(CORRECTED_CHROMA_PATH)
    aligned_chromatos = []
    for corrected_chroma in corrected_chroma_files:
        aligned_chromatos.append(read_aligned_chromato(CORRECTED_CHROMA_PATH + corrected_chroma))
    aligned_chromatos = np.array(aligned_chromatos)
    return aligned_chromatos

def find_discriminant_pixels(chromato_ref_obj, aligned_chromatos, vij, disp=False, max_pixel=500, local_max_filter=False, title=""):
    r"""Find discriminant pixels.

    Parameters
    ----------
    chromato_ref_obj :
        Chromatogram object wrapping chromato, time_rn and spectra object used as reference. 
    aligned_chromatos : optional
        List or (pre-aligned) chromatogram within the cohort.
    vij :
        List of 1 or -1 for each chromatogram within the cohort. 1 representing the first group and -1 others.
    disp : optional
        Whether to display figures or not
    max_pixel : optional
        The number of most discriminating pixels to be returned.
    local_max_filter : optional
        Whether to apply peak_local_max to merge pixels or not.
    title : optional
        Title prefixe for the figures.

    -------
    Returns
    -------
    pixels:
        The n (n < max_filter if max_filter is not None) most discriminating pixels.
    """
    cmap = compute_map(aligned_chromatos, vij)
    if (disp):
        plot.visualizer((cmap, chromato_ref_obj[1]), log_chromato=False, title=title + "cmap")
    pixels_sorted_by_disc_val = np.dstack(np.unravel_index(np.argsort((-cmap).ravel()), cmap.shape))[0]
    if (max_pixel):
        pixels_sorted_by_disc_val = pixels_sorted_by_disc_val[:max_pixel]
        
    res = pixels_sorted_by_disc_val
    if (disp):
            u = projection.matrix_to_chromato(pixels_sorted_by_disc_val[:max_pixel], chromato_ref_obj[1], 1.25, cmap.shape)
            if (max_pixel):
                t = title + str(max_pixel) + "_most_discriminant_pixels"
            else:
                t = title + "discriminant_pixels"
            plot.visualizer((aligned_chromatos[0], chromato_ref_obj[1]), log_chromato=False, points=u, title=t)
    
    if (local_max_filter):
        footprint = np.zeros_like(aligned_chromatos[0])
        for cd in pixels_sorted_by_disc_val:
            footprint[cd[0]][cd[1]] = 1

        cp = cmap.copy()
        cp[footprint == 0] = 0
        local_discriminant_pixels = peak_local_max(cp)
        res = local_discriminant_pixels
        if (disp):
            local_discriminant_pixels_in_chromato = projection.matrix_to_chromato(local_discriminant_pixels, chromato_ref_obj[1], 1.25, cmap.shape)
            plot.visualizer((aligned_chromatos[0], chromato_ref_obj[1]), log_chromato=False, points=local_discriminant_pixels_in_chromato, title=title + "discriminant_local_max")
            
    return res

def find_discriminant_compounds(chromato_ref_obj, aligned_chromatos, chromato_cube, vij, disp=False, max_pixel=500, local_max_filter=False,mod_time=1.25, title="", hit_prob_min=15, match_factor_min=800):
    r"""Finds discriminant compounds (finds discriminant pixels and identifies them).

    Parameters
    ----------
    chromato_ref_obj :
        Chromatogram object wrapping chromato, time_rn and spectra object used as reference. 
    aligned_chromatos : optional
        List or (pre-aligned) chromatogram within the cohort.
    chromato_cube :
        3D chromatogram.
    vij :
        List of 1 or -1 for each chromatogram within the cohort. 1 representing the first group and -1 others.
    disp : optional
        Whether to display figures or not
    max_pixel : optional
        The number of most discriminating pixels to be returned.
    local_max_filter : optional
        Whether to apply peak_local_max to merge pixels or not.
    mod_time : optional
        Modulation time.
    title : optional
        Title prefixe for the figures.
    hit_prob_min : optional
        Filter compounds with hit_prob < hit_prob_min.
    match_factor_min : optional
        Filter compounds with match_factor < match_factor_min.

    -------
    Returns
    -------
    pixels:
        The most discriminating compounds.
    """
    chromato, time_rn, spectra_obj = chromato_ref_obj
    discriminant_pixels = find_discriminant_pixels(chromato_ref_obj, aligned_chromatos, vij, disp=disp, max_pixel=max_pixel, local_max_filter=local_max_filter, title=title)
    matches = matching.matching_nist_lib_from_chromato_cube(chromato_ref_obj, chromato_cube, discriminant_pixels, mod_time = 1.25, hit_prob_min=hit_prob_min, match_factor_min=match_factor_min)
    if (disp):
        chromato_cd = projection.matrix_to_chromato(discriminant_pixels,time_rn, mod_time,chromato.shape)
        casnos_dict = (utils.get_name_dict(matches))
        plot.visualizer(chromato_obj=(chromato, time_rn), mod_time=mod_time, points=chromato_cd, casnos_dict=casnos_dict, title=title + "discriminant_matches")

    return matches

def align_and_find_discriminant_compounds(chromato_ref_obj, chromato_cube, CORRECTED_CHROMA_PATH, vij, disp=False, max_pixel=500, local_max_filter=False,mod_time=1.25, title="", hit_prob_min=15, match_factor_min=800):
    """
    ERROR DANS L APPEL A find_discriminant_compounds
    A REFACTO SI UTILE ?
    """
    aligned_chromatos = read_aligned_chromatos(CORRECTED_CHROMA_PATH)
    aligned_chromatos = np.array([baseline_correction.chromato_no_baseline(aligned_chromato) for aligned_chromato in aligned_chromatos])
    if (disp):
        for i, aligned_chromato in enumerate(aligned_chromatos):
            plot.visualizer((aligned_chromato, chromato_ref_obj[1]), log_chromato=False, title="aligned_chromato_" + str(i))
    return find_discriminant_compounds(chromato_ref_obj, chromato_cube, vij, disp=True, max_pixel=max_pixel, local_max_filter=local_max_filter, mod_time=1.25, title=title, hit_prob_min=hit_prob_min, match_factor_min=match_factor_min)