import numpy as np
import projection
import mass_spec
from write_masspec import mass_spectra_to_mgf
from matchms.importing import load_from_mgf
import pyms_nist_search
import pyms
import logging

present = {"HMDB0031018": [[23.43, 0.008]], "HMDB0061859": [[32.22, 0.042]], "HMDB0030469": [[28.15, 0.008]],
           "HMDB0031264": [[15.59, 0.017]], "HMDB0033848": [[18.41, 0.025]], "HMDB0031291": [[13.10, 1.231]],
           "HMDB0034154": [[36.08, 0.083]]}

present = ['111-82-0', '112-39-0', '124-10-7', '1731-84-6', '110-42-9', '111-11-5', '112-61-8', '1120-28-1', '929-77-1', '5802-82-4', '55682-92-3', '2442-49-1']

def check_match(match):
    #return np.array([databaseid for databaseid in [meta['databaseid'] for meta in match[:, 1]] if databaseid in present])
    return np.array([databaseid for databaseid in [meta['casno'] for meta in match[:, 1]] if databaseid in present])

'''def check_match_nist_lib(match):
    return np.array([casno for casno in match[:, 1] if casno in present])'''
    

def matching_nist_lib_from_chromato_cube(chromato_obj, chromato_cube, coordinates, mod_time = 1.25, hit_prob_min=15, match_factor_min=800):
    r"""Indentify retrieved peaks using NIST library.

    Parameters
    ----------
    chromato_obj :
        Chromatogram object wrapping chromato, time_rn and spectra_obj
    chromato_cube :
        3D chromatogram.
    coordinates :
        Peaks coordinates.
    mod_time : optional
        Modulation time
    hit_prob_min :
        Filter compounds with hit_prob < hit_prob_min 
    match_factor_min : optional
        Filter compounds with match_factor < match_factor_min 
    -------
    Returns
    -------
    matches:
        Array of match dictionary containing casno, name, formula and spectra for each of identified as well as hit_prob, match_factor and reverse_match_factor
    --------
    """
    chromato, time_rn, spectra_obj = chromato_obj
    coordinates_in_chromato = projection.matrix_to_chromato(
        coordinates, time_rn, mod_time, chromato.shape)

    search = pyms_nist_search.Engine(
                    "C:/NIST14/MSSEARCH/mainlib/",
                    pyms_nist_search.NISTMS_MAIN_LIB,
                    "C:/Users/Stan/Test",
                    )
    logger=logging.getLogger('pyms_nist_search')
    logger.setLevel('ERROR')
    logger=logging.getLogger('pyms')
    logger.setLevel('ERROR')

    match = []
    try:
        (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    except:
        range_min, range_max=spectra_obj
    mass_values = np.linspace(range_min, range_max, range_max - range_min + 1).astype(int)
    nb_analyte = 0
    print("nb_peaks: ", len(coordinates))
    for i, coord in enumerate(coordinates):
        
        d_tmp = dict()
        int_values = mass_spec.read_spectrum_from_chromato_cube(coord, chromato_cube=chromato_cube)
        mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
        res = search.full_search_with_ref_data(mass_spectrum)
        #res = search.full_spectrum_search(mass_spectrum)
        if (res[0][0].match_factor < match_factor_min):
            continue
        '''if (res[0][0].hit_prob < hit_prob_min):
            continue'''
        #print(res[0][1].formula)
        del mass_spectrum
        compound_casno = res[0][0].cas
        compound_name = res[0][0].name
        compound_formula = res[0][1].formula
        hit_prob = res[0][0].hit_prob
        match_factor = res[0][0].match_factor
        reverse_match_factor = res[0][0].reverse_match_factor
        d_tmp['casno'] = compound_casno
        d_tmp['compound_name'] = compound_name
        d_tmp['compound_formula'] = compound_formula
        d_tmp['hit_prob'] = hit_prob
        d_tmp['match_factor'] = match_factor
        d_tmp['reverse_match_factor'] = reverse_match_factor
        
        d_tmp['spectra'] = int_values
        '''if (res[0][0].hit_prob < hit_prob_min):
            nb_analyte = nb_analyte + 1
            d_tmp['compound_name'] = 'Analyte' + str(nb_analyte)'''
        
        match.append([[(coordinates_in_chromato[i][0]), (coordinates_in_chromato[i][1])], d_tmp, coord])
        
        del res
    print("nb match:")
    print(len(coordinates))
    return match


def matching_nist_lib(chromato_obj, spectra, coordinates, mod_time = 1.25):

    chromato, time_rn, spectra_obj = chromato_obj
    coordinates_in_chromato = projection.matrix_to_chromato(
        coordinates, time_rn, mod_time, chromato.shape)
    search = pyms_nist_search.Engine(
                    "C:/NIST14/MSSEARCH/mainlib/",
                    pyms_nist_search.NISTMS_MAIN_LIB,
                    "C:/Users/Stan/Test",
                    )
    match = []
    for i, coord in enumerate(coordinates):
        
        d_tmp = dict()
        mass_values, int_values = mass_spec.read_spectrum(chromato, coord, spectra)
        mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
        res = search.full_search_with_ref_data(mass_spectrum)
        
        
        del mass_spectrum
        
        
        '''if (res[0][0].hit_prob < 15):
            continue'''
        compound_casno = res[0][0].cas
        compound_name = res[0][0].name
        d_tmp['casno'] = compound_casno
        d_tmp['compound_name'] = compound_name
        match.append([[round(coordinates_in_chromato[i][0], 2), round(coordinates_in_chromato[i][1], 2)], d_tmp])
        
        del res
    return np.array(match)


def matching(chromato_obj, spectra, spectrum_path, spectrum_lib, coordinates, mod_time=1.25, min_score=None):
    chromato, time_rn, spectra_obj = chromato_obj
    coordinates_in_chromato = projection.matrix_to_chromato(
        coordinates, time_rn, mod_time, chromato.shape)

    mass_values_list = []
    intensity_values_list = []

    for coord in coordinates:
        mass_values, int_values = mass_spec.read_spectrum(chromato, coord, spectra)
        mass_values_list.append(mass_values)
        intensity_values_list.append(int_values)

    mass_spectra_to_mgf(spectrum_path, mass_values_list, intensity_values_list)

    spectrum = list(load_from_mgf(spectrum_path, metadata_harmonization=False))
    matching = mass_spec.peak_retrieval(spectrum, spectrum_lib, coordinates_in_chromato, min_score=min_score)

    return matching

def compute_metrics_from_chromato_cube(coordinates, chromato_obj, chromato_cube, spectrum_path, spectrum_lib, mod_time=1.25):
    if (not len(coordinates)):
        return [0,0,0,0]
    else:
        match = matching_nist_lib_from_chromato_cube(chromato_obj=chromato_obj, chromato_cube=chromato_cube, coordinates=coordinates)
        found_present = check_match(match)
        #nb_peaks,rappel,precision
        return [len(coordinates), len(np.unique(found_present)) / len(present), len(found_present) / len(coordinates), len(found_present)]
import gc
def compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25):
    if (not len(coordinates)):
        return [0,0,0,0]
    else:
        '''match = matching(chromato_obj, spectra, spectrum_path,
                     spectrum_lib, coordinates, mod_time=mod_time)'''
        match = matching_nist_lib(chromato_obj=chromato_obj, spectra=spectra, coordinates=coordinates)
        match_np_array=np.array(match)
        found_present = check_match(match_np_array)
        #nb_peaks,rappel,precision
        return [len(coordinates), len(np.unique(found_present)) / len(present), len(found_present) / len(coordinates), len(found_present)]