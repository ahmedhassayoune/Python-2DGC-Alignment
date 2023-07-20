import os
from molmass import Formula
import nistchempy as nist
import numpy as np
import itertools
import pandas as pd
import csv
import glob
import random
from matchms import calculate_scores, Spectrum
from matchms.similarity import CosineGreedy, CosineHungarian
import mass_spec
import pyms_nist_search
import pyms
from matchms.importing import load_from_mgf, scores_from_json
import logging
import write_masspec

def get_casno_dict(matches):
    casnos = dict()
    for match in matches:
        casno = match[1]['casno']
        if (casnos.get(casno)):
            casnos[casno].append(match[0])
        else:
            casnos[casno] = [match[0]]
    return casnos

def get_name_dict(matches):
    r"""Group coordinates in matches by compound name

    Parameters
    ----------
    matches :
        Retrieved peaks NIST infomartions.
    ----------
    Returns
    ----------
    names :
        Dictionary with compound names as keys and coordinates as values
    ----------
    Examples
    ----------
    >>> import utils
    >>> coordinates = ...
    >>> matches = matching_nist_lib_from_chromato_cube((chromato, time_rn, mass_range), chromato_cube, coordinates)
    >>> names = utils.get_name_dict(matches)
    """
    names = dict()
    for match in matches:
        name = match[1]['compound_name']
        if (names.get(name)):
            names[name].append(match[0])
        else:
            names[name] = [match[0]]
    return names

def get_casno_list(matches):
    casnos = []
    for match in matches:
        casno = match[1]['casno']
        if (not casno in casnos):
            casnos.append(casno)
    return casnos

def get_name_list(matches):
    names = []
    for match in matches:
        name = match[1]['compound_name']
        if (not name in names):
            names.append(name)
    return names

def colors(coordinates, matches):
    color_labels = []
    casnos = get_casno_list(matches)
    for i in range(len(coordinates)):
        casno = matches[i][1]['casno']
        id = casnos.index(casno)
        color_labels.append(id)
    return color_labels


def formula_to_nominal_mass(formula):
    r"""Retrieve and add formula in aligned peak table for each molecule.

    Parameters 
    ----------
    formula :
        Formula of the molecule
    ----------
    Returns
    ----------
        Nominal mass of the molecule
    """
    f = Formula(formula)
    return f.nominal_mass

def retrieve_formula_and_mass_from_compound_name(compound_name):
    search = nist.Search()
    search.find_compounds(identifier = compound_name, search_type = 'name')
    search.load_found_compounds()
    if (len(search.compounds)):
        return (search.compounds[0].formula).replace(' ', ''), search.compounds[0].mol_weight
    return "", ""

def unique_mol_list_formla_weight_dict(mol_list):
    res = dict()
    unique_mol_list = np.unique(list(itertools.chain.from_iterable(mol_list)))
    for mol in unique_mol_list:
        formula, weight = retrieve_formula_and_mass_from_compound_name(mol)
        res[mol] = formula, weight
    return res

def retrieve_mol_list_formula_weight(mol_list, mol_list_formla_weight_dict):
    res = []
    for mol in mol_list:
        formula, weight = mol_list_formla_weight_dict[mol]
        res.append((mol, formula, weight))
    return res

def add_formula_in_aligned_peak_table(filename, output_filename):
    r"""Retrieve and add formula in aligned peak table for each molecule. 

    Parameters
    ----------
    filename :
        Filename of the peak table.
    output_filename :
        Filename of the new peak table
    """
    df_unique_res = pd.read_csv(filename, header=None)
    mol_list = []
    for index, row in df_unique_res.iterrows():
        if (index == 0):
            labels = np.array(row[1:])
        if (index == 0):
            files = np.array(row[1:])
            continue
        
        mol = row[0]
        formula, weight = retrieve_formula_and_mass_from_compound_name(mol)
        row = row[1:]
        new_row = [mol] + [formula, weight] + list(row)
        mol_list.append(new_row)
        print(index)
    
    with open(output_filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(['','',''] + list(labels))
        writer.writerow(['','',''] + list(files))
        for mol in mol_list:
            writer.writerow(mol)

def add_formula_weight_in_aligned_peak_table_from_molecule_name_formula_dict(filename, output_filename, molecule_name_formula_dict):
    r"""Add formula weight in aligned peak table from molecule name/formula dict

    Parameters
    ----------
    filename :
        Filename of the peak table.
    output_filename :
        Filename of the new peak table
    molecule_name_formula_dict :
        Dictionary with names of the molecule as keys and formula and nominal mass as values
    ----------
    -------
    Examples
    -------
    >>> import utils
    >>> molecule_name_formula_dict=utils.build_molecule_name_formula_dict_from_peak_tables(PATH)
    >>> utils.add_formula_weight_in_aligned_peak_table_from_molecule_name_formula_dict(filename, output_filename, molecule_name_formula_dict)
    """
    df_unique_res = pd.read_csv(filename, header=None)
    mol_list = []
    for index, row in df_unique_res.iterrows():
        if (index == 0):
            labels = np.array(row[1:])
            continue
        if (index == 1):
            files = np.array(row[1:])
            continue
        mol = row[0]
        row = row[1:]
        try:
            new_row = [mol] + molecule_name_formula_dict[mol] + list(row)
        except:
            formula, weight = retrieve_formula_and_mass_from_compound_name(mol)
            new_row = [mol] + [formula, weight] + list(row)
            print(mol, "not found")
            
        mol_list.append(new_row)
        print(index)
    
    with open(output_filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['','',''] + list(labels))
        writer.writerow(['','',''] + list(files))
        for mol in mol_list:
            writer.writerow(mol)


def build_molecule_name_formula_dict_from_peak_tables(PATH):
    r"""Take aligned peak table and create a dictionary which associate formula and nominal mass to each molecule in the peak table

    Parameters
    ----------
    PATH :
        Path to the peak table.
    ----------
    Returns
    ----------
    res :
        Dictionary with names of the molecule as keys and formula and nominal mass as values.
    """
    res = dict()
    filenames = glob.glob(PATH + '*.csv')
    i = 0
    for filename in filenames:
        i = i + 1
        df_unique_res = pd.read_csv(filename, header=None)
        for index, row in df_unique_res.iterrows():
            if (index == 0):
                continue
            mol = row[0]
            formula = row[2]
            res[mol] = [formula, formula_to_nominal_mass(formula)]
        print(i)
    return res

def shuffle_dict(d):
    r"""Shuffle dictionary

    Parameters
    ----------
    d :
        Dictionary.
    ----------
    Returns
    ----------
    Shuffled dictionary
    """
    keys =  list(d.keys()) 
    random.shuffle(keys)
    d_res = dict()
    for key in keys:
        d_res[key] = d[key]
    return d_res

def compute_spectrum_mass_values(spectra_obj):
    r"""Compute nominal mass format mass values from spectra infos. NEED REFACTO

    Parameters
    ----------
    spectra_obj :
        Chromatogram spectra object.
    ----------
    Returns
    ----------
    mass_values :
        Array of nominal mass from range_min to range_max
    -------
    Examples
    -------
    >>> import utils
    >>> spectra_obj=(l1, l2, mv, iv, range_min, range_max)
    >>> mass_values=utils.compute_spectrum_mass_values(spectra_obj)
    """
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    return np.linspace(range_min, range_max, range_max - range_min + 1).astype(int)


def compute_spectra_similarity(spectrum1, spectrum2):
    r"""Compute similarity between two spectra. First spectrum and second spectrum must have the same dimensions.

    Parameters
    ----------
    spectrum1 :
        First spectrum as tuple (mass_values, intensities)
    spectrum2 :
        Second spectrum as tuple (mass_values, intensities)
    ----------
    Returns
    ----------
    similarity :
        Spectra cosine_greedy similarity between spectrum1 and spectrum2.
    """
    m1, i1 = spectrum1
    m2, i2 = spectrum2
    cosine_greedy = CosineGreedy(tolerance=0.0, mz_power=1.0)
    pt1_specturm = Spectrum(mz=np.array(m1).astype(float), intensities=np.array(i1))
    pt2_spectrum = Spectrum(mz=np.array(m2).astype(float), intensities=np.array(i2))
    return float(cosine_greedy.pair(pt1_specturm, pt2_spectrum)['score'])
    
def get_two_points_and_compute_spectra_similarity(pt1, pt2, spectra_obj, chromato_cube):
    r"""Get two points and compute spectra similarity.

    Parameters
    ----------
    pt1 :
        Coordinates of the first spectrum.
    pt2 :
        Coordinates of the second spectrum.
    spectra_obj :
        Chromatogram spectra object.
    chromato_cube :
        3D chromatogram.
    ----------
    Returns
    ----------
    similarity :
        Spectra similarity.
    """
    mass_values = compute_spectrum_mass_values(spectra_obj)
    pt1_int_values = mass_spec.read_spectrum_from_chromato_cube(pt1, chromato_cube)
    pt2_int_values = mass_spec.read_spectrum_from_chromato_cube(pt2, chromato_cube)
    return compute_spectra_similarity((mass_values, pt1_int_values), (mass_values, pt2_int_values))

def retrieved_nist_casnos_from_hmbd_spectra(lib_path):
    r"""NIST Search with HMDB spectra to retrieve NIST infos (casnos...).

    Parameters
    ----------
    lib_path :
        Path to HMDB spectra library file
    ----------
    Returns
    ----------
    spectra :
        Spectra in HMDB library file.
    casnos :
        Casnos of spectra in HMDB library file.
    names :
        Names of spectra in HMDB library file.
    hit_probs :
        Hits probabilities of spectra in HMDB library file.
    hmdb_ids :
        HMDB IDs of spectra in HMDB library file.
    -------
    Examples
    --------
    >>> import utils
    >>> spectra, casnos, names, hit_probs, hmdb_ids=utils.retrieved_nist_casnos_from_hmbd_spectra(filename)
    """
    logger=logging.getLogger('matchms')
    logger.setLevel('ERROR')
    logger=logging.getLogger('pyms')
    logger.setLevel('ERROR')
    logger=logging.getLogger('pyms_nist_search')
    logger.setLevel('ERROR')
    search = pyms_nist_search.Engine(
                "C:/NIST14/MSSEARCH/mainlib/",
                pyms_nist_search.NISTMS_MAIN_LIB,
                "C:/Users/Stan/Test",
                )
    lib_spectra=load_from_mgf(lib_path)
    hmdb_ids=[]
    spectra=[]
    for spectrum in lib_spectra:
        hmdb_ids.append(spectrum.metadata['databaseid'])
        spectra.append(pyms.Spectrum.MassSpectrum(spectrum.peaks.mz, spectrum.peaks.intensities))
    casnos=[]
    names=[]
    hit_probs=[]
    for spectrum in spectra:
        res = search.full_search_with_ref_data(spectrum)
        names.append(res[0][0].name)
        casnos.append(res[0][0].cas)
        hit_probs.append(res[0][0].hit_prob)
    return spectra, casnos, names, hit_probs, hmdb_ids

def add_nist_info_in_mgf_file(filename, output_filename='lib_EIB_gt.mgf'):
    r"""Retrieve NIST infos of spectra in HMDB library file (MGF format) and create a new library (MGF format) with retrieved NIST informations.

    Parameters
    ----------
    filename :
        Path to HMDB spectra library file
    output_filename : optional
        New library filename
    ----------
    -------
    Examples
    --------
    >>> import utils
    >>> utils.add_formula_in_aligned_peak_table('filename', 'output_filename')
    """
    spectra, casnos, names, hit_probs, hmdb_ids=retrieved_nist_casnos_from_hmbd_spectra(filename)
    mass_values_list=[]
    intensity_values_list=[]
    for spectrum in spectra:
        mass_values_list.append(spectrum.mass_list)
        intensity_values_list.append(spectrum.intensity_list)
    write_masspec.mass_spectra_to_mgf(output_filename, mass_values_list, intensity_values_list, meta_list = None, filter_by_instrument_type = None, names=names, casnos=casnos, hmdb_ids=hmdb_ids)

def generate_lib_scores_from_lib(lib_filename, output_path='./lib_scores.json'):
    r"""Read the input mgf library and compute pairwise similarity

    Parameters
    ----------
    lib_filename :
        Path to HMDB spectra library file
    output_filename : optional
        New library filename
    ----------
    -------
    Examples
    --------
    >>> import utils
    >>> utils.generate_lib_scores_from_lib('lib_filename', 'output_filename')
    """
    # disable matchms logger
    logger=logging.getLogger('matchms')
    logger.setLevel('ERROR')

    spectra = list(load_from_mgf(lib_filename))
    scores = calculate_scores(references=spectra,
                            queries=spectra,
                            similarity_function=CosineGreedy(),
                            is_symmetric=False, array_type="numpy")
    scores.to_json(output_path)


#add_formula_in_aligned_peak_table('C:/Users/Stan/pic/COVID-2020-2021/aligned_peak_table.csv', 'C:/Users/Stan/pic/COVID-2020-2021/peak_table.csv')