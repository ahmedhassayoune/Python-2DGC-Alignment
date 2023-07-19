import netCDF4 as nc
import numpy as np
import time
import math
from scipy.signal import argrelextrema
from projection import chromato_to_matrix
import os
import glob
from matchms.importing import load_from_mgf
from matchms import calculate_scores, Spectrum
from matchms.similarity import CosineGreedy, CosineHungarian
from matchms.similarity import ModifiedCosine
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities

def hmdb_txt_to_mgf(path, libname = "./hmdb_lib.mgf"):
    if (os.path.exists(libname)):
        os.remove(libname)
    res = ""
    files = os.listdir(path)[:10]
    for file in files:
        mass_values, int_values = read_hmdb_spectrum(path + "/" + file)
        res = res + "BEGIN IONS\n"
        res = res + "NAME=" + file[:7] + "\n"
        for i in range (len(mass_values)):
            mass = mass_values[i]
            intensity = int_values[i]
            res = res + (str(mass) + " " + str(intensity) + "\n")
        res = res + "END IONS\n"
    with open(libname, "a") as f:
        f.write(res)


def read_hmdb_spectrum(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        mass_values = []
        int_values = []
        for line in lines:
            mviv = line.split(" ")
            mv, iv = (mviv[0]), (mviv[1][:-2])
            mass_values.append(float(mv))
            int_values.append(float(iv))
        return mass_values, int_values
    return [], []


def read_spectra(spectra_obj, max_val = None):
    start_time = time.time()
    
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj

    debuts = np.where(mv == range_min)
    fins = np.where(mv ==  range_max)
    print(l1)
    print(l2)
    print(len(debuts[0]))
    #Stack 1-D arrays as columns into a 2-D array.
    mass_spectra_ind = np.column_stack((debuts[0],fins[0]))
    spectra = []
    for beg,end in mass_spectra_ind:
        mass_values = (mv[beg:end])
        int_values = (iv[beg:end])
        spectra.append((mass_values,int_values))

    print("--- %s seconds --- to compute spectra" % (time.time() - start_time))

    return spectra, debuts, fins

def read_full_spectra(spectra_obj, max_val = None):
    start_time = time.time()
    
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj

    debuts = np.where(mv == range_min)
    fins = np.where(mv ==  range_max)
    mass_spectra_ind = np.column_stack((debuts[0],fins[0]))
    print((mass_spectra_ind.shape))
    spectra = []
    for beg,end in mass_spectra_ind:
        tmp = np.linspace(range_min, range_max, range_max - range_min + 1).astype(int)
        new_iv= np.zeros((range_max - range_min + 1))
        new_iv[np.isin(tmp, (mv[beg:end + 1]))] = iv[beg:end+1]
        spectra.append((tmp,new_iv))
        
    print("--- %s seconds --- to compute full spectra" % (time.time() - start_time))
        
    return np.array(spectra), debuts, fins

'''
Read all ms in centroid data
'''
def read_spectra_centroid(spectra_obj, max_val = None):
    start_time = time.time()
    
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    minima = argrelextrema(mv, np.less)[0]
    fins = minima - 1
    fins = np.append(fins, len(mv - 1))
    debuts = np.insert(minima, 0, 0)
    #Stack 1-D arrays as columns into a 2-D array.
    mass_spectra_ind = np.column_stack((debuts,fins))
    spectra = []
    for beg,end in mass_spectra_ind:
        mass_values = (mv[beg:end])
        int_values = (iv[beg:end])
        spectra.append((mass_values,int_values))
    print("--- %s seconds --- to compute spectra centroid" % (time.time() - start_time))

    return spectra, debuts, fins

def compute_chromato_mass_from_spectra_loc(loc, mass_range, mass):
    (debuts, fins)=loc
    (range_min, range_max)=mass_range
    mass_spectra_ind = np.column_stack((debuts,fins))
    

def read_full_spectra_centroid(spectra_obj, max_val = None):
    r"""Build nominal mass mass spectra from centroided mass and intensity values.

    Parameters
    ----------
    spectra_obj :
        Spectra object wrapping chromato dims, all spectra masses, all spectra intensities, mass range_min and range_max.
    Returns
    -------
    A: tuple
        Return the nominal mass mass spectra, _, _.
    Examples
    --------
    >>> import read_chroma
    >>> import mass_spec
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> spectra, debuts, fins = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    """
    start_time = time.time()
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    minima = argrelextrema(mv, np.less)[0]
    fins = minima - 1
    fins = np.append(fins, len(mv - 1))
    debuts = np.insert(minima, 0, 0)
    #Stack 1-D arrays as columns into a 2-D array.
    mass_spectra_ind = np.column_stack((debuts,fins))
    spectra = []
    for beg,end in mass_spectra_ind:

        mass_values = (mv[beg:end])
        int_values = (iv[beg:end])
        spectra.append((mass_values,int_values))
    spectra_full_nom = []

    '''for i in range(len(spectra)):
        spectra_full_nom.append(centroid_to_full_nominal(spectra_obj, spectra[i][0], spectra[i][1]))'''
    cpu_count = multiprocessing.cpu_count()
    mv = np.linspace(range_min, range_max, range_max - range_min + 1).astype(int)
    with multiprocessing.Pool(processes = cpu_count) as pool:
        #for i, result in enumerate(pool.starmap(centroid_to_full_nominal, [(spectra_obj, spectra[i][0], spectra[i][1]) for i in range(len(spectra))])):
        for i, result in enumerate(pool.starmap(centroid_to_full_nominal, [((range_min, range_max), spectra[i][0], spectra[i][1]) for i in range(len(spectra))])):
            spectra_full_nom.append((mv, result))

    print("--- %s seconds --- to compute full spectra centroid" % (time.time() - start_time))

    return np.array(spectra_full_nom), debuts, fins

def centroid_to_full_nominal(spectra_obj, mass_values, int_values):
    #(l1, l2, mv, iv, range_min, range_max) = spectra_obj
    range_min, range_max = spectra_obj
    #print(range_min)
    #print(range_max)

    #mv = np.linspace(range_min, range_max, range_max - range_min + 1).astype(int)
    iv = np.zeros((range_max - range_min + 1))
    for i, mass in enumerate(mass_values):
        rounded_mass = round(mass)
        mass_ind = rounded_mass - range_min
        iv[mass_ind] = iv[mass_ind] + int_values[i]
    #return mv, iv
    return iv


def read_spectrum_from_chromato_cube(pic_coord, chromato_cube):
    #return spectra[pic_coord[1] * chromato.shape[0] + pic_coord[0]]
    ms_from_chromato_cube = chromato_cube[:, pic_coord[0],pic_coord[1]]
    #return np.linspace(range_min, range_max, range_max - range_min + 1).astype(int), ms_from_chromato_cube
    return ms_from_chromato_cube

def read_spectrum(chromato, pic_coord, spectra):
    #return spectra[pic_coord[1] * chromato.shape[0] + pic_coord[0]]
    return spectra[pic_coord[0] * chromato.shape[1] + pic_coord[1]]


def read_spectrum_tr_coord(chromato, pic_coord, spectra, time_rn, mod_time = 1.25):
    cd = chromato_to_matrix(pic_coord, time_rn, mod_time, chromato.shape)
    return read_spectrum(chromato, cd[0], spectra)

def filter_spectra(file):
    spectrums = []
    for spectrum in file:
        # Default filter is fully explained at https://matchms.readthedocs.io/en/latest/api/matchms.filtering.html .
        spectrum = default_filters(spectrum)
        spectrum = normalize_intensities(spectrum)
        spectrums.append(spectrum)
    return spectrums

def spectra_matching(spectrums, spectrums_lib, similarity_measure = CosineGreedy(tolerance=0.01, mz_power=1.0)):
    scores = calculate_scores(references=spectrums_lib, queries=spectrums, similarity_function=similarity_measure, is_symmetric=False)
    return scores

def peak_retrieval_from_path(spectrum_path, lib_path, coordinates_in_chromato=None):
    similarity_measure = CosineGreedy(tolerance=0.01, mz_power=1.0)
    spectrum = list(load_from_mgf(spectrum_path))
    spectrum_lib = list(load_from_mgf(lib_path))
    scores = spectra_matching(spectrum, spectrum_lib, similarity_measure)
    matching = []
    print(spectrum[0].metadata)
    print(spectrum[1].metadata)

    print(len(spectrum))
    for i, spectrum in enumerate(spectrum):
        print(i)
        selected_scores = scores.scores_by_query(spectrum, similarity_measure.__class__.__name__ + '_score', sort=True)
        #print(selected_scores[0][0].metadata, selected_scores[0][1], selected_scores[1][0].metadata, selected_scores[1][1], selected_scores[2][0].metadata,selected_scores[2][1])
        if (coordinates_in_chromato is not None):
            matching.append([[round(coordinates_in_chromato[i][0], 2), round(coordinates_in_chromato[i][1], 2)], selected_scores[0][0].metadata,  selected_scores[0][1]])
    return matching

def peak_retrieval(spectrums, spectrums_lib, coordinates_in_chromato, min_score=None):
    similarity_measure = CosineGreedy(tolerance=0.01, mz_power=1.0)
    scores = spectra_matching(spectrums, spectrums_lib, similarity_measure)
    matching = []
    for i, spectrum in enumerate(spectrums):
        selected_scores = scores.scores_by_query(spectrum, similarity_measure.__class__.__name__ + '_score', sort=True)
        if (min_score != None and selected_scores[0][1][0] < min_score):
            continue
        else:
            print(selected_scores)
            matching.append([[round(coordinates_in_chromato[i][0], 2), round(coordinates_in_chromato[i][1], 2)], selected_scores[0][0].metadata,  selected_scores[0][1], selected_scores[1][0].metadata, selected_scores[1][1], selected_scores[2][0].metadata, selected_scores[2][1]])
    return matching



import multiprocessing
from multiprocessing import Pool

def peak_retrieval_kernel(spectra, spectrums_lib, similarity_measure):
    scores = spectra_matching(spectra, spectrums_lib, similarity_measure)
    selected_scores = scores.scores_by_query(spectra, similarity_measure.__class__.__name__ + '_score', sort=True)
    return selected_scores


def peak_retrieval_multiprocessing(spectrums, spectrums_lib, coordinates_in_chromato, min_score=None):
    
    similarity_measure = CosineGreedy(tolerance=0.01, mz_power=1.0)
    cpu_count = multiprocessing.cpu_count()
    print(cpu_count)
    pool = multiprocessing.Pool(processes = cpu_count)
    matching = []
    for i, result in enumerate(pool.starmap(peak_retrieval_kernel, [(spectrums[i], spectrums_lib, similarity_measure) for i in range(len(spectrums))])):
        if (min_score != None and result[0][1][0] < min_score):
            continue
        else:
            matching.append([[round(coordinates_in_chromato[i][0], 2), round(coordinates_in_chromato[i][1], 2)], result[0][0].metadata,  result[0][1], result[1][0].metadata, result[1][1], result[2][0].metadata, result[2][1]])
    return matching
   
   
   
def centroid(spectra_obj, mv_index, mass_values, int_values):
    range_min, range_max = spectra_obj

    iv = np.zeros(len(mv_index))
    for i, mass in enumerate(mass_values):
        rounded_mass = round(mass,1)
        mass_ind = mv_index[rounded_mass]
        iv[mass_ind] = iv[mass_ind] + int_values[i]
    return iv

def read_full_spectra_full_centroid(spectra_obj, max_val = None):
    print("start")
    start_time = time.time()
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    minima = argrelextrema(mv, np.less)[0]
    fins = minima - 1
    fins = np.append(fins, len(mv - 1))
    debuts = np.insert(minima, 0, 0)
    #Stack 1-D arrays as columns into a 2-D array.
    mass_spectra_ind = np.column_stack((debuts,fins))
    spectra = []
    for beg,end in mass_spectra_ind:

        mass_values = (mv[beg:end])
        int_values = (iv[beg:end])
        spectra.append((mass_values,int_values))
    spectra_full_nom = []
    cpu_count = multiprocessing.cpu_count()
    mv = np.unique(np.around(mv,1))
    mv_index = dict()
    for i, mass in enumerate(mv):
        mv_index[mass] = i
    with multiprocessing.Pool(processes = cpu_count) as pool:
        for i, result in enumerate(pool.starmap(centroid, [((range_min, range_max), mv_index, spectra[i][0], spectra[i][1]) for i in range(len(spectra))])):
            spectra_full_nom.append((mv, result))

    print("--- %s seconds --- to compute full spectra centroid" % (time.time() - start_time))

    return np.array(spectra_full_nom), debuts, fins


#read_spectra("data/751340_blanc_28juillet.cdf")
#read_spectra("F:/Bureau/Nouveau dossier/751301_YBS8_J0-CDF-nominal-mass.cdf")

#peak_retrieval_from_path("./my_lib.mgf", "./lib_EIB.mgf", )