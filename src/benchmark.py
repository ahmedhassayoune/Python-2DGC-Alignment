from read_chroma import read_chroma, read_chromato_mass, chromato_cube, print_chroma_header, full_spectra_to_chromato_cube, full_spectra_to_chromato_cube_centroid
from projection import matrix_to_chromato, chromato_to_matrix
from mass_spec import read_spectra, read_spectrum, peak_retrieval, read_full_spectra, read_spectra_centroid, read_full_spectra_centroid
from write_masspec import mass_spectra_to_mgf
import numpy as np
import peak_detection
import json
from matchms.importing import load_from_mgf
import plot
from image_processing import gaussian_filter, gauss_laplace, gauss_multi_deriv, prewitt, sobel
from matching import compute_metrics, compute_metrics_from_chromato_cube

def threshold_filter_mass_per_mass(chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, chromato_cube, mod_time=1.25, cluster=False):
    
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    benchmark_threshold = dict()
    seuil_min = np.min(SEUILS)
    tmp = np.delete(coordinates.copy(), 0, -1)
    tmp = np.unique(tmp, axis=0)

    if (cluster):
        tmp = peak_detection.clustering(tmp, chromato)

    benchmark_threshold[seuil_min] = compute_metrics(tmp, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)

    for i in range(1, len(SEUILS)):
    #for i in range(len(SEUILS)):
        tmp = []
        for coordinate in coordinates:
            m,t1,t2 = coordinate
            # Check m coordinates to its relative mass slice
            if (chromato_cube[m][t1][t2] > np.max(chromato_cube[m]) * SEUILS[i]):
                tmp.append(coordinate)
        if (len(tmp)):
            tmp = np.delete(tmp, 0, -1)
            tmp = np.unique(tmp, axis=0)
            if (cluster):
                tmp = peak_detection.clustering(tmp, chromato)

        benchmark_threshold[SEUILS[i]] = compute_metrics(tmp, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)

    dict_seuil = dict()
    dict_seuil['seuil'] = benchmark_threshold
    return dict_seuil


def threshold_filter(chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25, mode='tic', chromato_cube=None):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    benchmark_threshold = dict()
    max_peak_val = np.max(chromato)

    seuil_min = np.min(SEUILS)
    #benchmark_threshold[seuil_min] = compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)

    #for i in range(1, len(SEUILS)):
    for i in range(len(SEUILS)):
        coordinates = np.array(
            [[x, y] for x, y in coordinates if chromato[x, y] > SEUILS[i] * max_peak_val])
        #benchmark_threshold[SEUILS[i]] = compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)
        benchmark_threshold[SEUILS[i]] = compute_metrics_from_chromato_cube(coordinates, chromato_obj, chromato_cube, "", "", mod_time=1.25)
  
    dict_seuil = dict()
    dict_seuil['seuil'] = benchmark_threshold
    return dict_seuil

def benchmark_pers_hom(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    seuil_min = np.min(SEUILS)
    coordinates = peak_detection.pers_hom(chromato_obj=(
            chromato, time_rn), mod_time=0, seuil=seuil_min)

    return threshold_filter(
            chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25, chromato_cube=chromato_cube)


def benchmark_LoG(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    seuil_min = np.min(SEUILS)
    num_sigmas = param['num_sigma']
    ABS_THRESHOLDS = param['abs_t']
    chromato_max_val = np.max(chromato)
    benchmark_log = dict()
    abs_thresholds_dict = dict()

    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        abs_thresholds_dict_tmp = dict()
        dict_sigma = dict()
        for num_sigma in num_sigmas:
            if (mode == "ehrgehriueheuh"):
                #if (mode == "mass_per_mass"):
                #coordinates = peak_detection.LoG_mass_per_mass_multiprocessing(chromato_cube, seuil=seuil_min, num_sigma=num_sigma, threshold_abs=ABS_THRESHOLDS * chromato_max_val)
                coordinates = peak_detection.LoG_mass_per_mass_multiprocessing(chromato_cube, seuil=seuil_min, num_sigma=num_sigma, threshold_abs=ABS_THRESHOLDS)

                # DoG_mass_per_mass return [[m, t1, t2, r]] so we delete radius
                for coord in coordinates:
                    print(coord)
                coordinates = np.delete(coordinates, 3 ,-1)
                print(len(coordinates))
                dict_sigma[num_sigma] = threshold_filter_mass_per_mass(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, chromato_cube, mod_time=1.25, cluster=cluster)
            else:
                coordinates, radius = peak_detection.LoG(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, num_sigma=num_sigma, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                print(len(coordinates))
                dict_sigma[num_sigma] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25, mode=mode, chromato_cube=chromato_cube)
        
        abs_thresholds_dict_tmp['num_sigma'] = dict_sigma
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp

    benchmark_log['abs_t'] = abs_thresholds_dict
    return benchmark_log
'''
def benchmark_LoG_ancien(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    seuil_min = np.min(SEUILS)
    num_sigmas = param['num_sigma']
    ABS_THRESHOLDS = param['abs_t']

    
    benchmark_log = dict()
    abs_thresholds_dict = dict()
    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        abs_thresholds_dict_tmp = dict()
        dict_sigma = dict()
        for num_sigma in num_sigmas:
            if (mode == "mass_per_mass"):
                threshold_dict = dict()
                benchmark_threshold_dict = dict()
                for SEUIL in SEUILS:
                    coordinates, radius = peak_detection.LoG(chromato_obj=(
                        chromato, time_rn), mod_time=0, seuil=seuil_min, num_sigma=num_sigma, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                    benchmark_threshold_dict[SEUIL] = compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)
                threshold_dict['seuil'] = benchmark_threshold_dict
                dict_sigma[num_sigma] = threshold_dict
            else:
                coordinates, radius = peak_detection.LoG(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, num_sigma=num_sigma, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                dict_sigma[num_sigma] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25)
        
        abs_thresholds_dict_tmp['num_sigma'] = dict_sigma
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp

    benchmark_log['abs_t'] = abs_thresholds_dict
    return benchmark_log
'''
def benchmark_DoH(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    seuil_min = np.min(SEUILS)
    num_sigmas = param['num_sigma']
    ABS_THRESHOLDS = param['abs_t']

    chromato_max_val = np.max(chromato)


    benchmark_doh = dict()
    abs_thresholds_dict = dict()
    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        abs_thresholds_dict_tmp = dict()
        dict_sigma = dict()
        for num_sigma in num_sigmas:
            if (mode == "ehrgehriueheuh"):
                #if (mode == "mass_per_mass"):
                #coordinates = peak_detection.DoH_mass_per_mass_multiprocessing(chromato_cube, seuil_min, num_sigma=num_sigma, threshold_abs=ABS_THRESHOLDS * chromato_max_val)
                coordinates = peak_detection.DoH_mass_per_mass_multiprocessing(chromato_cube, seuil_min, num_sigma=num_sigma, threshold_abs=ABS_THRESHOLDS)

                coordinates = np.delete(coordinates, 3 ,-1)
                print(len(coordinates))
                dict_sigma[num_sigma] = threshold_filter_mass_per_mass(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, chromato_cube, mod_time=1.25, cluster=cluster)
            else:
                coordinates, radius = peak_detection.DoH(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, num_sigma=num_sigma, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                print(len(coordinates))
                dict_sigma[num_sigma] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25, mode=mode, chromato_cube=chromato_cube)
        
        abs_thresholds_dict_tmp['num_sigma'] = dict_sigma
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp

    benchmark_doh['abs_t'] = abs_thresholds_dict
    return benchmark_doh

'''def benchmark_DoH_ancien(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    seuil_min = np.min(SEUILS)
    num_sigmas = param['num_sigma']
    ABS_THRESHOLDS = param['abs_t']

    benchmark_doh = dict()
    abs_thresholds_dict = dict()
    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        abs_thresholds_dict_tmp = dict()
        dict_sigma = dict()
        for num_sigma in num_sigmas:
            if (mode == "mass_per_mass"):
                threshold_dict = dict()
                benchmark_threshold_dict = dict()
                for SEUIL in SEUILS:
                    coordinates, radius = peak_detection.DoH(chromato_obj=(
                        chromato, time_rn), mod_time=0, seuil=seuil_min, num_sigma=num_sigma, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                    benchmark_threshold_dict[SEUIL] = compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)
                threshold_dict['seuil'] = benchmark_threshold_dict
                dict_sigma[num_sigma] = threshold_dict
            else:
                coordinates, radius = peak_detection.DoH(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, num_sigma=num_sigma, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                dict_sigma[num_sigma] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25)
        
        abs_thresholds_dict_tmp['num_sigma'] = dict_sigma
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp

    benchmark_doh['abs_t'] = abs_thresholds_dict
    return benchmark_doh'''


def benchmark_DoG(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    seuil_min = np.min(SEUILS)
    sigma_ratios = param['sigma_ratio']
    ABS_THRESHOLDS = param['abs_t']
    
    chromato_max_val = np.max(chromato)

    benchmark_dog = dict()
    abs_thresholds_dict = dict()
    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        abs_thresholds_dict_tmp = dict()
        dict_sigma = dict()
        for sigma_ratio in sigma_ratios:
            if (mode == "ehrgehriueheuh"):
                #if(mode=="mass_per_mass"):
                #coordinates = peak_detection.DoG_mass_per_mass_multiprocessing(chromato_cube, seuil=seuil_min, sigma_ratio=sigma_ratio, threshold_abs=ABS_THRESHOLDS * chromato_max_val)
                coordinates = peak_detection.DoG_mass_per_mass_multiprocessing(chromato_cube, seuil=seuil_min, sigma_ratio=sigma_ratio, threshold_abs=ABS_THRESHOLDS)

                # DoG_mass_per_mass return [[m, t1, t2, r]] so we delete radius
                coordinates = np.delete(coordinates, 3 ,-1)
                print(len(coordinates))
                dict_sigma[sigma_ratio] = threshold_filter_mass_per_mass(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, chromato_cube, mod_time=1.25, cluster=cluster)
            else:
                coordinates, radius = peak_detection.DoG(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, sigma_ratio=sigma_ratio, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                print(len(coordinates))
                dict_sigma[sigma_ratio] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25, mode=mode, chromato_cube=chromato_cube)
        abs_thresholds_dict_tmp['sigma_ratio'] = dict_sigma
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp
        
    benchmark_dog['abs_t'] = abs_thresholds_dict
    return benchmark_dog

'''def benchmark_DoG_ancien(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    seuil_min = np.min(SEUILS)
    sigma_ratios = param['sigma_ratio']
    ABS_THRESHOLDS = param['abs_t']
    

    benchmark_dog = dict()
    abs_thresholds_dict = dict()
    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        abs_thresholds_dict_tmp = dict()
        dict_sigma = dict()
        for sigma_ratio in sigma_ratios:
            if(mode=="mass_per_mass"):
                threshold_dict = dict()
                benchmark_threshold_dict = dict()
                for SEUIL in SEUILS:
                    print(SEUIL)
                    coordinates, radius = peak_detection.DoG(chromato_obj=(
                        chromato, time_rn), mod_time=0, seuil=seuil_min, sigma_ratio=sigma_ratio, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                    print(len(coordinates))
                    benchmark_threshold_dict[SEUIL] = compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)
                threshold_dict['seuil'] = benchmark_threshold_dict
                dict_sigma[sigma_ratio] = threshold_dict
            else:
                coordinates, radius = peak_detection.DoG(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, sigma_ratio=sigma_ratio, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                dict_sigma[sigma_ratio] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25)
        abs_thresholds_dict_tmp['min_distance'] = dict_sigma
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp
        
    benchmark_dog['abs_t'] = abs_thresholds_dict
    return benchmark_dog'''

def benchmark_pers_hom(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    seuil_min = np.min(SEUILS)
    
    ABS_THRESHOLDS = param['abs_t']
    benchmark_pers_hom = dict()
    abs_thresholds_dict = dict()
    
    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        #abs_thresholds_dict_tmp = dict()
    
        coordinates = peak_detection.pers_hom(chromato_obj=(
                chromato, time_rn), mod_time=0, seuil=seuil_min, mode=mode, chromato_cube=chromato_cube, cluster= cluster, threshold_abs=ABS_THRESHOLDS)
        '''abs_thresholds_dict_tmp['min_distance'] = threshold_filter(
            chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25)
    
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp'''
        abs_thresholds_dict[ABS_THRESHOLDS] = threshold_filter(
            chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25, chromato_cube=chromato_cube)
    
    benchmark_pers_hom['abs_t'] = abs_thresholds_dict
    return benchmark_pers_hom
    

def benchmark_peak_local_max(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    min_distances = param['min_distance']
    seuil_min = np.min(SEUILS)

    ABS_THRESHOLDS = param['abs_t']
    benchmark_plm = dict()
    abs_thresholds_dict = dict()
    chromato_max_val = np.max(chromato)
    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        abs_thresholds_dict_tmp = dict()
        min_distance_dict = dict()
        for min_distance in min_distances:
            # We have to compute plm for each relative thresholds because thresholds are relative to the mass slices
            if(mode=='rgedrgerrge'):
                '''coordinates = peak_detection.plm_mass_per_mass_multiprocessing(chromato_cube=chromato_cube, seuil=seuil_min,min_distance=min_distance, threshold_abs=ABS_THRESHOLDS * chromato_max_val)
                print(len(coordinates))
                min_distance_dict[min_distance] = threshold_filter_mass_per_mass(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, chromato_cube, mod_time=1.25, cluster=cluster)'''
                coordinates = peak_detection.plm(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, min_distance=min_distance, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                print(len(coordinates))
                min_distance_dict[min_distance] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25, mode=mode, chromato_cube=chromato_cube)
            else:
                coordinates = peak_detection.plm(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, min_distance=min_distance, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                print(len(coordinates))
                min_distance_dict[min_distance] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25, mode=mode, chromato_cube=chromato_cube)
    
        abs_thresholds_dict_tmp['min_distance'] = min_distance_dict
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp
    benchmark_plm['abs_t'] = abs_thresholds_dict
    return benchmark_plm


'''def benchmark_peak_local_ancien(chromato_obj, spectra, spectrum_path, spectrum_lib, param, mod_time=1.25, mode="tic", chromato_cube=None, cluster=False):
    chromato, time_rn, spectra_obj = chromato_obj
    SEUILS = param['seuil']
    min_distances = param['min_distance']
    seuil_min = np.min(SEUILS)

    ABS_THRESHOLDS = param['abs_t']
    benchmark_plm = dict()
    abs_thresholds_dict = dict()
    for ABS_THRESHOLDS in ABS_THRESHOLDS:
        abs_thresholds_dict_tmp = dict()
        min_distance_dict = dict()
        for min_distance in min_distances:
            print(min_distance)
            # We have to compute plm for each relative thresholds because thresholds are relative to the mass slices
            if(mode=='mass_per_mass'):
                threshold_dict = dict()
                benchmark_threshold_dict = dict()
                for SEUIL in SEUILS:
                    print(SEUIL)
                    coordinates = peak_detection.plm(chromato_obj=(
                        chromato, time_rn), mod_time=0, seuil=SEUIL, min_distance=min_distance, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                    print(len(coordinates))
                    benchmark_threshold_dict[SEUIL] = compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)
                threshold_dict['seuil'] = benchmark_threshold_dict
                min_distance_dict[min_distance] = threshold_dict
            else:
                coordinates = peak_detection.plm(chromato_obj=(
                    chromato, time_rn), mod_time=0, seuil=seuil_min, min_distance=min_distance, mode=mode, chromato_cube=chromato_cube, cluster=cluster, threshold_abs=ABS_THRESHOLDS)
                min_distance_dict[min_distance] = threshold_filter(
                    chromato_obj, spectra, spectrum_path, spectrum_lib, param, coordinates, mod_time=1.25)
        abs_thresholds_dict_tmp['min_distance'] = min_distance_dict
        abs_thresholds_dict[ABS_THRESHOLDS] = abs_thresholds_dict_tmp
    benchmark_plm['abs_t'] = abs_thresholds_dict
    return benchmark_plm'''


def benchmark(chromato_obj, spectra, chromato_cube, mod_time, spectrum_path, lib_path, params, mass=None, benchmark_filename="./benchmark.json", mode='tic' , cluster=False):
    #spectrum_lib = list(load_from_mgf(lib_path, metadata_harmonization=False))
    spectrum_lib = None
    benchmark = dict()
    #chromato_obj = read_chroma(filename, mod_time)
    chromato, time_rn, spectra_obj = chromato_obj

    '''chromato_cube = None
    if (mode == "tic"):
        spectra, debuts, fins = read_spectra(spectra_obj)
    else:
        #full_spectra = read_full_spectra(spectra_obj=spectra_obj)
        full_spectra = read_full_spectra_centroid(spectra_obj=spectra_obj)

        spectra, debuts, fins = full_spectra
    
        chromato_cube = full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
        #chromato_cube = full_spectra_to_chromato_cube_centroid(full_spectra=full_spectra, spectra_obj=spectra_obj)
        
        #chromato_cube = chromato_cube[:3]


    if (mass):
        chromato_mass, debuts = read_chromato_mass(spectra_obj, mass)
        chromato = chromato_mass'''
    for method in list(params.keys()):
        print(method)
        filter_dict = dict()
        filter_dict['No Filter'] = globals()['benchmark_' + method](chromato_obj=(chromato, time_rn, spectra_obj), spectra=spectra,
                                                    spectrum_path=spectrum_path, spectrum_lib=spectrum_lib, param=params[method], mod_time=1.25, mode=mode, cluster=cluster, chromato_cube=(chromato_cube))
        filters = params[method]['filters']
        for filter in filters:
            filter_dict[filter] = globals()['benchmark_' + method](chromato_obj=(globals()[filter](chromato), time_rn, spectra_obj), spectra=spectra,
                                                    spectrum_path=spectrum_path, spectrum_lib=spectrum_lib, param=params[method], mod_time=1.25, mode=mode, cluster=cluster, chromato_cube=globals()[filter](chromato_cube))
            
        benchmark[method] = filter_dict

    json_object = json.dumps(benchmark, indent=4)
    with open(benchmark_filename, "w") as outfile:
        outfile.write(json_object)
