import numpy as np
from pyteomics import mgf
import pyms_nist_search
from matchms import Pipeline
import matplotlib.pyplot as plt
import os
from matchms.importing import load_from_mgf
from matchms import calculate_scores, Spectrum
from matchms.similarity import CosineGreedy
from matchms.similarity import ModifiedCosine
from plot import plot_mass, plot_scores_array, visualizer
from projection import chromato_to_matrix, matrix_to_chromato
from read_chroma import read_chroma, read_chromato_mass, chromato_cube, print_chroma_header, full_spectra_to_chromato_cube
from mass_spec import read_spectra, read_spectrum,hmdb_txt_to_mgf,read_hmdb_spectrum, spectra_matching, read_spectra_centroid, centroid_to_full_nominal, read_full_spectra_centroid
import benchmark
import time
import plot
import os
import gc
import baseline_correction
from skimage.restoration import estimate_sigma

import tracemalloc

#from pympler.tracker import SummaryTracker

def bench_all_methods(file, full_filename):
    mod_time = 1.25
    filters = []

    try:
        chromato_obj = read_chroma(full_filename, mod_time)
        chromato, time_rn, spectra_obj = chromato_obj
        full_spectra = read_full_spectra_centroid(spectra_obj=spectra_obj)
        spectra, debuts, fins = full_spectra
        chromato_cube = full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
        
        '''CORRECTION BASELINE CHROMATO AND CUBE (SPECTRA)'''
        chromato = np.array(baseline_correction.chromato_no_baseline(chromato))
        chromato_obj = chromato, time_rn, spectra_obj
        chromato_cube = np.array(baseline_correction.chromato_cube_corrected_baseline(chromato_cube))
        
        '''MIN THRESHOLD 5 times the estimated noise standard deviation (presume it is gaussian noise)'''
        sigma = estimate_sigma(chromato, channel_axis=None)
        MIN_SEUIL = 5 * sigma * 100 / np.max(chromato)
        print(MIN_SEUIL)
        
        SEUIL = [MIN_SEUIL]
        
        os.makedirs(os.path.dirname('./benchmark/' + file[:-4] + '/'), exist_ok=True)
    except:
        return
    

    try:
        print("start 3D PLM")
        start_time = time.time()
        seuil = SEUIL
        peak_local_max_params = {"filters": filters,"min_distance":[1],"seuil": seuil, "abs_t": [0.01]}
        params = {"peak_local_max": peak_local_max_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/plm_3D_cluster.json', mode='3D', cluster=True)
        print("time 3D plm")
        print(time.time() - start_time)

    except Exception as e:
        print("ERR 3D peak_local_max")


    try:
        print("start TIC PLM")
        start_time = time.time()

        seuil = SEUIL
        peak_local_max_params = {"filters": filters,"min_distance":[1],"seuil": seuil, "abs_t": [0.024]}
        params = {"peak_local_max": peak_local_max_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/plm_TIC_cluster.json', mode='tic', cluster=True)
        print("time TIC PLM")
        print(time.time() - start_time)

    except:
        print("ERR TIC peak_local_max")
        
    try:
        print("start mpm PLM")
        start_time = time.time()
        seuil = SEUIL
        peak_local_max_params = {"filters": filters,"min_distance":[1],"seuil": seuil, "abs_t": [0.7, 0.9]}
        params = {"peak_local_max": peak_local_max_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/plm_mass_per_mass_cluster.json', mode='mass_per_mass', cluster=True)
        print("time mpm PLM")
        print(time.time() - start_time)
        
    except:
        print("ERR mass_per_mass peak_local_max")
        

    try:
        print("start tic dog")
        start_time = time.time()
        dog_params = {"filters": filters, "sigma_ratio": [1.6], "seuil": SEUIL, "abs_t": [0.01]}
        params = {"DoG":dog_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/dog_TIC_cluster.json', mode='tic', cluster=True)
        print("time tic dog")
        print(time.time() - start_time)

    except:
        print("ERR TIC DoG")

    try:
        print('start 3D dog')
        start_time = time.time()
        dog_params = {"filters": filters, "sigma_ratio": [1.6], "seuil": SEUIL, "abs_t": [0.01]}
        params = {"DoG":dog_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/dog_3D_cluster.json', mode='3D', cluster=True)
        print('time 3D dog')
        print(time.time() - start_time)

    except:
        print("ERR 3D DoG")
    try:
        print("start mpm DoG")
        start_time = time.time()
        dog_params = {"filters": filters, "sigma_ratio": [1.6], "seuil": SEUIL, "abs_t": [0.7, 0.9]}
        params = {"DoG":dog_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/dog_mass_per_mass_cluster.json', mode='mass_per_mass', cluster=True)
        print("time mpm DoG")
        print(time.time() - start_time)

    except:
        print("ERR mas_per_mass DoG")

    try:
        print("start TIC log")
        start_time = time.time()
        log_params = {"filters": filters, "num_sigma": [10], "seuil": SEUIL, "abs_t": [0.01]}
        params = {"LoG": log_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/log_TIC_cluster.json', mode='tic', cluster=True)
        print("time TIC log")
        print(time.time() - start_time)

    except:
        print("ERR TIC LoG")
        
    try:
        print("start mass_per_mass log")
        start_time = time.time()
        log_params = {"filters": filters, "num_sigma": [10], "seuil": SEUIL, "abs_t": [0.7, 0.9]}
        params = {"LoG": log_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/log_mass_per_mass_cluster.json', mode='mass_per_mass', cluster=True)
        print("time mass_per_mass log")
        print(time.time() - start_time)

    except:
        print("ERR mass_per_mass LoG")
        
    '''try:
        print("start 3D log")
        start_time = time.time()
        log_params = {"filters": filters, "num_sigma": [10], "seuil": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], "abs_t": [0.01]}
        params = {"LoG": log_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/log_3D_cluster.json', mode='3D', cluster=True)
        print("time 3D log")
        print(time.time() - start_time)

    except:
        print("ERR 3D LoG")'''


    '''try:
        print("start TIC DoH")
        start_time = time.time()
        doh_params = {"filters": filters, "num_sigma": [10],"seuil": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], "abs_t": [0.01]}
        params = {"DoH":doh_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/doh_TIC_cluster.json', mode='tic', cluster=True)
        print("time TIC DoH")
        print(time.time() - start_time)

    except:
        print("ERR TIC DoH")

    try:
        print("start mpm DoH ")
        start_time = time.time()
        doh_params = {"filters": filters, "num_sigma": [10],"seuil": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], "abs_t": [0.7, 0.9]}
        params = {"DoH":doh_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/doh_mass_per_mass_no_cluster.json', mode='mass_per_mass', cluster=True)
        print("time mpm DoH ")
        print(time.time() - start_time)

    except:
        print("ERR DoH mass_per_mass")'''
        
    try:
        print("start TIC pers_hom")
        start_time = time.time()

        seuil = SEUIL
        pers_hom_params = {"filters": filters, "seuil": seuil, "abs_t": [0.024]}
        params = {"pers_hom": pers_hom_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/pers_hom_TIC_cluster.json', mode='tic', cluster=True)
        print("time TIC pers_hom")
        print(time.time() - start_time)

    except:
        print("ERR TIC pers_hom")
        
    try:
        print("start mpm pers_hom")
        start_time = time.time()
        seuil = SEUIL
        pers_hom_params = {"filters": filters, "seuil": seuil, "abs_t": [0.7]}
        params = {"pers_hom": pers_hom_params}
        benchmark.benchmark(chromato_obj, spectra, chromato_cube, mod_time=1.25, spectrum_path = "./myspectra.mgf",lib_path = "./my_lib.mgf", params=params, benchmark_filename='./benchmark/' + file[:-4] + '/pers_hom_mass_per_mass_cluster.json', mode='mass_per_mass', cluster=True)
        print("time mpm pers_hom")
        print(time.time() - start_time)
        
    except:
        print("ERR mass_per_mass pers_hom")
        
    del chromato_cube
    del spectra
    del debuts
    del fins
    del full_spectra
    del chromato
    del chromato_obj
    del spectra_obj
    collected = gc.collect()
    print("Garbage collector: collected",
        "%d objects." % collected)
    

if __name__ == '__main__':
    filters = []
    print("start benchmark")
    PATH = 'G:/ELO_CDF/liquide-cdf-centroid/'
    files = os.listdir(PATH)
    
    already_bench = (os.listdir("./benchmark/"))

    #tracemalloc.start()

    #count = 0
    for file in files:
        print(file)
        
        '''other
        if ("NIST" in file or "G0" in file):
            continue'''
        '''NIST
        if (not "NIST" in file):
            continue'''
        '''G0
        if (not "G0" in file):
            continue'''
        
        if (file[:-4] in already_bench):
            continue
        
        full_filename = PATH + file
        print(full_filename)
        
        bench_all_methods(file, full_filename)
    
    '''snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)'''
            