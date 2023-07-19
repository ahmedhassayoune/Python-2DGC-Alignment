from read_chroma import read_chromato_cube
import numpy as np
from matching import matching_nist_lib_from_chromato_cube
from peak_detection import peak_detection
import integration
from identification import cohort_identification_to_csv
import logging
import warnings
def benchmark_integration(filename, methods, params, output_path, mod_time=1.25, seuil=5, suffix=""):
    warnings.filterwarnings("ignore")
    logger=logging.getLogger('pyms_nist_search')
    logger.setLevel('ERROR')
    logger=logging.getLogger('pyms')
    logger.setLevel('ERROR')
    chromato_obj, chromato_cube, sigma=read_chromato_cube(filename, mod_time=1.25, pre_process=True)
    print("chromato cube readed")
    chromato,time_rn,spectra_obj = chromato_obj

    MIN_SEUIL = seuil * sigma * 100 / np.max(chromato)
    print("min seuil: ", MIN_SEUIL)
    hit_prob_min=0
    match_factor_min=0

    for i, method in enumerate(methods):
        param=params[i]
        try:
            ABS_THRESHOLDS=param['ABS_THRESHOLDS']
        except:
            ABS_THRESHOLDS=None
        try:
            cluster=param['cluster']
        except:
            cluster=True
        mode=param['mode']
        try:

            coordinates = peak_detection(chromato_obj=(chromato, time_rn, spectra_obj), spectra=None, chromato_cube=chromato_cube, seuil=MIN_SEUIL,
                ABS_THRESHOLDS=ABS_THRESHOLDS, method=method, mode=mode, cluster=cluster)
            
            matches = matching_nist_lib_from_chromato_cube((chromato, time_rn, spectra_obj), chromato_cube, coordinates, mod_time = 1.25, hit_prob_min=hit_prob_min, match_factor_min=match_factor_min)

            matches_identification = []
            similarity_threshold=0.001
            for match in matches:
                coord =  match[2]
                blob = integration.peak_pool_similarity_check(chromato, np.stack(matches[:,2]), coord, chromato_cube, threshold=0.5, plot_labels=True, similarity_threshold=similarity_threshold)
                area = integration.compute_area(chromato, blob)
                height = chromato[coord[0], coord[1]]
                
                identification_data_dict = dict()
                identification_data_dict['casno'] = match[1]['casno']
                identification_data_dict['compound_name'] = match[1]['compound_name']
                identification_data_dict['compound_formula'] = match[1]['compound_formula']
                identification_data_dict['hit_prob'] = match[1]['hit_prob']
                identification_data_dict['match_factor'] = match[1]['match_factor']
                identification_data_dict['reverse_match_factor'] = match[1]['reverse_match_factor']
                identification_data_dict['rt1'] = match[0][0]
                identification_data_dict['rt2'] = match[0][1]
                identification_data_dict['area'] = area
                identification_data_dict['height'] = height
                
                matches_identification.append(identification_data_dict)

            cohort_identification_to_csv(suffix + method + "_" + mode + "_cluster_" + str(cluster), matches_identification, output_path)
            print("method", "mode", "done")
        except:
            print("ERROR in " + method)


methods=['peak_local_max','peak_local_max','peak_local_max', 'persistent_homology', 'persistent_homology', 'DoG', 'DoG', 'DoG', 'LoG', 'LoG']

peak_local_max_params = {"mode": "tic"}
peak_local_max_mpm_params = {"mode": "mass_per_mass", "cluster": True, 'ABS_THRESHOLDS':0.7}
peak_local_max_3d_params = {"mode": "3D", "cluster": True, 'ABS_THRESHOLDS':0.01}

drain_params = {"mode": "tic"}
drain_mpm_params = {"mode": "mass_per_mass", "cluster": True,'ABS_THRESHOLDS':0.7}

dog_params = {"mode": "tic"}
dog_mpm_params = {"mode": "mass_per_mass", "cluster": True, 'ABS_THRESHOLDS':0.7}
dog_3d_params = {"mode": "3D", "cluster": True, 'ABS_THRESHOLDS':0.01}

log_params = {"mode": "tic"}
log_mpm_params = {"mode": "mass_per_mass", "cluster": True, 'ABS_THRESHOLDS':0.7}

params=[peak_local_max_params, peak_local_max_mpm_params, peak_local_max_3d_params, drain_params, drain_mpm_params, dog_params, dog_mpm_params, dog_3d_params, log_params, log_mpm_params]

if __name__ == '__main__':
    '''benchmark_integration('./SIMULATION/simulation_weak_overlap.cdf', methods=methods, params=params, suffix="weak_overlap_")
    benchmark_integration('./SIMULATION/simulation_mid_overlap.cdf', methods=methods, params=params, suffix="mid_overlap_")
    benchmark_integration('./SIMULATION/simulation_strong_overlap.cdf', methods=methods, params=params, suffix="strong_overlap_")'''
    
    '''benchmark_integration('G:/SIMULATION/noisy_intense_simulation_weak_overlap.cdf', methods=methods, params=params, suffix="noisy_intense_weak_overlap_", output_path='G:/SIMULATION/noisy_peak_tables/')
    benchmark_integration('G:/SIMULATION/noisy_intense_simulation_strong_overlap.cdf', methods=methods, params=params, suffix="noisy_intense_strong_overlap_", output_path='G:/SIMULATION/noisy_peak_tables/')
    benchmark_integration('G:/SIMULATION/noisy_intense_simulation_mid_overlap.cdf', methods=methods, params=params, suffix="noisy_intense_mid_overlap_", output_path='G:/SIMULATION/noisy_peak_tables/')'''

    '''benchmark_integration('G:/SIMULATION/loc_1000_scale_500_poisson_09.cdf', methods=methods, params=params, suffix="loc_1000_", output_path='G:/SIMULATION/noisy_loc/')
    benchmark_integration('G:/SIMULATION/loc_1500_scale_500_poisson_09.cdf', methods=methods, params=params, suffix="loc_1500_", output_path='G:/SIMULATION/noisy_loc/')
    benchmark_integration('G:/SIMULATION/loc_2000_scale_500_poisson_09.cdf', methods=methods, params=params, suffix="loc_2000_", output_path='G:/SIMULATION/noisy_loc/')
    benchmark_integration('G:/SIMULATION/loc_4000_scale_500_poisson_09.cdf', methods=methods, params=params, suffix="loc_4000_", output_path='G:/SIMULATION/noisy_loc/')
    benchmark_integration('G:/SIMULATION/loc_8000_scale_500_poisson_09.cdf', methods=methods, params=params, suffix="loc_8000_", output_path='G:/SIMULATION/noisy_loc/')'''

    try:
        benchmark_integration('G:/SIMULATION/loc_1000_scale_500_poisson_12.cdf', methods=methods, params=params, suffix="poisson_rep_12_", output_path='G:/SIMULATION/noisy_poisson_rep/')
    except:
        print("ERROR loc_1000_scale_500_poisson_12")
    try:
        benchmark_integration('G:/SIMULATION/loc_1000_scale_500_poisson_15.cdf', methods=methods, params=params, suffix="poisson_rep_15_", output_path='G:/SIMULATION/noisy_poisson_rep/')
    except:
        print("ERROR loc_1000_scale_500_poisson_15")
    try:
        benchmark_integration('G:/SIMULATION/loc_1000_scale_500_poisson_18.cdf', methods=methods, params=params, suffix="poisson_rep_18_", output_path='G:/SIMULATION/noisy_poisson_rep/')
    except:
        print("ERROR loc_1000_scale_500_poisson_18")
    try:
        benchmark_integration('G:/SIMULATION/loc_1000_scale_500_poisson_21.cdf', methods=methods, params=params, suffix="poisson_rep_21_", output_path='G:/SIMULATION/noisy_poisson_rep/')
    except:
        print("ERROR loc_1000_scale_500_poisson_21")


    try:
        benchmark_integration('G:/SIMULATION/loc_2000_scale_500_poisson_12_overlap_07_099.cdf', methods=methods, params=params, suffix="loc_2000_poisson_rep_12_overlap_07_099", output_path='G:/SIMULATION/noisy_loc_poisson_rep_strong_overlap/')
    except:
        print("ERROR loc_2000_scale_500_poisson_12_overlap_07_09")
    try:
        benchmark_integration('G:/SIMULATION/loc_2000_scale_500_poisson_15_overlap_07_099.cdf', methods=methods, params=params, suffix="loc_2000_poisson_rep_15_overlap_07_099", output_path='G:/SIMULATION/noisy_loc_poisson_rep_strong_overlap/')
    except:
        print("ERROR loc_2000_scale_500_poisson_15_overlap_07_099")
    try:
        benchmark_integration('G:/SIMULATION/loc_2000_scale_500_poisson_18_overlap_07_099.cdf', methods=methods, params=params, suffix="loc_2000_poisson_rep_18_overlap_07_099", output_path='G:/SIMULATION/noisy_loc_poisson_rep_strong_overlap/')
    except:
        print("ERROR loc_2000_scale_500_poisson_18_overlap_07_099")
    try:
        benchmark_integration('G:/SIMULATION/loc_2000_scale_500_poisson_21_overlap_07_099.cdf', methods=methods, params=params, suffix="loc_2000_poisson_rep_21_overlap_07_099", output_path='G:/SIMULATION/noisy_loc_poisson_rep_strong_overlap/')
    except:
        print("ERROR loc_2000_scale_500_poisson_21_overlap_07_099")
