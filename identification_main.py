import glob
import time
import pyms_nist_search
import read_chroma
import numpy as np
import pyms
from scipy.signal import argrelextrema
import gc


import subprocess
import os
import sys
import argparse


def cohort_identification_and_alignment(PATH, OUTPUT_PATH, mod_time=1.25, method='persistent_homology', mode='tic', seuil=5, hit_prob_min=15, ABS_THRESHOLDS=None, cluster=True, min_distance=1, sigma_ratio=1.6, num_sigma=10, formated_spectra=True, match_factor_min=700):
    r"""Compute peak table for each chromatogram within the cohort and align samples.

    Parameters
    ----------
    PATH :
        Path to the cohort.
    OUTPUT_PATH :
        Path to the resulting peak tables. 
    -------
    --------
    Examples
    --------
    >>> python peak_table_alignment.py 'PATH' 'OUTPUT_PATH'
    """

    files = os.listdir(PATH)
    files=[file for file in files if ('.cdf' in file)]
    for i, file in enumerate(files):
        print(file)
        subprocess.run(["python", "identification.py", '-p' , PATH, '-f', file, '-op', OUTPUT_PATH, '--mod_time', str(mod_time), '-m', method, '--mode', mode, '--threshold', str(seuil), '--hit_prob_min', str(hit_prob_min),
                        '--abs_threshold', str(ABS_THRESHOLDS), '--cluster', str(cluster), '--min_distance', str(min_distance), '--sigma_ratio', str(sigma_ratio), '--num_sigma', str(num_sigma),
                        '--format_spectra', str(formated_spectra), '--match_factor_min', str(match_factor_min)])
    subprocess.run(["python", "peak_table_alignment.py", OUTPUT_PATH])


if __name__ == '__main__':

    parser=argparse.ArgumentParser(description="Launch cohort identification and alignment")
    parser.add_argument('-p', '--path', required=True, help="Path to the directory containing the chromatograms of the cohort")
    parser.add_argument('-op', '--output_path', required=True, help="Path where peaks table will be generated. The input path for the alignment and path where aligned peak table will be generated")
    parser.add_argument('--mod_time', default=1.25, help="Modulation time")
    parser.add_argument('-m', '--method', default='persistent_homology', help="Method used to detect peaks")
    parser.add_argument('--mode', default='tic', help="Mode used to detect peaks. Can be either tic or mass_per_mass or 3D.")
    parser.add_argument('--match_factor_min', default=0, help="Filter compounds with match_factor < match_factor_min")
    parser.add_argument('-t', '--threshold', default=5.0, help="Used to compute theshold as threshold * 100 * estimated gaussian white noise / np.max(chromato).")
    parser.add_argument('-hpm', '--hit_prob_min', default=0, help="Filter compounds with hit_prob < hit_prob_min")
    parser.add_argument('-at', '--abs_threshold', default=0.0, help="If mode='mass_per_mass' or mode='3D', ABS_THRESHOLDS is the threshold relative to a slice of the 3D chromatogram or a slice of the 3D chromatogram.")
    parser.add_argument('-c', '--cluster', default=True, help="Whether to cluster coordinates when mode is mass_per_mass or 3D.")
    parser.add_argument('-md', '--min_distance', default=1, help="peak_local_max method parameter. The minimal allowed distance separating peaks. To find the maximum number of peaks, use min_distance=1.")
    parser.add_argument('-sr', '--sigma_ratio', default=1.6, help="DoG method parameter. The ratio between the standard deviation of Gaussian Kernels used for computing the Difference of Gaussians.")
    parser.add_argument('-ns', '--num_sigma', default=10, help="LoG/DoH method parameter. The number of intermediate values of standard deviations to consider between min_sigma (1) and max_sigma (30).")
    parser.add_argument('-fs', '--format_spectra', default=True, help="If spectra need to be formatted for peak table based alignment.")
    args=parser.parse_args()

    cohort_identification_and_alignment(args.path, args.output_path, mod_time=args.mod_time, method=args.method, mode=args.mode, seuil=args.threshold, hit_prob_min=args.hit_prob_min, ABS_THRESHOLDS=args.abs_threshold, cluster=args.cluster, min_distance=args.min_distance, sigma_ratio=args.sigma_ratio, num_sigma=args.num_sigma, formated_spectra=args.format_spectra, match_factor_min=args.match_factor_min)
    





























'''def test_lib_leaks(mass_spectrum):
    search = pyms_nist_search.Engine(
                    "C:/NIST14/MSSEARCH/mainlib/",
                    pyms_nist_search.NISTMS_MAIN_LIB,
                    "C:/Users/Stan/Test",
                    )
    print('start lib search will start in', '20')
    time.sleep(20)
    print('start lib search')
    for i in range(8000):
        res = search.full_search_with_ref_data(mass_spectrum)

def tmp():
    chromato_obj = read_chroma.read_chroma('F:/Bureau/Elodie_CDF\\818826-etalon-VOC-melange1-split20-2D-d5000-MeOH.cdf')
    start_time = time.time()
    
    print('start read spectrum',start_time)
    chromato,time_rn,spectra_obj = chromato_obj
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    minima = argrelextrema(mv, np.less)[0]
    fins = minima - 1
    fins = np.append(fins, len(mv - 1))
    debuts = np.insert(minima, 0, 0)
    #Stack 1-D arrays as columns into a 2-D array.
    mass_spectra_ind = np.column_stack((debuts,fins))
    print('end read spectrum', time.time() - start_time)
    beg,end=mass_spectra_ind[20]
    mass_values=mv[beg:end]
    i1=iv[beg:end]
    mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, i1)
    test_lib_leaks(mass_spectrum)
    
if __name__ == '__main__':
    tmp()
    print(gc.collect())
    print('start last sleep')
    time.sleep(300)'''
    
'''if __name__ == '__main__':
    subprocess.run(['python', 'test_lib_leaks.py'])
    print('start last sleep')
    time.sleep(60)'''
