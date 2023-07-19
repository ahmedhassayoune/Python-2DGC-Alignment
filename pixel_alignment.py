import subprocess
import os

def pixel_alignment(PATH = "F:/Bureau/032023-data-Tenax-VOC-COVID/"):
    r"""Pixel based chromatogram alignment.

    Parameters
    ----------
    PATH :
        Path to the directory containing chromatograms of the cohort.
    """
    files = os.listdir(PATH)
    files = [PATH + file for file in files]
    print(files)
    ref_index = 1
    ref = files[ref_index]
    files.pop(ref_index)
    print(ref)
    print(files)
    p = subprocess.Popen(['Rscript', './pixel_alignment.r', ref] + files[:-1])
    p.wait()