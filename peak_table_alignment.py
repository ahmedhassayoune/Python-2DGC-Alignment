import subprocess
import os
import time
import utils
import sys 
def peak_table_alignment():
    r"""Peak table based alignment.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    ----------
    Examples
    --------
    >>> python peak_table_alignment.py 'PATH'
    """
    if (len(sys.argv) > 1):
        PATH=sys.argv[1]
        start_time=time.time()
        ABS_PATH = os.path.abspath(PATH)
        print(ABS_PATH)
        p = subprocess.Popen(['Rscript', './peak_table_alignment.r', ABS_PATH])
        p.wait()
        print(time.time() - start_time)
        utils.add_formula_in_aligned_peak_table(PATH + '/aligned_peak_table.csv', PATH + '/peak_table.csv')

if __name__ == '__main__':
    peak_table_alignment()