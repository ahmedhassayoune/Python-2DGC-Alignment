import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import multiprocessing
from scipy.signal import savgol_filter
import pybaselines

import time


def baseline_als(y, lam, p, niter=10):

  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def chromato_no_baseline(chromato, j=None):
    r"""Correct baseline and apply savgol filter.
    ----------
    chromato : ndarray
        Input chromato.
    Returns
    -------
    chromato :
        The input chromato without baseline
    Examples
    --------
    >>> import read_chroma
    >>> import baseline_correction
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> chromato = baseline_correction.chromato_no_baseline(chromato)
    """
    tmp = np.empty_like(chromato)
    for i in range (tmp.shape[1]):
        tmp[:,i] = savgol_filter(chromato[:,i] - pybaselines.whittaker.asls(chromato[:,i], lam=1000.0, p=0.05)[0], 5, 2, mode='nearest')
    tmp[tmp < .0] = 0
    return tmp



def chromato_cube_corrected_baseline(chromato_cube):
    r"""Apply baseline correction on each chromato of the input.
    ----------
    chromato_cube :
        Input chromato.
    Returns
    -------
    chromato_cube:
        List of chromato from input list without baseline
    Examples
    --------
    >>> import read_chroma
    >>> import baseline_correction
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    >>> chromato_cube = np.array(baseline_correction.chromato_cube_corrected_baseline(chromato_cube))
    """
    cpu_count = multiprocessing.cpu_count()
    chromato_cube_no_baseline = []
    with multiprocessing.Pool(processes = cpu_count) as pool:
        for i, result in enumerate(pool.starmap(chromato_no_baseline, [(m_chromato, j) for j, m_chromato in enumerate(chromato_cube)])):
            chromato_cube_no_baseline.append(result)
    return chromato_cube_no_baseline