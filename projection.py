import numpy as np
def matrix_to_chromato(points, time_rn, mod_time, chromato_dim):
    r"""Project points from chromatogram matrix (ndarray) into chromatogram (in time).

    Parameters
    ----------
    points :
        Tuple wrapping spectra, debuts and fins.
    time_rn :
        Scan acquisition_time.
    mod_time :
        Modulation time.
    chromato_dim :
        Chromatogram shape.
    Returns
    -------
    A: ndarray
        Return the created 3D chromatogram. An array containing all mass chromatogram for each unique centroided mass.
    Examples
    --------
    >>> import projection
    >>> chromato_obj = read_chroma.read_chroma(file, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> pts_in_matrix = np.array([[20, 20]])
    >>> pts_in_chromato = projection.matrix_to_chromato(pts_in_matrix, time_rn, 1.25, chromato.shape)
    """
    if (points is None):
        return None
    #return np.column_stack((points[:,0] * (time_rn[1] - time_rn[0]) / (chromato_dim[0]) + time_rn[0], points[:,1] * mod_time / chromato_dim[1]))
    return np.column_stack((points[:,0] * (time_rn[1] - time_rn[0]) / (chromato_dim[0] - 1) + time_rn[0], points[:,1] * mod_time / (chromato_dim[1] - 1)))

def chromato_to_matrix(points, time_rn, mod_time, chromato_dim):
    r"""Project points from chromatogram (in time) into matrix chromatogram (ndarray).

    Parameters
    ----------
    points :
        Tuple wrapping spectra, debuts and fins.
    time_rn :
        Scan acquisition_time.
    mod_time :
        Modulation time.
    chromato_dim :
        Chromatogram shape.
    Returns
    -------
    A: ndarray
        Return the created 3D chromatogram. An array containing all mass chromatogram for each unique centroided mass.
    Examples
    --------
    >>> import projection
    >>> chromato_obj = read_chroma.read_chroma(file, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> pts_in_chromato = np.array([[19.1, 0.85]])
    >>> pts_in_matrix = projection.chromato_to_matrix(pts_in_matrix, time_rn, 1.25, chromato.shape)
    """
    if (points is None):
        return None
    #return np.rint(np.column_stack(((points[:,0] -  time_rn[0]) * chromato_dim[0] / (time_rn[1] - time_rn[0]), points[:,1] / mod_time * chromato_dim[1]))).astype(int)
    return np.rint(np.column_stack(((points[:,0] -  time_rn[0]) * (chromato_dim[0] - 1) / (time_rn[1] - time_rn[0]), points[:,1] / mod_time * (chromato_dim[1] - 1)))).astype(int)

