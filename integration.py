import projection
import numpy as np
import plot
import mass_spec
import utils

from scipy import ndimage as ndi
from skimage.segmentation import watershed
import skimage
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
def peak_pool(chromato, coordinates, coordinate, threshold=0.25, plot_labels=False):
    mask = np.zeros(chromato.shape, dtype=bool)
    mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(mask)
    peak_apex_int = chromato[coordinate[0]][coordinate[1]]
    img = np.where(chromato < threshold * peak_apex_int, 0, 1)
    labels = watershed(-chromato, markers, mask=img)
    '''if (plot_labels):
        plot.visualizer((labels, time_rn), log_chromato=False)'''
    coordinate_label = labels[coordinate[0]][coordinate[1]]
    blob = np.where(labels != coordinate_label, 0, 1)
    return blob


def get_all_area(chromato, coordinates, threshold=0.25):
    blobs = []
    for coordinate in coordinates:
        blobs.append(peak_pool(chromato, coordinate, threshold=threshold))
    return np.array(blobs)

def compute_area(chromato, blob):
    r"""Compute area of a blob.

    Parameters
    ----------
    chromato:
        TIC chromatogram
    blob:
        An ndarray of boolean values with the same shape as the TIC chromatogram passed in parameter which indicate for each pixel if it belongs to the peak blob and computed by peak_pool_similarity_check function. 
    Returns
    -------
    A: float
        The computed area/volume of the blob/peak.
    Examples
    --------
    >>> import peak_detection
    >>> import read_chroma
    >>> import integration
    >>> # First detect some peaks in the chromatogram.
    >>> from skimage.restoration import estimate_sigma
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> # seuil=MIN_SEUIL is computed as the ratio between 5 times the estimated gaussian white noise standard deviation (sigma) in the chromatogram and the max value in the chromatogram.
    >>> sigma = estimate_sigma(chromato, channel_axis=None)
    >>> MIN_SEUIL = 5 * sigma * 100 / np.max(chromato)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    >>> coordinates = peak_detection.peak_detection(chromato_obj=(chromato, time_rn, spectra_obj), chromato_cube=chromato_cube, seuil=MIN_SEUIL)
    >>> matches = matching.matching_nist_lib_from_chromato_cube((chromato, time_rn, spectra_obj), chromato_cube, coordinates, mod_time = 1.25, hit_prob_min=hit_prob_min)
    >>> blob = integration.peak_pool_similarity_check(chromato, np.stack(matches[:,2]), matches[0][2], chromato_cube, threshold=0.5, plot_labels=True, similarity_threshold=similarity_threshold)
    >>> area = integration.compute_area(chromato, blob)
    """
    '''blob = np.argwhere(blob != 0)
    return np.sum(chromato[blob])'''
    cds=np.argwhere(blob != 0)
    return np.sum(chromato[cds[:,0], cds[:,1]])

def get_contour(blob, chromato, time_rn):
    contour=skimage.segmentation.expand_labels(blob, distance=1) - blob
    contour_pts = np.argwhere(contour != 0)
    return projection.matrix_to_chromato(contour_pts, time_rn, 1.25, chromato.shape)


def get_all_contour(blobs):
    all_contour=[]
    for blob in blobs:
        blob_contour = get_contour(blob)
        for contour in blob_contour:
            all_contour.append(contour)
    return np.unique(all_contour, axis=-0)


def similarity_cluestering(chromato_cube, coordinates, ref_point, similarity_threshold=0.01):
    intensity_values_list = []
    ref_point_index = 0

    for i, coordinate in enumerate(coordinates):
        if (coordinate[0]==ref_point[0] and coordinate[1]==ref_point[1]):
            ref_point_index = i
        int_values = mass_spec.read_spectrum_from_chromato_cube(coordinate, chromato_cube=chromato_cube)
        intensity_values_list.append(int_values)
    intensity_values_list = np.array(intensity_values_list)
    clustering = DBSCAN(eps=similarity_threshold, min_samples=1, metric='cosine').fit(intensity_values_list)
    ref_label = (clustering.labels_[ref_point_index])
    res = []
    for i, coordinate in enumerate(coordinates):
        if (clustering.labels_[i] == ref_label):
            res.append(coordinate)
    return np.array(res)

def peak_pool_similarity_check_coordinates(chromato, coordinates, coordinate, chromato_cube, threshold=0.25, similarity_threshold=0.01, plot_labels=False):
    r"""Peak integration. Find pixels in the neighborhood of the peak passed in parameter which belong to the peak blob. It is usefull to determine the area of the peak.

    Parameters
    ----------
    chromato:
        TIC chromatogram
    coordinates:
        Detected peaks coordinates.
    coordinate:
        Coordinate of the peak to be integrated.
    chromato_cube:
        3D chromatogram to read mass spectra from.
    threshold: optional
        Intensity threshold. Only pixels that have associated intensities greather than coordinate (center of the blob) intensity * threshold are considered to be part of the blob.
    similarity_threshold: optional
        Similarity threshold. Only pixels that have associated mass spectra similar to the coordinate (center of the blob) mass spectrum greather than 0.99 are considered to be part of the blob.
    Returns
    -------
    A: 1D list
        A list with blobs coordinates.
    Examples
    --------
    >>> import peak_detection
    >>> import read_chroma
    >>> import integration
    >>> # First detect some peaks in the chromatogram.
    >>> from skimage.restoration import estimate_sigma
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> # seuil=MIN_SEUIL is computed as the ratio between 5 times the estimated gaussian white noise standard deviation (sigma) in the chromatogram and the max value in the chromatogram.
    >>> sigma = estimate_sigma(chromato, channel_axis=None)
    >>> MIN_SEUIL = 5 * sigma * 100 / np.max(chromato)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    >>> coordinates = peak_detection.peak_detection(chromato_obj=(chromato, time_rn, spectra_obj), chromato_cube=chromato_cube, seuil=MIN_SEUIL)
    >>> matches = matching.matching_nist_lib_from_chromato_cube((chromato, time_rn, spectra_obj), chromato_cube, coordinates, mod_time = 1.25, hit_prob_min=hit_prob_min)
    >>> cds = integration.peak_pool_similarity_check_coordinates(chromato, np.stack(matches[:,2]), matches[0][2], chromato_cube, threshold=0.5, plot_labels=True, similarity_threshold=similarity_threshold)
    """
    mask = np.zeros(chromato.shape, dtype=bool)
    mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(mask)
    peak_apex_int = chromato[coordinate[0]][coordinate[1]]
    img = np.where(chromato < threshold * peak_apex_int, 0, 1)
    labels = watershed(-chromato, markers, mask=img)
    '''if (plot_labels):
        plot.visualizer((labels, time_rn), log_chromato=False)'''
    coordinate_label = labels[coordinate[0]][coordinate[1]]
    blob = np.where(labels != coordinate_label, 0, 1)
    cds = []
    for cd in np.argwhere(blob == 1):
        cds.append([cd[0], cd[1]])
    cds = np.array(cds)
    cds = similarity_cluestering(chromato_cube, cds, ref_point=coordinate, similarity_threshold=similarity_threshold)
    return cds

def peak_pool_similarity_check(chromato, coordinates, coordinate, chromato_cube, threshold=0.25, similarity_threshold=0.01, plot_labels=False):
    r"""Peak integration. Find pixels in the neighborhood of the peak passed in parameter which belong to the peak blob. It is usefull to determine the area of the peak.

    Parameters
    ----------
    chromato:
        TIC chromatogram
    coordinates:
        Detected peaks coordinates.
    coordinate:
        Coordinate of the peak to be integrated.
    chromato_cube:
        3D chromatogram to read mass spectra from.
    threshold: optional
        Intensity threshold. Only pixels that have associated intensities greather than coordinate (center of the blob) intensity * threshold are considered to be part of the blob.
    similarity_threshold: optional
        Similarity threshold. Only pixels that have associated mass spectra similar to the coordinate (center of the blob) mass spectrum greather than 0.99 are considered to be part of the blob.
    Returns
    -------
    A: ndarray
        An ndarray of boolean values with the same shape as the TIC chromatogram passed in parameter which indicate for each pixel if it belongs to the peak blob.
    Examples
    --------
    >>> import peak_detection
    >>> import read_chroma
    >>> import integration
    >>> # First detect some peaks in the chromatogram.
    >>> from skimage.restoration import estimate_sigma
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> # seuil=MIN_SEUIL is computed as the ratio between 5 times the estimated gaussian white noise standard deviation (sigma) in the chromatogram and the max value in the chromatogram.
    >>> sigma = estimate_sigma(chromato, channel_axis=None)
    >>> MIN_SEUIL = 5 * sigma * 100 / np.max(chromato)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    >>> coordinates = peak_detection.peak_detection(chromato_obj=(chromato, time_rn, spectra_obj), chromato_cube=chromato_cube, seuil=MIN_SEUIL)
    >>> matches = matching.matching_nist_lib_from_chromato_cube((chromato, time_rn, spectra_obj), chromato_cube, coordinates, mod_time = 1.25, hit_prob_min=hit_prob_min)
    >>> blob = integration.peak_pool_similarity_check(chromato, np.stack(matches[:,2]), matches[0][2], chromato_cube, threshold=0.5, plot_labels=True, similarity_threshold=similarity_threshold)
    """
    mask = np.zeros(chromato.shape, dtype=bool)
    mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(mask)
    peak_apex_int = chromato[coordinate[0]][coordinate[1]]
    img = np.where(chromato < threshold * peak_apex_int, 0, 1)
    labels = watershed(-chromato, markers, mask=img)
    '''if (plot_labels):
        plot.visualizer((labels, time_rn), log_chromato=False)'''
    coordinate_label = labels[coordinate[0]][coordinate[1]]
    blob = np.where(labels != coordinate_label, 0, 1)
    cds = []
    for cd in np.argwhere(blob == 1):
        cds.append([cd[0], cd[1]])
    cds = np.array(cds)
    cds = similarity_cluestering(chromato_cube, cds, ref_point=coordinate, similarity_threshold=similarity_threshold)
    res = np.zeros_like(chromato)
    for cd in cds:
        res[cd[0], cd[1]] = 1
    return res

def tmp(chromato, coordinates, coordinate, threshold=0.25):
    mask = np.zeros(chromato.shape, dtype=bool)
    mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(mask)
    peak_apex_int = chromato[coordinate[0]][coordinate[1]]
    img = np.where(chromato < threshold * peak_apex_int, 0, 1)
    labels = watershed(-chromato, markers, mask=img)
    coordinate_label = labels[coordinate[0]][coordinate[1]]
    blob = np.where(labels != coordinate_label, 0, 1)
    cds = []
    for cd in np.argwhere(blob == 1):
        cds.append([cd[0], cd[1]])
    return cds

def estimate_num_components_aic(X, max_components=3):
    n_components_range = range(1, max_components+1)
    aics = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)
        aic = gmm.aic(X)
        aics.append(aic)
    
    optimal_num_components = np.argmin(aics) + 1
    print(aics)
    return optimal_num_components

def estimate_num_components(X, max_components=3):
    n_components_range = range(1, max_components+1)
    bics = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)
        bic = gmm.bic(X)
        bics.append(bic)
    
    optimal_num_components = np.argmin(bics) + 1
    print(bics)
    return optimal_num_components

def estimate_num_components_with_spectra_dist(X, chromato_cube, mass_values, max_components=3, threshold=0.9):
    dists=[1.0]
    for n_components in range(2,max_components+1):
        gmm = GaussianMixture(n_components=n_components, tol=0.001)
        gmm.fit(X)
        gaussian_peak_centers=np.around(gmm.means_,0).astype(int)
        d=[]
        for i in range (n_components):
            for j in range (i + 1, n_components):
                print(gaussian_peak_centers[i], gaussian_peak_centers[j])
                d_tmp=utils.compute_spectra_similarity((mass_values, mass_spec.read_spectrum_from_chromato_cube(gaussian_peak_centers[i], chromato_cube)), (mass_values,mass_spec.read_spectrum_from_chromato_cube(gaussian_peak_centers[j], chromato_cube)))
                d.append(d_tmp)
        dists.append(np.mean(d))
        print(gmm.n_iter_)
    print(dists)
    optimal_num_components = np.argmin(dists) + 1
    return optimal_num_components if np.min(dists) < threshold else 1

def estimate_peaks_center_bic(cds, num_components=None, max_components=3):
    if (num_components is None):
        num_components=estimate_num_components(cds, max_components=max_components)
    gm = GaussianMixture(n_components=num_components, random_state=0).fit(cds)
    return gm.means_, num_components

def estimate_peaks_center_with_dist(cds, chromato_cube, mass_values, num_components=None, threshold=0.90, max_components=3):
    if (num_components is None):
        num_components=estimate_num_components_with_spectra_dist(cds, chromato_cube, mass_values, max_components=max_components, threshold=threshold)
    gm = GaussianMixture(n_components=num_components, random_state=0).fit(cds)
    return gm.means_, num_components

def estimate_peaks_center_aic(cds, num_components=None, max_components=3):
    if (num_components is None):
        num_components=estimate_num_components_aic(cds, max_components=max_components)
    gm = GaussianMixture(n_components=num_components, random_state=0).fit(cds)
    return gm.means_, num_components
