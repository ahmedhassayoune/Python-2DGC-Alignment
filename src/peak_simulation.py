import numpy as np
import random
from matchms.importing import load_from_mgf, scores_from_json
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
import logging
from mass_spec import centroid_to_full_nominal
import os
import math
import netCDF4 as nc
import json


#RADIUS_FACTOR
#CONVERT BLOB STANDARD DEVIATION TO RADIUS
RADIUS_FACTOR=2.0
RADIUS_FACTOR=math.sqrt(2)

def add_noise(noise_typ,image):
    r"""Print chromato header.

    Parameters
    ----------
    noise_typ :
        Noise type.
    image :
        The image to be noisy
    ----------
    Returns
    ----------
    noisy_image:
        Noisy image
    -------
    """
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    
def add_noise_3D(image,noise_typ="custom_gauss", noise_loc=1000.0, noise_scale=500.0, poisson_rep=0.9):
    r"""Add noise in 3D chromatogram

    Parameters
    ----------
    image :
        3D chromatogram ot be noisy.
    noise_typ : optional
        Noise type.
    noise_loc : optional
        Mean of the custom_gauss noise.
    noise_scale : optional
        Standard deviation of the custom_gauss noise.
    poisson_rep : optional
        Poisson parameter of the custom_gauss noise.
    ----------
    Returns
    ----------
    noisy_image:
        Noisy 3D chromatogram.
    -------
    """
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 10
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col, ch))
        gauss = gauss.reshape(row,col, ch)
        noisy = image + gauss
        return noisy
    if noise_typ == "custom_poisson":
        row,col,ch= image.shape
        s = np.random.poisson(1.5, row*col*ch)
        id_g=(np.argwhere(s > 0)).flatten()
        s[id_g]=np.random.poisson(4, len(id_g))
        s=s.reshape(row,col, ch)
        noisy=image+s
        return noisy
    if noise_typ == "custom_gauss":
        row,col,ch= image.shape
        s = np.random.poisson(poisson_rep, row*col*ch)
        id_g=(np.argwhere(s > 0)).flatten()
        #s[id_g]=np.exp(np.abs(np.random.normal(loc=10.0, scale=3.0, size=len(id_g)))) / ch
        s[id_g]=(np.abs(np.random.normal(loc=noise_loc, scale=noise_scale, size=len(id_g))))

        s=s.reshape(row,col,ch)
        noisy=image+s
        return noisy
    

def asym_gauss_kernel(size: int, sizey: int=None, sigma=1.0) -> np.array:
    """
    Returns a 2D asymetric Gaussian kernel.
    
    Parameters
    ----------
    size: int
        Size of the kernel to build
    sigma: float or tuple of float
        sigma_y_1, sigma_y_2, sigma_x_1, sigma_x_2=sigma

    Returns
    -------
    kernel:
        Resulting Gaussian kernel where kernel[i,j] = Gaussian(i, j, mu=(0,0), sigma=sigma
    """
    if isinstance(sigma, tuple):
        sigma_y_1, sigma_y_2, sigma_x_1, sigma_x_2=sigma
    else:
        sigma_x_1=sigma
        sigma_x_2=sigma
        sigma_y_1=sigma
        sigma_y_2=sigma

    size = int(size)
    sizey = int(sizey) if sizey is not None else size
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]
    g=np.zeros_like((x)).astype(float)
    for i in range(len(x)):
        for j in range(len(x)):
            sigma_x=sigma_x_1 if x[i][j] >= 0 else sigma_x_2
            sigma_y=sigma_y_1 if y[i][j] >= 0 else sigma_y_2
            g[i][j]=np.exp(-(x[i][j]**2/(2*(sigma_x)**2)+y[i][j]**2/(2*(sigma_y)**2)))

    return g
    
def gauss_kernel(size: int, sizey: int=None, sigma=1.0) -> np.array:
    """
    Returns a 2D Gaussian kernel.
    
    Parameters
    ----------
    size: int
        Size of the kernel to build
    sigma: float or tuple of float
        sigma_y, sigma_x = sigma

    Returns
    -------
    kernel:
        Resulting Gaussian kernel where kernel[i,j] = Gaussian(i, j, mu=(0,0), sigma=sigma
    """
    if isinstance(sigma, tuple):
        sigma_y, sigma_x = sigma
    else:
        sigma_x=sigma
        sigma_y=sigma

    size = int(size)
    sizey = int(sizey) if sizey is not None else size
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/(2*(sigma_x)**2)+y**2/(2*(sigma_y)**2)))
    return g

def create_peak(size, intensity, sigma=1.0, random_noise=False):
    r"""Returns a peak (guassien kernel multiply by intensity).

    Parameters
    ----------
    size :
       Size of the created gaussian kernel.
    intensity :
       intensity of the peak.
    sigma : optional
       Standard deviation of the gaussian kernel.
    random_noise : optional
       If noise needs to be added to the created peak.
    ----------
    Returns
    -------
    peak:
        The created peak.
    """
    peak = intensity * gauss_kernel(size, sigma=sigma)
    if random_noise:
        peak = add_noise("gauss", peak)
    return peak


def add_peak(chromato, peak, mu):
    r"""Take a chromatogram and a peak and add the peak at the location mu in the TIC chromatogram.

    Parameters
    ----------
    chromato :
       TIC chromatogram in which the peaks need to added.
    peak :
       The peak to be added in the TIC chromatogram.
    mu : optional
       Peak location in the chromatogram.
    ----------
    Returns
    -------
    chromato:
        The chromatogram TIC with a new peak.
    """
    x_size=peak.shape[0]//2
    y_size=peak.shape[1]//2
    for i_in_peak, i in  enumerate(range(mu[0] - x_size, mu[0] + x_size + 1)):
        for j_in_peak, j in enumerate(range(mu[1] - y_size, mu[1] + y_size + 1)):
            chromato[i][j] = chromato[i][j] + peak[i_in_peak][j_in_peak]
    return chromato

def create_and_add_peak(chromato, size, intensity, mu, sigma=1.0):
    r"""Take a chromatogram, create a peak and add the peak at the location mu in the TIC chromatogram.

    Parameters
    ----------
    chromato :
       TIC chromatogram in which the peaks need to added.
    size :
        Size of the created gaussian kernel used to build the new peak.
    intensity :
        Intensity of the created peak.
    mu : optional
       Peak location in the chromatogram.
    sigma : optional

    ----------
    Returns
    -------
    chromato:
        The chromatogram TIC with a new peak.
    """
    peak=create_peak(size, intensity, sigma=sigma, random_noise=False)
    return add_peak(chromato, peak, mu)

def add_peak_and_spectrum(chromato, chromato_cube, peak, mu):
    x_size=peak.shape[0]//2
    y_size=peak.shape[1]//2
    for i_in_peak, i in  enumerate(range(mu[0] - x_size, mu[0] + x_size + 1)):
        for j_in_peak, j in enumerate(range(mu[1] - y_size, mu[1] + y_size + 1)):
            chromato[i][j] = chromato[i][j] + peak[i_in_peak][j_in_peak]
            chromato_cube[i][j] = chromato_cube[i][j]
    return chromato


def create_random_mass_spectra(nb):
    intensities = np.zeros(nb)
    for i in range(random.randint(1, 4)):
        mean = random.randint(0, nb-1)
        s = np.random.poisson(mean, nb)
        for j in s:
            if (j > nb - 1):
                continue
            intensities[j] = intensities[j] + 1
    return intensities

def create_and_add_peak_spectrum(chromato_cube, size, intensity, mu, spectrum, sigma=1.0):
    peak=create_peak(size, 1, sigma=sigma, random_noise=False)
    perc = spectrum / np.sum(spectrum)
    x_size=peak.shape[0]//2
    y_size=peak.shape[1]//2
    area=0
    for i_in_peak, i in  enumerate(range(mu[0] - x_size, mu[0] + x_size + 1)):
        for j_in_peak, j in enumerate(range(mu[1] - y_size, mu[1] + y_size + 1)):
            if (i < 0 or j < 0 or i >= chromato_cube.shape[0] or j >= chromato_cube.shape[1]):
                continue
            chromato_cube[i][j] = chromato_cube[i][j] + perc * intensity * peak[i_in_peak][j_in_peak]
            area+=np.sum(perc * intensity * peak[i_in_peak][j_in_peak])
    return chromato_cube, area

def add_chromato_cube_gaussian_white_noise(chromato_cube, noise_typ="gauss"):
    return add_noise_3D(noise_typ,chromato_cube)

def delete_peak(chromato_cube, size, intensity, mu, spectrum, sigma=1.0):
    peak=create_peak(size, 1, sigma=sigma, random_noise=False)
    perc = spectrum / np.sum(spectrum)
    x_size=peak.shape[0]//2
    y_size=peak.shape[1]//2
    for i_in_peak, i in  enumerate(range(mu[0] - x_size, mu[0] + x_size + 1)):
        for j_in_peak, j in enumerate(range(mu[1] - y_size, mu[1] + y_size + 1)):
            chromato_cube[i][j] = chromato_cube[i][j] - perc * intensity * peak[i_in_peak][j_in_peak]
    return chromato_cube

def get_similar_spectrum(spectrum_index, array_scores, min_similarity=0.0, max_similariy=1.1):
    #similarity can be 1.000000000000002
    spectrum_sim_scores=array_scores[spectrum_index]
    idx=np.concatenate(np.argwhere((spectrum_sim_scores>=min_similarity) & (spectrum_sim_scores<=max_similariy)))
    if (len(idx) <= 0):
        return -1
    return idx[random.randint(0, len(idx) - 1)]

def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.

    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres."""

    vol = (math.pi / (12 * d) * (r1 + r2 - d)**2 *
           (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2))
    return vol / (4./3 * math.pi * min(r1, r2) ** 3)


def create_chromato_cube(chromato_shape, nb_peaks, mu_center=[], center_dist=3, size=80, min_similarity=0.0, max_similarity=1.1, lib_path="./lib_EIB.mgf", scores_path='./lib_scores.json'):
    logger=logging.getLogger('matchms')
    logger.setLevel('ERROR')
    lib_spectra=list(load_from_mgf(lib_path))[:100]
    scores=scores_from_json(scores_path)
    array_scores=scores.to_array()['CosineGreedy_score']
    MARGIN=size//2
    #std around center
    '''if isinstance(center_dist, tuple):
        center_y, center_x = center_dist
    else:
        center_x=random.randint(0, center_dist)
        center_y=center_dist-center_x'''
    #if center not provided
    if (not len(mu_center)):
        mu_center = [random.randint(0+MARGIN, chromato_shape[0]-MARGIN), random.randint(0+MARGIN, chromato_shape[1]-MARGIN)]
    #generate coordinates around center mu_center
    '''xs = np.around(np.random.normal(mu_center[0], center_sigma_x, nb_peaks)).astype(int)
    ys = np.around(np.random.normal(mu_center[1], center_sigma_y, nb_peaks)).astype(int)
    pts=np.stack((xs, ys), axis=-1)
    pts=np.unique(pts, axis=0)'''
    pts=[]
    for i in range(nb_peaks):
        center_x_dist=random.randint(-center_dist, center_dist)
        center_x=mu_center[0]-center_x_dist
        abs_dst=center_dist-np.abs(center_x_dist)
        center_y=mu_center[1]+abs_dst if random.randint(0, 1) else mu_center[1]-abs_dst
        pts.append([center_x, center_y])

    pts=np.unique(pts, axis=0)
    #peaks intensities
    intensities = []
    for i in range(len(pts)):
        intensities.append(random.uniform(10000, 40000))

    peaks_sigma=[]
    spectrum_index=random.randint(0, len(lib_spectra) - 1)
    spectrum_obj=lib_spectra[spectrum_index]
    spectra=[]
    spectra_id=[]
    created_points=[]
    #create spectra
    for i in range(len(pts)):
        #peak sigma
        peak_sigma_1=random.uniform(1, 5)
        peak_sigma_2=random.uniform(1, 5)
        #peak spectrum
        if (i > 0):
            #random spectrum with similarity > sim_score and < max_similarity
            spectrum_id=get_similar_spectrum(spectrum_index, array_scores, min_similarity=min_similarity, max_similariy=max_similarity)
            if (id != -1):
                spectrum_obj=lib_spectra[spectrum_id]
                print('spectrum shape: ', len(spectrum_obj.peaks.mz))
            else:
                continue
        spectra.append((spectrum_obj.peaks.mz, spectrum_obj.peaks.intensities))
        spectra_id.append(spectrum_obj.metadata['databaseid'])
        created_points.append([int(pts[i][0]), int(pts[i][1])])
    spectra=np.array(spectra)
    #create new chromato cube with spectra
        #create zeros 3D chromatogram
    tmp=np.concatenate(spectra[:,0])
    range_min=int(round(np.min(tmp)))
    range_max=int(round(np.max(tmp)))
    print(range_min, range_max)
    chromato_cube = np.zeros((chromato_shape[0], chromato_shape[1], range_max-range_min+1))
    print("cube shape: ", chromato_cube.shape)
        #compute nominal mv
    new_mv=np.linspace(range_min, range_max, range_max-range_min+1)
        #correc spectra dim
    tmp_spectra=[]
    for i, spectrum in enumerate(spectra):
        mass_values, int_values=spectrum
        new_iv=centroid_to_full_nominal((range_min, range_max), mass_values, int_values)
        #spectra[i]=(new_mv, new_iv)
        tmp_spectra.append((new_mv, new_iv))
        #add peaks to chromato cube
    tmp_spectra=np.array(tmp_spectra)
    spectra=tmp_spectra

    for i in range(len(created_points)):
        chromato_cube,_ = create_and_add_peak_spectrum(chromato_cube, size=size, intensity=intensities[i], mu=pts[i], spectrum=spectra[i][1], sigma=(peak_sigma_1, peak_sigma_2))
        peaks_sigma.append((peak_sigma_1, peak_sigma_2))

    chromato_TIC=np.sum(chromato_cube, -1)
    chromato_cube = np.moveaxis(chromato_cube, -1, 0)
    params=dict()
    params['peaks_mu']=created_points
    params['peaks_sigma']=peaks_sigma
    params['spectra_id']=spectra_id
    params['peaks_center']=mu_center
    params['center_dist']=center_dist
    params['range']=(range_min, range_max)
    return chromato_TIC, chromato_cube, params, spectra

'''
def simulation_from_cdf_model(center_dist, min_similarity=0.0, max_similarity=1.1, model_filename='./data/ELO_CDF/model.cdf', nb_chromato=1, mod_time=1.25, new_cdf_path='./data/ELO_CDF/', cdf_name="new_cdf"):
    # load model cdf chromatogram
    ds = nc.Dataset(model_filename, "r")
    chromato = ds['total_intensity']
    sam_rate = 1 / ds['scan_duration'][0]
    l1 = math.floor(sam_rate * mod_time)
    l2 = math.floor(len(chromato) / l1)
    chromato = np.reshape(chromato[:l1*l2], (l2,l1))

    # create new cdf chromatogram
    new_cdf_path+=cdf_name
    if os.path.exists(new_cdf_path+'.cdf'):
        os.remove(new_cdf_path+'.cdf')
    new_cdf = nc.Dataset(new_cdf_path+'.cdf', "w")
    new_cdf.setncatts(ds.__dict__)

    error=[]
    nbs_peak=[]
    peaks_mu=[]
    peaks_sigma=[]

    for i in range(nb_chromato):
        #random peak number in [1,3]
        nb_peak=random.randint(1,3)
        nbs_peak.append(nb_peak)
        #random peak std in [1,7]
        peak_sigma=random.uniform(1, 7)
        peaks_sigma.append(peak_sigma)
        #create new chromato cube with random spectra
        new_chromato_TIC, new_chromato_cube, params=create_chromato_cube(chromato.shape, nb_peak, center_dist=center_dist, min_similarity=min_similarity, max_similarity=max_similarity, size=80)
        print("new_chromato_cube shape: ", new_chromato_cube.shape)
        #retrieve mv and iv from chromato cube and delete useless values (0)
        (range_min, range_max)=params['range']
        mv=np.tile(np.linspace(range_min, range_max, range_max-range_min+1).astype(int), l1*l2)
        iv=np.moveaxis(new_chromato_cube, 0, -1).flatten()
        id_to_delete=np.argwhere((iv==0) & (mv != range_min) & (mv != range_max)).flatten()
        mv=np.delete(mv, id_to_delete)
        iv=np.delete(iv, id_to_delete)
    
        # copy dimensions from model and change point_number dim
        for name, dimension in ds.dimensions.items():
            if (name=='point_number'):
                new_cdf.createDimension(
                    name,len(mv))
            else:
                new_cdf.createDimension(
                name, (len(dimension)))
        # create variable from model except for mass_values, intensity_values, total_intensity, mass_range_min and mass_range_max
        for name, variable in ds.variables.items():
            if (name=='mass_values'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill mv values
                new_cdf[name][:] = mv
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            elif (name=='intensity_values'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill iv values
                new_cdf[name][:] = iv
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            elif (name == 'total_intensity'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill TIC values
                new_cdf[name][:l1*l2] = new_chromato_TIC.flatten()
                new_cdf[name][l1*l2:] = 0
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            elif (name == 'mass_range_min'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill mass_range_min values
                new_cdf[name][:l1*l2]=[range_min]*(l1*l2)
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            elif (name == 'mass_range_max'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill mass_range_max values
                new_cdf[name][:l1*l2]=[range_max]*(l1*l2)
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)

            else:
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill from model
                new_cdf[name][:] = ds[name][:]
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            # write chromatogram characteristics
            d=json.dumps(params, indent=4)
            with open(new_cdf_path + '.json', "w") as f:
                f.write(d)
    new_cdf.close()
'''

def simulation_from_cdf_model(lib_path="./lib_EIB.mgf", scores_path='./lib_scores.json', min_similarity=0.0, max_similarity=1.1, min_overlap=0.7, max_overlap=0.99, intensity_range_min=20000, intensity_range_max=40000, model_filename='./data/ELO_CDF/model.cdf', nb_chromato=1, mod_time=1.25, new_cdf_path='./data/ELO_CDF/', cdf_name="new_cdf", noise_loc=1000.0, noise_scale=500.0, poisson_rep=0.9):
    r"""Creates a new chromatogram with overlapped clusters based on a model chromatogram and saves it as cdf file.
    Parameters
    ----------
    lib_path : optional
        Path to the library used to select random spectra.
    scores_path : optional
        Pairwise similarity between spectra within the library.
    min_similarity : optional
        Minimum spectra similarity between spectra within clusters.
    max_similarity : optional
        Maximuum spectra similarity between spectra within clusters.
    min_overlap : optional
        Minimum overlap between spectra within clusters.
    max_overlap : optional
        Maximum overlap between spectra within clusters.
    intensity_range_min : optional
        Minimum intensity for a peak.
    intensity_range_max : optional
        Maximum intensity for a peak.
    model_filename : optional
        Filename of the chromatogram used as a model.
    nb_chromato : optional
        Number of chromatogram to generate.
    mod_time : optional
        Modulation time
    new_cdf_path : optional
        Path of the new cdf file.
    cdf_name : optional
        Name of the new cdf file.
    noise_loc : optional
        Mean of the custom_gauss noise.
    noise_scale : optional
        Standard deviation of the custom_gauss noise.
    poisson_rep : optional
        Poisson parameter of the custom_gauss noise.
    -------
    Returns
    -------
    A: tuple
        Return the created chromato object wrapping (chromato TIC), (time_rn).
    --------
    Examples
    --------
    >>> import read_chroma
    >>> chromato = read_chroma.read_only_chroma(filename, mod_time)
    """
    # load model cdf chromatogram
    ds = nc.Dataset(model_filename, "r")
    chromato = ds['total_intensity']
    sam_rate = 1 / ds['scan_duration'][0]
    l1 = math.floor(sam_rate * mod_time)
    l2 = math.floor(len(chromato) / l1)
    chromato = np.reshape(chromato[:l1*l2], (l2,l1))
    ID_MAX=100
    # disable matchms logger
    logger=logging.getLogger('matchms')
    logger.setLevel('ERROR')
    lib_spectra=list(load_from_mgf(lib_path))[:]
    scores=scores_from_json(scores_path)
    array_scores=scores.to_array()['CosineGreedy_score']

    # create new cdf chromatogram
    new_cdf_path+=cdf_name
    if os.path.exists(new_cdf_path+'.cdf'):
        os.remove(new_cdf_path+'.cdf')
    new_cdf = nc.Dataset(new_cdf_path+'.cdf', "w")
    new_cdf.setncatts(ds.__dict__)

    error=[]
    nbs_peak=[]
    peaks_mu=[]
    peaks_sigma=[]

    for i in range(nb_chromato):
        #random peak number in [1,3]
        nb_peak=random.randint(1,3)
        nbs_peak.append(nb_peak)
        #random peak std in [1,7]
        peak_sigma=random.uniform(1, 7)
        peaks_sigma.append(peak_sigma)
        #create new chromato cube with random spectra
        #new_chromato_TIC, new_chromato_cube, params=create_chromato_cube(chromato.shape, nb_peak, center_dist=center_dist, min_similarity=min_similarity, max_similarity=max_similarity, size=80)
        new_chromato_TIC, new_chromato_cube, params=create_chromato_cube_with_overlapped_cluster(chromato.shape, lib_spectra, array_scores,min_similarity=min_similarity, max_similarity=max_similarity,  min_overlap=min_overlap, max_overlap=max_overlap, intensity_range_min=intensity_range_min, intensity_range_max=intensity_range_max, noise_loc=noise_loc, noise_scale=noise_scale, poisson_rep=poisson_rep)
        print("new_chromato_cube shape: ", new_chromato_cube.shape)
        #retrieve mv and iv from chromato cube and delete useless values (0)
        (range_min, range_max)=params['range']
        mv=np.tile(np.linspace(range_min, range_max, range_max-range_min+1).astype(int), l1*l2)
        iv=np.moveaxis(new_chromato_cube, 0, -1).flatten()
        id_to_delete=np.argwhere((iv==0) & (mv != range_min) & (mv != range_max)).flatten()
        mv=np.delete(mv, id_to_delete)
        iv=np.delete(iv, id_to_delete)
    
        # copy dimensions from model and change point_number dim
        for name, dimension in ds.dimensions.items():
            if (name=='point_number'):
                new_cdf.createDimension(
                    name,len(mv))
            else:
                new_cdf.createDimension(
                name, (len(dimension)))
        # create variable from model except for mass_values, intensity_values, total_intensity, mass_range_min and mass_range_max
        for name, variable in ds.variables.items():
            if (name=='mass_values'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill mv values
                new_cdf[name][:] = mv
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            elif (name=='intensity_values'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill iv values
                new_cdf[name][:] = iv
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            elif (name == 'total_intensity'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill TIC values
                new_cdf[name][:l1*l2] = new_chromato_TIC.flatten()
                new_cdf[name][l1*l2:] = 0
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            elif (name == 'mass_range_min'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill mass_range_min values
                new_cdf[name][:l1*l2]=[range_min]*(l1*l2)
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            elif (name == 'mass_range_max'):
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill mass_range_max values
                new_cdf[name][:l1*l2]=[range_max]*(l1*l2)
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)

            else:
                x = new_cdf.createVariable(name, variable.datatype, variable.dimensions)
                # fill from model
                new_cdf[name][:] = ds[name][:]
                # copy variable attributes all at once via dictionary
                new_cdf[name].setncatts(ds[name].__dict__)
            # write chromatogram characteristics
            d=json.dumps(params, indent=4)
            with open(new_cdf_path + '.json', "w") as f:
                f.write(d)
    new_cdf.close()
    return params
 
def _compute_disk_overlap(d, r1, r2):
    """
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.

    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
            0.5 * math.sqrt(abs(a * b * c * d)))
    return area / (math.pi * (min(r1, r2) ** 2))


def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.

    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres.

    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (math.pi / (12 * d) * (r1 + r2 - d)**2 *
           (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2))
    return vol / (4./3 * math.pi * min(r1, r2) ** 3)


def _blob_overlap(blob1, blob2, *, sigma_dim=1):
    """Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area. Note that 0.0
    is *always* returned for dimension greater than 3.

    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    sigma_dim : int, optional
        The dimensionality of the sigma value. Can be 1 or the same as the
        dimensionality of the blob space (2 or 3).

    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    blob1=np.array(blob1)
    blob2=np.array(blob2)
    ndim = len(blob1) - sigma_dim
    if ndim > 3:
        return 0.0
    root_ndim = math.sqrt(ndim)

    # we divide coordinates by sigma * sqrt(ndim) to rescale space to isotropy,
    # giving spheres of radius = 1 or < 1.
    if blob1[-1] == blob2[-1] == 0:
        return 0.0
    elif blob1[-1] > blob2[-1]:
        max_sigma = blob1[-sigma_dim:]
        r1 = 1
        r2 = blob2[-1] / blob1[-1]
    else:
        max_sigma = blob2[-sigma_dim:]
        r2 = 1
        r1 = blob1[-1] / blob2[-1]

    pos1 = blob1[:ndim] / (max_sigma * root_ndim)
    pos2 = blob2[:ndim] / (max_sigma * root_ndim)

    d = np.sqrt(np.sum((pos2 - pos1)**2))
    if d > r1 + r2:  # centers farther than sum of radii, so no overlap
        return 0.0

    # one blob is inside the other
    if d <= abs(r1 - r2):
        return 1.0

    if ndim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:  # ndim=3 http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return _compute_sphere_overlap(d, r1, r2)

def create_random_blob(mu_x, mu_y, mu_x_range=0, mu_y_range=0, sigma_x_range_min=0.5, sigma_x_range_max=5, sigma_y_range_min=0.5, sigma_y_range_max=5):
    r"""Create random blob.

    Parameters
    ----------
    mu_x :
        Coordinate in the first dimension.
    mu_y : optional
        Coordinate in the second dimension.
    -------
    Returns
    -------
    A: list
        Created blob as a list of x coordinate, y coordinate, x radius and y radius.
    --------
    """
    random_mu_x=int(random.randint(mu_x-mu_x_range, mu_x+mu_x_range))
    random_mu_y=int(random.randint(mu_y-mu_y_range, mu_y+mu_y_range))
    sigma_x=np.random.uniform(sigma_x_range_min, sigma_x_range_max)
    sigma_y=np.random.uniform(sigma_y_range_min, sigma_y_range_max)
    return [random_mu_x, random_mu_y, RADIUS_FACTOR*sigma_x, RADIUS_FACTOR*sigma_y]


def create_overlapped_cluster(chrom, mu0, sigma_x_range_min=2, sigma_x_range_max=5, sigma_y_range_min=2, sigma_y_range_max=5, size=40, min_overlap=0.7, max_overlap=0.99):
    blob_list=[]
    overlaps=[]
    blob1=create_random_blob(mu0[0], mu0[1])
    blob_list.append(blob1)
    chrom=create_and_add_peak(chrom, size, np.random.uniform(20000, 40000), [int(blob1[0]), int(blob1[1])], (blob1[2], blob1[3]))
    min_overlap=min_overlap
    max_overlap=max_overlap
    for i in range (random.randint(1,2)):
        it=0
        it_max=10000
        while (it<=it_max):
            tmp_blob=create_random_blob(mu0[0], mu0[1], mu_x_range=20, mu_y_range=20)
            tmp_min_overlap=1.0
            tmp_max_overlap=0.
            for j in range(len(blob_list)):
                tmp_overlap=_blob_overlap(blob_list[j], tmp_blob, sigma_dim=2)
                tmp_min_overlap=min(tmp_min_overlap, tmp_overlap)
                tmp_max_overlap=max(tmp_max_overlap, tmp_overlap)
            if ((tmp_min_overlap >= min_overlap and tmp_max_overlap <= max_overlap)):
                chrom=create_and_add_peak(chrom, size, np.random.uniform(20000, 40000), [int(tmp_blob[0]), int(tmp_blob[1])], (tmp_blob[2], tmp_blob[3]))
                blob_list.append(tmp_blob)
                overlaps.append((tmp_min_overlap, tmp_max_overlap))
                break
            it+=1
    blob_list=np.array(blob_list)
    return blob_list

def create_chromato_with_overlapped_cluster(shape, min_overlap=0.7, max_overlap=0.99, sigma_x_range_min=2, sigma_x_range_max=5, sigma_y_range_min=2, sigma_y_range_max=5):
    size=40
    chrom=np.ones((shape))
    chromato_clusters=[]
    rg1=range(size, shape[0] - size, 4 * sigma_x_range_max)
    rg2=range(size, shape[1] - size, 4 * sigma_y_range_max)
    nb1=len(rg1)
    nb2=len(rg2)
    for it1, i in enumerate(rg1):
        for it2, j in enumerate(rg2):
            print(str(it1 + 1) + "/" + str(nb1) + " " + str(it2 + 1) + "/" + str(nb2))
            cluster=create_overlapped_cluster(chrom, mu0=[i, j])
            chromato_clusters.append(cluster)
    return chrom, chromato_clusters


def create_overlapped_cluster_in_cube(mu0, min_overlap=0.7, max_overlap=0.99):
    r"""Create overlapped cluster.

    Parameters
    ----------
    mu0 :
        Chromatogram full filename.
    min_overlap : optional
        Minimum overlap within clusters.
    max_overlap : optional
        Maximum overlap within clusters.
    -------
    Returns
    -------
    blob_list:
        List of the created blobs.
    A: tuple
        Minimum and maximum overlap within the created cluster
    --------
    """
    IT_MAX=10000
    blob_list=[]
    #overlaps=[]
    blob1=create_random_blob(mu0[0], mu0[1])
    blob_list.append(blob1)
    cluster_min_overlap=1.0
    cluster_max_overlap=0.
    for i in range (random.randint(1,2)):
        it=0
        while (it<=IT_MAX):
            tmp_blob=create_random_blob(mu0[0], mu0[1], mu_x_range=20, mu_y_range=20)
            tmp_min_overlap=1.0
            tmp_max_overlap=0.
            for j in range(len(blob_list)):
                tmp_overlap=_blob_overlap(blob_list[j], tmp_blob, sigma_dim=2)
                tmp_min_overlap=min(tmp_min_overlap, tmp_overlap)
                tmp_max_overlap=max(tmp_max_overlap, tmp_overlap)
            #if ((tmp_min_overlap >= min_overlap and tmp_max_overlap <= max_overlap)):
            #au moins une valeur supérieur à min_overlap mais pas besoin que ce soit le cas pour tous et au pire en dessous du max_overlap
            if ((tmp_max_overlap >= min_overlap and tmp_max_overlap <= max_overlap)):
                blob_list.append(tmp_blob)
                cluster_min_overlap=min(tmp_min_overlap, cluster_min_overlap)
                cluster_max_overlap=max(tmp_max_overlap, cluster_max_overlap)
                #overlaps.append((tmp_min_overlap, tmp_max_overlap))
                break
            it+=1
    blob_list=(blob_list)
    return blob_list, (cluster_min_overlap, cluster_max_overlap)

def create_chromato_cube_with_overlapped_cluster(shape, lib_spectra, array_scores, min_overlap=0.7, max_overlap=0.99,  min_similarity=0.0, max_similarity=1.1, add_noise=True, intensity_range_min=20000, intensity_range_max=40000, noise_loc=1000.0, noise_scale=500.0, poisson_rep=0.9):
    r"""Create a 3D chromatogram with overlapped clusters

    Parameters
    ----------
    shape :
        Dimensions of the created 3D chromatogram.
    lib_spectra :
        Library used to select random spectra.
    array_scores :
        Pairwise similarity of spectra in lib_spectra 
    min_overlap : optional
        Minimum overlap within clusters.
    max_overlap : optional
        Maximum overlap within clusters.
    min_similarity : optional
        Minimum spectra similarity between spectra within clusters.
    max_similarity : optional
        Maximum spectra similarity between spectra within clusters.
    add_noise : optional
        If noise needs to be added to the created 3D chromatogram.
    intensity_range_min : optional
        Minimum intensity for a peak.
    intensity_range_max : optional
        Maximum intensity for a peak.
    noise_loc : optional
        Mean of the custom_gauss noise.
    noise_scale : optional
        Standard deviation of the custom_gauss noise.
    poisson_rep : optional
        Poisson parameter of the custom_gauss noise.
    -------
    Returns
    -------
    chromato_TIC:
        Created TIC chromatogram.
    chromato_cube:
        Created 3D chromatogram.
    params:
        Created peaks parameters (specturm, location, intensity...).
    --------
    """
    size=20

    # peaks shapes range
    sigma_x_range_min=2
    sigma_x_range_max=5
    sigma_y_range_min=2
    sigma_y_range_max=5

    # create clusters blobs array of array of blobs [[mu0, mu1, sigma0, sigma1], ... [[mu0, mu1, sigma0, sigma1]]]
    chromato_clusters=[]
    chromato_clusters_overlap=[]
    rg1=range(size, shape[0] - size, 32 * sigma_x_range_max)
    rg2=range(size, shape[1] - size, 8 * sigma_y_range_max)
    nb1=len(rg1)
    nb2=len(rg2)

    for it1, i in enumerate(rg1):
        for it2, j in enumerate(rg2):
            print(str(it1 + 1) + "/" + str(nb1) + " " + str(it2 + 1) + "/" + str(nb2))
            cluster, (cluster_min_overlap, cluster_max_overlap)=create_overlapped_cluster_in_cube(mu0=[i, j], min_overlap=min_overlap, max_overlap=max_overlap)
            chromato_clusters.append(cluster)
            chromato_clusters_overlap.append(chromato_clusters_overlap)

    # create spectra for each peaks in each clusters
    clusters_created_points=[]
    clusters_created_points_spectra=[]
    clusters_created_points_spectra_id=[]
    clusters_created_points_spectra_name=[]

    range_min=float('inf')
    range_max=float('-inf')
    for cluster in chromato_clusters:
        # create ref spectra in cluster
        spectrum_index=random.randint(0, len(lib_spectra) - 1)
        spectrum_obj=lib_spectra[spectrum_index]
        while (np.max(spectrum_obj.peaks.mz) > 500):
            spectrum_index=random.randint(0, len(lib_spectra) - 1)
            spectrum_obj=lib_spectra[spectrum_index]


        spectra=[]
        spectra_id=[]
        spectra_name=[]
        created_points=[]

        for i, point in enumerate(cluster):
            # if its the first point we do not need to create new spectrum
            if (i > 0):
                #random spectrum with similarity > sim_score and < max_similarity
                spectrum_id=get_similar_spectrum(spectrum_index, array_scores, min_similarity=min_similarity, max_similariy=max_similarity)
                if (spectrum_id != -1 and np.max(lib_spectra[spectrum_id].peaks.mz) <= 500):
                    spectrum_obj=lib_spectra[spectrum_id]
                else:
                    continue
            range_min=min(range_min, (np.min(spectrum_obj.peaks.mz)))
            range_max=max(range_max, (np.max(spectrum_obj.peaks.mz)))
            spectra.append((spectrum_obj.peaks.mz, spectrum_obj.peaks.intensities))
            spectra_id.append(spectrum_obj.metadata['databaseid'])
            spectra_name.append(spectrum_obj.metadata['compound_name'])
            created_points.append(point)

        clusters_created_points.append(created_points)
        clusters_created_points_spectra.append(np.array(spectra))
        clusters_created_points_spectra_id.append(spectra_id)
        clusters_created_points_spectra_name.append(spectra_name)
    
    #clusters_created_points_spectra=np.array(clusters_created_points_spectra)
    # compute chromato cube range min and range max

    range_min=int(round(range_min))
    range_max=int(round(range_max))
    print(range_min, range_max)

    #create new chromato cube
    """
        FAUT PAS METTRE NP ONES ICI SINON LE CHROMATO CUBE VA ETRE TROP GROS
    """
    #chromato_cube = np.ones((shape[0], shape[1], range_max-range_min+1))
    chromato_cube = np.zeros((shape[0], shape[1], range_max-range_min+1))

    # create mn mass values
    new_mv=np.linspace(range_min, range_max, range_max-range_min+1)

    tmp_spectra=[]
    for i, created_points_spectra in enumerate(clusters_created_points_spectra):
        tmp_cluster_spectra=[]
        for j, spectrum in enumerate(created_points_spectra):
            mass_values, int_values=spectrum
            new_iv=centroid_to_full_nominal((range_min, range_max), mass_values, int_values)
            tmp_cluster_spectra.append((new_mv, new_iv))
        tmp_spectra.append(tmp_cluster_spectra)

    # add points and spectra in chromato cube
    clusters_created_points_intensities=[]
    clusters_created_points_area=[]
    for i in range(len(clusters_created_points)):
        cluster_created_points_intensities=[]
        cluster_created_points_area=[]
        for j in range(len(clusters_created_points[i])):
            mu=[int(clusters_created_points[i][j][0]), int(clusters_created_points[i][j][1])]
            spectrum=tmp_spectra[i][j][1]
            sigma=(clusters_created_points[i][j][2], clusters_created_points[i][j][3])
            intensity=np.random.uniform(intensity_range_min, intensity_range_max)
            chromato_cube, area = create_and_add_peak_spectrum(chromato_cube, size=size, intensity=intensity, mu=mu, spectrum=spectrum, sigma=(sigma[0] / RADIUS_FACTOR, sigma[1] / RADIUS_FACTOR))
            cluster_created_points_intensities.append(intensity)
            cluster_created_points_area.append(area)
        clusters_created_points_intensities.append(cluster_created_points_intensities)
        clusters_created_points_area.append(cluster_created_points_area)

    if (add_noise):
        chromato_cube=add_noise_3D(chromato_cube, noise_loc=noise_loc, noise_scale=noise_scale, poisson_rep=poisson_rep)
    chromato_TIC=np.sum(chromato_cube, -1)
    chromato_cube = np.moveaxis(chromato_cube, -1, 0)
    params=dict()

    params['peaks_mu']=clusters_created_points
    params['peaks_int']=clusters_created_points_intensities
    params['peaks_area']=clusters_created_points_area
    params['spectra_id']=clusters_created_points_spectra_id
    params['spectra_name']=clusters_created_points_spectra_name
    params['range']=(range_min, range_max)
    return chromato_TIC, chromato_cube, params
