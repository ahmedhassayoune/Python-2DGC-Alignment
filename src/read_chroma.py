import netCDF4 as nc
import numpy as np
import math
import time
import mass_spec
from projection import matrix_to_chromato, chromato_to_matrix
from scipy.signal import argrelextrema
import baseline_correction
from skimage.restoration import estimate_sigma
import os

def print_chroma_header(filename):
    r"""Print chromato header.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    ----------
    Returns
    -------
    Examples
    --------
    >>> import read_chroma
    >>> read_chroma.print_chroma_header(filename)
    """
    ds = nc.Dataset(filename)
    print(ds)
    range_min = math.ceil(ds["mass_range_min"][0])
    range_max = math.floor(ds["mass_range_max"][0])
    print(range_min)
    print(range_max)

def read_only_chroma(filename, mod_time = 1.25):
    r"""Read chromatogram file.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    mod_time : optional
        Modulation time
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
    ds = nc.Dataset(filename)
    chromato = ds['total_intensity']
    sam_rate = 1 / ds['scan_duration'][0]
    l1 = math.floor(sam_rate * mod_time)
    l2 = math.floor(len(chromato) / l1)
    chromato = np.reshape(chromato[:l1*l2], (l2,l1))

    return chromato, (ds['scan_acquisition_time'][0] / 60, ds['scan_acquisition_time'][-1] / 60)

def read_chroma(filename, mod_time = 1.25, max_val = None):
    r"""Read chromatogram file.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    mod_time : optional
        Modulation time
    -------
    Returns
    -------
    A: tuple
        Return the created chromato object wrapping (chromato TIC), (time_rn), (spectra and mass range min and max).
    -------
    Examples
    --------
    >>> import read_chroma
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    """
    ds = nc.Dataset(filename)
    chromato = ds['total_intensity']
    Timepara = ds["scan_acquisition_time"][np.abs(ds["point_count"]) < np.iinfo(np.int32).max]
    sam_rate = 1 / np.mean(Timepara[1:] - Timepara[:-1])
    l1 = math.floor(sam_rate * mod_time)
    l2 = math.floor(len(chromato) / l1)

    if (max_val):
        mv = ds["mass_values"][:max_val]
        iv = ds["intensity_values"][:max_val]
    else:
        mv = ds["mass_values"][:]
        iv = ds["intensity_values"][:]

    range_min = math.ceil(ds["mass_range_min"][:].min())
    range_max = math.floor(ds["mass_range_max"][:].max())

    chromato = np.reshape(chromato[:l1*l2], (l2,l1))
    return chromato, (ds['scan_acquisition_time'][0] / 60, ds['scan_acquisition_time'][-1] / 60), (l1, l2, mv, iv, range_min, range_max)

def read_chromato_cube(filename, mod_time=1.25, pre_process=True):
    r"""Read chromatogram file and compute TIC chromatogram, 3D chromatogram and noise std.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    mod_time : optional
        Modulation time
    pre_process : optional
        If pre_process is True the background of the TIC chromatogram and the background of the 3D chromatogram are corrected.
    -------
    Returns
    -------
    A: tuple
        Return the created chromato object wrapping (chromato TIC), (time_rn), (spectra_obj).
    chromato_cube:
        3D chromatogram.
    sigma:
        Gaussian white noise std.
    -------
    Examples
    --------
    >>> import read_chroma
    >>> chromato_obj, chromato_cube, sigma=read_chroma.read_chromato_cube(filename, mod_time=1.25, pre_process=True)
    """
    start_time=time.time()
    chromato_obj = read_chroma(filename, mod_time)
    chromato,time_rn,spectra_obj = chromato_obj
    print("chromato read", time.time()-start_time)
    start_time=time.time()
    full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    print("full spectra computed", time.time()-start_time)

    spectra, debuts, fins = full_spectra
    chromato_cube = full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    
    # baseline correction
    if(pre_process):
        chromato = baseline_correction.chromato_no_baseline(chromato)
        chromato_cube = np.array(baseline_correction.chromato_cube_corrected_baseline(chromato_cube))
    
    sigma = estimate_sigma(chromato, channel_axis=None)
    return (chromato,time_rn,spectra_obj), chromato_cube, sigma

def read_chromato_and_chromato_cube(filename, mod_time=1.25, pre_process=True):
    r"""Same as read_chromato_cube but do not returns full spectra_obj (only range_min and range_max) because of RAM issue.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    mod_time : optional
        Modulation time
    pre_process : optional
        If pre_process is True the background of the TIC chromatogram and the background of the 3D chromatogram are corrected.
    -------
    Returns
    -------
    chromato:
        hromato TIC
    time_rn : 
        Time range
    chromato_cube:
        3D chromatogram.
    sigma:
        Gaussian white noise std.
    A: 
    range_min and range_max
    -------
    Examples
    --------
    >>> import read_chroma
    >>> chromato, time_rn, chromato_cube, sigma, (range_min, range_max)=read_chroma.read_chromato_and_chromato_cube(filename, mod_time=1.25, pre_process=True)
    """
    start_time=time.time()
    chromato_obj = read_chroma(filename, mod_time)
    chromato,time_rn,spectra_obj = chromato_obj
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    print("chromato read", time.time()-start_time, 's')
    start_time=time.time()
    full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    print("full spectra computed", time.time()-start_time, 's')

    spectra, debuts, fins = full_spectra
    chromato_cube = full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    
    # baseline correction
    if(pre_process):
        chromato = baseline_correction.chromato_no_baseline(chromato)
        chromato_cube = np.array(baseline_correction.chromato_cube_corrected_baseline(chromato_cube))
        print("baseline corrected")
    sigma = estimate_sigma(chromato, channel_axis=None)
    return chromato, time_rn, chromato_cube, sigma, (range_min, range_max)

def centroided_to_mass_nominal_chromatogram(filename, cdf_name, mod_time = 1.25):
    r"""Read centroided chromatogram, convert and save it as nominal mass chromatogramm cdf file.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    cdf_name :
        Name of the new cdf file
    mod_time : optional
        Modulation time
    """
    # load centroided chromatogram
    ds = nc.Dataset(filename)
    chromato = ds['total_intensity']
    sam_rate = 1 / ds['scan_duration'][0]
    l1 = math.floor(sam_rate * mod_time)
    l2 = math.floor(len(chromato) / l1)
    mv = ds["mass_values"][:]
    iv = ds["intensity_values"][:]
    range_min = math.ceil(ds["mass_range_min"][0])
    range_max = math.floor(ds["mass_range_max"][1])
    # compute centroided chromaogram
    chromato = np.reshape(chromato[:l1*l2], (l2,l1))
    # compute mass nominal chromato cube from centroided
    full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=(l1, l2, mv, iv, range_min, range_max))
    chromato_cube = full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=(l1, l2, mv, iv, range_min, range_max))
    # compute chromatogram from chromato cube
    new_chromato_TIC=np.sum(chromato_cube, -1)
    ## write new chromatogram as cdf file
    # compute mass values and intensities
    mv=np.tile(np.linspace(range_min, range_max, range_max-range_min+1).astype(int), l1*l2)
    iv=np.moveaxis(chromato_cube, 0, -1).flatten()
    # delete 0 in mv and iv
    id_to_delete=np.argwhere((iv==0) & (mv != range_min) & (mv != range_max)).flatten()
    mv=np.delete(mv, id_to_delete)
    iv=np.delete(iv, id_to_delete)

    # create new cdf chromatogram
    new_cdf_path+=cdf_name
    if os.path.exists(new_cdf_path+'.cdf'):
        os.remove(new_cdf_path+'.cdf')
    new_cdf = nc.Dataset(new_cdf_path+'.cdf', "w")
    new_cdf.setncatts(ds.__dict__)
    
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
    new_cdf.close()


def full_spectra_to_chromato_cube(full_spectra, spectra_obj, mass_range_min=None, mass_range_max=None):
    r"""Compute 3D chromatogram from mass spectra. Then it is possible to read specific mass spectrum from this 3D chromatogram or detect peaks in 3D.

    Parameters
    ----------
    full_spectra :
        Tuple wrapping spectra, debuts and fins.
    spectra_obj :
        Tuple wrapping l1, l2, mv, iv, range_min and range_max
    -------
    Returns
    -------
    A: ndarray
        Return the created 3D chromatogram. An array containing all mass chromatogram.
    -------
    Examples
    --------
    >>> import read_chroma
    >>> chromato_obj = read_chroma.read_chroma(file, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    """
    spectra, debuts, fins = full_spectra
    l1, l2, mv, iv, range_min, range_max = spectra_obj

    if (not mass_range_min):
        mass_range_min = range_min
    if (not mass_range_max):
        mass_range_max = range_max
    chromato_mass_list = []

    for tm in range(mass_range_min - range_min, mass_range_max - range_min + 1):

        chromato_mass = spectra[:,1,tm]
        chromato_mass_tm = np.reshape(chromato_mass[:l1*l2], (l2,l1))
        chromato_mass_list.append(chromato_mass_tm)
    return np.array(chromato_mass_list)


def full_spectra_to_chromato_cube_from_centroid(full_spectra, spectra_obj, mass_range_min=None, mass_range_max=None):
    r"""Compute 3D chromatogram from centroided mass spectra to handle centroided details.

    Parameters
    ----------
    full_spectra :
        Tuple wrapping spectra, debuts and fins.
    spectra_obj :
        Tuple wrapping l1, l2, mv, iv, range_min and range_max
    -------
    Returns
    -------
    A: ndarray
        Return the created 3D chromatogram. An array containing all mass chromatogram for each unique centroided mass.
    -------
    Examples
    --------
    >>> import read_chroma
    >>> chromato_obj = read_chroma.read_chroma(file, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra_full_centroid = mass_spec.read_full_spectra_full_centroid(spectra_obj=spectra_obj)
    >>> chromato_cube_full_centroid = read_chroma.full_spectra_to_chromato_cube_from_centroid(full_spectra=full_spectra_full_centroid, spectra_obj=spectra_obj)
    """
    spectra, debuts, fins = full_spectra
    l1, l2, mv, iv, range_min, range_max = spectra_obj

    if (not mass_range_min):
        mass_range_min = range_min
    if (not mass_range_max):
        mass_range_max = range_max
    chromato_mass_list = []
    for i, tm in enumerate(spectra[0][0]):
        chromato_mass = spectra[:,1,i]
        chromato_mass_tm = np.reshape(chromato_mass[:l1*l2], (l2,l1))
        chromato_mass_list.append(chromato_mass_tm)
    return np.array(chromato_mass_list)

def full_spectra_to_chromato_cube_centroid(full_spectra, spectra_obj, mass_range_min=None, mass_range_max=None):
    spectra, debuts, fins = full_spectra
    l1, l2, mv, iv, range_min, range_max = spectra_obj

    spectra_full_nom = []
    lg = len(spectra)
    for i, spectrum in enumerate(spectra):
        spectra_full_nom.append(mass_spec.centroid_to_full_nominal(spectra_obj, spectrum[0], spectrum[1]))
        print(str(i) + "/" + str(lg))
    spectra_full_nom = np.array(spectra_full_nom)
    if (not mass_range_min):
        mass_range_min = range_min
    if (not mass_range_max):
        mass_range_max = range_max
    chromato_mass_list = []
    for tm in range(mass_range_min - range_min, mass_range_max - range_min + 1):
        chromato_mass = spectra_full_nom[:,1,tm]
        chromato_mass_tm = np.reshape(chromato_mass[:l1*l2], (l2,l1))
        chromato_mass_list.append(chromato_mass_tm)
    return np.array(chromato_mass_list)

def read_chromato_mass(spectra_obj, tm, debuts = None):
    start_time = time.time()
    l1, l2, mv, iv, range_min, range_max = spectra_obj
    if (debuts is None):
        debuts = np.where(mv == range_min)
    offset = tm - range_min
    debuts_tm = debuts[0] + offset
    chromato_mass = iv[debuts_tm]
    chromato_mass = np.reshape(chromato_mass[:l1*l2], (l2,l1))
    print("--- %s seconds ---" % (time.time() - start_time), "to compute chromato slice " + str(tm))

    return chromato_mass, debuts

def read_chroma_spectra_loc(filename, mod_time = 1.25):
    r"""Read chromatogram file.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    mod_time : optional
        Modulation time
    -------
    Returns
    -------
    A: tuple
        Return the created chromato object wrapping (chromato TIC), (time_rn), (spectra and mass range min and max).
    -------
    Examples
    --------
    >>> import read_chroma
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    """
    ds = nc.Dataset(filename)
    chromato = ds['total_intensity']
    sam_rate = 1 / ds['scan_duration'][0]
    l1 = math.floor(sam_rate * mod_time)
    l2 = math.floor(len(chromato) / l1)

    mv = ds["mass_values"][:]
    iv = ds["intensity_values"][:]

    range_min = math.ceil(ds["mass_range_min"][0])
    range_max = math.floor(ds["mass_range_max"][1])

    chromato = np.reshape(chromato[:l1*l2], (l2,l1))

    minima = argrelextrema(mv, np.less)[0]
    fins = minima - 1
    fins = np.append(fins, len(mv - 1))
    debuts = np.insert(minima, 0, 0)
    
    return (debuts, fins), (range_min, range_max)        
    
def chromato_cube(spectra_obj, mass_range_min=None, mass_range_max=None, debuts=None):
    #chromato_mass_list = [read_chromato_mass(spectra_obj,tm,debuts) for tm in range(mass_range_min, mass_range_max)]
    start_time = time.time()
    chromato_mass_list = []
    for tm in range(mass_range_min, mass_range_max):
        chromato_mass, ind = read_chromato_mass(spectra_obj,tm,debuts)
        chromato_mass_list.append(chromato_mass)
        debuts=ind
    cube = np.stack(chromato_mass_list)
    print("--- %s seconds ---" % (time.time() - start_time), "to compute cube from " + str(mass_range_min) + " to " + str(mass_range_max))

    return cube


def chromato_part(chromato_obj, rt1, rt2, mod_time = 1.25, rt1_window = 5, rt2_window = 0.1):
    chromato,time_rn = chromato_obj
    shape = chromato.shape
    position_in_chromato = np.array([[rt1 - rt1_window, rt2 - rt2_window], [rt1 + rt1_window, rt2 + rt2_window]])
    indexes = chromato_to_matrix(position_in_chromato,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
    return chromato[indexes[0][0]:indexes[1][0], indexes[0][1]:indexes[1][1]]



'''
from pyteomics import mzml, auxiliary

def read_profile_spectrum(coord):
    start_time = time.time()
    with mzml.MzML("F:/Bureau/Nouveau dossier/751301_YBS8_J0-mzml-profil.mzML") as reader:
        auxiliary.print_tree(next(reader))
    print("--- %s seconds --- to compute spectra" % (time.time() - start_time))


def plot_all_format(filename_nm, filename_centroid, filename_profile, coordinates):
    start_time = time.time()

    mod_time= 1.25
    
    #nm
    chromato_obj_nm = read_chroma(filename_nm, mod_time)
    chromato_nm,time_rn,spectra_obj_nm = chromato_obj_nm
    spectra_nm,debuts,fins = read_spectra(spectra_obj_nm)
    
    #centroid
    chromato_obj_centroid = read_chroma(filename_centroid, mod_time)
    chromato_centroid,time_rn,spectra_obj_centroid = chromato_obj_centroid
    spectra_centroid,debuts,fins = read_spectra_centroid(spectra_obj_centroid)

    #profile
    reader = mzml.MzML(filename_profile)
    for coord in coordinates:
        coord_str = str(coord[0]) + "-" + str(coord[1])
        #nm
        mass_values_nm, int_values_nm = read_spectrum(chromato_nm, coord, spectra_nm)
        ind_nm = np.where(mass_values_nm > 100)[0][0]
        print(ind_nm)
        plot_mass(mass_values_nm[:ind_nm], int_values_nm[:ind_nm], coord_str + "_nm_zoom")
        #centroid
        mass_values_centroid, int_values_centroid = read_spectrum(chromato_centroid, coord, spectra_centroid)
        ind_centroid = np.where(mass_values_centroid > 100)[0][0]
        print(ind_centroid)
        plot_mass(mass_values_centroid[:ind_centroid], int_values_centroid[:ind_centroid], coord_str + "_centroid_zoom")
        #profile
        mass_values_profile, int_values_profile = reader[coord[0] * chromato_nm.shape[1] + coord[1]]['m/z array'], reader[coord[0] * chromato_nm.shape[1] + coord[1]]['intensity array']
        ind_profile = np.where(mass_values_profile > 100)[0][0]
        print(ind_profile)
        plot_mass(mass_values_profile[:ind_profile], int_values_profile[:ind_profile], coord_str + "_profile_zoom")
        
        plot.mass_overlay([mass_values_nm[:ind_nm],mass_values_centroid[:ind_centroid],mass_values_profile[:ind_profile]], [int_values_nm[:ind_nm],int_values_centroid[:ind_centroid], int_values_profile[:ind_profile]], "nm_centroid_profile_" + coord_str + "_overlay_zoom")
    
    print("--- %s seconds --- to plot nm + centroid + profile" % (time.time() - start_time))

    
plot_all_format("F:/Bureau/Nouveau dossier/751301_YBS8_J0-CDF-nominal-mass.cdf", "F:/Bureau/Nouveau dossier/751301_YBS8_J0-CDF-centroid.cdf", "F:/Bureau/Nouveau dossier/751301_YBS8_J0-mzml-profil.mzML", [[800,100], [400, 150]])
'''
