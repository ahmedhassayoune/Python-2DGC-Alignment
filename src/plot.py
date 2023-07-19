import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from projection import matrix_to_chromato, chromato_to_matrix
from matplotlib import cm
import time
import glob
from mass_spec import read_hmdb_spectrum
import pandas as pd
import seaborn as sn
import pandas as pd

def plot_confusion_matrix(conf_mat):
    df_cm = pd.DataFrame(conf_mat, index = [i for i in ["0" , "1"]],
                  columns = [i for i in ["0", "1"]])
    plt.figure(figsize = (5,3))
    sn.heatmap(df_cm, annot=True, cmap=sn.color_palette("coolwarm_r", as_cmap=True)
)

def plot_corr_matrix(corr_matrix):
    plt.imshow(corr_matrix)

def plot_feature_corr(index, corr_matrix, mol_list, id_max=20):
    r"""Plot the correlation of a feature with all other feature

    Parameters
    ----------
    index :
        Index of the feature in the molecule list or the name of the molecule.
    corr_matrix:
        The correlation matrix
    mol_list:
        The name of all the molecule associated with the feature
    id_max: optional
        Number max of the correlated feature
    Returns
    -------
    Examples
    --------
    >>> import plot
    >>> import find_biom
    >>> import numpy as np
    >>> features = np.array(find_biom.compute_sample_features(PEAK_TABLE_PATH))
    >>> corr_matrix = np.corrcoef(np.transpose(features))
    >>> mol_list, mol_data_list, labels = find_biom.group_by_labels_all_molecules_2(PEAK_TABLE_PATH)
    >>> plot.print_chroma_header("Benzophenone", corr_matrix,mol_list)
    """
    if (isinstance(index, str)):
        index = np.argwhere(mol_list == index)[0][0]
    series = pd.Series(
    corr_matrix[index],
    index=mol_list
    ).sort_values(ascending=True)[-id_max:]
    series.plot.barh(title=mol_list[index] + " correlation")

def plot_feature_and_permutation_importance(feature_importance, permutation_importance, mol_list, id_max=10):
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos[-id_max:], feature_importance[sorted_idx][-id_max:], align="center")
    plt.yticks(pos[-id_max:], np.array(mol_list)[sorted_idx][-id_max:])
    plt.title("Feature Importance (MDI)")


    sorted_idx = permutation_importance.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        permutation_importance.importances[sorted_idx][-id_max:].T,
        vert=False,
        labels=np.array(mol_list)[sorted_idx][-id_max:],
    )
    plt.title("Permutation Importance")
    fig.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance, mol_list, id_max=10):
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos[-id_max:], feature_importance[sorted_idx][-id_max:], align="center")
    plt.yticks(pos[-id_max:], np.array(mol_list)[sorted_idx][-id_max:])
    plt.title("Feature Importance (MDI)")

def plot_acp(features_disc_mol_new_cd, labels, projection=None, figsize=(10,5)):
    r"""Plot ACP

    Parameters
    ----------
    features_disc_mol_new_cd :
        Array of sample features
    labels:
        Labels associated with the samples
    projection: optional
        2D scatter plot with None projection and 3D with projection = "3d"
    figsize: optional
        Figure size
    Returns
    -------
    Examples
    --------
    >>> import plot
    >>> import find_biom
    >>> mol_list, mol_data_list, labels = find_biom.group_by_labels_all_molecules_2(PEAK_TABLE_PATH)
    >>> features_new_cd, pca = find_biom.acp(features)
    >>> plot.plot_acp(features_new_cd, labels)
    """

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121, projection=projection)

    cdict = {'negatif': 'blue', 'positif faible': 'orange', 'positif': 'red'}
    for g in ['negatif', 'positif faible', 'positif']:
        index = np.where(labels == g)
        p = features_disc_mol_new_cd[index]
        if (projection=="3d"):
            ax1.scatter(p[:,0], p[:,1], p[:,0], c = cdict[g], label = g)
        else:
            ax1.scatter(p[:,0], p[:,1], c = cdict[g], label = g)
    ax1.set_title("Positif vs Positif Faible vs Negatif")
    ax1.legend()
    
    ax2 = fig.add_subplot(122, projection=projection)

    cdict = {'negatif': 'blue', 'positif faible': 'red', 'positif': 'red'}
    for g in ['negatif', 'positif faible', 'positif']:
        index = np.where(labels == g)
        p = features_disc_mol_new_cd[index]
        if (projection=="3d"):
            ax2.scatter(p[:,0], p[:,1], p[:,0], c = cdict[g], label = g)
        else:
            ax2.scatter(p[:,0], p[:,1], c = cdict[g], label = g)
    ax2.set_title("Positif vs Negatif")
    ax2.legend()

    plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def mass_overlay(mass_values_list, intensity_values_list, title="mass_overlay", top_n_mass=10, figsize=(32, 18)):
    r"""Plot multiple mass spectra

    Parameters
    ----------
    mass_values_list :
        Masses of each mass spectra
    intensity_values_list:
        Intensities of each mass spectra
    title: optional
        Title of the plot
    figsize: optional
        Figure size
    Returns
    -------
    Examples
    --------
    >>> import plot
    >>> import find_biom
    >>> import mass_spec
    >>> import read chroma
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    >>> spectra, debuts, fins = full_spectra
    >>> m1, i1 = mass_spec.read_spectrum(chromato=chromato, pic_coord=coord1, spectra=spectra)
    >>> m2, i2 = mass_spec.read_spectrum(chromato=chromato, pic_coord=coord2, spectra=spectra)
    >>> plot.mass_overlay([m1, m2], [i1, i2])
    """
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 32,
            }
    plt.figure(figsize=figsize)
    spectrum = zip(mass_values_list[0], intensity_values_list[0])
    sorted_mz = sorted(spectrum, key = lambda x: x[1],reverse=True)
    for i in range (len(mass_values_list)):
        plt.bar(mass_values_list[i],intensity_values_list[i], width=0.4)
    for i, mi in enumerate(sorted_mz):
        if (i >= top_n_mass):
            break
        m = mi[0]
        i = mi[1]
        plt.text(m, i + 20, str(m), color='black', fontdict=font)
    #plt.xticks(mass_values_list[-1])
    plt.title(title)  
    plt.savefig("figs/" + title + ".png")
    plt.show() 


def plot_mass(mass_values, int_values, title="", top_n_mass=10, figsize=(32, 18)):
    r"""Plot mass spectrum

    Parameters
    ----------
    mass_values_list :
        Masses of the mass spectrum
    intensity_values_list:
        Intensities of the mass spectrum
    title: optional
        Title of the plot
    figsize: optional
        Figure size
    Returns
    -------
    Examples
    --------
    >>> import plot
    >>> import find_biom
    >>> import mass_spec
    >>> import read chroma
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    >>> spectra, debuts, fins = full_spectra
    >>> m1, i1 = mass_spec.read_spectrum(chromato=chromato, pic_coord=coord1, spectra=spectra)
    >>> plot.plot_mass(m1, i1)
    """
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 32,
            }
    plt.figure(figsize=figsize)
    plt.bar(mass_values,int_values, width=0.4)
    spectrum = zip(mass_values, int_values)
    sorted_mz = sorted(spectrum, key = lambda x: x[1],reverse=True)
    if (title):
        plt.title(title)
    for i, mi in enumerate(sorted_mz):
        if (i >= top_n_mass):
            break
        m = mi[0]
        i = mi[1]
        plt.text(m, i + 20, str(m), color='black', fontdict=font)
    plt.savefig("figs/mass_spectrum_" + title + ".png")
    plt.xticks(fontsize=30)

    plt.show()


def point_is_visible(point, indexes):
    x,y = point[0], point[1]
    if (x <= indexes[0][0] or x >= indexes[1][0] or y <= indexes[0][1] or y >= indexes[1][1]):
        return False
    return True

def plot(chromato_obj):
    """USELESS"""
    chromato, time_rn = chromato_obj
    plt.contourf(chromato)
    plt.colorbar()
    plt.show()
    

def visualizer(chromato_obj, mod_time = 1.25, rt1 = None, rt2 = None, rt1_window = 5, rt2_window = 0.1, plotly = False, title = "", points = None, radius=None, pt_shape = ".", log_chromato=True, casnos_dict=None, contour=[], center_pt=None, center_pt_window_1 = None, center_pt_window_2 = None, save=False):
    r"""Plot mass spectrum

    Parameters
    ----------
    chromato_obj :
        chromato_obj=(chromato, time_rn)
    rt1: optional
        Center the plot in the first dimension around rt1
    rt2: optional
        Center the plot in the second dimension around rt2
    rt1_window: optional
        If rt1, window size in the first dimension around rt1
    rt2_window: optional
        If rt2, window size in the second dimension around rt2
    points: optional
        Coordinates to displayed on the chromatogram
    radius: optional
        If points, dislay their radius (blobs detection)
    pt_shape: optional
        Shape of the points to be displayed.
    log_chromato: optional
        Apply logarithm function to the chromato before visualization
    contour: optional
        Displays stitch outlines
    center_pt: optional
        Center the plot around center_pt coordinates
    center_pt_window_1: optional
        If center_pt, window size in the first dimension around center_pt first coordinate
    center_pt_window_2: optional
        If center_pt, window size in the second dimension around center_pt second coordinate
    casnos_dict: optional
        If center_pt, window size in the second dimension around center_pt second coordinate
    title: optional
        Title of the plot
    Returns
    -------
    Examples
    --------
    >>> import plot
    >>> import read chroma
    >>> import utils
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> matches = matching.matching_nist_lib(chromato_obj, spectra, some_pixel_coordinates)
    >>> casnos_dict = utils.get_name_dict(matches)
    >>> coordinates_in_time = projection.matrix_to_chromato(u,time_rn, mod_time,chromato.shape)
    >>> plot.visualizer(chromato_obj=(chromato, time_rn), mod_time=mod_time, points=coordinates_in_time, casnos_dict=casnos_dict)
    """
    chromato, time_rn = chromato_obj
    shape = chromato.shape
    X = np.linspace(time_rn[0], time_rn[1], shape[0])
    Y = np.linspace(0, mod_time, shape[1])
    if (rt1 is not None and rt2 is not None):
        rt1minusrt1window = rt1 - rt1_window
        rt1plusrt1window = rt1 + rt1_window
        rt2minusrt2window = rt2 - rt2_window
        rt2plusrt2window = rt2 + rt2_window
        if (rt1minusrt1window < time_rn[0]):
            rt1minusrt1window = time_rn[0]
            rt1plusrt1window = rt1 + rt1_window
        if (rt1plusrt1window > time_rn[1]):
            rt1plusrt1window = time_rn[1]
            rt1minusrt1window = rt1 - rt1_window
        if (rt2minusrt2window < 0):
            rt2minusrt2window = 0
            rt2plusrt2window = rt2 + rt2_window
        if (rt2plusrt2window > mod_time):
            rt2plusrt2window = mod_time
            rt2minusrt2window = rt2 - rt2_window
        position_in_chromato = np.array([[rt1minusrt1window, rt2minusrt2window], [rt1plusrt1window, rt2plusrt2window]])
        indexes = chromato_to_matrix(position_in_chromato,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
        indexes_in_chromato = matrix_to_chromato(indexes,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
        chromato = chromato[indexes[0][0]:indexes[1][0], indexes[0][1]:indexes[1][1]]
        X = np.linspace(rt1minusrt1window, rt1plusrt1window, indexes[1][0] - indexes[0][0])
        Y = np.linspace(rt2minusrt2window, rt2plusrt2window, indexes[1][1] - indexes[0][1])
    elif (center_pt_window_1 and center_pt_window_2):
        center_pt1_minusrt1window = center_pt[0] - center_pt_window_1
        center_pt1_plusrt1window =  center_pt[0] + center_pt_window_1
        center_pt2_minusrt2window =  center_pt[1] - center_pt_window_2
        center_pt2_plusrt2window =  center_pt[1] + center_pt_window_2
        if (center_pt1_minusrt1window < 0):
            center_pt1_minusrt1window = 0
            center_pt1_plusrt1window = 2 * center_pt[0]
        if (center_pt1_plusrt1window >= shape[0]):
            center_pt1_plusrt1window = shape[0] - 1
            center_pt1_minusrt1window = center_pt[0] - abs(center_pt[0] - center_pt1_plusrt1window)
        if (center_pt2_minusrt2window < 0):
            center_pt2_minusrt2window = 0
            center_pt2_plusrt2window = 2 * center_pt[1]
        if (center_pt2_plusrt2window >= shape[1]):
            center_pt2_plusrt2window = shape[1] - 1
            center_pt2_minusrt2window = center_pt[1] - abs(center_pt[1] - center_pt2_plusrt2window)

        chromato = chromato[center_pt1_minusrt1window:center_pt1_plusrt1window + 1, center_pt2_minusrt2window:center_pt2_plusrt2window + 1]
        position_in_chromato = np.array([[center_pt1_minusrt1window, center_pt2_minusrt2window], [center_pt1_plusrt1window, center_pt2_plusrt2window]])
        indexes = matrix_to_chromato(position_in_chromato,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
        #indexes_in_chromato = matrix_to_chromato(indexes,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
        indexes_in_chromato=indexes

        X = np.linspace(indexes[0][0], indexes[1][0], chromato.shape[0])
        Y = np.linspace(indexes[0][1], indexes[1][1], chromato.shape[1])

        indexes = np.array([[center_pt1_minusrt1window, center_pt2_minusrt2window], [center_pt1_plusrt1window + 1, center_pt2_plusrt2window + 1]])
    if (log_chromato):
        chromato = np.log(chromato)
    chromato = np.transpose(chromato)
    fig, ax = plt.subplots()

    #tmp = ax.pcolormesh(X, Y, chromato)
    tmp = ax.contourf(X, Y, chromato)
    plt.colorbar(tmp)
    if (title != ""):
        plt.title(title)
    if (points is not None):
        if ((rt1 and rt2) or (center_pt_window_1 and center_pt_window_2)):
            tmp = []
            point_indexes = chromato_to_matrix(points,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
            for i, point in enumerate(point_indexes):
                if (point_is_visible(point, indexes)):
                    tmp.append(points[i])

            points = np.array(tmp)
        if (radius is not None and len(points) > 0):
            for i in range(len(points)):
                c = plt.Circle((points[i][0], points[i][1]), radius[i] / shape[1] , color="red", linewidth=2, fill=False)
                ax.add_patch(c)
        if (len(points) > 0):
            if (casnos_dict != None):
                mol_name = []
                scatter_list = []
                comp_list = list(casnos_dict.keys())
                nb_comp = len(comp_list)
                cmap = get_cmap(nb_comp)
                for i, casno in enumerate(comp_list):
                    tmp_pt_list = []
                    for pt in casnos_dict[casno]:
                        if (not((rt1 and rt2) or (center_pt_window_1 and center_pt_window_2)) or point_is_visible(pt, indexes_in_chromato)):
                            print(casno)
                            tmp_pt_list.append(pt)
                    '''x_pts = np.array(casnos_dict[casno])[:,0]
                    y_pts = np.array(casnos_dict[casno])[:,1]'''
                    if len(tmp_pt_list) == 0:
                        continue
                    print("----")

                    mol_name.append(comp_list[i])
                    tmp_pt_list = np.array(tmp_pt_list)
                    x_pts = tmp_pt_list[:,0]
                    y_pts = tmp_pt_list[:,1]
                    tmp = ax.scatter(x_pts,y_pts, c=cmap(i), marker=pt_shape, cmap='hsv')
                    scatter_list.append(tmp)
                print(mol_name)
                plt.legend(scatter_list,
                    mol_name,
                    scatterpoints=1, fontsize=8, ncol=1, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand")
            else:
                ax.plot(points[:,0], points[:,1], "r" + pt_shape)
    if (len(contour)):
        if (center_pt_window_1 and center_pt_window_2):
            indexes_in_chromato = matrix_to_chromato(indexes,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
            tmp = []
            for i in range(len(contour)):
                if (point_is_visible(contour[i], indexes_in_chromato)):
                    tmp.append(contour[i])
            tmp=np.array(tmp)
            ax.plot(tmp[:,0], tmp[:,1], "b.")
        else:
            ax.plot(contour[:,0], contour[:,1], "b.")
    if (save):
        plt.savefig("figs/chromato_" + title + ".png")

    plt.show()
    if (plotly):
        fig = go.Figure(data =
        go.Contour(
            z=np.transpose(chromato),
            x = np.linspace(time_rn[0], time_rn[1], shape[0]),
            y = np.linspace(0, mod_time, shape[1])
        ))
        fig.show()

def plot_3d_chromato(chromato,rstride=10, cstride=10, plot_map=True):
    sub_chrom = chromato.copy()
    wsize = sub_chrom.shape[0] // 2
    hsize = sub_chrom.shape[1] // 2
    x_sup = wsize if sub_chrom.shape[0] % 2 == 0 else wsize + 1
    y_sup = hsize if sub_chrom.shape[1] % 2 == 0 else hsize + 1
    Y,X = np.mgrid[-wsize:x_sup,-hsize:y_sup]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, sub_chrom)
    if (plot_map):
        ax.plot_surface(X, Y, sub_chrom, rstride=rstride, cstride=cstride, cmap=cm.coolwarm)
    else:
        ax.plot_surface(X, Y, sub_chrom, rstride=rstride, cstride=cstride)

def plot_hmdb_id_spectra(path, hmdb_id):
    tmp = path + "/" + hmdb_id + "*"
    hmdb_id_files = glob.glob(tmp)
    for hmdb_id_file in hmdb_id_files:
        m, v = read_hmdb_spectrum(hmdb_id_file.replace("\\", "/"))
        plot_mass(m, v, hmdb_id_file[len(path + "/") - 1:])

def plot_scores_array(scores_array, similarity_measure):
    plt.imshow(scores_array[similarity_measure.__class__.__name__ + "_score"], cmap="viridis")
    plt.colorbar(shrink=0.7)
    plt.title("Modified Cosine spectra similarities")
    plt.xlabel("Spectrum #ID")
    plt.ylabel("Spectrum #ID")
    plt.show()

    