import os
import pandas as pd
import math
import csv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import utils

group_dict = {0: 'neg_vs_posi', 1: 'neg_vs_posi_f_plus_posi', 2: 'neg_plus_posi_f_vs_posi', 3: 'neg_posi_vs_posi_f'}
group_dict_name = {'neg_vs_posi':0, 'neg_vs_posi_f_plus_posi':1, 'neg_plus_posi_f_vs_posi': 2, 'neg_vs_posi_f_vs_posi': 3}


def acp(features,n_components=2):
    pca = PCA(n_components=n_components)
    features_new_cd = pca.fit_transform(features)
    print(pca.explained_variance_ratio_, pca.singular_values_)
    return features_new_cd, pca


def compute_sample_features_filter_by_disc_mol(filename, mol_list):
    df = pd.read_csv(filename, header=None)
    features = []
    for index, row in df.iterrows():
        if (index > 1):
            name = row[0]
            if (name in mol_list):
                row = row[1:].astype(float)
                row[pd.isnull(row)] = 0
                features.append(np.array(row))
    return np.transpose(features)

def compute_sample_features_filter_by_disc_mol_ordered(filename, mol_list):
    df = pd.read_csv(filename, header=None)
    features = []
    ordered_mol_list = []
    for index, row in df.iterrows():
        if (index > 1):
            name = row[0]
            if (name in mol_list):
                row = row[1:].astype(float)
                row[pd.isnull(row)] = 0
                features.append(np.array(row))
                ordered_mol_list.append(name)
    return np.transpose(features), np.transpose(ordered_mol_list)
'''
    Unique molecules between samples
'''
def compute_unique_mol_list(PATH = './COVID/'):
    files = os.listdir(PATH)
    unique_mol_list = []
    for file in files[:-7]:
        df = pd.read_csv(PATH + file)
        for index, row in df.iterrows():
            unique_mol_list.append(row['Name'])
    unique_mol_list = np.unique(unique_mol_list)
    return unique_mol_list

'''
    Merge molecule (area) in each sample
'''
def merge_molecule_in_sample(file, unique_mol_list, PATH_RES='./COVID_UNIQUE/'):
    df = pd.read_csv(file)
    with open(PATH_RES + 'unique_mol_' + file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for unique_mol in unique_mol_list:
            tmp = df.loc[df.Name == unique_mol,:]
            area = 0
            for i in range(tmp.shape[0]):
                area = area + tmp.iloc[i]['Area']
            if (area):
                row = [tmp.iloc[i]['Name'], tmp.iloc[i]['Casno'], tmp.iloc[i]['Formula'], area]
                writer.writerow(row)

def merge_molecule_in_samples(unique_mol_list, PATH ='./COVID/', PATH_RES='./COVID_UNIQUE/'):
    files = os.listdir(PATH)
    for file in files[:-7]:
        merge_molecule_in_sample(PATH + file, unique_mol_list, PATH_RES=PATH_RES)
        
'''
    Write concentration in samples for each molecule
'''
def write_cohort_feature(unique_mol_list, UNIQUE_PATH ='./COVID_UNIQUE/', res_filename='./unique_res.csv'):
    lg = len(unique_mol_list)
    files_unique = os.listdir(UNIQUE_PATH)
    with open(res_filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = files_unique.copy()
        header.insert(0, '')
        writer.writerow(header)
        for i, molecule in enumerate(unique_mol_list):
            res = [molecule]
            for file in files_unique:
                df = pd.read_csv(UNIQUE_PATH + file, header=None)
                b = False
                for index, row in df.iterrows():
                    #row = NAME, CASNO, FORMULA, AREA
                    if (row[0] == molecule):
                        #AREA
                        area = (row[3])
                        res.append(area)
                        b = True
                if (not b):
                    res.append('')
            print("----", i, '/', lg - 1, "----")
            writer.writerow(res)
            
            

def compute_p_value(data, plot=False):
    if (len(data) > 2):
        #statistic, pvalue
        return stats.kruskal(data[0], data[1], data[2])
    else:
        return stats.kruskal(data[0], data[1])


def group_by_labels_zero_value_for_nan(row, labels):
    
    row[pd.isnull(row)] = 0
    negatif = np.array(row[labels == 'negatif'])
    positif_f = np.array(row[labels == 'positif faible'])
    positif = np.array(row[labels == 'positif'])
    negatif_plus_posi_f = np.array(row[labels != 'positif'])
    posi_plus_posi_f = np.array(row[labels != 'negatif'])

    data = [negatif.astype(np.float), positif_f.astype(np.float), positif.astype(np.float),
            negatif_plus_posi_f.astype(np.float), posi_plus_posi_f.astype(np.float)]
    
    return data

def group_by_labels(row, labels):
    negatif = np.array(row[labels == 'negatif'])
    positif_f = np.array(row[labels == 'positif faible'])
    positif = np.array(row[labels == 'positif'])
    negatif_plus_posi_f = np.array(row[labels != 'positif'])
    posi_plus_posi_f = np.array(row[labels != 'negatif'])

    filtered_negatif = negatif[~pd.isnull(negatif)]
    filtered_positif_f = positif_f[~pd.isnull(positif_f)]
    filtered_positif = positif[~pd.isnull(positif)]

    filtered_negatif_plus_posi_f = negatif_plus_posi_f[~pd.isnull(negatif_plus_posi_f)]
    filtered_posi_plus_posi_f = posi_plus_posi_f[~pd.isnull(posi_plus_posi_f)]

    data = [filtered_negatif.astype(np.float), filtered_positif_f.astype(np.float), filtered_positif.astype(np.float),
            filtered_negatif_plus_posi_f.astype(np.float), filtered_posi_plus_posi_f.astype(np.float)]
    
    return data

def group_by_labels_all_molecules(filename = './unique_res.csv'):
    mol_list = []
    mol_data_list = []
    labels = None
    df_unique_res = pd.read_csv('./unique_res.csv', header=None)
    for index, row in df_unique_res.iterrows():
        if (index == 0):
            labels = np.array(row[1:])
            continue
        if (index == 1):
            files = np.array(row[1:])
            continue
        title = row[0]
        row = row[1:]
        mol_list.append(title)
        mol_data_list.append(group_by_labels(row , labels))
    return mol_list, mol_data_list, labels


def group_by_labels_all_molecules_2(filename):
    mol_list = []
    mol_data_list = []
    labels = None
    df_unique_res = pd.read_csv(filename, header=None)
    for index, row in df_unique_res.iterrows():
        if (index == 0):
            labels = np.array(row[1:])
            continue
        if (index == 1):
            files = np.array(row[1:])
            continue
        title = row[0]
        row = row[1:]
        mol_list.append(title)
        mol_data_list.append(group_by_labels_zero_value_for_nan(row , labels))
    return mol_list, mol_data_list, labels

def filter_molecules_by_p_value(mol_list, mol_data_list, labels, mode=0, p_value_thresold = 0.1, group_threshold=0.5, plot = False, skip=False, skip_nan=True):
    mol_of_interest = []
    nb_neg = len(np.argwhere(labels == 'negatif'))
    nb_posi_f = len(np.argwhere(labels == 'positif faible'))
    nb_posi = len(np.argwhere(labels == 'positif'))
    for i, mol in enumerate(mol_list):
        #mol_data = [neg, posi_f, posi, neg_plus_posi_f, posi_plus_posi_f]
        mol_data = mol_data_list[i]
        #statistic, pvalue = None, None
        data = None
        if (mode == 0):
            #neg_vs_posi
            if (not len(mol_data[0]) and not len(mol_data[2])):
                continue
            if (skip):
                ratio_neg = len(mol_data[0]) / nb_neg
                ratio_posi = len(mol_data[2]) / nb_posi
                if ((ratio_neg > 0 and ratio_neg < group_threshold) or (ratio_posi > 0 and ratio_posi < group_threshold)):
                    continue
                if (not len(mol_data[0]) and ratio_posi <= group_threshold):
                    continue
                if (not len(mol_data[2]) and ratio_neg <= group_threshold):
                    continue
                
            data = [mol_data[0], mol_data[2]]
        elif (mode == 1):
            #neg_vs_posi_f_plus_posi
            if (not len(mol_data[0]) and not len(mol_data[4])):
                continue
            if (skip):
                ratio_neg = len(mol_data[0]) / nb_neg
                ratio_posi_plus_posi_f = len(mol_data[4]) / (nb_posi + nb_posi_f)
                if ((ratio_neg > 0 and ratio_neg < group_threshold) or (ratio_posi_plus_posi_f > 0 and ratio_posi_plus_posi_f < group_threshold)):
                    continue
                if (not len(mol_data[0]) and ratio_posi_plus_posi_f <= group_threshold):
                    continue
                if (not len(mol_data[4]) and ratio_neg <= group_threshold):
                    continue

            data = [mol_data[0], mol_data[4]]
        elif (mode == 2):
            #neg_plus_posi_f_vs_posi
            if (not len(mol_data[3]) and not len(mol_data[2])):
                continue
            if (skip):
                ratio_neg_plus_posi_f = len(mol_data[3]) / (nb_neg + nb_posi_f)
                ratio_posi = len(mol_data[2]) / nb_posi
                if ((ratio_neg_plus_posi_f > 0 and ratio_neg_plus_posi_f < group_threshold) or (ratio_posi > 0 and ratio_posi < group_threshold)):
                    continue
                if (not len(mol_data[3]) and ratio_posi <= group_threshold):
                    continue
                if (not len(mol_data[2]) and ratio_neg_plus_posi_f <= group_threshold):
                    continue

            data = [mol_data[3], mol_data[2]]
        else:
            #neg_vs_posi_f_vs_posi
            if (not len(mol_data[0]) and not len(mol_data[1]) and not len(mol_data[2])):
                continue
            if (skip):
                ratio_neg = len(mol_data[0]) / nb_neg
                ratio_posi_f = len(mol_data[1]) / nb_posi_f
                ratio_posi = len(mol_data[2]) / nb_posi
                if ((ratio_neg > 0 and ratio_neg < group_threshold) or (ratio_posi_f > 0 and ratio_posi_f < group_threshold) or (ratio_posi > 0 and ratio_posi < group_threshold)):
                    continue
            if (not len(mol_data[0]) and (ratio_posi_f <= group_threshold or ratio_posi <= group_threshold)):
                continue
            if (not len(mol_data[1]) and (ratio_neg <= group_threshold or ratio_posi <= group_threshold)):
                continue
            if (not len(mol_data[2]) and (ratio_neg <= group_threshold or ratio_posi_f <= group_threshold)):
                continue

            data = [mol_data[0], mol_data[1], mol_data[2]]

        statistic, pvalue = compute_p_value(data)
        if (pvalue < p_value_thresold or (not skip_nan and math.isnan(pvalue))):
            mol_of_interest.append(mol)
            if (plot):
                fig = plt.figure(figsize =(5, 4))
                ax = fig.add_axes([0, 0, 1, 1])
                bp = ax.boxplot(data)
                plt.title(mol + '_' + group_dict[mode])
                plt.show()
    return mol_of_interest


def filter_molecules_by_p_value_2(mol_list, mol_data_list, mode=0, p_value_thresold = 0.1, plot = False, x_axis=['negatif','positif']):
    mol_of_interest = []
    for i, mol in enumerate(mol_list):
        mol_data = mol_data_list[i]
        data = None
        if (mode == 0):
            #neg_vs_posi
            data = [mol_data[0], mol_data[2]]
        elif (mode == 1):
            #neg_vs_posi_f_plus_posi
            data = [mol_data[0], mol_data[4]]
        elif (mode == 2):
            #neg_plus_posi_f_vs_posi
            data = [mol_data[3], mol_data[2]]
        else:
            #neg_vs_posi_f_vs_posi
            data = [mol_data[0], mol_data[1], mol_data[2]]
        statistic, pvalue = compute_p_value(data)
        if (pvalue < p_value_thresold):
            mol_of_interest.append(mol)
            '''formula, weight = utils.retrieve_formula_and_mass_from_compound_name(mol)
            mol_of_interest.append([mol, formula, weight])'''
            if (plot):
                fig = plt.figure(figsize =(5, 4))
                '''ax = fig.add_axes([0, 0, 1, 1])
                bp = ax.boxplot(data)'''
                plt.boxplot(data)
                plt.xticks(np.arange(1,len(x_axis) + 1, 1), x_axis)
                m = utils.retrieve_formula_and_mass_from_compound_name(mol)
                plt.title(mol + ' ' + str(m))
                plt.show()
    return mol_of_interest

'''def compute_features(filename):
    df = pd.read_csv(filename, header=None)
    features = []
    for index, row in df.iterrows():
        if (index == 0):
            labels = np.array(row[1:])
            continue
        if (index == 1):
            files = np.array(row[1:])
            continue
        row = row[1:]
        row[pd.isnull(row)] = 0
        features.append(row)
    return features'''
    
def compute_sample_features(filename):
    df = pd.read_csv(filename, header=None)
    i = 0
    features = []
    for column in df:
        if (i > 0):
            col = df[column][2:].astype(float)
            col[pd.isnull(col)] = 0
            features.append(np.array(col))
        i = i + 1
    return features

def pca_most_important_features(pca, mol_list, id_max=10):
    n_pcs= pca.components_.shape[0]
    most_important = [np.argsort(np.abs(pca.components_[i]))[-id_max:] for i in range(n_pcs)]
    most_important_features = np.flip([mol_list[most_important[i]] for i in range(n_pcs)], -1)
    return most_important_features