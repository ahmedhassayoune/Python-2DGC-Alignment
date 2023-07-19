import math
import numpy as np 
import pandas as pd

present = ['Octanoic acid, methyl ester', 'Nonanoic acid, methyl ester', 'Decanoic acid, methyl ester', 'Dodecanoic acid, methyl ester', 'Methyl tetradecanoate'
           ,'Hexadecanoic acid, methyl ester', 'Methyl stearate', 'Eicosanoic acid, methyl ester', 'Docosanoic acid, methyl ester', 'Tetracosanoic acid, methyl ester'
           ,'Hexacosanoic acid, methyl ester', 'Octacosanoic acid, methyl ester']


def parse_chromato_xlsx(filename='./data/Classeur1.xlsx', block_w = 6, offset = 3):
    df = pd.read_excel(filename)
    sample_comp = dict()
    sample = []
    step = int((len(df.columns) - offset) / block_w)
    for index, row in df.iterrows():
        if (index == 0):
            for i in range(0, step):
                sample_comp[row['Column' + str(i * block_w + offset)]] = []
                sample.append(row['Column' + str(i * block_w + offset)])
        elif (index == 1):
            continue
        else:
            comp_name = row['Column1']
            for i in range(0, step):
                start_block = i * block_w + offset
                area = float(row['Column' + str(start_block + 1)])
                if (not math.isnan(area)):
                    sample_comp[sample[i]].append(comp_name)
    return sample_comp, sample

def check_match_chromaTOF_verif(databaseids):
    not_found = []
    for comp in present:
        if (not comp in databaseids):
            not_found.append(comp)
    return not_found

def check_match_chromaTOF(match):
    return np.array([databaseid for databaseid in match if databaseid in present])

def compute_metrics_chromaTOF(comp):
    if (not len(comp)):
        return [0,0,0,0]
    else:
        found_present = check_match_chromaTOF(match=np.array(comp))
        #nb_peaks,rappel,precision
        return [len(comp), len(np.unique(found_present)) / len(present), len(found_present) / len(comp), len(found_present)]

def compute_sample_metrics(sample_comp):
    total = 0
    metrics = []
    sample_list = list(sample_comp.keys())
    for sample in sample_list:
        tmp = len(sample_comp[sample])
        total = total + tmp
        metrics.append(compute_metrics_chromaTOF(sample_comp[sample]))
    print(total / len(sample_list))
    return metrics

def compute_not_found_dict(sample_comp):
    not_found_dict = dict()
    sample_list = list(sample_comp.keys())
    for sample in sample_list:
        for not_found in (check_match_chromaTOF_verif(check_match_chromaTOF(match=np.array(sample_comp[sample])))):
            try:
                not_found_dict[not_found] = not_found_dict[not_found] + 1
            except:
                not_found_dict[not_found] = 1
    return not_found_dict

#class_name = G0/NIST/other
def compute_metrics_per_class(sample_list, metrics):

    #G0
    G0_nb_comp = 0
    G0_recall = 0
    G0_accuracy = 0
    G0_nb_class_comp = 0
    #NIST
    NIST_nb_comp = 0
    NIST_recall = 0
    NIST_accuracy = 0
    NIST_nb_class_comp = 0
    #other
    other_nb_comp = 0
    other_recall = 0
    other_accuracy = 0
    other_nb_class_comp = 0

    lg_list = len(sample_list)
    for i in range(lg_list):
        sample = sample_list[i]
        sample_nb_comp, sample_recall, sample_accuracy,_ = metrics[i]

        if ("G0" in sample):
            G0_nb_comp = G0_nb_comp + sample_nb_comp
            G0_recall = G0_recall + sample_recall
            G0_accuracy = G0_accuracy + sample_accuracy
            G0_nb_class_comp = G0_nb_class_comp + 1
        elif ("NIST" in sample):
            NIST_nb_comp = NIST_nb_comp + sample_nb_comp
            NIST_recall = NIST_recall + sample_recall
            NIST_accuracy = NIST_accuracy + sample_accuracy
            NIST_nb_class_comp = NIST_nb_class_comp + 1
        else:
            other_nb_comp = other_nb_comp + sample_nb_comp
            other_recall = other_recall + sample_recall
            other_accuracy = other_accuracy + sample_accuracy
            other_nb_class_comp = other_nb_class_comp + 1
    G0_metrics,NIST_metrics,other_metrics = 0,0,0
    if (G0_nb_comp):
        G0_metrics = [G0_nb_comp / G0_nb_class_comp, G0_recall / G0_nb_class_comp, G0_accuracy / G0_nb_class_comp, 0]
    if (NIST_nb_comp):
        NIST_metrics = [NIST_nb_comp / NIST_nb_class_comp, NIST_recall / NIST_nb_class_comp, NIST_accuracy / NIST_nb_class_comp, 0]
    if (other_nb_comp):
        other_metrics =  [other_nb_comp / other_nb_class_comp, other_recall / other_nb_class_comp, other_accuracy / other_nb_class_comp, 0]
    return G0_metrics,NIST_metrics,other_metrics
    return [G0_nb_comp / G0_nb_class_comp, G0_recall / G0_nb_class_comp, G0_accuracy / G0_nb_class_comp, 0], [NIST_nb_comp / NIST_nb_class_comp, NIST_recall / NIST_nb_class_comp, NIST_accuracy / NIST_nb_class_comp, 0], [other_nb_comp / other_nb_class_comp, other_recall / other_nb_class_comp, other_accuracy / other_nb_class_comp, 0] 