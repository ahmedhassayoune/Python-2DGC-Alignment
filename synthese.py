import json
import matplotlib.pyplot as plt
import os
import numpy as np
def get_keys_from_value(d, val):
    for k, v in d.items():
        if v == val:
            return k
    return -1

plm_params = {"filter","min_distance","min_distance_value", "seuil", "seuil value"}

def nb_peak_nb_found_present_retrieval(obj, seuil_min=None, seuil_max=None):
    keys_array = list((obj['seuil']).keys())
    if (seuil_min or seuil_max):
        keys_array = [seuil for seuil in keys_array if (not ((seuil_min and float(seuil) < seuil_min) or (seuil_max and float(seuil) > seuil_max)))]
    nb_peak = [obj['seuil'][str(seuil)][0] for seuil in keys_array]
    nb_found_present = [obj['seuil'][str(seuil)][3] for seuil in keys_array]
    return nb_peak, nb_found_present

def acc_recall_retrieval(obj, seuil_min=None, seuil_max=None):
    keys_array = list((obj['seuil']).keys())
    if (seuil_min or seuil_max):
        keys_array = [seuil for seuil in keys_array if (not ((seuil_min and float(seuil) < seuil_min) or (seuil_max and float(seuil) > seuil_max)))]
    recall = [obj['seuil'][str(seuil)][1] for seuil in keys_array]
    acc = [obj['seuil'][str(seuil)][2] for seuil in keys_array]
    return acc, recall

'''def nb_peak_nb_found_present_retrieval(obj, seuil_min=None, seuil_max=None):
    keys_array = list((obj['seuil']).keys())
    nb_peak = [obj['seuil'][str(seuil)][0] for seuil in keys_array]
    nb_found_present = [obj['seuil'][str(seuil)][3] for seuil in keys_array]
    return nb_peak, nb_found_present

def acc_recall_retrieval(obj, seuil_min=None, seuil_max=None):
    keys_array = list((obj['seuil']).keys())
    recall = [obj['seuil'][str(seuil)][1] for seuil in keys_array]
    acc = [obj['seuil'][str(seuil)][2] for seuil in keys_array]
    return acc, recall'''

def plot_nb_peak_nb_found_present(seuil_values, nb_peak, nb_found_present, seuil_min=None, seuil_max=None, title=""):
    if (seuil_min or seuil_max):
        seuil_values = [seuil for seuil in seuil_values if (not ((seuil_min and float(seuil) < seuil_min) or (seuil_max and float(seuil) > seuil_max)))]
    plt.plot(seuil_values, nb_peak, label='nb_peaks')
    plt.plot(seuil_values, nb_found_present, label='relevant')
    plt.title(title)
    plt.xlabel("seuils")
    plt.ylabel('nb_peaks')
    plt.legend()
    plt.show()

def plot_acc_recall(seuil_values, recall, acc, seuil_min=None, seuil_max=None, title=""):

    if (seuil_min or seuil_max):
        seuil_values = [seuil for seuil in seuil_values if (not ((seuil_min and float(seuil) < seuil_min) or (seuil_max and float(seuil) > seuil_max)))]
    plt.plot(seuil_values, recall, label='recall')
    plt.plot(seuil_values, acc, label='accuracy')
    plt.title(title)
    plt.xlabel("seuils")
    plt.ylabel('%')
    plt.legend()
    plt.show()

def benchmark_LoG(method_obj, param, plot=False, seuil_min=None, seuil_max=None):
    filters = param['filters']
    num_sigmas = param['num_sigma']
    filters.insert(0, "No Filter")
    ABS_THRESHOLDS = param['abs_t']
    for filter in filters:
        try:
            obj_filter_dict = method_obj[filter]
        except:
            print(filter + " not found in benchmark file")
            continue
        for ABS_THRESHOLD in ABS_THRESHOLDS:
            try:
                obj_abs_t_dict = obj_filter_dict['abs_t'][str(ABS_THRESHOLD)]
            except:
                print(str(ABS_THRESHOLD) + " not found in benchmark file")
                continue
            for num_sigma in num_sigmas:
                try:
                    obj_num_sigma_dict = obj_abs_t_dict['num_sigma'][str(num_sigma)]
                except:
                    print(str(num_sigma) + " not found in benchmark file")
                    continue

                keys_array = list((obj_num_sigma_dict['seuil']).keys())
                seuil_values = [float(seuil) for seuil in keys_array]
                nb_peak, nb_found_present = nb_peak_nb_found_present_retrieval(obj_num_sigma_dict, seuil_min, seuil_max)
                acc, recall = acc_recall_retrieval(obj_num_sigma_dict, seuil_min, seuil_max)
                if (plot):
                    title = "LoG_" + filter + "_abs_t_" + str(ABS_THRESHOLD) + "_num_sigma_" + str(num_sigma)
                    plot_nb_peak_nb_found_present(seuil_values, nb_peak, nb_found_present, seuil_min, seuil_max, title=title)
                    plot_acc_recall(seuil_values, acc, recall,  seuil_min, seuil_max, title=title)
    return nb_peak, nb_found_present, acc, recall


def benchmark_DoG(method_obj, param, plot=False, seuil_min=None, seuil_max=None):
    filters = param['filters']
    sigma_ratios = param['sigma_ratio']
    filters.insert(0, "No Filter")
    ABS_THRESHOLDS = param['abs_t']

    for filter in filters:
        try:
            obj_filter_dict = method_obj[filter]
        except:
            print(filter + " not found in benchmark file")
            continue
        for ABS_THRESHOLD in ABS_THRESHOLDS:
            try:
                obj_abs_t_dict = obj_filter_dict['abs_t'][str(ABS_THRESHOLD)]
            except:
                print(str(ABS_THRESHOLD) + " not found in benchmark file")
                continue
            for sigma_ratio in sigma_ratios:
                try:
                    obj_sigma_ratio_dict = obj_abs_t_dict['sigma_ratio'][str(sigma_ratio)]
                except:
                    print("sigma ratio: " + str(sigma_ratio) + " not found in benchmark file")

                title = "DoG_" + filter + "_abs_t_" + str(ABS_THRESHOLD) + "_sigma_ratio_" + str(sigma_ratio)
                keys_array = list((obj_sigma_ratio_dict['seuil']).keys())
                seuil_values = [float(seuil) for seuil in keys_array]
                nb_peak, nb_found_present = nb_peak_nb_found_present_retrieval(obj_sigma_ratio_dict, seuil_min, seuil_max)
                acc, recall = acc_recall_retrieval(obj_sigma_ratio_dict, seuil_min, seuil_max)
                if (plot):
                    plot_nb_peak_nb_found_present(seuil_values, nb_peak, nb_found_present,  seuil_min, seuil_max, title=title)
                    plot_acc_recall(seuil_values, acc, recall,  seuil_min, seuil_max, title=title)
    return nb_peak, nb_found_present, acc, recall


def benchmark_DoH(method_obj, param, plot=False, seuil_min=None, seuil_max=None):
    filters = param['filters']
    num_sigmas = param['num_sigma']
    filters.insert(0, "No Filter")
    ABS_THRESHOLDS = param['abs_t']

    for filter in filters:
        try:
            obj_filter_dict = method_obj[filter]
        except:
            print(filter + " not found in benchmark file")
            continue
        for ABS_THRESHOLD in ABS_THRESHOLDS:
            try:
                obj_abs_t_dict = obj_filter_dict['abs_t'][str(ABS_THRESHOLD)]
            except:
                print(str(ABS_THRESHOLD) + " not found in benchmark file")
                continue
            for num_sigma in num_sigmas:
                try:
                    obj_num_sigma_dict = obj_abs_t_dict['num_sigma'][str(num_sigma)]
                except:
                    print(str(num_sigma) + " not found in benchmark file")
                    continue

                keys_array = list((obj_num_sigma_dict['seuil']).keys())
                seuil_values = [float(seuil) for seuil in keys_array]
                nb_peak, nb_found_present = nb_peak_nb_found_present_retrieval(obj_num_sigma_dict, seuil_min, seuil_max)
                acc, recall = acc_recall_retrieval(obj_num_sigma_dict, seuil_min, seuil_max)
                if (plot):
                    title = "DoH_" + filter + "_abs_t_" + str(ABS_THRESHOLD) + "_num_sigma_" + str(num_sigma)
                    plot_nb_peak_nb_found_present(seuil_values, nb_peak, nb_found_present,  seuil_min, seuil_max, title=title)
                    plot_acc_recall(seuil_values, acc, recall,  seuil_min, seuil_max, title=title)
    return nb_peak, nb_found_present, acc, recall

def benchmark_pers_hom(method_obj, param, plot=False, seuil_min=None, seuil_max=None):
    filters = param['filters']
    ABS_THRESHOLDS = param['abs_t']

    filters_copy = filters.copy()
    filters_copy.insert(0, "No Filter")

    for filter in filters_copy:
        try:
            obj_filter_dict = method_obj[filter]
        except:
            print(filter + " not found in benchmark file")
            continue
        for ABS_THRESHOLD in ABS_THRESHOLDS:
            try:
                obj_abs_t_dict = obj_filter_dict['abs_t'][str(ABS_THRESHOLD)]
            except:
                print(str(ABS_THRESHOLD) + " not found in benchmark file")
                continue

            keys_array = list((obj_abs_t_dict['seuil']).keys())
            seuil_values = [float(seuil) for seuil in keys_array]
            nb_peak, nb_found_present = nb_peak_nb_found_present_retrieval(obj_abs_t_dict, seuil_min, seuil_max)
            acc, recall = acc_recall_retrieval(obj_abs_t_dict, seuil_min, seuil_max)


            if (plot):
                title = "peak_local_max_" + filter + "_abs_t_" + str(ABS_THRESHOLD)
                plot_nb_peak_nb_found_present(seuil_values, nb_peak, nb_found_present,  seuil_min, seuil_max, title=title)
                plot_acc_recall(seuil_values, acc, recall,  seuil_min, seuil_max, title=title)

    return np.array(nb_peak), np.array(nb_found_present), np.array(acc), np.array(recall)

def benchmark_peak_local_max(method_obj, param, plot=False, seuil_min=None, seuil_max=None):
    filters = param['filters']
    min_distances = param['min_distance']
    ABS_THRESHOLDS = param['abs_t']

    filters_copy = filters.copy()
    filters_copy.insert(0, "No Filter")

    for filter in filters_copy:
        try:
            obj_filter_dict = method_obj[filter]
        except:
            print(filter + " not found in benchmark file")
            continue
        for ABS_THRESHOLD in ABS_THRESHOLDS:
            try:
                obj_abs_t_dict = obj_filter_dict['abs_t'][str(ABS_THRESHOLD)]
            except:
                print(str(ABS_THRESHOLD) + " not found in benchmark file")
                continue
            for min_distance in min_distances:
                try:
                    obj_min_distance_dict = obj_abs_t_dict['min_distance'][str(min_distance)]
                except:
                    print(str(min_distance) + " not found in benchmark file")
                    continue

                keys_array = list((obj_min_distance_dict['seuil']).keys())
                seuil_values = [float(seuil) for seuil in keys_array]
                nb_peak, nb_found_present = nb_peak_nb_found_present_retrieval(obj_min_distance_dict, seuil_min, seuil_max)
                acc, recall = acc_recall_retrieval(obj_min_distance_dict, seuil_min, seuil_max)
                if (plot):
                    title = "peak_local_max_" + filter + "_abs_t_" + str(ABS_THRESHOLD) + "_min_distance_" + str(min_distance)
                    plot_nb_peak_nb_found_present(seuil_values, nb_peak, nb_found_present, seuil_min=seuil_min, seuil_max=seuil_max, title=title)
                    plot_acc_recall(seuil_values, acc, recall, seuil_min=seuil_min, seuil_max=seuil_max, title=title)

    return np.array(nb_peak), np.array(nb_found_present), np.array(acc), np.array(recall)


def synthese(params, filename="./benchmark.json"):
    with open(filename, "r") as f:
        obj = json.load(f)

    for method in list(params.keys()):
        print(method)
        globals()['benchmark_' + method](obj[method], params[method])
import glob
def synthese_group(params, filename_format, PATH, method='peak_local_max', group='G0', plot=False, seuil_min=None, seuil_max=None):
    #files = os.listdir(PATH)
    files = glob.glob(PATH + "/" + group + "*/*")
    #tic, 3d, mpm
    nb_peaks = []
    nb_found_presents = []
    recalls = []
    accuracies = []
    for file in files:
        if (group == "*" and ("G0" in file or "NIST" in file)):
            continue
        file = file.replace("\\", "/")
        if (filename_format in file):
            with open(file, "r") as f:
                #print(file)
                obj = json.load(f)
                nb_peak, nb_found_present, acc, recall = globals()['benchmark_' + method](obj[method], params[method], seuil_min=seuil_min, seuil_max=seuil_max)
                nb_peaks.append(nb_peak)
                nb_found_presents.append(nb_found_present)
                recalls.append(recall)
                accuracies.append(acc)
    nb_peaks = np.array(nb_peaks)
    nb_found_presents = np.array(nb_found_presents)
    recalls = np.array(recalls)
    accuracies = np.array(accuracies)

    nb_peaks_mean, nb_found_present_mean, recall_mean, accuracy_mean = np.mean(nb_peaks, 0), np.mean(nb_found_presents, 0), np.mean(recalls, 0), np.mean(accuracies, 0)
    if (plot):
        seuil_values = params[method]['seuil']
        plot_nb_peak_nb_found_present(seuil_values, nb_peak=nb_peaks_mean, nb_found_present=nb_found_present_mean, seuil_min=seuil_min, seuil_max=seuil_max, title=filename_format + "_mean_nb_peaks_nb_found_presents")
        plot_acc_recall(seuil_values=seuil_values, recall=recall_mean, acc=accuracy_mean, seuil_min=seuil_min, seuil_max=seuil_max, title=filename_format + "_mean_acc_recall")
    return nb_peaks_mean, nb_found_present_mean, recall_mean,accuracy_mean

filters = ["gaussian_filter", "gauss_laplace", "gauss_multi_deriv", "prewitt", "sobel"]
filters = []

seuil = [0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99]
dog_params = {"filters": filters, "sigma_ratio": [1.6], "seuil": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], "abs_t": [0.024]}
log_params = {"filters": filters, "num_sigma": [10], "seuil": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], "abs_t": [0., 0.024, 0.025]}

peak_local_max_params = {"filters": filters,"min_distance":[1],"seuil": [0.01, 0.1,0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], "abs_t": [0.024]}

params = {"LoG": log_params}
params = {"DoG":dog_params}
params = {"peak_local_max": peak_local_max_params}

#nb_peaks_mean, nb_found_present_mean, recall_mean, accuracy_mean = synthese_group(params, "plm_TIC_cluster.json", "./benchmark/", method='peak_local_max', group='G0', plot=True)
#synthese(params, "./benchmark_DoG/dog_3D_cluster_min_sigma.json")