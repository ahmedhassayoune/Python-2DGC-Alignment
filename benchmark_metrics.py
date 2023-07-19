import json
import numpy as np
import pandas as pd
import projection
from skimage.morphology import binary_dilation

def read_chromato_gt_json(filename):
    with open(filename, "r") as f:
        data=json.load(f)
    return data


def read_peak_table(filename, chromato_obj):
    retrieved_peaks=np.zeros_like(chromato_obj[0])
    df_unique_res = pd.read_csv(filename, header=0)
    retrieved_cmp=np.full_like(chromato_obj[0], '', dtype='<U64')
    retrieved_peaks_area=np.zeros_like(chromato_obj[0])

    for index, row in df_unique_res.iterrows():
        cd_in_matrix=projection.chromato_to_matrix(np.array([[row[6], row[7]]]), chromato_obj[1], 1.25, chromato_obj[0].shape)[0]
        retrieved_peaks[cd_in_matrix[0], cd_in_matrix[1]]+=1
        retrieved_cmp[cd_in_matrix[0], cd_in_matrix[1]]=row[0]
        try:
            retrieved_peaks_area[cd_in_matrix[0], cd_in_matrix[1]]=row[8]
        except:
            continue
    return retrieved_peaks, retrieved_cmp, retrieved_peaks_area

def compute_gt(filename, chromato_obj):
    data_json=read_chromato_gt_json(filename)
    peaks_gt=np.zeros_like(chromato_obj[0])
    gt_cmp=np.full_like(chromato_obj[0], '', dtype='<U64')
    area_gt=np.zeros_like(chromato_obj[0])
    for i, cluster in enumerate(data_json['peaks_mu']):
        for j, peaks in enumerate(cluster):
            peaks_gt[peaks[0], peaks[1]]+=1
            gt_cmp[peaks[0], peaks[1]]=data_json['spectra_name'][i][j]
            try:
                area_gt[peaks[0], peaks[1]]=data_json['peaks_area'][i][j]
            except:
                continue
    peaks_gt_dilated=np.where(binary_dilation(peaks_gt, footprint=None, out=None) == True, 1, 0)
    return peaks_gt, gt_cmp, peaks_gt_dilated, area_gt

def read_and_compute_metrics(gt_filename, peak_table_filename, chromato_obj):
    retrieved_peaks, retrieved_cmp, retrieved_peaks_area=read_peak_table(peak_table_filename, chromato_obj)
    peaks_gt, gt_cmp, peaks_gt_dilated, area_gt=compute_gt(gt_filename, chromato_obj)
    return compute_metrics(peaks_gt, gt_cmp, peaks_gt_dilated, retrieved_peaks, retrieved_cmp)

def compute_metrics(peaks_gt, gt_cmp, peaks_gt_dilated, retrieved_peaks, retrieved_cmp):
    # nb retrieved peaks
    # ratio nb retrieved peaks / nb true peak
    # recall strict
    # recall
    # identification recall
    # precision strict
    # precision
    return (len(np.argwhere(retrieved_cmp != '')),
            len(np.argwhere(retrieved_cmp != '')) / len(np.argwhere(gt_cmp != '')),
            len(np.argwhere((retrieved_peaks==peaks_gt) & (peaks_gt != 0))) / len(np.argwhere(gt_cmp != '')),
            len(np.argwhere((retrieved_peaks==peaks_gt_dilated) & (peaks_gt_dilated != 0))) / len(np.argwhere(gt_cmp != '')),
            len(np.argwhere((retrieved_cmp==gt_cmp) & (gt_cmp != ''))) / len(np.argwhere(gt_cmp != '')),
            len(np.argwhere((retrieved_peaks==peaks_gt) & (peaks_gt != 0))) / len(np.argwhere(retrieved_cmp != '')),
            len(np.argwhere((retrieved_peaks==peaks_gt_dilated) & (peaks_gt_dilated != 0))) / len(np.argwhere(retrieved_cmp != '')))
