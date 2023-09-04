# -*- coding: utf-8 -*-
"""
Created on 06.05.2020 14:14 2020

@author: GEIRI_EU_Kai Gu

"""
import itertools

import numpy as np
from scipy.cluster.hierarchy import fcluster, cophenet
from fastcluster import linkage
from scipy.spatial.distance import pdist
import time
from GPU_cross_correlation import GPUTensorCC
import scipy.spatial.distance as ssd
from sklearn.manifold import MDS


def multi_sensor_cc(df, data_list):
    # get the analysis parameters

    device_number = 2
    channel_number = 2

    # make the data_list an array
    data_array = np.array(data_list)

    # get the unique indices of instance
    instance_index = np.unique(df['inst_id'])
    instance_count = len(instance_index)

    # create a (number_channels, number_instances,number_instances) multi-sensor matrix
    instance_index_combinations = list(itertools.combinations(instance_index, 2))
    multi_sensor_ccm = np.empty((device_number * channel_number, instance_count, instance_count))

    # initialize the matrix with nan
    multi_sensor_ccm[:] = np.nan

    # container for df,cc_matrix, lag_matrix from different channels
    df_slices = []
    ccm_mats = []
    lag_mats = []

    # get the flattenend device-channel index
    df['flattened_index'] = df['device'] * device_number + df['channel']
    flattenend_chan_index = np.unique(df['flattened_index'].to_numpy())
    # run cc and clustering on all the channels

    for i in range(device_number * channel_number):
        df_single_channel_slice = df[df['flattened_index'] == i]
        # check if the df is empty
        if df_single_channel_slice.empty:
            df_slices.append(df_single_channel_slice)
            ccm_mats.append(None)
            lag_mats.append(None)
        else:
            # get the dafa slice for cc
            data_slice = data_array[df_single_channel_slice['row_id']]

            cc_single = GPUTensorCC(data_slice, )
            ccm_matrix, lag_matrix = cc_single.get_result()

            # fill the cc value to the multi-sensor matrix
            flattened_ccm = ccm_matrix[np.triu_indices(ccm_matrix.shape[0], k=1)]
            inst_indices = df_single_channel_slice['inst_id']
            inst_pairs = itertools.combinations(inst_indices.tolist(), 2)

            for inst_pair, ccm in zip(inst_pairs, flattened_ccm):
                multi_sensor_ccm[i, inst_pair[0], inst_pair[1]] = ccm

            # run single channel clustering
            Z, c, coph_dists, clusters = single_dimensional_fcluster(ccm_matrix)
            df_single_channel_slice['single_channel_cluster'] = clusters

            df_slices.append(df_single_channel_slice)
            ccm_mats.append(ccm_matrix)
            lag_mats.append(lag_matrix)

    # make the matrix symmetric and fill in the diagonal with 1
    upper_ind = np.triu_indices(multi_sensor_ccm.shape[1], k=1)
    multi_sensor_ccm[:, upper_ind[1], upper_ind[0]] = multi_sensor_ccm[:, upper_ind[0], upper_ind[1]]
    diag_ind = np.diag_indices(multi_sensor_ccm.shape[1], multi_sensor_ccm.shape[2])
    multi_sensor_ccm[:, diag_ind[0], diag_ind[1]] = 0

    if True:
        #  Computes the CC for pairs of instances detected from different sensors
        single_dimensional_ccm_matrix, multi_sensor_ccm_imputation = cc_matrix_imputation(multi_sensor_ccm, df_slices,
                                                                                          df)

    return single_dimensional_ccm_matrix, multi_sensor_ccm_imputation, df_slices, ccm_mats, lag_mats


def cc_matrix_imputation(multi_sensor_ccm, df_slices, df):
    """

    :param multi_sensor_ccm: ccm_matrix of all the channels shape (number_channels, number_instances,number_instances)
    :param df_slices: list of dataframes of different channels
    :param df: dataframe of all instances
    :return:
    """

    # count of device and channel
    device_number = 2
    channel_number = 2

    # unique indices of instance
    instance_index = np.unique(df['inst_id'])

    # combination of indices: [[0,1],[0,2],.....[1,2], [1,3],...]
    instance_index_combinations = np.array(list(itertools.combinations(instance_index, 2)))

    # get the upper indices of the upper-triangle and get the flattened cc_matrix
    ind = np.triu_indices(multi_sensor_ccm.shape[1], k=1)
    flattened_ccm = multi_sensor_ccm[:, ind[0], ind[1]]
    # find all nan columns which are non-common detection pairs
    non_common_detection_pairs = instance_index_combinations[np.all(np.isnan(flattened_ccm), axis=0)]

    # iterate over these instance pairs
    for progress, pair in enumerate(non_common_detection_pairs, ):
        # print the progress
        print(progress, '/', len(non_common_detection_pairs))
        t1 = time.perf_counter()

        inst_0, inst_1 = pair

        # container for df of instances
        df_insts = []
        # container for flattened index of instances
        ch_insts = []
        # container for inst_ids of same single channel cluster as the selected instance
        inst_ids_same_cluster_insts = []

        """
        
        assume  inst_0 detected on channel 0, 1
                inst_1 detected on channel 2, 3
        (1)
        find all the instance that has the same single_channel_cluster id as inst_0_ch_0 as same_clust_inst_0_ch_0
            e.g (2,3,6,9,12)
        find all the instance that has the same single_channel_cluster id as inst_0_ch_1 as same_clust_inst_0_ch_1
            e.g (2,6,9,11,13)
        get the union of (same_clust_inst_0_ch_0,same_clust_inst_0_ch_1) as inst_ids_same_cluster_inst_0
            (2,3,6,9,11,12,13)
        
        (2)
        find the cc_value of inst_0 to these instance on chan 0 and 1
        
                    0 to (2,    3,      6,      9,      11,     12,     13)
        channel 0         0.7   0.6     0.75    0.8     nan     0.79    0.1
        channel 1         0.67  nan     0.85    0.75    0.65    nan     0.75
        
        (3)
        get the maximum along the channel and sort the id by the value
        max = (0.7,0.6,0.85,0.8,0.65,0.79,0.75)
        inst_ids_same_cluster_inst_0_sorted = (6,9,12,13,2,11,3)
        
        repeat the same for the inst_1                        
        """

        for inst_i in pair:
            df_inst_i = df[df['inst_id'] == inst_i]
            ch_inst_i = df_inst_i['flattened_index'].to_numpy()
            inst_ids_same_cluster_inst_i = np.array([], dtype='int')

            # (1)
            for single_ch_inst_i in ch_inst_i:
                df_single_ch = df_slices[single_ch_inst_i]
                cluster_single_ch = \
                    df_single_ch[df_single_ch['inst_id'] == inst_i]['single_channel_cluster'].to_list()[0]
                inst_ids_single_ch_cluster = \
                    df_single_ch[df_single_ch['single_channel_cluster'] == cluster_single_ch]['inst_id'].to_numpy()
                inst_ids_same_cluster_inst_i = np.union1d(inst_ids_same_cluster_inst_i, inst_ids_single_ch_cluster)

            df_insts.append(df_inst_i)
            ch_insts.append(ch_inst_i)

            # (2)
            ccm_same_cluster_inst_i = np.nan_to_num(
                multi_sensor_ccm[ch_inst_i][:, inst_0, inst_ids_same_cluster_inst_i])

            # (3)
            inst_ids_same_cluster_inst_i_sorted = inst_ids_same_cluster_inst_i[
                np.argsort(np.max(ccm_same_cluster_inst_i, axis=0))]
            inst_ids_same_cluster_insts.append(inst_ids_same_cluster_inst_i_sorted)

        """
        get the ref value for channel 0:
            (4)
            find all the inst_ids that has channel 0 detection as inst_ids_ch_0
            intersect it with inst_ids_same_cluster_inst_1 
                (4.1)and keep the order in inst_ids_same_cluster_inst_1 (as they are sorted by cc_value_to_inst_1)
            as inst_ids_same_cluster_inst_1_on_ch_0
            
            take [:-15] if needed
                    
            (5)
            get the cc_value of inst_0 to the ids in inst_ids_same_cluster_inst_1_on_ch_0
            average them as the ref value for channel 0
            
        repeat the same for channel 1
        
        (6)
        then repeat the same for channels on inst_1
                        
        """

        for ch_i in ch_insts[0]:
            # (4)
            inst_ids_of_ch_i = df[df['flattened_index'] == ch_i]['inst_id']
            _, indices_intersect, _ = np.intersect1d(inst_ids_same_cluster_insts[1], inst_ids_of_ch_i,
                                                     return_indices=True)
            # (4.1)
            inst_ids_intersected = inst_ids_same_cluster_insts[1][np.sort(indices_intersect)][:-15]

            # (5)
            average_ccm_on_ch_i = np.mean(multi_sensor_ccm[ch_i, inst_0, inst_ids_intersected])
            multi_sensor_ccm[ch_i, inst_0, inst_1] = average_ccm_on_ch_i
            multi_sensor_ccm[ch_i, inst_1, inst_0] = average_ccm_on_ch_i

        # (6)
        for ch_i in ch_insts[1]:
            inst_ids_of_ch_i = df[df['flattened_index'] == ch_i]['inst_id']
            _, indices_intersect, _ = np.intersect1d(inst_ids_same_cluster_insts[0], inst_ids_of_ch_i,
                                                     return_indices=True)
            inst_ids_intersected = inst_ids_same_cluster_insts[0][np.sort(indices_intersect)][:-15]
            average_ccm_on_ch_i = np.mean(multi_sensor_ccm[ch_i, inst_0, inst_ids_intersected])
            multi_sensor_ccm[ch_i, inst_0, inst_1] = average_ccm_on_ch_i
            multi_sensor_ccm[ch_i, inst_1, inst_0] = average_ccm_on_ch_i

    # take the 75 percentile of the multi-sensor cc matrix and make it single-dimensional
    upper_ind = np.triu_indices(multi_sensor_ccm.shape[1], k=1)
    flattened = multi_sensor_ccm[:, upper_ind[0], upper_ind[1]]

    single_dimensional_flattened = np.nan_to_num(np.nanpercentile(flattened, 75, axis=0))
    single_dimensional_ccm_matrix = np.ones((multi_sensor_ccm.shape[1], multi_sensor_ccm.shape[2]))

    single_dimensional_ccm_matrix[upper_ind] = np.array(single_dimensional_flattened)
    single_dimensional_ccm_matrix[upper_ind[::-1]] = np.array(single_dimensional_flattened)

    return single_dimensional_ccm_matrix, multi_sensor_ccm


def single_dimensional_fcluster(cc_matrix, clustering_method='ward', clustering_criter='distance',
                                clustering_threshold=0.1):
    Z = linkage(cc_matrix, clustering_method)
    c, coph_dists = cophenet(Z, pdist(cc_matrix))
    delta_dist = []
    for z_dist in Z[:, 2]:
        delta_dist.append(z_dist / np.max(coph_dists))
    index_min_incr_distances = np.min(np.where(np.array(delta_dist) >= clustering_threshold)[0])
    cut_off_threshold = Z[index_min_incr_distances, 2] - 1e-2
    clusters = fcluster(Z, cut_off_threshold, criterion=clustering_criter)
    # embedding = MDS(n_components = 3, random_state = 0, dissimilarity= 'precomputed')
    # cc_matrix_transformed = embedding.fit_transform(1 - cc_matrix)

    return Z, c, coph_dists, clusters


def single_dimensional_fcluster_process(cc_matrix, clustering_method='ward', clustering_criter='distance',
                                        clustering_threshold=0.25, parent=None):
    new_method = True
    if new_method:
        if len(cc_matrix.shape) > 1:

            upper_ind = np.triu_indices(cc_matrix.shape[1], k=1)
            cc_array = cc_matrix[upper_ind[0], upper_ind[1]]

        else:
            cc_array = cc_matrix

        parent.update_progress('.linkage', 0)
        Z = linkage(1 - cc_array, clustering_method)
        parent.update_progress('.linkage', 1)
        parent.update_progress('.cal_threshold', 0)
        cut_off_threshold = Z[-1, 2] * clustering_threshold

    else:
        parent.update_progress('.linkage', 0)
        pair_wise_dist = pdist(cc_matrix, metric='correlation')

        Z = linkage(pair_wise_dist, clustering_method)
        parent.update_progress('.linkage', 1)
        parent.update_progress('.cal_threshold', 0)
        # c, coph_dists = cophenet(Z, pair_wise_dist)
        # delta_dist = Z[:,2]/np.max(coph_dists)
        # index_min_incr_distances = np.min(np.where(np.array(delta_dist) >= clustering_threshold)[0])
        # cut_off_threshold = Z[index_min_incr_distances, 2] - 1e-2
        cut_off_threshold = Z[-1, 2] * clustering_threshold
    c, coph_dists = None, None
    parent.update_progress('.cal_threshold', 1)
    parent.update_progress('.fcluster', 0)
    clusters = fcluster(Z, cut_off_threshold, criterion=clustering_criter)
    parent.update_progress('.fcluster', 1)
    return Z, c, coph_dists, clusters


if __name__ == '__main__':
    from mc_database.sql_loader import SQLLoader
    from mc_database.evaluation_tools import get_common_event
    # import matplotlib.pyplot as plt
    import time
    import pandas as pd
    from .dynamic_threshold import get_VRC, optimize_threshold

    # database = 'china_20190115_tuesday_wuxi'
    # table = 'mc_075313_calibration_2'
    database = 'china_20190117_thursday_xuyi'
    table = 'thursday_xuyi_20190117_094202_th20'

    loader1 = SQLLoader(database, table, )
    # host = '192.168.31.10')
    dataset = loader1.get_data_by_range(range(0, 2000))


    def get_signals(row):
        return row[4].y_raw


    def get_list(row):
        Snapshot, Device, Channel, raw_time_stamp, data = row
        return (Snapshot, Device, Channel, data.x_origin)


    #
    un_ziped_data_lsts = list(zip(*list(map(get_list, dataset))))

    d = {'snapshot': un_ziped_data_lsts[0],
         'device': un_ziped_data_lsts[1],
         'channel': un_ziped_data_lsts[2],
         'time_stamp': un_ziped_data_lsts[3],
         'row_id': range(len(un_ziped_data_lsts[0]))}
    df = pd.DataFrame(data=d)
    df = get_common_event(df)
    df = df.rename(columns={'EV_CUM': 'inst_id'})
    data_list = list(map(get_signals, dataset))

    single_dimensional_ccm, ccm_imputation, *_ = multi_sensor_cc(df, data_list)

    Z, c, coph_dists, clusters = single_dimensional_fcluster(single_dimensional_ccm)

    # df['cluster'] = clusters
    thresh_array, vrc_array, clusters_array = optimize_threshold(single_dimensional_ccm, Z)

    # print(clusters, clusters_opt)
