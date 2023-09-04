import pickle
import math
import os
import src.plotting as plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from onset_detection import onset_detection_fun
from clustering_module import multi_sensor_cc, single_dimensional_fcluster




def main(opt):

    # 1. LOAD DATA - depending on the data supplied this step has to be performed differently
    # The main parameters that are necessary to run the code are: signals (full_waveforms), sensor information,
    # instance id, relative phase information and power carried by the signal.
    # 2. CLUSTER DATA - depending on the case the data has to be clustered in a different way.
    # For 'hueng' 'stutt' and 'lab_multi' cases the clustering is done with CCM
    # For 'lab_single' the data is from single source and so it is naturally labeled

    if opt['case'] == 'hueng':
        with open('./data/data_huengyang', 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            time_vectors = data[2]
            signals = data[3]
        with open('./data/data_huengyang_df', 'rb') as pickle_file:
            df_sorted = pickle.load(pickle_file)
            sensor = np.array(df_sorted['flattened_index'])
            instance = np.array(df_sorted['inst_id'])
            phase = np.array(df_sorted['phase_relative'])
            power = np.array(df_sorted['power_watt'])
    if opt['case'] == 'stutt':
        with open(
                './data/data_for_polimi_2021-06-11-115301UniStuuttgart_lab_test_20210611_PD_Bushing 4 RF 11dB 89.4kV.data',
                'rb') as pickle_file:
            data = pickle.load(pickle_file)
            df_sorted = data[0]
            signals = data[1]
            sensor = np.array(df_sorted['flattened_index'])
            instance = np.array(df_sorted['inst_id'])
            phase = np.array(df_sorted['phase_absolute'])
            power = np.array(df_sorted['power_watt'])
    # Open single signal - no clustering
    if opt['case'] == 'lab_single_signal':
        with open('./data/lab_data/single/single_source_scenario_1.1.data', 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            df_sorted = data[0]
            signals = data[1]
            sensor = np.array(df_sorted['flattened_index'])
            instance = np.array(df_sorted['inst_id'])
            phase = np.array(df_sorted['phase_absolute'])
            power = np.array(df_sorted['power_watt'])
    # Open signals from single source - they are naturally labeled
    if opt['case'] == 'lab_single':

        # get the number of files located in the data folder
        dir_path = r'./data/lab_data/single'
        source_file_nr = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

        # define the lists of exported information
        signals = []
        sensor = []
        instance = []
        cluster = []

        for number in range(source_file_nr):
            with open('./data/lab_data/single/single_source_scenario_' + str(number+1) + '.data',
                      'rb') as pickle_file:
                data = pickle.load(pickle_file)
                df_sorted = data[0]
                signals_single = data[1]
                sensor_single = np.array(df_sorted['flattened_index'])
                instance_single = np.array(df_sorted['inst_id'])
                cluster_single = np.full(len(signals_single), (number+1)) # Cluster labels should begin from 1

            signals.append(signals_single)
            sensor.append(sensor_single)
            instance.append(instance_single)
            cluster.append(cluster_single)
    # Open signals from multiple source - they have to be clustered - we need a label for accuracy definition
    if opt['case'] == 'lab_multi':

        # get the number of files located in the data folder
        dir_path = r'./data/lab_data/multiple'
        source_file_nr = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

        # define the lists of exported information
        signals = []
        sensor = []
        instance = []
        cluster = []

        for number in range(source_file_nr):
            if number+1 == 8:  # There was a problem with CCM generation for scenario 8
                continue
            with open('./data/lab_data/multiple/multiple_source_scenario_' + str(number+1) + '.data',
                      'rb') as pickle_file:
                data = pickle.load(pickle_file)
                df_sorted = data[0]
                signals_single = data[1]
                sensor_single = np.array(df_sorted['flattened_index'])
                instance_single = np.array(df_sorted['inst_id'])

            # Hierarchical agglomerative clustering
            # Compute the cross-correlation matrices from both a single-sensor and a multi-sensor perspective
            # single_dimensional_ccm_matrix, multi_sensor_ccm_imputation, df_slices, ccm_mats, lag_mats = \
            #     multi_sensor_cc(df_sorted, signals_single)
            # # Save the generated matrix: in this way, it is possible to avoid to compute it again
            # with open('./clustering/' + opt['case'] + '/single_dimensional_ccm_matrix_' + str(number), 'wb') \
            #         as file:
            #     pickle.dump(single_dimensional_ccm_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
            # Load the previously generated matrix: in this way, it is possible to avoid to compute it again
            with open('./clustering/lab_multi/single_dimensional_ccm_matrix_'+str(number+1), 'rb') \
                    as pickle_file:
                single_dimensional_ccm_matrix = pickle.load(pickle_file)
            # HAC: divide instances in groups on the basis of the cross-correlation matrices
            z, c, coph_dists, clusters = single_dimensional_fcluster(single_dimensional_ccm_matrix,
                                                                     clustering_threshold=opt[
                                                                         'clustering_threshold'])
            # Add a "cluster" column to the dataframe
            # Variable clusters contains instance-based information,
            # it is first needed to translate it to event-based information to add it to the dataframe
            # Add an empty column
            df_sorted['clusters_events'] = np.nan
            # List all the indexes corresponding to the instances in the available data
            unique_inst = np.unique(instance_single)
            # Compile the new column with the cluster number assigned to each event
            for m in range(len(unique_inst)):
                df_sorted.loc[instance_single == unique_inst[m], 'clusters_events'] = clusters[m]
            # Extract cluster labels from dataframe
            cluster_single = np.array([int(ind) for ind in df_sorted['clusters_events']])
            print('Clustering done for scenario_' + str(number+1))

            signals.append(signals_single)
            sensor.append(sensor_single)
            instance.append(instance_single)
            cluster.append(cluster_single)
    print('Data loaded.')

    # CCM clustering for Huengyang and Stuttgart datasets
    if opt['case'] == 'hueng' or opt['case'] == 'stutt':
        # Hierarchical agglomerative clustering
        # Compute the cross-correlation matrices from both a single-sensor and a multi-sensor perspective
        # single_dimensional_ccm_matrix, multi_sensor_ccm_imputation, df_slices, ccm_mats, lag_mats = \
        #     multi_sensor_cc(df_sorted, signals)
        # Save the generated matrix: in this way, it is possible to avoid to compute it again
        # with open('./clustering/' + opt['case'] + '/single_dimensional_ccm_matrix', 'wb') as file:
        #     pickle.dump(single_dimensional_ccm_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
        # Load the previously generated matrix: in this way, it is possible to avoid to compute it again
        with open('./clustering/'+opt['case']+'/single_dimensional_ccm_matrix', 'rb') as pickle_file:
            single_dimensional_ccm_matrix = pickle.load(pickle_file)
        # HAC: divide instances in groups on the basis of the cross-correlation matrices
        z, c, coph_dists, clusters = single_dimensional_fcluster(single_dimensional_ccm_matrix,
                                                                 clustering_threshold=opt['clustering_threshold'])

        # Add a "cluster" column to the dataframe
        # Variable clusters contains instance-based information,
        # it is first needed to translate it to event-based information to add it to the dataframe
        # Add an empty column
        df_sorted['clusters_events'] = np.nan
        # List all the indexes corresponding to the instances in the available data
        unique_inst = np.unique(instance)
        # Compile the new column with the cluster number assigned to each event
        for m in range(len(unique_inst)):
            df_sorted.loc[instance == unique_inst[m], 'clusters_events'] = clusters[m]
        # Extract cluster labels from dataframe
        cluster = np.array([int(ind) for ind in df_sorted['clusters_events']])
        print('Clustering done.')

    # 3. GRAPHS - There is an option to plot graphs of the dataset; it is available through the options on the bottom.
    plotting.plot_dendrogram(z, coph_dists)


    # 4. ONSET DETECTION - the feature extraction part of the code; done on the basis of the onset detection function
    # Feature extraction for lab_multi and lab_single files

    features = onset_detection_fun(signal_ar=signals,
                                       case=('stutt'),
                                       save=False)

        # Prepare information about instance id, sensor id and cluster id to connect to the feature data
        # These information is needed for training of the ANN, the expected format has to be:
        # 0        1       2        3 - (...)
        # inst_id  sens_id clust_id signal_data
        # This array potentially should be changed to a df for easier handling.

    info_acquisition = np.vstack((instance, sensor, cluster)).T
    signal_extracted = np.concatenate((info_acquisition, features), axis=1)

    # Save the extracted data as a pickle file - this data will have to be merged before training
    with open('./features/'+opt['case']+'/onset_'+opt['case'], 'wb') as file:
        pickle.dump(signal_extracted, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('scenario_saved.')


    for number in range(source_file_nr):
        # if number+1 == 8:     # for 'lab_multi' case the scenario 8 is broken
        #     continue
        # elif number+1 > 8:
        #     number = number - 1
        # Call the feature extraction file without saving
        features = onset_detection_fun(signal_ar=signals[number],
                                       case=('scenario_multi_' + str(number+1)),
                                       save=False)

        # Prepare information about instance id, sensor id and cluster id to connect to the feature data
        # These information is needed for training of the ANN, the expected format has to be:
        # 0        1       2        3 - (...)
        # inst_id  sens_id clust_id signal_data
        # This array potentially should be changed to a df for easier handling.

        info_acquisition = np.vstack((instance[number], sensor[number], cluster[number])).T
        signal_extracted = np.concatenate((info_acquisition, features), axis=1)

        # Save the extracted data as a pickle file - this data will have to be merged before training
        with open('./features/'+opt['case']+'/onset_scenario_' + str(number + 1), 'wb') as file:
            pickle.dump(signal_extracted, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('scenario_' + str(number + 1) + ' saved.')


if __name__ == '__main__':
    options = {'clustering_threshold': 0.2,

               'onset_detection/window_length': 500,
               'onset_detection/window_step': 5,
               'onset_detection/sample_start_search': 500,
               'onset_detection/sample_end_search': 2000,
               'onset_detection/onset_threshold': 0.1,
               'onset_detection/number_extracted_samples': 625,
               'onset_detection/energy_gap': 250,

               'case': 'stutt'}

    main(options)
