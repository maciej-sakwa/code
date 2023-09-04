import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import dendrogram
from onset_detection import onset_detection_fun

# General parameters
#General plot parameters
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.linewidth'] = 1.5
plt.style.use('bmh')



def plot_dendrogram(linkage, cophenetic_dist, clustering_threshold=0.2):
    """
    Plots dendrogram of the HAC

    :param linkage: ndarray; linkage matrix of the clustering
    :param cophenetic_dist: ndarray; cophenetic distance matrix of the clustering
    :param clustering_threshold: float; default; threshold at which the clustering has been done

    :return: none
    """

    # Compute the cut-off distance threshold, used as stopping condition to the HAC procedure
    cut_off_threshold = clustering_threshold * np.max(cophenetic_dist)
    # Plot the dendrogram
    dendrogram(linkage, no_labels=True)
    # Add the cut-off threshold to the plot
    plt.axhline(y=cut_off_threshold, c='black', lw=1, linestyle='dashed')
    plt.ylabel("Cophenetic distance")
    plt.xlabel("Cluster members")
    # Display the plot
    plt.show()


def plot_prpd(phase, power, cluster_id, scale=1e7):
    """
    Plots PRPDs for identified clusters

    :param phase: ndarray; array of the recorded phase of each event
    :param power: ndarray; array of the power of each event
    :param cluster_id: ndarray; array of cluster labels of each recorded event
    :param scale: float; default; a scale parameter for power - the values are very low

    :return: none
    """
    # Consider a phase range of 360 degrees
    power_degrees = np.arange(360)
    # Define a sine wave representing the power cycle
    power_cycle = np.array([np.sin(math.radians(deg)) for deg in power_degrees])
    # Plot a diagram for each of the clusters identified
    for clus in np.unique(cluster_id):
        # Extract phase information (for PRPD plot)
        phase_cluster = phase[cluster_id == clus]
        for i in range(len(phase_cluster)):
            if 360 < phase_cluster[i] < 720:
                phase_cluster[i] = phase_cluster[i] - 360
            elif 720 < phase_cluster[i] < 1080:
                phase_cluster[i] = phase_cluster[i] - 720
            elif 1080 < phase_cluster[i] < 1440:
                phase_cluster[i] = phase_cluster[i] - 1080

        # Extract power information (for PRPD plot)
        power_cluster = power[cluster_id == clus]
        # Open a new figure
        plt.figure()
        # Plot the signals as function of their power and phase; a scale factor defined in the code options is used
        # in order to achieve a better representation
        plt.scatter(phase_cluster, power_cluster * scale)
        # Plot the power cycle
        plt.plot(power_degrees, power_cycle)
        # Display the plot
        plt.show()


def plot_prpd_multiclass(power, phase, cluster):
    """
    Plots a multiclass plot of PRPD

    :param power: ndarray; power of plotted signals
    :param phase: ndarray; phase of plotted signals
    :param cluster: ndarray; class of plotted signals

    :return: none
    """

    # Consider a phase range of 360 degrees
    degrees = np.arange(360)
    # Define a sine wave representing the power cycle
    power_range = (max(power) - min(power)) / 2
    power_mid = (max(power) + min(power)) / 2
    power_cycle = np.array([power_range * np.sin(math.radians(deg)) + power_mid for deg in degrees])

    
    fig, ax = plt.subplots()
    ax.plot(power_cycle, c='k', alpha=0.5)
    for i in np.unique(cluster):
        # if i == np.amax(cluster):
        #     continue
        ax.scatter(phase[cluster == i], power[cluster == i], label='Cluster '+str(int(i)))
    ax.set_ylabel('Power [dbm]')
    ax.set_xlabel('Phase [degrees]')
    ax.legend()

    plt.show()


def plot_member_example(signals, cluster, class_labels):
    """
    Plots random examples of members of each class

    :param signals: ndarray; sorted array of all the signals
    :param cluster: ndarray; sorted array of corresponding clusters

    :return: none
    """

    # Find the number of clusters to identify
    unique_clusters = np.unique(cluster)
    # Define the plot
    fig, axs = plt.subplots(len(unique_clusters), sharex='all', figsize = (8, 3*len(unique_clusters)))
    fig.suptitle("Randomly picked signal from each cluster", y=0.95)
    # Plot random member of each class
    for c, ax in zip(range(len(unique_clusters)), axs.ravel()):
        signals_cluster = signals[cluster == c]
        id_cluster = random.randint(0, len(signals_cluster))
        ax.plot(signals_cluster[id_cluster], label=class_labels[c])
        ax.legend(loc="upper right")
        ax.set_ylabel('[p.u.]')
    ax.set_xlabel('Sample [-]')
    plt.tight_layout()
    plt.show()

# TODO: Move the plot here but first fix the onset detection algorithm so it works depending on the shape of the input
def plot_onset():
    pass