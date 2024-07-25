from time import process_time
import numpy as np
import pickle


def onset_detection_fun(signal_ar, #save=False,
                        window_length=500, window_step=5, sample_start_search=500, sample_end_search=2000,
                        onset_threshold=0.1, number_extracted_samples=625, energy_gap=250):
    """

    Onset detection function with SNR-based picking.
    Return value can be modified to provide sample index of the onset.
    Default parameters optimized for case with 5120 samples per signal.

    :param signal_ar: numpy array; each signal occupies one row;
    :param case: string; case name to be used as a save file name;

    :return: numpy array; shortened signals of a length equal to: number_extracted_samples.
        This array can be also saved in './features'.

    """

    print('Extracting onset: ')
    ### Pulse onset detection
    # Define an empty list to store the extracted signals and the corresponding onset indexes
    t_start = process_time()
    signal_extracted = []
    onset_picked = []

    # Cycle over all the available events
    for x in range(len(signal_ar)):
        # Extract a signal at every iteration
        signal = signal_ar[x].astype(np.int64)
        # Extract the pulse samples in the selected interval for onset detection (between sample_start_search and sample_end_search)
        signal_range = signal[sample_start_search:sample_end_search]
        # Extract the pulse samples in an interval which is required for HOS moments calculation
        signal_range_hos = signal[(sample_start_search - window_length + 1):sample_end_search]
        # Define empty lists storing, for each event, the statistic moments values computed on all the the time windows
        s1 = []
        s2 = []
        s6 = []
        # High order statistic moments calculation for each time window
        for xx in range(int(len(signal_range) / window_step)):
            # Compute the time window limits as function of the iterative step
            begin_window = int(xx * window_step)
            end_window = int(xx * window_step + window_length)
            # Extract the waveform contained in the moving windows for a specific iterative step
            wave = signal_range_hos[begin_window:end_window]
            # First statistic moment (mean) calculation
            s1.append(np.mean(wave))
            # Second statistic moment (variance) calculation
            s2.append(np.sum((wave - s1[xx]) ** 2) / (window_length - 1))
            # Sixth statistic (S6) moment calculation
            s6.append(np.sum((wave - s1[xx]) ** 6) / ((window_length - 1) * (s2[xx] ** 3)) - 15)
        # Calculation of the derivative of the sixth statistic moment (dS6)
        ds6 = np.gradient(s6)
        # Compute the threshold for pulse onset detection (equal to a given percentage of the maximum value of ds6)
        threshold = onset_threshold * np.nanmax(ds6)
        # Identify all the samples that are above the threshold
        onset_index_ds6 = np.where(abs(ds6) >= threshold)[0].astype(np.int16)
        # Define an empty array for storage of the sign of each identified sample
        onset_sign = np.empty(len(onset_index_ds6))
        # Iterate over the identified samples to find crossings
        # For crossing the product of (sample_value_n - threshold)*(sample_value_(n-1) - threshold) is negative
        for xx in range(len(onset_index_ds6)):
            onset_sign[xx] = (ds6[onset_index_ds6[xx]] - threshold) * (ds6[onset_index_ds6[xx] - 1] - threshold)
        # For crossing the sample is passed, else the 0 value is passed (or the sample_start_search value)
        onset_indices = (np.where(np.sign(onset_sign) == -1, onset_index_ds6, 0) * window_step) + sample_start_search
        # Indexes with value of 0 (or the sample_start_search value) are cut
        onset_indices = onset_indices[np.where(onset_indices > sample_start_search)]
        # Definition of the empty SNR list
        onset_ratios = []
        # Iterate over the crossings to calculate the SNR of each crossing
        for xx in range(len(onset_indices)):
            potential_onset = onset_indices[xx]
            pre_onset = np.sum(signal[(potential_onset - energy_gap): potential_onset] ** 2)
            post_onset = np.sum(signal[potential_onset: (potential_onset + energy_gap)] ** 2)
            SN_ratio = post_onset / pre_onset
            onset_ratios.append(SN_ratio)
        # The crossing with the highest SNR is identified as the onset
        onset_index = onset_indices[np.argmax(np.array(onset_ratios))]

        # Define the number of samples to be extracted after pulse onset (set in code options)
        num_samples = number_extracted_samples
        # Extraction of first num_samples samples after pulse onset
        waveform_from_onset = signal[(onset_index):(onset_index + num_samples)]
        # Add the extracted waveform and the onset index to the previously defined lists
        signal_extracted.append(waveform_from_onset)
        onset_picked.append(onset_index)

    t_stop = process_time()
    print("Elapsed onset detection time equal to: " + str(t_stop - t_start))

    # if save:    # If save is True the feature file is saved
    #     with open('./features/onset_' + case, 'wb') as file:
    #         pickle.dump(signal_extracted, file, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('Features saved.')

    return signal_extracted, onset_picked, ds6


class OnsetDetection():
    def __init__(self, window_length=500, out_length=625, window_step=5, start_search=None, end_search=None, threshold=0.1, SNR=False, energy_gap=None) -> None:
        
        self.out_length = out_length
        self.window_length = window_length

        self.threshold = threshold

        self.window_step = window_step
        self.search_start = start_search
        self.search_end = end_search
        
        # Fill the params if not set
        if self.search_start is None:
            self.search_start = window_length
        if self.search_end is None:
            self.search_end = self.search_start * 3

        # Define the search range
        self.search_range = range(self.search_start, self.search_end, self.window_step)    

        # SNR picking
        self.SNR = SNR
        if SNR: 
            if energy_gap is None: raise ValueError(f"With SNR picking energy gap cannot be set to {energy_gap}")
            else: self.energy_gap = energy_gap


    def _check_length(self, x):
        self.input_length = len(x)
        
        if self.input_length <= self.out_length:
            raise ValueError(f'Input length cannot be smaller then output length and is: \
                             {self.input_length} and {self.out_length}')
        
        if self.input_length <= self.window_length:
            raise ValueError(f'Input length cannot be smaller then the selected window size and is: \
                             {self.input_length} and {self.window_length}')
        
        if self.input_length < self.search_end:
            raise ValueError(f'Input length cannot be smaller then the searched area and is: \
                             {self.input_length} and {self.search_end}')
        
        if self.input_length < self.search_end+self.out_length:
            raise ValueError(f'Input length cannot be smaller sum of the searched area and out vector and is: \
                             {self.input_length} and {self.search_end+self.out_length}')

        return

    def _calculate_moment_6(self, x, moment_1, moment_2):
        
        # Prealocation
        moment_6 = []

        # Loop through the windows in search range
        for n, i in enumerate(self.search_range): 

            # Get the waveform piece of the window
            x_i = x[i-self.window_length:i]

            # Calculate the fraction nom and denom
            nom = np.sum((x_i - moment_1[n]) ** 6)
            denom = ((self.window_length - 1) * (moment_2[n] ** 3))

            if denom == 0:
                print('Zero division, replaced denom with epsilon')
                denom += 0.001


            moment_6.append(nom/denom - 15)

        return moment_6
    

    def _first_positive_index(self, x, threshold=0):
        """Finds the first index above threshold"""

        # Find where the values are positive
        positive_indices = np.where(x > threshold)[0]
        
        # Check if there is any positive value (rahter not possible with the threshold definition)
        if positive_indices.size == 0:
            return None 
        
        # Return the index of the first positive value
        return positive_indices[0]

    def __call__(self, x):
        
        # Make sure the input is an array
        x = np.asarray(x).astype(int)

        # Check if the length is correct
        self._check_length(x)

        # Calculate statistical moments based on waveform pieces
        moment_1 = [np.mean(x[i-self.window_length:i], dtype=float) for i in self.search_range]
        moment_2 = [np.var(x[i-self.window_length:i], dtype=float) for i in self.search_range]
        moment_6 = self._calculate_moment_6(x, moment_1, moment_2)

        # Get the change rate (absolute)
        d_moment_6 = np.abs(np.gradient(moment_6))

        # Get threshold
        threshold = self.threshold * np.nanmax(d_moment_6)

        # Find the first index above threshold and adapt it to the search area
        onset_index = self._first_positive_index(d_moment_6, threshold)
        onset_index = onset_index*self.window_step + self.search_start

        # Short waveform
        out = x[onset_index:onset_index+self.out_length].astype(int)

        # Signal to noise ratio picking
        if self.SNR:
            pass # TODO: finish the SNR picking


        return onset_index, out


if __name__ == 'main':
    print('This is not the main function! hehe')
