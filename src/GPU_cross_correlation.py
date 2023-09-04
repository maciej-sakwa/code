# -*- coding: utf-8 -*-
"""
Created in 2020

@author: GEIRI_EU_Kai Gu

"""

import itertools
import numpy as np
import scipy.fftpack
import scipy.signal as sp_sig
import torch
from ivi_trace import TraceYT
from torch.multiprocessing import set_start_method
from threading import Event
import time
import matplotlib.pyplot as plt

try:
     set_start_method('spawn')
except RuntimeError:
    pass

def complex_multiplication_nd(t1, t2):
    # complex multiplication on nd tensors
    result = t1.clone()
    real1 = t1[:, :, 0]
    imag1 = t1[:, :, 1]
    real2 = t2[:, :, 0]
    imag2 = t2[:, :, 1]
    result[:, :, 0] = real1 * real2 - imag1 * imag2
    result[:, :, 1] = real1 * imag2 + imag1 * real2
    return result

class GPUTensorCC:

    def __init__(self, signals:list, pre_processing=False, progress_queue= None, use_gpu=True, interrupt = None):
        """
            signals:list of np.ndarray or 2d np.ndarray or list of TranceYT signal
            pre_processing: boolean, whether to pad the signal to same length
            progress_queue: queue interface for realtime evaluation
            use_gpu: allocate the tensor on GPU, otherwise on CPU.
            interrupt:interrupt event for realtime evaluation

        """


        # define the type of list:
        first_signal = next((item for item in signals if item is not None), None)
        # get the detrended signal list
        if isinstance(first_signal, TraceYT):
            self.signals = list(map(lambda y: scipy.signal.detrend(y.y_raw) if y is not None else None, signals))
        else:
            self.signals = list(map(lambda y: scipy.signal.detrend(y) if y is not None else None, signals))
            # pre-process the signal of variable length
        if pre_processing:
            self.pre_processing()
        self.signal_length = len(self.signals[0])
        #if (torch.cuda.device_count()>0) and (torch.cuda.is_available()):
        #    self.use_gpu = use_gpu
        #else:
        self.use_gpu = False

        self.cpu_memory = 32*1024
        self.gpu_memory = 4*1024
        self.progress_queue = progress_queue

        if interrupt is not None:
            self._interrupt = interrupt
        else:
            self._interrupt = Event()


    def pre_processing(self):
        # make the signals of the same length
        len_list = list(map(lambda y: len(y) if y is not None else 0, self.signals))
        max_len = np.max(len_list)

        padded_signals = list(
            map(lambda y: np.pad(y, (0, max_len - len(y)), 'constant') if y is not None else None, self.signals))

        self.signals = padded_signals

    def get_result(self, get_negative=False, samples_shift=None):
        """"
        get_negative:boolean, whether get the min of cc function and its lag,
                    if True, return 4 matrices as cc_matrix, lag_matrix, cc_min_matrix, lag_min_matrix
                    if False, return 2 matrices cc_matrix, lag_matrix
        samples_shift: int, how many samples will be shifted during the cross-correlation,
                    when None, a full cross-correlation is performed
        """

        t1 = time.perf_counter()
        if samples_shift is None:
            padding_length = len(self.signals[0])
        else:
            if samples_shift<0 or samples_shift>len(self.signals[0]):
                padding_length = len(self.signals[0])
            else:
                padding_length = int(samples_shift)

        if (len(self.signals[0])+ padding_length)%2 >0:
            padding_length+=1

        if self.use_gpu:
            self.update_log('cc using gpu')
        else:
            self.update_log('cc using cpu')
        self.update_progress('cc_preprocess',0)
        self.signals = np.array(self.signals)
        padded_array = np.pad(self.signals, [[0,0],[0,padding_length]],'constant')

        if self.use_gpu:
            pulse_tensor = torch.cuda.FloatTensor(self.signals)
            padded_tensor = torch.cuda.FloatTensor(padded_array)
        else:
            pulse_tensor = torch.FloatTensor(self.signals)
            padded_tensor = torch.FloatTensor(padded_array)

        self.update_progress('cc_preprocess',0.5)
        # preparing tensors on GPU
        # norm tensor of signals
        norms = torch.norm(pulse_tensor,dim=1)
        # fft and conj of fft tensors
        fft_tensor = torch.rfft(padded_tensor,1,False,True)
        fft_conj_tensor = fft_tensor.clone()
        fft_conj_tensor[:,:,1] = fft_conj_tensor[:,:,1]*-1
        #indices combinations
        ind = list(range(len(self.signals)))
        ind_combinations = list(zip(*list(itertools.combinations(ind, 2))))

        ind_0 = np.array(ind_combinations[0])
        ind_1 = np.array(ind_combinations[1])
        #TODO memory estimation
        # N_samples = fft_tensor.size()[0]
        # n_features = fft_tensor.size()[1]
        # memory_usage_single_sample = N_samples*32/((1024**2)*8)

        # length of segments that are processed each time
        seg = 1000
        i = 0
        loops = len(ind_0)//seg +1
        if self.use_gpu:
            cc_tensor = torch.cuda.FloatTensor(np.zeros((len(ind_0),)))
        else:
            cc_tensor = torch.FloatTensor(np.zeros((len(ind_0),)))
        lag_tensor = cc_tensor.clone()

        if get_negative:
            cc_min_tensor = cc_tensor.clone()
            lag_min_tensor = lag_tensor.clone()

        self.update_progress('cc_preprocess',1)
        self.update_progress('cc',0)
        while (i * seg < len(ind_0)):

            if self._interrupt.is_set():
                return None

            ii0 = ind_0[i * seg:i * seg + seg]
            ii1 = ind_1[i * seg:i * seg + seg]

            # get the slice of selected index combinations
            fft_slice = fft_tensor[ii0]
            conj_slice = fft_conj_tensor[ii1]

            # do the complex multiplication in freq-domain
            nd_mult = complex_multiplication_nd(fft_slice, conj_slice)
            # tt3 = time.perf_counter()
            # norm_slice = norms[ii0] * norms[ii1]
            # tt4 = time.perf_counter()

            # get the inverse fft of freq-cc function
            new = torch.irfft(nd_mult, signal_ndim=1,
                              signal_sizes=[len(self.signals[0])+ padding_length, ])



            # instead of rolling the array to reconstruct the cc function
            # do the index operation for lag_array later
            # new = torch.roll(new, int(len(new[0]) / 2), 1)[:,1:]
            maximum, max_ind = torch.max(new, dim=1)
            minimum, min_ind = torch.min(new, dim=1)
            cc_tensor[i * seg:i * seg + seg] = maximum
            lag_tensor[i * seg:i * seg + seg] = max_ind

            if get_negative:
                cc_min_tensor[i * seg:i * seg + seg] = minimum
                lag_min_tensor[i * seg:i * seg + seg] = min_ind

            tt4 = time.perf_counter()
            # del fft_slice,conj_slice,norm_slice
            i += 1
            self.update_progress('cc', i / loops)

        cc_tensor = cc_tensor / (norms[ind_0] * norms[ind_1])
        if get_negative:
            cc_min_tensor = cc_min_tensor/(norms[ind_0]*norms[ind_1])

        if self.use_gpu:
            array_cc = cc_tensor.cpu().numpy()
            array_lag = lag_tensor.cpu().numpy()
            if get_negative:
                array_cc_min = cc_min_tensor.cpu().numpy()
                array_lag_min = lag_min_tensor.cpu().numpy()
        else:
            array_cc = cc_tensor.numpy()
            array_lag = lag_tensor.numpy()
            if get_negative:
                array_cc_min = cc_min_tensor.numpy()
                array_lag_min = lag_min_tensor.numpy()


        cc_matrix = np.ones([len(self.signals), len(self.signals)])
        lag_matrix = np.zeros([len(self.signals), len(self.signals)])
        # assign ccm values to the cc_matrix
        cc_matrix[ind_0,ind_1] = array_cc
        cc_matrix[ind_1,ind_0] = array_cc
        # recover the index of lags
        # peak of the inverse fft of cc-function will be on at the beginning and end of the array
        # index operation needed to get the correct lags
        array_lag[np.where(array_lag>self.signal_length)[0]] -= (self.signal_length+padding_length)
        array_lag[np.where(array_lag < -self.signal_length)[0]] += (self.signal_length+padding_length)
        lag_matrix[ind_0,ind_1] = array_lag
        lag_matrix[ind_1,ind_0] = -array_lag

        if get_negative:
            cc_min_matrix = np.ones([len(self.signals), len(self.signals)])
            lag_min_matrix = np.zeros([len(self.signals), len(self.signals)])

            cc_min_matrix[ind_0, ind_1] = array_cc_min
            cc_min_matrix[ind_1, ind_0] = array_cc_min
            array_lag_min[np.where(array_lag_min > self.signal_length)[0]] -= (self.signal_length + padding_length)
            array_lag_min[np.where(array_lag_min < -self.signal_length)[0]] += (self.signal_length + padding_length)
            lag_min_matrix[ind_0, ind_1] = array_lag_min
            lag_min_matrix[ind_1, ind_0] = -array_lag_min


            t2 = time.perf_counter()
            print(f'{t2-t1}s cc {len(self.signals)} pulses')

            return cc_matrix, lag_matrix, cc_min_matrix, lag_min_matrix

        return cc_matrix,lag_matrix

    def update_progress(self, job, progress):
        job = '..' +job
        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait(('progress',(job,progress)))
            except:
                pass
        else:
            print(job, progress)

    def update_log(self, log):
        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait(('log',log))
            except:
                pass
        else:
            print(log)

if __name__ == '__main__':
    from mc_database.sql_loader import SQLLoader
    # import matplotlib.pyplot as plt
    from mc_database.multi_process_cc_freq import MultiProcessCC_Freq
    import scipy.signal

    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    database = 'china_20190117_thursday_xuyi'
    table = 'thursday_xuyi_20190117_102442_th30'
    database = 'beijing_tegaoyajidi_2019_07'
    table = 'mc20190724_092145_test_workstation_gap_200pc'
    # '10.200.128.112'
    loader1 = SQLLoader(database, table,)
    data_lst = loader1.get_data_by_range(range(0, 500))

    def get_signals(row):
        return row[4].y

    signal_lst = list(map(get_signals, data_lst))


    t1 = time.perf_counter()


    gpu_cc = GPUTensorCC(signal_lst)
    cc1, lag1 = gpu_cc.get_result()

    #
    # t2 = time.perf_counter()
    # print(t2-t1)
    # #
    # t3 = time.perf_counter()
    # multi_cc = MultiProcessCC_Freq(8, signal_lst, indicator=False)
    # t4 = time.perf_counter()
    # cc, lag = multi_cc.get_result()
    #
    # print(np.allclose(cc1,cc))
    # print(np.allclose(lag1, lag))
    # print(np.where(np.abs(lag1-lag)>0))
    # # print(np.max(np.abs(cc1-cc)))
