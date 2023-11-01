# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from Q1 import generate_1D_LIF_neurons
from Q2 import generate_signal, get_neurons_spike_response_to_stimulus, \
                get_pos_syn_filt, filter_spikes



if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(189)

    Tref = 2 / 1000 # Converted to seconds
    Trc = 20 / 1000 # Converted to seconds
    
    T = 1
    dt = 0.001
    N_time_samples = int(T / dt)
    rnd_stim_time, rnd_stim, _, _ = generate_signal(T, dt, 1, 5, 200)
    
    N_neurons = 20
    radius = 2
    
    # 3A) [DONE?]
    neuron_counts = [8, 16, 32, 64, 128, 256]
    run_len = 5
    avgd_RMSEs = []
    _, h = get_pos_syn_filt(T, N_time_samples)
    
    for N_neurons in neuron_counts:
        
        RMSEs = []
        
        for _ in range(run_len):
        
            # Change up the group of neurons each iteration
            neurons = generate_1D_LIF_neurons( 
                Tref, 
                Trc, 
                N_neurons,
                radius
            )
            
            # Some neurons do not spike at all!! This can cause matrix inversion
            # problems with A * A.T during decoder calculation
            spikes = get_neurons_spike_response_to_stimulus(neurons, rnd_stim, dt)
            
            A = filter_spikes(spikes, h)
            
            ro = 0.00000000001 * 200
            normalizer = N_neurons * ro * ro * np.eye(N_neurons)
            decoders = np.linalg.inv(A * A.T + normalizer) * A * np.matrix(rnd_stim).T 
            decoders = decoders.T
            
            reconstructed_stim = (decoders * A).T

            rmse = np.sqrt(np.mean(np.square(reconstructed_stim.T - rnd_stim)))
            RMSEs.append(rmse)
            
        avgd_RMSEs.append(sum(RMSEs) / run_len)
        print("Avgd RMSE for " + str(N_neurons) + " = " + str(avgd_RMSEs[-1]))

    plt.loglog(neuron_counts, avgd_RMSEs)
    plt.xlabel("Neuron Count")
    plt.ylabel("RMSE")
    plt.grid()
    plt.title("Log-Log plot of RMSE vs Neuron Count")


    # 3B) [DONE]
    
    discussion_3B = """
        It is observed that as the number of neurons increases, the RMSE decreases.
        On a log-log plot, the behaviour is observed to be linear, suggesting some
        kind of exponential relationship between the two variables, up until the
        observed inflection point.
    """










