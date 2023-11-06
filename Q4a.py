# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
try:
    from IPython import get_ipython
except:
    pass
import matplotlib.pyplot as plt

from Q1 import generate_1D_LIF_neurons
from Q2 import get_neurons_spike_response_to_stimulus, filter_spikes, get_pos_syn_filt
    
def decode_spiky_output(A, stimulus, desired_output):
    
    decoders = np.linalg.pinv(A * A.T) * A * np.matrix(desired_output).T 
    decoders = decoders.T
    
    reconstructed_stim = (decoders * A).T
    
    return decoders, reconstructed_stim


def plot_decoded_spiking_output(x_hat, stimulus_x, time, title):
    plt.plot(time, x_hat, label="Reconstructed Stimulus")    
    plt.plot(time, stimulus_x, label="Real Stimulus")
    plt.grid()
    plt.xlabel("Stimulus")
    plt.ylabel("Response")
    plt.legend()
    plt.title(title + ": Real vs Reconstructed Stimulus")
    plt.show()

if __name__ == "__main__":
    
    try:
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass
    
    np.random.seed(18945)

    Tref = 2 / 1000 # Converted to seconds
    Trc = 20 / 1000 # Converted to seconds
    N_neurons = 200
    r = 1
    
    neuronsX = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    neuronsY = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    
    T = 1
    dt = 0.001
    Tau_filter = 10/1000
    N_time_samples = int(T / dt)
    time = np.linspace(0, T, N_time_samples)
    
    _, h = get_pos_syn_filt(T, N_time_samples, Tau_filter)
    
    # Define encoders
    stimulus_x = time
    desired_response_x = 2 * stimulus_x + 1
    x_spike_response = get_neurons_spike_response_to_stimulus(neuronsX, stimulus_x, dt)
    x_spike_response = filter_spikes(x_spike_response, h)
    
    stimulus_y = time
    desired_response_y = stimulus_y
    y_spike_response = get_neurons_spike_response_to_stimulus(neuronsY, stimulus_y, dt)
    y_spike_response = filter_spikes(y_spike_response, h)
    
    # Define decoders
    decoders_x, reconstructed_x = decode_spiky_output(x_spike_response, stimulus_x, desired_response_x)
    decoders_y, reconstructed_y = decode_spiky_output(y_spike_response, stimulus_y, desired_response_y)

    """    
    plt.plot(time, reconstructed_x, label="Actual Output", color='blue')
    plt.plot(time, stimulus_x, label = "Stimulus", color='black') 
    plt.plot(time, desired_response_x, label = "Desired Output", color='red', linestyle='dashed') 
    plt.grid()
    plt.legend()
    plt.title("X Encoding: Stimulus, desired output, and Actual output")
    plt.show()
    
    plt.plot(time, reconstructed_y, label="Reconstructed y", color='blue')
    plt.plot(time, stimulus_y, label = "Original Y", color='black')    
    plt.plot(time, desired_response_y, label = "Desired Output", color='red', linestyle='dashed') 
    plt.grid()
    plt.legend()
    plt.title("Y Encoding: Stimulus, desired output, and Actual output")
    plt.show()
    """
    
    
    # 4A)
    
    
    # Apply decoders
    stimulus_x = time - 1
    desired_response_x = 2 * stimulus_x + 1
    x_spike_response = get_neurons_spike_response_to_stimulus(neuronsX, stimulus_x, dt)
    x_spike_response = filter_spikes(x_spike_response, h)
    decoded_spiking_output_x = (decoders_x * x_spike_response).T 
    
    stimulus_y = decoded_spiking_output_x
    desired_response_y = stimulus_y
    y_spike_response = get_neurons_spike_response_to_stimulus(neuronsY, stimulus_y, dt) 
    y_spike_response = filter_spikes(y_spike_response, h)
    decoded_spiking_output_y = (decoders_y * y_spike_response).T 
    
    final_desired_response = desired_response_x # Since desired response in y is just standard rep. decoder
    
    """
    plt.plot(time, decoded_spiking_output_x, label="Actual Output", color='blue')
    plt.plot(time, stimulus_x, label = "Stimulus", color='black')    
    plt.plot(time, desired_response_x, label = "Desired Output", color='red', linestyle='dashed')    
    plt.title("X Decoding: Stimulus, desired output, actual output")
    plt.legend()
    plt.grid()
    plt.show()
    """
    
    plt.plot(time, stimulus_x, label="Input Stimulus x(t)")
    plt.plot(time, final_desired_response, label="Desired Final Response") 
    plt.plot(time, decoded_spiking_output_y, label="Actual final response")
    plt.title("Y Decoding: Stimulus, desired output, actual output")
    plt.legend()
    plt.grid()
    plt.show()
    
    









