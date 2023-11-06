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
from Q4a import decode_spiky_output

if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(189)

    Tref = 2 / 1000 # Converted to seconds
    Trc = 20 / 1000 # Converted to seconds
    
    T = 1
    dt = 0.001
    N_time_samples = int(T / dt)
    time = np.linspace(0, T, N_time_samples)
    Tau_filter = 10/1000
    _, h = get_pos_syn_filt(T, N_time_samples, Tau_filter)
    
    N_neurons = 200
    r = 1
    
    # Define encoders
    neuronsX = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    neuronsY = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    neuronsZ = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    
    
    # Define encoders
    stimulus_x = time
    desired_response_x = 0.5 * stimulus_x
    x_spike_response = get_neurons_spike_response_to_stimulus(neuronsX, stimulus_x, dt)
    x_spike_response = filter_spikes(x_spike_response, h)
    
    stimulus_y = time 
    desired_response_y = 2 * stimulus_y
    y_spike_response = get_neurons_spike_response_to_stimulus(neuronsY, stimulus_y, dt)
    y_spike_response = filter_spikes(y_spike_response, h)
    
    # The f(z) = z was not given, I am assuming it is intended to be the std. repr. decoder.
    stimulus_z = time 
    desired_response_z = stimulus_z
    z_spike_response = get_neurons_spike_response_to_stimulus(neuronsZ, stimulus_z, dt)
    z_spike_response = filter_spikes(z_spike_response, h)
    
    
    # Define decoders
    decoders_x, reconstructed_x = decode_spiky_output(x_spike_response, stimulus_x, desired_response_x)
    decoders_y, reconstructed_y = decode_spiky_output(y_spike_response, stimulus_y, desired_response_y)
    decoders_z, reconstructed_z = decode_spiky_output(z_spike_response, stimulus_z, desired_response_z)
    

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
    
    plt.plot(time, reconstructed_z, label="Reconstructed z", color='blue')
    plt.plot(time, stimulus_z, label = "Original Z", color='black')    
    plt.plot(time, desired_response_z, label = "Desired Output", color='red', linestyle='dashed') 
    plt.grid()
    plt.legend()
    plt.title("Z Encoding: Stimulus, desired output, and Actual output")
    plt.show()
    """
    
    #5A)
    
    stimulus_x = np.cos(3 * np.pi * time)
    x_spike_response = get_neurons_spike_response_to_stimulus(neuronsX, stimulus_x, dt)
    x_spike_response = filter_spikes(x_spike_response, h)
    decoded_spiking_output_x = (decoders_x * x_spike_response).T 
    
    stimulus_y = 0.5 * np.sin(2 * np.pi * time)
    y_spike_response = get_neurons_spike_response_to_stimulus(neuronsY, stimulus_y, dt) 
    y_spike_response = filter_spikes(y_spike_response, h)
    decoded_spiking_output_y = (decoders_y * y_spike_response).T 
    
    stimulus_z = 2 * decoded_spiking_output_y + 0.5 * decoded_spiking_output_x
    z_spike_response = get_neurons_spike_response_to_stimulus(neuronsZ, stimulus_z, dt) 
    z_spike_response = filter_spikes(z_spike_response, h)
    decoded_spiking_output_z = (decoders_z * z_spike_response).T 
    
    desired_response_y = 2 * stimulus_y
    desired_response_x = 0.5 * stimulus_x
    desired_response_z = 2 * desired_response_y.T + 0.5 * desired_response_x
    desired_response_z = desired_response_z.T
    
    plt.plot(time, decoded_spiking_output_x, label="Actual Output", color='blue')
    plt.plot(time, stimulus_x, label = "Stimulus", color='black')    
    plt.plot(time, desired_response_x, label = "Desired Output", color='red', linestyle='dashed')    
    plt.title("X Decoding: Stimulus, desired output, actual output")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.plot(time, decoded_spiking_output_y, label="Actual Output", color='blue')
    plt.plot(time, stimulus_y, label = "Stimulus", color='black')    
    plt.plot(time, desired_response_y, label = "Desired Output", color='red', linestyle='dashed')    
    plt.title("Y Decoding: Stimulus, desired output, actual output")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    plt.plot(time, decoded_spiking_output_y, label="Actual Output", color='blue')
    plt.plot(time, stimulus_x, label = "Stimulus x", color='black')
    plt.plot(time, stimulus_y, label = "Stimulus y", color='brown')    
    plt.plot(time, desired_response_z, label = "Desired Output", color='red')    
    plt.title("Z Decoding: Stimulus, desired output, actual output")
    plt.legend()
    plt.grid()
    plt.show()
    
    










