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
from Q4 import decode_spiky_output

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
    
    N_neurons = 200
    r = 1
    
    # Define encoders
    neuronsX = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    neuronsY = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    neuronsZ = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    
    
    # Define decoders
    stimulusX =  0.5 * time
    x_spiking_output = get_neurons_spike_response_to_stimulus(neuronsX, stimulusX, dt)
    x_spiking_output = np.matrix(x_spiking_output)
    decodersX, reconstructedX, = decode_spiky_output(x_spiking_output, stimulusX)
    
    stimulusY =  2 * time
    y_spiking_output = get_neurons_spike_response_to_stimulus(neuronsY, stimulusY, dt)
    y_spiking_output = np.matrix(y_spiking_output)
    decodersY, reconstructedY, = decode_spiky_output(y_spiking_output, stimulusY)  
    
    stimulusZ =  time
    z_spiking_output = get_neurons_spike_response_to_stimulus(neuronsZ, stimulusZ, dt)
    z_spiking_output = np.matrix(z_spiking_output)
    decodersZ, reconstructedZ, = decode_spiky_output(z_spiking_output, stimulusZ)  
    
    plt.plot(time, reconstructedX, label="Reconstructed X", color='blue')
    plt.plot(time, stimulusX, label = "Original X", color='black')    
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.plot(time, reconstructedY, label="Reconstructed Y", color='blue')
    plt.plot(time, stimulusY, label = "Original Y", color='black')    
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.plot(time, reconstructedZ, label="Reconstructed Z", color='blue')
    plt.plot(time, stimulusZ, label = "Original Z", color='black')    
    plt.grid()
    plt.legend()
    plt.show()
    
    

    # 5A)
    
    _, h = get_pos_syn_filt(T, N_time_samples)
    
    stimulusX = np.cos(3 * np.pi * time)
    x_spiking_output = get_neurons_spike_response_to_stimulus(neuronsX, stimulusX, dt)
    x_spiking_output = np.matrix(x_spiking_output)
    reconstructedX = (decodersX * x_spiking_output).T
    
    stimulusY = 0.5 * np.sin(2 * np.pi * time)
    y_spiking_output = get_neurons_spike_response_to_stimulus(neuronsY, stimulusY, dt)
    reconstructedY = (decodersY * y_spiking_output).T
    
    stimulusZ = 2 * reconstructedY + 0.5 * reconstructedX
    z_spiking_output = get_neurons_spike_response_to_stimulus(neuronsZ, stimulusZ, dt)
    reconstructedZ = (decodersZ * z_spiking_output).T
    
    plt.plot(time, reconstructedX, label="Reconstructed X", color='blue')
    plt.plot(time, stimulusX, label = "Original X", color='black')    
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.plot(time, reconstructedY, label="Reconstructed Y", color='blue')
    plt.plot(time, stimulusY, label = "Original Y", color='black')    
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.plot(time, reconstructedZ, label="Reconstructed Z", color='blue')
    plt.plot(time, stimulusZ, label = "Original Z", color='black')    
    plt.grid()
    plt.legend()
    plt.show()
    
    
    """
    
    
    summed_output = 2 * y_hat_A + 0.5 * x_hat_A
    z_spiking_output = generate_spiking_output(encoders_z, alphas_z, J_biases_z, summed_output)
    z_hat = (decoders_z * z_spiking_output).T
    z = 2 * stimulus_y_A + 0.5 * stimulus_x_A
    
    plot_decoded_spiking_output(x_hat_A, stimulus_x_A, time)
    plot_decoded_spiking_output(y_hat_A, stimulus_y_A, time)
    plot_decoded_spiking_output(z_hat, z , time)
    
    """
    
   
    













