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
    
def decode_spiky_output(A, stimulus):
    
    decoders = np.linalg.pinv(A * A.T) * A * np.matrix(stimulus).T 
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
    
    np.random.seed(189)

    Tref = 2 / 1000 # Converted to seconds
    Trc = 20 / 1000 # Converted to seconds
    N_neurons = 200
    r = 1
    
    neuronsX = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    neuronsY = generate_1D_LIF_neurons(Tref, Trc, N_neurons, r)
    
    T = 1
    dt = 0.001
    N_time_samples = int(T / dt)
    time = np.linspace(0, T, N_time_samples)
    
    _, h = get_pos_syn_filt(T, N_time_samples)
    
    # Define encoders
    stimulus_x = 2 * time + 1
    x_spike_response = get_neurons_spike_response_to_stimulus(neuronsX, stimulus_x, dt)
    x_spike_response = filter_spikes(x_spike_response, h)
    
    stimulus_y = time
    y_spike_response = get_neurons_spike_response_to_stimulus(neuronsY, stimulus_y, dt)
    y_spike_response = filter_spikes(y_spike_response, h)
    
    # Define decoders
    decoders_x, reconstructed_x = decode_spiky_output(x_spike_response, stimulus_x)
    decoders_y, reconstructed_y = decode_spiky_output(y_spike_response, stimulus_y)

    plt.plot(time, reconstructed_x, label="Reconstructed X", color='blue')
    plt.plot(time, stimulus_x, label = "Original X", color='black')    
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.plot(time, reconstructed_y, label="Reconstructed y", color='blue')
    plt.plot(time, stimulus_y, label = "Original Y", color='black')    
    plt.grid()
    plt.legend()
    plt.show()
    
    # 4A)
    
    # Apply decoders
    stimulus_x = time - 1
    x_spike_response = get_neurons_spike_response_to_stimulus(neuronsX, stimulus_x, dt)
    x_spike_response = filter_spikes(x_spike_response, h)
    
    decoded_spiking_output = (decoders_x * x_spike_response).T 
    
    plt.plot(time, decoded_spiking_output, label="Reconstructed X", color='blue')
    plt.plot(time, stimulus_x, label = "Original X", color='black')    
    plt.grid()
    plt.show()
    
    
    """
    stimulus_y_A = x_hat_A # Feed-forward best effort at t-1
    y_spiking_output = generate_spiking_output(encoders_y, alphas_y, J_biases_y, stimulus_y_A) # Generate spikes from best effort at t-1
    y_hat_A = (decoders_y * y_spiking_output).T # Decode spikes, should hopefully look like t-1 
    
    
    plot_decoded_spiking_output(x_hat_A, stimulus_x_A, time, "x(t) = t - 1")
    plot_decoded_spiking_output(y_hat_A, stimulus_x_A, time, "x(t) = t - 1")
    
    
    # 4B)
     
    stimulus_x_B = np.zeros((len(time), 1))
    
    for i in range(10):
        low = int(i * N_time_samples/10)
        high = int((i+1) * N_time_samples/10)
        rand_float = np.random.uniform(-1, 0)
        stimulus_x_B[low : high] = rand_float
    
    x_spiking_output_B = generate_spiking_output(encoders_x, alphas_x, J_biases_x, stimulus_x_B)
    x_hat_B = (decoders_x * x_spiking_output_B).T
    
    stimulus_y_B = x_hat_B
    y_spiking_output_B = generate_spiking_output(encoders_y, alphas_y, J_biases_y, stimulus_y_B)
    y_hat_B = (decoders_y * y_spiking_output_B).T
    
    plot_decoded_spiking_output(x_hat_B, stimulus_x_B, time, "Input = random step input")
    plot_decoded_spiking_output(y_hat_B, stimulus_x_B, time, "Input = random step input")
    
    
    #4C)
    
    stimulus_x_C = 0.2 * np.sin(6 * np.pi * time)
    x_spiking_output_C = generate_spiking_output(encoders_x, alphas_x, J_biases_x, stimulus_x_C)
    x_hat_C = (decoders_x * x_spiking_output_C).T
    
    stimulus_y_C = x_hat_C
    y_spiking_output_C = generate_spiking_output(encoders_y, alphas_y, J_biases_y, stimulus_y_C)
    y_hat_C = (decoders_y * y_spiking_output_C).T

    plot_decoded_spiking_output(x_hat_C, stimulus_x_C, time, "f(t) = 0.2 * sin(2*pi*t)")    
    plot_decoded_spiking_output(y_hat_C, stimulus_x_C, time, "f(y) = 0.2 * sin(2*pi*t)")

    """









