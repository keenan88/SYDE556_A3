# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from math import exp
from Q1 import G_LIF_at_x, generate_LIF_tuning_curves
from Q2 import generate_signal

def generate_spiking_output(encoders, alphas, J_biases, stimulus):
    
    spikes = np.matrix(np.zeros((len(encoders), len(stimulus))))
    Vth = 1
    
    neuron_idx = 0
    for encoder, alpha, J_bias in zip(encoders, alphas, J_biases):
        v = 0
        time_idx = 0
        
        while time_idx < len(stimulus):
            if v >= Vth:
                v = 0
                spikes[int(neuron_idx), int(time_idx)] = 1
                time_idx += Tref * 1000 # Scaled to ms, since that is our step size here
                
            else:
                time_idx += 1
                
            if time_idx < len(stimulus):
                J = alpha * np.dot(encoder, stimulus[int(time_idx)]) + J_bias
                v += dt * (J - v) / Trc
                
        neuron_idx += 1
        
    return spikes
    
def decode_spiky_output(A, N_neurons, stimulus):
    
    ro = 0.0001 * 200
    normalizer = N_neurons * ro * ro * np.eye(N_neurons)
    decoders = np.linalg.inv(A * A.T + normalizer) * A * np.matrix(stimulus).T 
    decoders = decoders.T
    
    x_hat = (decoders * A).T
    
    return x_hat, decoders

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
    range_of_stims = np.linspace(-1, 1, N_time_samples)
    r = 1
    
    # Define encoders
    _, alphas_x, J_biases_x, encoders_x = \
    generate_LIF_tuning_curves(range_of_stims, 
                               Tref, 
                               Trc, 
                               N_neurons,
                               r)
    stimulus_x = 2 * time + 1
    x_spiking_output = generate_spiking_output(encoders_x, alphas_x, J_biases_x, stimulus_x)
    
    # Define decoders
    x_hat, decoders_x = decode_spiky_output(x_spiking_output, N_neurons, stimulus_x)
    
    # Define encoders
    _, alphas_y, J_biases_y, encoders_y = \
    generate_LIF_tuning_curves(range_of_stims, 
                               Tref, 
                               Trc, 
                               N_neurons,
                               r)
    stimulus_y = time
    y_spiking_output = generate_spiking_output(encoders_y, alphas_y, J_biases_y, stimulus_y)
    
    # Define decoders
    y_hat, decoders_y = decode_spiky_output(y_spiking_output, N_neurons, stimulus_y)
    
    # Validation. Looks good, x_hat is roughly decoding x, y_hay is roughly decoding y.
    plot_decoded_spiking_output(x_hat, stimulus_x, time, "f(x) = 2x + 1")
    plot_decoded_spiking_output(y_hat, stimulus_y, time, "f(y) = y")
    
    # 4A)
    
    stimulus_x_A = time - 1
    x_spiking_output_A = generate_spiking_output(encoders_x, alphas_x, J_biases_x, stimulus_x_A) # Resultant spike train from input t-1
    # Why are we using the 2x + 1 decoder??
    x_hat_A = (decoders_x * x_spiking_output_A).T # Best attempt at getting spikes back to state space, using original 2x+1 decoder
    
    
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











