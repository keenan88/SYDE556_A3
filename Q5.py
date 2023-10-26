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
    
    j = 0
    for encoder, alpha, J_bias in zip(encoders, alphas, J_biases):
        v = 0
        i = 0
        
        while i < len(stimulus):
            if v >= Vth:
                v = 0
                spikes[int(j), int(i)] = 1
                i += Tref * 1000 # Scaled to ms, since that is our step size here
                
            else:
                i += 1
                
            if i < len(stimulus):
                J = alpha * np.dot(encoder, stimulus[int(i)]) + J_bias
                v += dt * (J - v) / Trc
                
        j += 1
        
    return spikes
    
def decode_spiky_output(A, N_neurons, stimulus):
    
    ro = 0.0001 * 200
    normalizer = N_neurons * ro * ro * np.eye(N_neurons)
    decoders = np.linalg.inv(A * A.T + normalizer) * A * np.matrix(stimulus).T 
    decoders = decoders.T
    
    x_hat = (decoders * A).T
    
    return x_hat, decoders

def plot_decoded_spiking_output(x_hat, stimulus_x, time):
    plt.plot(time, x_hat, label="Reconstructed Stimulus")    
    plt.plot(time, stimulus_x, label="Real Stimulus")
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Stimulus")
    plt.legend()
    plt.title("Real vs Reconstructed Stimulus")
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
    
    stimulus_x =  0.5 * time
    x_spiking_output = generate_spiking_output(encoders_x, alphas_x, J_biases_x, stimulus_x)
    
    # Define decoders
    x_hat, decoders_x = decode_spiky_output(x_spiking_output, N_neurons, stimulus_x)
    plot_decoded_spiking_output(x_hat, stimulus_x, time)
    
    # Define Encoders
    _, alphas_y, J_biases_y, encoders_y = \
    generate_LIF_tuning_curves(range_of_stims, 
                               Tref, 
                               Trc, 
                               N_neurons,
                               r)
    
    stimulus_y = 2 * time
    y_spiking_output = generate_spiking_output(encoders_y, alphas_y, J_biases_y, stimulus_y)
    
    # Define Decoders
    y_hat, decoders_y = decode_spiky_output(y_spiking_output, N_neurons, stimulus_y)
    plot_decoded_spiking_output(y_hat, stimulus_y, time)
    
    # Define encoders
    tuning_curves_z, alphas_z, J_biases_z, encoders_z = \
    generate_LIF_tuning_curves(range_of_stims, 
                               Tref, 
                               Trc, 
                               N_neurons,
                               r)
    
    # Define decoders
    # Encoder for third neuron not specified, assumed 1 to 1 output
    stimulus_z = time
    z_spiking_output = generate_spiking_output(encoders_z, alphas_z, J_biases_z, stimulus_z)
    z_hat, decoders_z = decode_spiky_output(z_spiking_output, N_neurons, stimulus_z)
    plot_decoded_spiking_output(z_hat, stimulus_z, time)
    
    
    # 5A)
    
    stimulus_x_A = np.cos(3 * np.pi * time)
    x_spiking_output_A = generate_spiking_output(encoders_x, alphas_x, J_biases_x, stimulus_x_A)
    x_hat_A = (decoders_x * x_spiking_output_A).T
    
    stimulus_y_A = 0.5 * np.sin(2 * np.pi * time)
    y_spiking_output = generate_spiking_output(encoders_y, alphas_y, J_biases_y, stimulus_y_A)
    y_hat_A = (decoders_y * y_spiking_output).T
    
    summed_output = 2 * y_hat_A + 0.5 * x_hat_A
    z_spiking_output = generate_spiking_output(encoders_z, alphas_z, J_biases_z, summed_output)
    z_hat = (decoders_z * z_spiking_output).T
    z = 2 * stimulus_y_A + 0.5 * stimulus_x_A
    
    plot_decoded_spiking_output(x_hat_A, stimulus_x_A, time)
    plot_decoded_spiking_output(y_hat_A, stimulus_y_A, time)
    plot_decoded_spiking_output(z_hat, z , time)
    
    
    
    
    # 5B)
    
    _, stimulus_x_A, _, _ = generate_signal(T, dt, 1, 8, 18976)
    x_spiking_output_A = generate_spiking_output(encoders_x, alphas_x, J_biases_x, stimulus_x_A)
    x_hat_A = (decoders_x * x_spiking_output_A).T
    
    _, stimulus_y_A, _, _= generate_signal(T, dt, 0.5, 5, 111)
    y_spiking_output = generate_spiking_output(encoders_y, alphas_y, J_biases_y, stimulus_y_A)
    y_hat_A = (decoders_y * y_spiking_output).T
    
    z = 2 * stimulus_y_A + 0.5 * stimulus_x_A
    summed_output = 2 * y_hat_A + 0.5 * x_hat_A
    z_spiking_output = generate_spiking_output(encoders_z, alphas_z, J_biases_z, summed_output)

    z_hat = (decoders_z * z_spiking_output).T
    
    plot_decoded_spiking_output(x_hat_A, stimulus_x_A, time)
    plot_decoded_spiking_output(y_hat_A, stimulus_y_A, time)
    plot_decoded_spiking_output(z_hat, z , time)
        
    













