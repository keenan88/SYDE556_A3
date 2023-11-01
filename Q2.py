# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from math import exp
from scipy.fft import fftfreq

    
"""
def generate_LIF_tuning_curve(x_linspace, Tref, Trc, alpha, J_bias, encoder):
        
    tuning_curve = []
    
    for x in x_linspace:
        a = G_LIF_at_x(alpha, x, encoder, J_bias, Tref, Trc)
        tuning_curve.append(a)
    
    return tuning_curve
"""

def generate_signal(T, dt, power_desired, limit_hz, seed):
    
    np.random.seed(seed)
    
    limit_rads = limit_hz * 2 * np.pi
    
    N = int(T / dt)
    t = np.linspace(0, T, N, endpoint=False)
    
    X_w = np.zeros((N,), dtype=complex)
    X_w[0] = np.random.normal(0, 1) + 1j*np.random.normal(0, 1) # Set 0 frequency
    xf_rads = fftfreq(N, dt) * 2 * np.pi # Gives frequency at each index

    for freq_idx in range(1, len(X_w)//2):    
        if xf_rads[freq_idx] < limit_rads: # Only generate signals for frequencies that are below the band limit
            signal = np.random.normal(0, 1) + 1j*np.random.normal(0, 1)
            # Each index of X_w represents a frequency to be fed into
            # ifft, in radians/second, NOT hz.
            X_w[freq_idx] = signal
            X_w[-freq_idx] = np.conjugate(signal) # Set the negative frequency too, ifft needs the pos and neg frequency
            
    y = np.real(np.fft.ifft(X_w))
    
    scaling_factor = power_desired / np.sqrt(np.sum(y**2) / N)
    
    y = y * scaling_factor
    
    X_w = X_w * scaling_factor
    
    return t, y, xf_rads, X_w

def get_neurons_spike_response_to_stimulus(neurons, stimulus, dt):
    
    spike_response = np.zeros((len(neurons), len(stimulus)))
    Vth = 1
    
    j = 0
    for neuron in neurons:
        v = 0
        i = 0
        
        while i < len(stimulus):
            if v >= Vth:
                v = 0
                spike_response[int(j), int(i)] = 1
                i += neuron['Tref'] * 1000 # Scaled to ms, since that is our step size here
            else:
                i += 1
                
            if i < len(stimulus):
                dot = np.dot(neuron['encoder'], stimulus[int(i)])
                J = neuron['alpha'] * dot + neuron['J_bias']
                v += dt * (J - v) / neuron['Trc']
                
        j += 1
                
    
    return spike_response

def filter_spikes(spikes, h):
    A = np.zeros(spikes.shape)

    for j in range(A.shape[0]):
        for i in range(A.shape[1]):
            
            if spikes[j][i]:
                
                A[j, i:] += spikes[j][i] * np.array(h[0: len(A[j]) - i])
                
                
    return np.matrix(A)

def get_pos_syn_filt(T, N):
    h = []
    time = np.linspace(0, T, N)
    Tau = 5 / 1000 # ms to s
    for t in time:
       h_t = exp(-t / Tau) / Tau
       h.append(h_t)
       
    h = h / np.sum(h)
    
    return time, h

if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(18945)   
    
    #x = 0.0 
    #a = 44.876
    #alpha = 1.844
    #J_bias = 1.569
    neurons = [
        {'alpha': 1.844, 'J_bias' : 1.569, 'encoder': 1, 'Tref': 0.002, 'Trc': 0.02}, 
        {'alpha': 1.844, 'J_bias' : 1.569, 'encoder': -1, 'Tref': 0.002, 'Trc': 0.02}
    ]
    
    
    T = 1
    dt = 0.001
    N_time_samples = int(T / dt)
    rnd_stim_timescale, rnd_stim, rnd_stim_freq_scale, rnd_stim_comps = generate_signal(T, dt, 1, 5, 18945)
    
    spikes = get_neurons_spike_response_to_stimulus(neurons, rnd_stim, dt)
    
    plt.plot(rnd_stim_timescale, spikes[0], color='blue', label="Positive neuron spikes")
    plt.plot(rnd_stim_timescale, spikes[1], color='green', label="Negative neuron spikes")
    plt.plot(rnd_stim_timescale, rnd_stim, label="Stimulus", color="black")
    plt.grid()
    plt.legend()
    plt.title("2) 2-neuron spiketrain response to random stimulus")
    plt.xlabel("Time (s)")
    plt.ylabel("Stimulus & Voltage spikes")
    plt.show()
    
    # 2A) [DONE]
    filter_time, h = get_pos_syn_filt(T, N_time_samples)
       
    plt.plot(filter_time, h)
    plt.grid()
    plt.title("2A) h(t) filter in time domain")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.show()
    
    # 2B) [DONE]
    A = filter_spikes(spikes, h)
                

    A = np.matrix(A)

    decoders = (np.linalg.inv(A * A.T) * A * np.matrix(rnd_stim).T).T
    
    reconstructed_stim = (decoders * A).T

    plt.plot(rnd_stim_timescale, reconstructed_stim, label="Reconstructed Stimulus")    
    plt.plot(rnd_stim_timescale, rnd_stim, label="Real Stimulus")
    plt.grid()
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Stimulus")
    plt.title("Real vs Reconstructed Stimulus")
    plt.show()
    
    # 2C [DONE]
    rmse = np.sqrt(np.mean(np.square(reconstructed_stim.T - rnd_stim)))
    
    print("RMSE (Reconstructed vs Actual Sim): ", rmse)


    














