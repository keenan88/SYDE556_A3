# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from math import exp
from Q1 import G_LIF_at_x
from scipy.fft import fft, fftfreq

    

def generate_LIF_tuning_curves(x_linspace, Tref, Trc, alpha, J_bias):
    
    tuning_curves = []
    

    for encoder in [-1, 1]:
    
        tuning_curve = []
        
        for x in x_linspace:
            a = G_LIF_at_x(alpha, x, encoder, J_bias, Tref, Trc)
            tuning_curve.append(a)
        
        tuning_curves.append(np.array(tuning_curve))

    return tuning_curves

def generate_signal(T, dt, power_desired, limit_hz, seed):
    
    np.random.seed(seed)
    
    limit_rads = limit_hz * 2 * np.pi
    
    N = int(T / dt)
    x = np.linspace(0, T, N, endpoint=False)
    
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
    
    return x, y, xf_rads, X_w


    

if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(18945)

    Tref = 2 / 1000 # Converted to seconds
    Trc = 20 / 1000 # Converted to seconds
    
    N = 20
    S = 41
    dt = 0.001
    x_linspace = np.linspace(-2, 2, S)
    
    x = 0.0 
    a = 44.876
    alpha = 1.844
    J_bias = 1.569
    
    tuning_curves = generate_LIF_tuning_curves(x_linspace, Tref, Trc, alpha, J_bias)
    
    for tuning_curve in tuning_curves:
        plt.plot(x_linspace, tuning_curve)

    plt.title("1A) " + str(N) + " LIF Tuning Curves")
    plt.xlabel("Current")
    plt.ylabel("Firitng Rate (Hz)")
    plt.xlim([x_linspace[0], x_linspace[-1]])
    plt.grid()
    plt.show() 
    
    T = 1
    dt = 0.001
    N = int(T / dt)
    x, stimulus, xf_rads, X_w = generate_signal(T, dt, 1, 5, 18945)
    
    
    voltages = []
    spikes = []
    Vth = 1
    encoders = [-1, 1]
    
    for encoder in encoders:
        v = 0
        i = 0
        
        spikes.append([])
        voltages.append([])
        
        while i < len(stimulus):
            if v >= Vth:
                v = 0
                i += Tref * 1000 # Scaled to ms, since that is our step size here
                voltages[-1].append(0)
                voltages[-1].append(0)
                spikes[-1].append(1)
                spikes[-1].append(0)
            else:
                voltages[-1].append(v)
                spikes[-1].append(0)
                i += 1
                
            if i < len(stimulus):
                J = alpha * np.dot(encoder, stimulus[int(i)]) + J_bias
                v += dt * (J - v) / Trc
        
        
    
    plt.plot(x, spikes[0], color='black')
    plt.plot(x, -1 * np.array(spikes[1]), color='black')
    
    plt.plot(x, voltages[0])
    plt.plot(x, -1 * np.array(voltages[1]))
    plt.plot(x, stimulus)
    plt.grid()
    plt.show()
    
    h = []
    time = np.linspace(0, T, N)
    Tau = 5 / 1000 # ms to s
    for t in time:
       h_t = exp(-t / Tau) / Tau
       h.append(h_t)
       
    h = h / np.sum(h)
       
    plt.plot(time, h)
    plt.show()
    
    A = np.zeros((2, N))

    for i in range(N):
        
        if spikes[0][i]:
            
            A[0, i:] += spikes[0][i] * np.array(h[0: len(A[0]) - i])
            
        if spikes[1][i]:
            
            A[1, i:] += spikes[1][i] * np.array(h[0: len(A[1]) - i])
            
            
    A = np.matrix(A)

    decoders = np.linalg.inv(A * A.T) * A * np.matrix(stimulus).T
    
    decoders = decoders.T
    
    x_hat = (decoders * A).T

    plt.plot(time, x_hat, label="Reconstructed Stimulus")    
    plt.plot(time, stimulus, label="Real Stimulus")
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Stimulus")
    plt.title("Real vs Reconstructed Stimulus")
    
    rmse = np.sqrt(np.mean(np.square(x_hat.T - stimulus)))
    
    print("RMSE: ", rmse)


    














