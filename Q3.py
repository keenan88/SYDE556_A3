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


if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(189)

    Tref = 2 / 1000 # Converted to seconds
    Trc = 20 / 1000 # Converted to seconds
    
    T = 1
    dt = 0.001
    N_time_samples = int(T / dt)
    time, stimulus, _, _ = generate_signal(T, dt, 1, 5, 200)
    
    N_neurons = 20
    S = 41
    range_of_stims = np.linspace(-2, 2, S)
    
    for N_neurons in [8, 16, 32, 64, 128, 256]:
        
        RMSEs = []
        
        for _ in range(5):
        
            tuning_curves, \
            alphas, \
            J_biases, \
            encoders = generate_LIF_tuning_curves(range_of_stims, 
                                                  Tref, 
                                                  Trc, 
                                                  N_neurons)
            
            spikes = []
            Vth = 1
            
            for encoder, alpha, J_bias in zip(encoders, alphas, J_biases):
                v = 0
                i = 0
                
                spikes.append([])
                
                while i < len(stimulus):
                    if v >= Vth:
                        v = 0
                        i += Tref * 1000 # Scaled to ms, since that is our step size here
                        spikes[-1].append(1)
                        spikes[-1].append(0)
                    else:
                        spikes[-1].append(0)
                        i += 1
                        
                    if i < len(stimulus):
                        J = alpha * np.dot(encoder, stimulus[int(i)]) + J_bias
                        v += dt * (J - v) / Trc
                
            
            h = []
            time = np.linspace(0, T, N_time_samples)
            Tau = 5 / 1000 # ms to s
            for t in time:
               h_t = exp(-t / Tau) / Tau
               h.append(h_t)
               
            h = h / np.sum(h)
            
            A = np.zeros((N_neurons, N_time_samples))
        
            for i in range(N_neurons):
                
                for j in range(N_time_samples):
                    
                    if spikes[i][j]:
                        
                        A[i, j:] += spikes[i][j] * np.array(h[0: len(A[i]) - j])
            
            A = np.matrix(A)
            # Becomes singular pretty fast. Do ridge regression?
            #decoders = np.linalg.inv(A * A.T) * A * np.matrix(stimulus).T 
            
            ro = 0.00000000001 * 200
            normalizer = N_neurons * ro * ro * np.eye(N_neurons)
            decoders = np.linalg.inv(A * A.T + normalizer) * A * np.matrix(stimulus).T 
            decoders = decoders.T
            
            x_hat = (decoders * A).T
        
            plt.plot(time, x_hat, label="Reconstructed Stimulus")    
            plt.plot(time, stimulus, label="Real Stimulus")
            plt.grid()
            plt.xlabel("Time")
            plt.ylabel("Stimulus")
            plt.title("Real vs Reconstructed Stimulus, " + str(N_neurons) + " Neurons")
            plt.show()
            
            rmse = np.sqrt(np.mean(np.square(x_hat.T - stimulus)))
            RMSEs.append(rmse)
            
            #print("RMSE: ", rmse)
            
        print("Avgd RMSE for " + str(N_neurons) + " = " + str(sum(RMSEs) / 5))

        













